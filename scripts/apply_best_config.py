#!/usr/bin/env python3
"""Apply configs/best_config.json for Parakeet TDT INT8 ONNX CPU inference.

Usage (from project root, venv active):
  python scripts/apply_best_config.py
  python scripts/apply_best_config.py --audio data/real_speech.wav
  python scripts/apply_best_config.py --benchmark --warmup 1 --repeats 3
  python scripts/apply_best_config.py --config configs/baseline.json

  # Multi-minute audio (bounds peak RAM; app-level chunk+concat, not streaming):
  python scripts/apply_best_config.py --config configs/production.json \\
    --audio long_talk.wav --chunk-window-s 30
  python scripts/apply_best_config.py --audio long_talk.wav \\
    --chunk-window-s 30 --chunk-overlap-s 2
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import soundfile as sf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "best_config.json"
DEFAULT_AUDIO = PROJECT_ROOT / "data" / "real_speech.wav"
MODEL_ID = "nemo-parakeet-tdt-0.6b-v3"


def load_config(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def apply_env(env: dict[str, str] | None) -> dict[str, str | None]:
    prev: dict[str, str | None] = {}
    for k, v in (env or {}).items():
        prev[k] = os.environ.get(k)
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[str(k)] = str(v)
    return prev


def restore_env(prev: dict[str, str | None]) -> None:
    for k, v in prev.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


def make_session_options(cfg: dict[str, Any]):
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.intra_op_num_threads = int(cfg.get("intra_op_num_threads", 8))
    so.inter_op_num_threads = int(cfg.get("inter_op_num_threads", 1))
    level = cfg.get("graph_optimization_level", "ORT_ENABLE_ALL")
    so.graph_optimization_level = getattr(
        ort.GraphOptimizationLevel, level, ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    )
    so.enable_mem_pattern = bool(cfg.get("enable_mem_pattern", True))
    so.enable_cpu_mem_arena = bool(cfg.get("enable_cpu_mem_arena", True))
    exec_mode = str(cfg.get("execution_mode", "ORT_SEQUENTIAL")).upper()
    if exec_mode in ("ORT_PARALLEL", "PARALLEL"):
        so.execution_mode = ort.ExecutionMode.ORT_PARALLEL
    else:
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    return so


def resolve_providers(cfg: dict[str, Any]) -> list[Any]:
    provider = cfg.get("provider", "CPUExecutionProvider")
    if provider == "OpenVINOExecutionProvider":
        opts = cfg.get("provider_options") or {"device_type": "CPU"}
        return [("OpenVINOExecutionProvider", opts), "CPUExecutionProvider"]
    return [provider]


def resolve_model_path(cfg: dict[str, Any]) -> Path:
    opt = cfg.get("optimized_model_dir")
    if opt:
        p = Path(opt)
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        if (p / "encoder-model.int8.onnx").exists():
            return p
    mp = Path(cfg.get("model_path", PROJECT_ROOT / "models" / "parakeet-tdt-0.6b-v3-onnx"))
    if not mp.is_absolute():
        mp = PROJECT_ROOT / mp
    return mp


def load_model(cfg: dict[str, Any]):
    import onnx_asr

    model_dir = resolve_model_path(cfg)
    so = make_session_options(cfg)
    providers = resolve_providers(cfg)
    model = onnx_asr.load_model(
        cfg.get("model_id", MODEL_ID),
        path=str(model_dir),
        quantization=cfg.get("quantization", "int8"),
        providers=providers,
        sess_options=so,
    )
    return model, model_dir


def audio_duration_s(path: Path) -> float:
    audio, sr = sf.read(str(path), always_2d=False)
    if hasattr(audio, "shape"):
        n = audio.shape[0]
    else:
        n = len(audio)
    return float(n / sr)


def recognize_chunked(model, wav_path: Path, chunking: dict[str, Any]) -> str:
    import numpy as np
    import soundfile as sf

    audio, sr = sf.read(str(wav_path), always_2d=False)
    if getattr(audio, "ndim", 1) > 1:
        audio = audio.mean(axis=1)
    audio = audio.astype("float32")
    window_s = float(chunking.get("window_s", 12.0))
    overlap_s = float(chunking.get("overlap_s", 1.0))
    win = max(1, int(window_s * sr))
    hop = max(1, int((window_s - overlap_s) * sr))
    if hop >= len(audio):
        out = model.recognize(str(wav_path))
        return out if isinstance(out, str) else str(out)

    tmp_dir = PROJECT_ROOT / ".tmp" / "chunk_audio_apply"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    parts: list[str] = []
    start = 0
    idx = 0
    try:
        while start < len(audio):
            end = min(len(audio), start + win)
            chunk = audio[start:end]
            if len(chunk) < int(0.4 * sr) and parts:
                break
            cpath = tmp_dir / f"chunk_{idx:03d}.wav"
            sf.write(str(cpath), chunk, sr)
            text = model.recognize(str(cpath))
            text = text if isinstance(text, str) else str(text)
            parts.append(text.strip())
            idx += 1
            if end >= len(audio):
                break
            start += hop
    finally:
        for p in tmp_dir.glob("chunk_*.wav"):
            try:
                p.unlink()
            except OSError:
                pass

    return merge_transcripts(parts)


def merge_transcripts(parts: list[str]) -> str:
    cleaned = [p.strip() for p in parts if p and p.strip()]
    if not cleaned:
        return ""
    out = cleaned[0]
    for nxt in cleaned[1:]:
        a_toks = out.split()
        b_toks = nxt.split()
        max_k = min(12, len(a_toks), len(b_toks))
        overlap = 0
        for k in range(max_k, 0, -1):
            if [t.lower() for t in a_toks[-k:]] == [t.lower() for t in b_toks[:k]]:
                overlap = k
                break
        if overlap:
            out = (out + " " + " ".join(b_toks[overlap:])).strip()
        else:
            out = (out + " " + nxt).strip()
    return out


def resolve_chunking(cfg: dict[str, Any], args: argparse.Namespace) -> dict[str, Any] | None:
    """Merge config chunking with CLI overrides.

    Production configs leave chunking null (full-file path — best RTF on short clips).
    For multi-minute audio, full-file encoder activations can OOM; use --chunk-window-s.
    """
    ch = cfg.get("chunking")
    if isinstance(ch, dict):
        out = dict(ch)
    else:
        out = {}

    if args.no_chunk:
        return None

    if args.chunk_window_s is not None:
        out["enabled"] = True
        out["window_s"] = float(args.chunk_window_s)
        if args.chunk_overlap_s is not None:
            out["overlap_s"] = float(args.chunk_overlap_s)
        else:
            out.setdefault("overlap_s", 2.0)
        return out

    if args.chunk_overlap_s is not None and out.get("enabled"):
        out["overlap_s"] = float(args.chunk_overlap_s)

    if out.get("enabled"):
        out.setdefault("window_s", 30.0)
        out.setdefault("overlap_s", 2.0)
        return out
    return None


def recognize_once(model, audio: Path, cfg: dict[str, Any]) -> str:
    ch = cfg.get("chunking")
    if ch and ch.get("enabled"):
        return recognize_chunked(model, audio, ch)
    out = model.recognize(str(audio))
    return out if isinstance(out, str) else str(out)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Apply best_config.json for Parakeet CPU inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Long audio: full-file recognize loads the whole utterance into the encoder and can\n"
            "OOM on multi-minute files (e.g. ~16GB+ then kill). Use --chunk-window-s 30 (or 15–60).\n"
            "Chunking is app-level window+concat — not true streaming; may add boundary artifacts\n"
            "and is not the frozen production RTF path."
        ),
    )
    p.add_argument("--config", type=str, default=str(DEFAULT_CONFIG))
    p.add_argument("--audio", type=str, default=str(DEFAULT_AUDIO))
    p.add_argument("--benchmark", action="store_true", help="Time warmup+repeats and print RTF")
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument(
        "--chunk-window-s",
        type=float,
        default=None,
        metavar="SEC",
        help="Enable app-level chunking with this window (seconds). "
        "Use for multi-minute audio to bound peak RAM (e.g. 30).",
    )
    p.add_argument(
        "--chunk-overlap-s",
        type=float,
        default=None,
        metavar="SEC",
        help="Chunk overlap in seconds (default 2.0 when --chunk-window-s is set).",
    )
    p.add_argument(
        "--no-chunk",
        action="store_true",
        help="Force full-file recognize even if config enables chunking.",
    )
    args = p.parse_args(argv)

    os.chdir(PROJECT_ROOT)
    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = PROJECT_ROOT / cfg_path
    if not cfg_path.exists():
        print(f"ERROR: config not found: {cfg_path}", file=sys.stderr)
        print("Run: python scripts/autoresearch_cpu_opts.py", file=sys.stderr)
        return 1

    audio = Path(args.audio)
    if not audio.is_absolute():
        audio = PROJECT_ROOT / audio
    if not audio.exists():
        print(f"ERROR: audio not found: {audio}", file=sys.stderr)
        return 1

    cfg = load_config(cfg_path)
    chunking = resolve_chunking(cfg, args)
    # Effective runtime config (do not mutate frozen production JSON on disk)
    run_cfg = dict(cfg)
    run_cfg["chunking"] = chunking

    print(f"Config: {cfg_path}", flush=True)
    print(f"Audio:  {audio}", flush=True)
    if chunking and chunking.get("enabled"):
        print(
            f"Chunking: window_s={chunking.get('window_s')} "
            f"overlap_s={chunking.get('overlap_s')} (app-level; bounds peak RAM)",
            flush=True,
        )
    else:
        print("Chunking: off (full-file; multi-minute audio may OOM)", flush=True)

    prev = apply_env(run_cfg.get("env") or {})
    try:
        model, model_dir = load_model(run_cfg)
        print(f"Model:  {model_dir}", flush=True)
        print(
            f"Threads intra={run_cfg.get('intra_op_num_threads')} inter={run_cfg.get('inter_op_num_threads')} "
            f"provider={run_cfg.get('provider')}",
            flush=True,
        )

        if args.benchmark:
            for _ in range(args.warmup):
                _ = recognize_once(model, audio, run_cfg)
            lats: list[float] = []
            transcript = ""
            for _ in range(args.repeats):
                t0 = time.perf_counter()
                transcript = recognize_once(model, audio, run_cfg)
                lats.append(time.perf_counter() - t0)
            dur = audio_duration_s(audio)
            mean_lat = statistics.mean(lats)
            rtf = mean_lat / dur if dur > 0 else float("inf")
            rtfx = dur / mean_lat if mean_lat > 0 else 0.0
            print(f"latency_mean_s={mean_lat:.4f} rtf={rtf:.6f} rtfx={rtfx:.2f}", flush=True)
            print(f"latencies_s={lats}", flush=True)
        else:
            transcript = recognize_once(model, audio, run_cfg)

        print("--- transcript ---", flush=True)
        print(transcript, flush=True)
        if not (transcript and str(transcript).strip()):
            print("ERROR: empty transcript", file=sys.stderr)
            return 2
        return 0
    finally:
        restore_env(prev)


if __name__ == "__main__":
    raise SystemExit(main())

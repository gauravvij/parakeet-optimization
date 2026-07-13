#!/usr/bin/env python3
"""Portable CPU autoresearch ladder for nvidia/parakeet-tdt-0.6b-v3 INT8 ONNX.

Sequential keep/discard experiments (E0–E6). Keep only configs that improve the
primary metric (geometric mean of mean RTF on medium_15s + long_30s) by ≥5%
with no quality regression on real_speech.

Usage (from project root, venv active):
  python scripts/autoresearch_cpu_opts.py
  python scripts/autoresearch_cpu_opts.py --warmup 1 --repeats 3
  python scripts/autoresearch_cpu_opts.py --skip-openvino --skip-chunking
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import platform
import re
import resource
import shutil
import statistics
import sys
import time
import traceback
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
import psutil
import soundfile as sf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_DIR = PROJECT_ROOT / "models" / "parakeet-tdt-0.6b-v3-onnx"
DEFAULT_OPT_MODEL_DIR = PROJECT_ROOT / "models" / "parakeet-tdt-0.6b-v3-onnx-opt"
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "results" / "autoresearch"
DEFAULT_CONFIGS_DIR = PROJECT_ROOT / "configs"

AUDIO_FILES = {
    "short_5s": "short_5s.wav",
    "medium_15s": "medium_15s.wav",
    "long_30s": "long_30s.wav",
    "real_speech": "real_speech.wav",
}
PRIMARY_KEYS = ("medium_15s", "long_30s")
KEEP_THRESHOLD = 0.05  # ≥5% RTF reduction
MODEL_ID = "nemo-parakeet-tdt-0.6b-v3"


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def peak_rss_mb() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return usage / (1024 * 1024)
    return usage / 1024.0


def current_rss_mb() -> float:
    return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)


def load_wav(path: Path) -> tuple[np.ndarray, int, float]:
    audio, sr = sf.read(str(path), always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float32)
    dur = float(len(audio) / sr)
    return audio, int(sr), dur


def host_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "cpu_count_logical": os.cpu_count(),
        "onnxruntime": ort.__version__,
        "providers": ort.get_available_providers(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as f:
            lines = f.readlines()
        model = next(
            (ln.split(":", 1)[1].strip() for ln in lines if ln.startswith("model name")),
            None,
        )
        flags_line = next(
            (ln.split(":", 1)[1].strip() for ln in lines if ln.startswith("flags")),
            "",
        )
        flag_set = set(flags_line.split())
        interesting = [
            c
            for c in (
                "avx",
                "avx2",
                "avx512f",
                "avx512dq",
                "avx512bw",
                "avx512vl",
                "avx512_vnni",
                "avx512_bf16",
                "avx512vnni",
                "avx512bf16",
            )
            if c in flag_set
        ]
        info["cpu_model"] = model
        info["cpu_flags_relevant"] = sorted(set(interesting))
    except OSError:
        info["cpu_model"] = platform.processor()
        info["cpu_flags_relevant"] = []
    try:
        vm = psutil.virtual_memory()
        info["ram_total_gb"] = round(vm.total / (1024**3), 2)
        info["ram_available_gb"] = round(vm.available / (1024**3), 2)
    except Exception:
        pass
    return info


def normalize_text(text: str) -> str:
    t = (text or "").lower().strip()
    t = re.sub(r"[^a-z0-9\s']", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def token_overlap(a: str, b: str) -> float:
    ta = set(normalize_text(a).split())
    tb = set(normalize_text(b).split())
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / max(len(ta), len(tb))


def geometric_mean(values: list[float]) -> float:
    vals = [v for v in values if v is not None and v > 0]
    if not vals:
        return float("inf")
    return math.exp(sum(math.log(v) for v in vals) / len(vals))


def primary_rtf(per_audio: dict[str, dict[str, Any]]) -> float:
    rtfs = []
    for k in PRIMARY_KEYS:
        m = per_audio.get(k)
        if m and "rtf" in m and m["rtf"] is not None:
            rtfs.append(float(m["rtf"]))
    return geometric_mean(rtfs)


# ---------------------------------------------------------------------------
# Session / model loading
# ---------------------------------------------------------------------------

def make_session_options(cfg: dict[str, Any]) -> ort.SessionOptions:
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


def apply_env(env: dict[str, str] | None) -> dict[str, str | None]:
    """Set env vars; return previous values for restore."""
    prev: dict[str, str | None] = {}
    env = env or {}
    for k, v in env.items():
        prev[k] = os.environ.get(k)
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = str(v)
    return prev


def restore_env(prev: dict[str, str | None]) -> None:
    for k, v in prev.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


def resolve_providers(cfg: dict[str, Any]) -> list[Any]:
    provider = cfg.get("provider", "CPUExecutionProvider")
    if provider == "OpenVINOExecutionProvider":
        opts = cfg.get("provider_options") or {"device_type": "CPU"}
        return [("OpenVINOExecutionProvider", opts), "CPUExecutionProvider"]
    return [provider]


def load_asr_model(cfg: dict[str, Any]):
    import onnx_asr

    model_dir = Path(cfg.get("model_path", DEFAULT_MODEL_DIR))
    so = make_session_options(cfg)
    providers = resolve_providers(cfg)
    model = onnx_asr.load_model(
        MODEL_ID,
        path=str(model_dir),
        quantization=cfg.get("quantization", "int8"),
        providers=providers,
        sess_options=so,
    )
    return model, so


def unload_model(model: Any) -> None:
    try:
        del model
    except Exception:
        pass
    gc.collect()


# ---------------------------------------------------------------------------
# Timing / quality
# ---------------------------------------------------------------------------

def timed_recognize(
    model,
    wav_path: Path,
    repeats: int,
    warmup: int,
    chunking: dict[str, Any] | None = None,
) -> dict[str, Any]:
    def _recognize_once() -> str:
        if chunking and chunking.get("enabled"):
            return recognize_chunked(model, wav_path, chunking)
        out = model.recognize(str(wav_path))
        return out if isinstance(out, str) else str(out)

    for _ in range(warmup):
        _ = _recognize_once()

    latencies: list[float] = []
    transcript = ""
    rss_before = current_rss_mb()
    for _ in range(repeats):
        t0 = time.perf_counter()
        transcript = _recognize_once()
        t1 = time.perf_counter()
        latencies.append(t1 - t0)

    _, _, audio_dur = load_wav(wav_path)
    mean_lat = statistics.mean(latencies)
    std_lat = statistics.pstdev(latencies) if len(latencies) > 1 else 0.0
    rtf = mean_lat / audio_dur if audio_dur > 0 else float("inf")
    rtfx = audio_dur / mean_lat if mean_lat > 0 else 0.0
    return {
        "audio_file": wav_path.name,
        "audio_duration_s": round(audio_dur, 4),
        "warmup": warmup,
        "repeats": repeats,
        "latencies_s": [round(x, 6) for x in latencies],
        "latency_mean_s": round(mean_lat, 6),
        "latency_std_s": round(std_lat, 6),
        "latency_min_s": round(min(latencies), 6),
        "latency_max_s": round(max(latencies), 6),
        "rtf": round(rtf, 6),
        "rtfx": round(rtfx, 4),
        "transcript": transcript,
        "transcript_nonempty": bool(transcript and str(transcript).strip()),
        "rss_before_mb": round(rss_before, 2),
        "rss_after_mb": round(current_rss_mb(), 2),
        "peak_rss_mb": round(peak_rss_mb(), 2),
        "chunking": bool(chunking and chunking.get("enabled")),
    }


def recognize_chunked(model, wav_path: Path, chunking: dict[str, Any]) -> str:
    """App-level chunk+concat for long audio throughput experiment.

    Note: this is NOT true streaming with encoder cache. Overlap is used only
    to reduce boundary artifacts; transcripts are concatenated with a simple
    overlap-dedup heuristic.
    """
    audio, sr, dur = load_wav(wav_path)
    window_s = float(chunking.get("window_s", 12.0))
    overlap_s = float(chunking.get("overlap_s", 1.0))
    win = max(1, int(window_s * sr))
    hop = max(1, int((window_s - overlap_s) * sr))
    if hop >= len(audio):
        out = model.recognize(str(wav_path))
        return out if isinstance(out, str) else str(out)

    tmp_dir = PROJECT_ROOT / ".tmp" / "chunk_audio"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    parts: list[str] = []
    start = 0
    idx = 0
    try:
        while start < len(audio):
            end = min(len(audio), start + win)
            chunk = audio[start:end]
            # skip tiny trailing fragments
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
        # best-effort cleanup of temp chunks
        for p in tmp_dir.glob("chunk_*.wav"):
            try:
                p.unlink()
            except OSError:
                pass

    return merge_transcripts(parts)


def merge_transcripts(parts: list[str]) -> str:
    """Concatenate chunk transcripts with light overlap dedup on trailing/leading words."""
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


def quality_ok(transcript: str, baseline_transcript: str | None) -> tuple[bool, str]:
    if not transcript or not str(transcript).strip():
        return False, "empty real_speech transcript"
    if not baseline_transcript:
        return True, "no baseline transcript yet (E0)"
    if normalize_text(transcript) == normalize_text(baseline_transcript):
        return True, "exact normalized match"
    ov = token_overlap(transcript, baseline_transcript)
    if ov >= 0.85:
        return True, f"high token overlap={ov:.3f}"
    return False, f"quality regression overlap={ov:.3f}"


# ---------------------------------------------------------------------------
# Offline ORT graph optimize (E3)
# ---------------------------------------------------------------------------

def optimize_onnx_models(src_dir: Path, dst_dir: Path, threads: int = 8) -> dict[str, Any]:
    """Load each ONNX with optimized_model_filepath and save optimized copies."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    # copy non-onnx assets
    for name in ("config.json", "vocab.txt"):
        src = src_dir / name
        if src.exists():
            shutil.copy2(src, dst_dir / name)

    files = [
        "nemo128.onnx",
        "encoder-model.int8.onnx",
        "decoder_joint-model.int8.onnx",
    ]
    report: dict[str, Any] = {"dst": str(dst_dir), "files": {}}
    for fname in files:
        src = src_dir / fname
        if not src.exists():
            report["files"][fname] = {"status": "missing"}
            continue
        dst = dst_dir / fname
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = 1
        so.optimized_model_filepath = str(dst)
        t0 = time.perf_counter()
        try:
            _ = ort.InferenceSession(
                str(src), so, providers=["CPUExecutionProvider"]
            )
            elapsed = time.perf_counter() - t0
            src_sz = src.stat().st_size
            dst_sz = dst.stat().st_size if dst.exists() else 0
            report["files"][fname] = {
                "status": "ok",
                "seconds": round(elapsed, 3),
                "src_mb": round(src_sz / 1e6, 2),
                "dst_mb": round(dst_sz / 1e6, 2),
            }
            del _
            gc.collect()
        except Exception as e:
            report["files"][fname] = {
                "status": "error",
                "error": str(e),
                "traceback": traceback.format_exc(),
            }
            # fallback: copy original so dir remains loadable
            if not dst.exists():
                shutil.copy2(src, dst)
    return report


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------

def default_config() -> dict[str, Any]:
    return {
        "model_path": str(DEFAULT_MODEL_DIR),
        "quantization": "int8",
        "provider": "CPUExecutionProvider",
        "provider_options": None,
        "intra_op_num_threads": 8,
        "inter_op_num_threads": 1,
        "graph_optimization_level": "ORT_ENABLE_ALL",
        "enable_mem_pattern": True,
        "enable_cpu_mem_arena": True,
        "execution_mode": "ORT_SEQUENTIAL",
        "env": {
            "OMP_NUM_THREADS": "8",
            "OMP_WAIT_POLICY": "PASSIVE",
        },
        "chunking": None,
        "optimized_model_dir": None,
    }


def _portable_path(p: Any) -> Any:
    """Prefer project-relative paths in emitted configs for other developers."""
    if not p:
        return p
    try:
        path = Path(str(p))
        if path.is_absolute():
            return str(path.resolve().relative_to(PROJECT_ROOT.resolve()))
        return str(path)
    except Exception:
        return str(p) if p is not None else p


def config_to_public(cfg: dict[str, Any]) -> dict[str, Any]:
    """Serialize config for best_config.json / ledger (portable)."""
    out = {
        "model_path": _portable_path(cfg.get("model_path")),
        "quantization": cfg.get("quantization", "int8"),
        "provider": cfg.get("provider", "CPUExecutionProvider"),
        "provider_options": cfg.get("provider_options"),
        "intra_op_num_threads": cfg.get("intra_op_num_threads", 8),
        "inter_op_num_threads": cfg.get("inter_op_num_threads", 1),
        "graph_optimization_level": cfg.get("graph_optimization_level", "ORT_ENABLE_ALL"),
        "enable_mem_pattern": cfg.get("enable_mem_pattern", True),
        "enable_cpu_mem_arena": cfg.get("enable_cpu_mem_arena", True),
        "execution_mode": cfg.get("execution_mode", "ORT_SEQUENTIAL"),
        "env": cfg.get("env") or {},
        "chunking": cfg.get("chunking"),
        "optimized_model_dir": _portable_path(cfg.get("optimized_model_dir")),
    }
    return out


def run_config(
    name: str,
    cfg: dict[str, Any],
    data_dir: Path,
    audio_keys: list[str],
    warmup: int,
    repeats: int,
    baseline_transcript: str | None,
    reference_primary_rtf: float | None,
) -> dict[str, Any]:
    """Load model with cfg, time all audio keys, apply keep/discard gates."""
    print(f"\n=== {name} ===", flush=True)
    print(f"  params: {json.dumps(config_to_public(cfg), sort_keys=True)}", flush=True)

    prev_env = apply_env(cfg.get("env") or {})
    per_audio: dict[str, Any] = {}
    error: str | None = None
    model = None
    t_load0 = time.perf_counter()
    try:
        model, _ = load_asr_model(cfg)
        load_s = time.perf_counter() - t_load0
        print(f"  model loaded in {load_s:.2f}s", flush=True)
        for key in audio_keys:
            wav = data_dir / AUDIO_FILES[key]
            if not wav.exists():
                per_audio[key] = {"error": f"missing {wav}"}
                print(f"  SKIP missing {wav}", flush=True)
                continue
            # chunking only for long_30s when enabled
            chunking = None
            ch = cfg.get("chunking")
            if ch and ch.get("enabled") and key == "long_30s":
                chunking = ch
            print(f"  timing {key} ...", flush=True)
            try:
                metrics = timed_recognize(
                    model, wav, repeats=repeats, warmup=warmup, chunking=chunking
                )
                metrics["audio_key"] = key
                per_audio[key] = metrics
                print(
                    f"    lat={metrics['latency_mean_s']:.3f}s RTF={metrics['rtf']:.4f} "
                    f"RTFx={metrics['rtfx']:.2f} text={metrics['transcript'][:70]!r}",
                    flush=True,
                )
            except Exception as e:
                per_audio[key] = {
                    "audio_key": key,
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                }
                print(f"    ERROR {key}: {e}", flush=True)
    except Exception as e:
        error = str(e)
        print(f"  LOAD/RUN ERROR: {e}", flush=True)
        traceback.print_exc()
    finally:
        if model is not None:
            unload_model(model)
        restore_env(prev_env)

    prim = primary_rtf(per_audio) if not error else float("inf")
    improvement = None
    if reference_primary_rtf is not None and reference_primary_rtf > 0 and math.isfinite(prim):
        improvement = (reference_primary_rtf - prim) / reference_primary_rtf

    rs = per_audio.get("real_speech") or {}
    transcript = rs.get("transcript", "") if isinstance(rs, dict) else ""
    q_ok, q_reason = quality_ok(transcript, baseline_transcript)

    keep = False
    reason_parts: list[str] = []
    if error:
        reason_parts.append(f"error: {error}")
    elif any(isinstance(v, dict) and "error" in v for v in per_audio.values()):
        reason_parts.append("per-audio error(s)")
    elif not math.isfinite(prim):
        reason_parts.append("invalid primary RTF")
    elif not q_ok:
        reason_parts.append(f"quality fail: {q_reason}")
    elif improvement is None:
        # E0 baseline — always "keep" as reference
        keep = True
        reason_parts.append("baseline reference")
    elif improvement >= KEEP_THRESHOLD:
        keep = True
        reason_parts.append(
            f"primary RTF improved {improvement*100:.2f}% (≥{KEEP_THRESHOLD*100:.0f}%)"
        )
        reason_parts.append(f"quality: {q_reason}")
    else:
        reason_parts.append(
            f"improvement {0.0 if improvement is None else improvement*100:.2f}% < {KEEP_THRESHOLD*100:.0f}%"
        )
        reason_parts.append(f"quality: {q_reason}")

    record = {
        "name": name,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "params": config_to_public(cfg),
        "primary_rtf": None if not math.isfinite(prim) else round(prim, 6),
        "reference_primary_rtf": (
            None if reference_primary_rtf is None else round(reference_primary_rtf, 6)
        ),
        "improvement_pct": (
            None if improvement is None else round(improvement * 100.0, 3)
        ),
        "keep": keep,
        "reason": "; ".join(reason_parts),
        "quality_ok": q_ok,
        "quality_reason": q_reason,
        "transcripts": {
            k: (v.get("transcript") if isinstance(v, dict) else None)
            for k, v in per_audio.items()
        },
        "metrics": {
            k: {
                kk: vv
                for kk, vv in v.items()
                if kk not in ("traceback",)
            }
            if isinstance(v, dict)
            else v
            for k, v in per_audio.items()
        },
        "error": error,
        "peak_rss_mb": round(peak_rss_mb(), 2),
    }
    print(
        f"  primary_rtf={record['primary_rtf']} improvement_pct={record['improvement_pct']} "
        f"keep={keep} reason={record['reason']}",
        flush=True,
    )
    return record


def append_ledger(ledger_path: Path, record: dict[str, Any]) -> None:
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    with open(ledger_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_summary(
    summary_path: Path,
    host: dict[str, Any],
    records: list[dict[str, Any]],
    e0_rtf: float | None,
    best_cfg: dict[str, Any],
    kept: list[str],
) -> None:
    lines = [
        "# Autoresearch CPU Optimization Ladder — Summary",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Host",
        "",
        f"- CPU: {host.get('cpu_model')}",
        f"- Logical CPUs: {host.get('cpu_count_logical')}",
        f"- Flags: {', '.join(host.get('cpu_flags_relevant') or [])}",
        f"- ORT: {host.get('onnxruntime')} providers={host.get('providers')}",
        f"- RAM: {host.get('ram_total_gb')} GB",
        "",
        "## Protocol",
        "",
        "- Model: INT8 ONNX `nemo-parakeet-tdt-0.6b-v3` via onnx-asr",
        "- Primary metric: geometric mean of mean RTF on `medium_15s` + `long_30s` (lower better)",
        f"- Keep gate: ≥{KEEP_THRESHOLD*100:.0f}% primary RTF improvement vs rolling best (E0 for first keeps; stack vs E0)",
        "- Quality gate: non-empty `real_speech` transcript; normalized match or ≥85% token overlap vs E0",
        "",
        "## Results",
        "",
        "| Experiment | primary_rtf | improvement_pct | keep | reason |",
        "|---|---:|---:|:---:|---|",
    ]
    for r in records:
        lines.append(
            f"| {r['name']} | {r.get('primary_rtf')} | {r.get('improvement_pct')} | "
            f"{'YES' if r.get('keep') else 'no'} | {r.get('reason', '')} |"
        )

    lines.extend(
        [
            "",
            "## Kept winners",
            "",
        ]
    )
    if kept:
        for k in kept:
            lines.append(f"- `{k}`")
    else:
        lines.append("- *(none — no experiment cleared the ≥5% gate with quality OK)*")

    lines.extend(
        [
            "",
            "## Best config",
            "",
            "```json",
            json.dumps(best_cfg, indent=2),
            "```",
            "",
            "## Notes",
            "",
        ]
    )
    e6 = next((r for r in records if r["name"].startswith("E6")), None)
    if e6 and e0_rtf and e6.get("primary_rtf"):
        imp = (e0_rtf - e6["primary_rtf"]) / e0_rtf * 100.0
        lines.append(
            f"- E6 stacked best vs E0: primary_rtf {e6['primary_rtf']} "
            f"(E0={e0_rtf:.6f}, improvement={imp:.2f}%)"
        )
        if kept and imp < KEEP_THRESHOLD * 100:
            lines.append(
                "- Stack did not add ≥5% vs E0 even though individual keeps existed "
                "(possible non-additive interactions or measurement noise)."
            )
        elif not kept:
            lines.append("- No individual keeps; E6 equals baseline stack.")
    elif not kept:
        lines.append("- No keeps; best_config is the E0 baseline.")
    lines.append("")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text("\n".join(lines), encoding="utf-8")


def try_openvino_available() -> tuple[bool, str]:
    providers = ort.get_available_providers()
    if "OpenVINOExecutionProvider" in providers:
        return True, "already available in onnxruntime providers"
    # Optional quick install — only if disk allows (~1GB+ free recommended)
    try:
        free_gb = shutil.disk_usage(PROJECT_ROOT).free / (1024**3)
    except Exception:
        free_gb = 0.0
    if free_gb < 2.0:
        return False, f"skip install: only {free_gb:.1f} GB free disk"
    # Do not force install onnxruntime-openvino (often conflicts with existing ORT wheel).
    # Document skip cleanly.
    return False, (
        f"OpenVINO EP not in providers={providers}; "
        "skipping optional install to avoid replacing existing onnxruntime CPU wheel"
    )


# ---------------------------------------------------------------------------
# Ladder
# ---------------------------------------------------------------------------

def run_ladder(args: argparse.Namespace) -> int:
    model_dir = Path(args.model_dir)
    data_dir = Path(args.data_dir)
    results_dir = Path(args.results_dir)
    configs_dir = Path(args.configs_dir)
    opt_dir = Path(args.opt_model_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    configs_dir.mkdir(parents=True, exist_ok=True)

    ledger_path = results_dir / "ledger.jsonl"
    summary_path = results_dir / "summary.md"
    # fresh ledger each full run
    if ledger_path.exists() and not args.append_ledger:
        ledger_path.unlink()

    host = host_info()
    print("Host:", json.dumps(host, indent=2), flush=True)

    audio_keys = ["short_5s", "medium_15s", "long_30s", "real_speech"]
    warmup = args.warmup
    repeats = args.repeats

    records: list[dict[str, Any]] = []
    kept_names: list[str] = []
    kept_deltas: list[dict[str, Any]] = []  # partial config overlays that won

    # ---- E0 baseline (also confirm 4 vs 8 threads) ----
    base = default_config()
    base["model_path"] = str(model_dir)
    base["intra_op_num_threads"] = 8
    base["env"] = {"OMP_NUM_THREADS": "8", "OMP_WAIT_POLICY": "PASSIVE"}

    # Write baseline.json early
    baseline_public = config_to_public(base)
    (configs_dir / "baseline.json").write_text(
        json.dumps(baseline_public, indent=2) + "\n", encoding="utf-8"
    )

    # Thread confirmation: 4 vs 8 (quick, only primary + real_speech for 4)
    e0_8 = run_config(
        "E0_baseline_threads8",
        base,
        data_dir,
        audio_keys,
        warmup,
        repeats,
        baseline_transcript=None,
        reference_primary_rtf=None,
    )
    records.append(e0_8)
    append_ledger(ledger_path, e0_8)

    base4 = deepcopy(base)
    base4["intra_op_num_threads"] = 4
    base4["env"] = {"OMP_NUM_THREADS": "4", "OMP_WAIT_POLICY": "PASSIVE"}
    e0_4 = run_config(
        "E0_confirm_threads4",
        base4,
        data_dir,
        audio_keys,
        warmup,
        repeats,
        baseline_transcript=e0_8.get("transcripts", {}).get("real_speech"),
        reference_primary_rtf=e0_8.get("primary_rtf"),
    )
    # E0_confirm is informational; keep only if better by ≥5% (unlikely vs 8)
    records.append(e0_4)
    append_ledger(ledger_path, e0_4)

    # Choose better of 4/8 as E0 reference
    if (
        e0_4.get("keep")
        and e0_4.get("primary_rtf") is not None
        and e0_8.get("primary_rtf") is not None
        and e0_4["primary_rtf"] < e0_8["primary_rtf"]
    ):
        e0 = e0_4
        best_cfg = deepcopy(base4)
        print("  E0 winner: threads=4", flush=True)
    else:
        e0 = e0_8
        best_cfg = deepcopy(base)
        # mark e0_4 as not the baseline keep for stack purposes
        print("  E0 winner: threads=8", flush=True)

    e0_rtf = e0.get("primary_rtf")
    baseline_transcript = e0.get("transcripts", {}).get("real_speech") or ""
    rolling_best_rtf = e0_rtf
    rolling_cfg = deepcopy(best_cfg)

    def consider(record: dict[str, Any], cfg: dict[str, Any], overlay: dict[str, Any]) -> None:
        nonlocal rolling_best_rtf, rolling_cfg
        records.append(record)
        append_ledger(ledger_path, record)
        if record.get("keep") and record["name"] not in (
            "E0_baseline_threads8",
            "E0_confirm_threads4",
        ):
            # For non-baseline: keep means beat reference (rolling or E0 depending)
            kept_names.append(record["name"])
            kept_deltas.append({"name": record["name"], "overlay": overlay, "cfg": deepcopy(cfg)})
            if record.get("primary_rtf") is not None:
                rolling_best_rtf = record["primary_rtf"]
                rolling_cfg = deepcopy(cfg)

    # ---- E1 thread/env matrix ----
    e1_candidates = [
        (
            "E1_intra2_omp2_passive",
            {
                "intra_op_num_threads": 2,
                "inter_op_num_threads": 1,
                "env": {"OMP_NUM_THREADS": "2", "OMP_WAIT_POLICY": "PASSIVE"},
            },
        ),
        (
            "E1_intra4_omp4_passive",
            {
                "intra_op_num_threads": 4,
                "inter_op_num_threads": 1,
                "env": {"OMP_NUM_THREADS": "4", "OMP_WAIT_POLICY": "PASSIVE"},
            },
        ),
        (
            "E1_intra8_omp8_active",
            {
                "intra_op_num_threads": 8,
                "inter_op_num_threads": 1,
                "env": {"OMP_NUM_THREADS": "8", "OMP_WAIT_POLICY": "ACTIVE"},
            },
        ),
        (
            "E1_intra8_omp8_passive",
            {
                "intra_op_num_threads": 8,
                "inter_op_num_threads": 1,
                "env": {"OMP_NUM_THREADS": "8", "OMP_WAIT_POLICY": "PASSIVE"},
            },
        ),
        (
            "E1_intra8_omp8_active_kmp_affinity",
            {
                "intra_op_num_threads": 8,
                "inter_op_num_threads": 1,
                "env": {
                    "OMP_NUM_THREADS": "8",
                    "OMP_WAIT_POLICY": "ACTIVE",
                    "KMP_AFFINITY": "granularity=fine,compact,1,0",
                },
            },
        ),
    ]
    for name, overlay in e1_candidates:
        cfg = deepcopy(rolling_cfg)
        cfg.update({k: v for k, v in overlay.items() if k != "env"})
        if "env" in overlay:
            cfg["env"] = dict(overlay["env"])
        # skip exact duplicate of current rolling
        if config_to_public(cfg) == config_to_public(rolling_cfg) and name.endswith("passive"):
            # still record a quick note? skip pure duplicate of E0
            if name == "E1_intra8_omp8_passive":
                print(f"\n=== {name} SKIP (duplicate of rolling best) ===", flush=True)
                rec = {
                    "name": name,
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "params": config_to_public(cfg),
                    "primary_rtf": rolling_best_rtf,
                    "reference_primary_rtf": rolling_best_rtf,
                    "improvement_pct": 0.0,
                    "keep": False,
                    "reason": "duplicate of rolling best; skipped remeasure",
                    "quality_ok": True,
                    "quality_reason": "n/a",
                    "transcripts": {},
                    "metrics": {},
                    "error": None,
                }
                records.append(rec)
                append_ledger(ledger_path, rec)
                continue
        rec = run_config(
            name,
            cfg,
            data_dir,
            audio_keys,
            warmup,
            repeats,
            baseline_transcript=baseline_transcript,
            reference_primary_rtf=rolling_best_rtf,
        )
        consider(rec, cfg, overlay)

    # ---- E2 session/memory opts ----
    e2_candidates = [
        (
            "E2_mem_pattern_off",
            {"enable_mem_pattern": False},
        ),
        (
            "E2_cpu_arena_off",
            {"enable_cpu_mem_arena": False},
        ),
        (
            "E2_mem_pattern_off_arena_off",
            {"enable_mem_pattern": False, "enable_cpu_mem_arena": False},
        ),
        (
            "E2_execution_parallel",
            {"execution_mode": "ORT_PARALLEL", "inter_op_num_threads": 2},
        ),
    ]
    for name, overlay in e2_candidates:
        cfg = deepcopy(rolling_cfg)
        cfg.update(overlay)
        rec = run_config(
            name,
            cfg,
            data_dir,
            audio_keys,
            warmup,
            repeats,
            baseline_transcript=baseline_transcript,
            reference_primary_rtf=rolling_best_rtf,
        )
        consider(rec, cfg, overlay)

    # ---- E3 offline ORT optimize ----
    print("\n=== E3 offline ORT graph optimize (build) ===", flush=True)
    opt_report = optimize_onnx_models(model_dir, opt_dir, threads=int(rolling_cfg.get("intra_op_num_threads", 8)))
    (results_dir / "e3_optimize_report.json").write_text(
        json.dumps(opt_report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(opt_report, indent=2), flush=True)
    opt_ok = all(
        (info.get("status") in ("ok",) or (opt_dir / fname).exists())
        for fname, info in opt_report.get("files", {}).items()
    )
    if opt_ok and (opt_dir / "encoder-model.int8.onnx").exists():
        cfg = deepcopy(rolling_cfg)
        cfg["model_path"] = str(opt_dir)
        cfg["optimized_model_dir"] = str(opt_dir)
        rec = run_config(
            "E3_optimized_onnx_dir",
            cfg,
            data_dir,
            audio_keys,
            warmup,
            repeats,
            baseline_transcript=baseline_transcript,
            reference_primary_rtf=rolling_best_rtf,
        )
        consider(rec, cfg, {"model_path": str(opt_dir), "optimized_model_dir": str(opt_dir)})
    else:
        rec = {
            "name": "E3_optimized_onnx_dir",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "params": {},
            "primary_rtf": None,
            "reference_primary_rtf": rolling_best_rtf,
            "improvement_pct": None,
            "keep": False,
            "reason": f"optimize failed or incomplete: {opt_report}",
            "quality_ok": False,
            "quality_reason": "n/a",
            "transcripts": {},
            "metrics": {},
            "error": "optimize incomplete",
        }
        records.append(rec)
        append_ledger(ledger_path, rec)

    # ---- E4 optional OpenVINO ----
    if args.skip_openvino:
        ov_ok, ov_msg = False, "skipped by --skip-openvino"
    else:
        ov_ok, ov_msg = try_openvino_available()
    print(f"\n=== E4 OpenVINO: available={ov_ok} ({ov_msg}) ===", flush=True)
    if ov_ok:
        cfg = deepcopy(rolling_cfg)
        cfg["provider"] = "OpenVINOExecutionProvider"
        cfg["provider_options"] = {"device_type": "CPU"}
        rec = run_config(
            "E4_openvino_cpu",
            cfg,
            data_dir,
            audio_keys,
            warmup,
            repeats,
            baseline_transcript=baseline_transcript,
            reference_primary_rtf=rolling_best_rtf,
        )
        consider(
            rec,
            cfg,
            {"provider": "OpenVINOExecutionProvider", "provider_options": {"device_type": "CPU"}},
        )
    else:
        rec = {
            "name": "E4_openvino_cpu",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "params": {"provider": "OpenVINOExecutionProvider"},
            "primary_rtf": None,
            "reference_primary_rtf": rolling_best_rtf,
            "improvement_pct": None,
            "keep": False,
            "reason": f"skipped: {ov_msg}",
            "quality_ok": False,
            "quality_reason": "n/a",
            "transcripts": {},
            "metrics": {},
            "error": None,
        }
        records.append(rec)
        append_ledger(ledger_path, rec)

    # ---- E5 chunked long audio ----
    if args.skip_chunking:
        rec = {
            "name": "E5_chunked_long30s",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "params": {},
            "primary_rtf": None,
            "reference_primary_rtf": rolling_best_rtf,
            "improvement_pct": None,
            "keep": False,
            "reason": "skipped by --skip-chunking",
            "quality_ok": False,
            "quality_reason": "n/a",
            "transcripts": {},
            "metrics": {},
            "error": None,
        }
        records.append(rec)
        append_ledger(ledger_path, rec)
    else:
        # E5 keep gate is special: ≥5% on long_30s RTF and quality OK;
        # still report primary metric for ledger consistency.
        for win, ov in ((12.0, 1.0), (15.0, 1.5), (10.0, 0.5)):
            name = f"E5_chunk_w{int(win)}_o{ov}"
            cfg = deepcopy(rolling_cfg)
            cfg["chunking"] = {
                "enabled": True,
                "window_s": win,
                "overlap_s": ov,
                "note": "app-level chunk+concat; not true streaming with encoder cache",
            }
            rec = run_config(
                name,
                cfg,
                data_dir,
                audio_keys,
                warmup,
                repeats,
                baseline_transcript=baseline_transcript,
                reference_primary_rtf=rolling_best_rtf,
            )
            # E5 gate: ≥5% on long_30s vs E0 baseline long_30s (not vs prior chunk runs)
            long_new = (rec.get("metrics") or {}).get("long_30s", {}).get("rtf")
            long_ref = (e0.get("metrics") or {}).get("long_30s", {}).get("rtf")

            long_imp = None
            if long_new and long_ref and long_ref > 0:
                long_imp = (long_ref - long_new) / long_ref

            # re-evaluate keep for E5 (must also not break primary badly / quality OK)
            q_ok = rec.get("quality_ok", False)
            if (
                long_imp is not None
                and long_imp >= KEEP_THRESHOLD
                and q_ok
                and rec.get("error") is None
            ):
                rec["keep"] = True
                rec["reason"] = (
                    f"long_30s RTF improved {long_imp*100:.2f}% vs E0 (≥5%); "
                    f"primary_imp={rec.get('improvement_pct')}; quality OK"
                )
            else:
                rec["keep"] = False
                rec["reason"] = (
                    f"E5 gate vs E0 long_30s: long_30s_imp="
                    f"{None if long_imp is None else round(long_imp*100, 2)}% "
                    f"(need ≥5%), quality_ok={q_ok}, primary_imp={rec.get('improvement_pct')}"
                )
            consider(rec, cfg, {"chunking": cfg["chunking"]})
            if rec["keep"]:
                break  # keep first winning chunk config

    # ---- E6 stack all kept winners ----
    stacked = deepcopy(best_cfg)  # start from E0 winner (4 or 8)
    # Apply kept overlays in order (later may override earlier)
    for kd in kept_deltas:
        ov = kd["overlay"]
        for k, v in ov.items():
            if k == "env" and isinstance(v, dict):
                stacked.setdefault("env", {}).update(v)
            else:
                stacked[k] = v
    # If optimized model dir was kept, ensure model_path points there
    if stacked.get("optimized_model_dir"):
        stacked["model_path"] = stacked["optimized_model_dir"]

    rec = run_config(
        "E6_stacked_best",
        stacked,
        data_dir,
        audio_keys,
        warmup,
        repeats,
        baseline_transcript=baseline_transcript,
        reference_primary_rtf=e0_rtf,  # always vs E0
    )
    # E6 keep is informational vs E0
    if rec.get("improvement_pct") is not None and rec["improvement_pct"] >= KEEP_THRESHOLD * 100 and rec.get("quality_ok"):
        rec["keep"] = True
        rec["reason"] = f"stacked vs E0 improvement {rec['improvement_pct']:.2f}% with quality OK"
    elif not kept_names:
        rec["keep"] = True
        rec["reason"] = "no individual keeps; stacked equals E0 baseline"
    else:
        # still save stacked as best if better than E0 even if <5%, else E0
        rec["keep"] = bool(
            rec.get("primary_rtf") is not None
            and e0_rtf is not None
            and rec["primary_rtf"] < e0_rtf
            and rec.get("quality_ok")
        )
        if not rec["keep"]:
            rec["reason"] = (
                f"stack did not beat E0 by ≥5% (imp={rec.get('improvement_pct')}); "
                f"quality_ok={rec.get('quality_ok')}"
            )
    records.append(rec)
    append_ledger(ledger_path, rec)

    # Final best_config: use stack if it beat E0 with quality, else rolling, else E0
    if (
        rec.get("primary_rtf") is not None
        and e0_rtf is not None
        and rec["primary_rtf"] <= e0_rtf
        and rec.get("quality_ok")
    ):
        final_cfg = config_to_public(stacked)
        final_cfg["e0_primary_rtf"] = e0_rtf
        final_cfg["best_primary_rtf"] = rec["primary_rtf"]
        final_cfg["improvement_pct_vs_e0"] = rec.get("improvement_pct")
        final_cfg["kept_experiments"] = kept_names
        final_cfg["baseline_real_speech_transcript"] = baseline_transcript
    else:
        final_cfg = config_to_public(best_cfg)
        final_cfg["e0_primary_rtf"] = e0_rtf
        final_cfg["best_primary_rtf"] = e0_rtf
        final_cfg["improvement_pct_vs_e0"] = 0.0
        final_cfg["kept_experiments"] = kept_names
        final_cfg["baseline_real_speech_transcript"] = baseline_transcript
        final_cfg["note"] = "No stack beat E0; shipping E0 baseline as best_config"

    final_cfg["host_snapshot"] = {
        "cpu_model": host.get("cpu_model"),
        "cpu_count_logical": host.get("cpu_count_logical"),
        "onnxruntime": host.get("onnxruntime"),
    }
    final_cfg["model_id"] = MODEL_ID
    final_cfg["generated_utc"] = datetime.now(timezone.utc).isoformat()

    (configs_dir / "best_config.json").write_text(
        json.dumps(final_cfg, indent=2) + "\n", encoding="utf-8"
    )
    write_summary(summary_path, host, records, e0_rtf, final_cfg, kept_names)

    print("\n==== DONE ====", flush=True)
    print(f"Ledger:  {ledger_path}", flush=True)
    print(f"Summary: {summary_path}", flush=True)
    print(f"Best:    {configs_dir / 'best_config.json'}", flush=True)
    print(f"Kept:    {kept_names}", flush=True)
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CPU autoresearch ladder for Parakeet TDT INT8 ONNX")
    p.add_argument("--model-dir", type=str, default=str(DEFAULT_MODEL_DIR))
    p.add_argument("--opt-model-dir", type=str, default=str(DEFAULT_OPT_MODEL_DIR))
    p.add_argument("--data-dir", type=str, default=str(DEFAULT_DATA_DIR))
    p.add_argument("--results-dir", type=str, default=str(DEFAULT_RESULTS_DIR))
    p.add_argument("--configs-dir", type=str, default=str(DEFAULT_CONFIGS_DIR))
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--skip-openvino", action="store_true")
    p.add_argument("--skip-chunking", action="store_true")
    p.add_argument("--append-ledger", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    os.chdir(PROJECT_ROOT)
    return run_ladder(args)


if __name__ == "__main__":
    raise SystemExit(main())

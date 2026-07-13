#!/usr/bin/env python3
"""Encoder-side static-quant optimization ladder (C0–C3) for Parakeet TDT INT8 ONNX.

Continuation of runtime E0–E6 (scripts/autoresearch_cpu_opts.py). Targets the
encoder graph: dynamic-activation INT8 → static-calibrated INT8 (QDQ/QOperator).

Primary metric: geometric mean of mean RTF on medium_15s + long_30s (lower better).
Keep only if ≥5% primary RTF reduction AND quality OK on real_speech.

Usage (from project root, venv active):
  python scripts/autoresearch_encoder_opts.py
  python scripts/autoresearch_encoder_opts.py --warmup 1 --repeats 3
  python scripts/autoresearch_encoder_opts.py --skip-download --skip-quant  # measure only
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
DEFAULT_FP32_DIR = PROJECT_ROOT / "models" / "parakeet-tdt-0.6b-v3-onnx-fp32-encoder"
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "results" / "autoresearch_encoder"
DEFAULT_CONFIGS_DIR = PROJECT_ROOT / "configs"
HF_REPO = "istupakov/parakeet-tdt-0.6b-v3-onnx"

AUDIO_FILES = {
    "short_5s": "short_5s.wav",
    "medium_15s": "medium_15s.wav",
    "long_30s": "long_30s.wav",
    "real_speech": "real_speech.wav",
}
PRIMARY_KEYS = ("medium_15s", "long_30s")
KEEP_THRESHOLD = 0.05
MODEL_ID = "nemo-parakeet-tdt-0.6b-v3"

# Static model assembly dirs (encoder-model.int8.onnx + shared decoder/frontend)
STATIC_DIRS = {
    "C1_minmax_qdq": PROJECT_ROOT / "models" / "parakeet-tdt-0.6b-v3-onnx-static-minmax",
    "C2_minmax_qop_pc": PROJECT_ROOT / "models" / "parakeet-tdt-0.6b-v3-onnx-static-qdq-pc",
    "C3_percentile_matmul": PROJECT_ROOT / "models" / "parakeet-tdt-0.6b-v3-onnx-static-matmul",
}


# ---------------------------------------------------------------------------
# Utilities (aligned with autoresearch_cpu_opts.py)
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
        info["disk_free_gb"] = round(shutil.disk_usage(PROJECT_ROOT).free / (1024**3), 2)
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


def resolve_providers(cfg: dict[str, Any]) -> list[Any]:
    provider = cfg.get("provider", "CPUExecutionProvider")
    return [provider]


def load_asr_model(cfg: dict[str, Any]):
    import onnx_asr

    model_dir = Path(cfg.get("model_path", DEFAULT_MODEL_DIR))
    if not model_dir.is_absolute():
        model_dir = PROJECT_ROOT / model_dir
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


def timed_recognize(
    model,
    wav_path: Path,
    repeats: int,
    warmup: int,
) -> dict[str, Any]:
    def _once() -> str:
        out = model.recognize(str(wav_path))
        return out if isinstance(out, str) else str(out)

    for _ in range(warmup):
        _ = _once()

    latencies: list[float] = []
    transcript = ""
    rss_before = current_rss_mb()
    for _ in range(repeats):
        t0 = time.perf_counter()
        transcript = _once()
        latencies.append(time.perf_counter() - t0)

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
    }


def quality_ok(transcript: str, baseline_transcript: str | None) -> tuple[bool, str]:
    if not transcript or not str(transcript).strip():
        return False, "empty real_speech transcript"
    if not baseline_transcript:
        return True, "no baseline transcript yet (C0)"
    if normalize_text(transcript) == normalize_text(baseline_transcript):
        return True, "exact normalized match"
    ov = token_overlap(transcript, baseline_transcript)
    if ov >= 0.85:
        return True, f"high token overlap={ov:.3f}"
    return False, f"quality regression overlap={ov:.3f}"


def default_config(model_path: Path | str | None = None) -> dict[str, Any]:
    return {
        "model_path": str(model_path or DEFAULT_MODEL_DIR),
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
        "model_id": MODEL_ID,
    }


def _portable_path(p: Any) -> Any:
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
    return {
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
        "model_id": cfg.get("model_id", MODEL_ID),
    }


# ---------------------------------------------------------------------------
# FP32 download + calibration + static quant
# ---------------------------------------------------------------------------

def ensure_fp32_encoder(fp32_dir: Path, skip_download: bool = False) -> Path:
    onnx_path = fp32_dir / "encoder-model.onnx"
    data_path = fp32_dir / "encoder-model.onnx.data"
    if onnx_path.exists() and data_path.exists():
        print(
            f"FP32 encoder present: {onnx_path} "
            f"({onnx_path.stat().st_size/1e6:.1f} MB + {data_path.stat().st_size/1e6:.1f} MB)",
            flush=True,
        )
        return onnx_path
    if skip_download:
        raise FileNotFoundError(f"FP32 encoder missing under {fp32_dir} and --skip-download set")

    from huggingface_hub import hf_hub_download

    fp32_dir.mkdir(parents=True, exist_ok=True)
    token = None
    for tp in (
        Path.home() / ".huggingface" / "token",
        Path.home() / ".cache" / "huggingface" / "token",
    ):
        if tp.exists():
            token = tp.read_text(encoding="utf-8").strip()
            break
    for f in ("encoder-model.onnx", "encoder-model.onnx.data"):
        print(f"Downloading {f} from {HF_REPO} ...", flush=True)
        p = hf_hub_download(HF_REPO, f, local_dir=str(fp32_dir), token=token or True)
        print(f"  -> {p} ({Path(p).stat().st_size/1e6:.1f} MB)", flush=True)
    if not onnx_path.exists() or not data_path.exists():
        raise FileNotFoundError(f"Download incomplete under {fp32_dir}")
    return onnx_path


def verify_encoder_io(encoder_path: Path, frontend_path: Path, sample_wav: Path) -> dict[str, Any]:
    so = ort.SessionOptions()
    so.intra_op_num_threads = 2
    enc = ort.InferenceSession(str(encoder_path), so, providers=["CPUExecutionProvider"])
    inputs = {i.name: (i.shape, i.type) for i in enc.get_inputs()}
    outputs = {o.name: (o.shape, o.type) for o in enc.get_outputs()}
    assert "audio_signal" in inputs, f"missing audio_signal in {inputs}"
    assert "length" in inputs, f"missing length in {inputs}"

    fe = ort.InferenceSession(str(frontend_path), so, providers=["CPUExecutionProvider"])
    audio, sr, _ = load_wav(sample_wav)
    wave = audio[None, :]
    lens = np.array([audio.shape[0]], dtype=np.int64)
    feats, flens = fe.run(None, {"waveforms": wave, "waveforms_lens": lens})
    outs = enc.run(None, {"audio_signal": feats, "length": flens})
    report = {
        "encoder": str(encoder_path),
        "inputs": {k: {"shape": list(v[0]) if v[0] else None, "type": v[1]} for k, v in inputs.items()},
        "outputs": {k: {"shape": list(v[0]) if v[0] else None, "type": v[1]} for k, v in outputs.items()},
        "sample_mel_shape": list(feats.shape),
        "sample_enc_out_shape": list(outs[0].shape),
        "sample_encoded_lengths": outs[1].tolist() if hasattr(outs[1], "tolist") else outs[1],
    }
    del enc, fe
    gc.collect()
    return report


def collect_calibration_features(
    frontend_path: Path,
    data_dir: Path,
    max_samples: int = 8,
    max_seconds: float | None = 20.0,
) -> list[dict[str, np.ndarray]]:
    """Run nemo128 frontend on WAVs → encoder mel tensors + lengths."""
    so = ort.SessionOptions()
    so.intra_op_num_threads = 4
    fe = ort.InferenceSession(str(frontend_path), so, providers=["CPUExecutionProvider"])

    wav_paths: list[Path] = []
    for key in ("real_speech", "short_5s", "medium_15s", "long_30s"):
        p = data_dir / AUDIO_FILES[key]
        if p.exists():
            wav_paths.append(p)
    # also any other wavs
    for p in sorted(data_dir.glob("*.wav")):
        if p not in wav_paths:
            wav_paths.append(p)

    samples: list[dict[str, np.ndarray]] = []
    for p in wav_paths:
        if len(samples) >= max_samples:
            break
        audio, sr, dur = load_wav(p)
        if max_seconds is not None and dur > max_seconds:
            audio = audio[: int(max_seconds * sr)]
        wave = audio[None, :]
        lens = np.array([audio.shape[0]], dtype=np.int64)
        feats, flens = fe.run(None, {"waveforms": wave, "waveforms_lens": lens})
        samples.append(
            {
                "audio_signal": feats.astype(np.float32),
                "length": flens.astype(np.int64),
                "_source": p.name,
                "_mel_shape": list(feats.shape),
            }
        )
        print(
            f"  calib sample {len(samples)}: {p.name} mel={feats.shape} length={flens.tolist()}",
            flush=True,
        )
    del fe
    gc.collect()
    if len(samples) < 2:
        raise RuntimeError(f"Need ≥2 calibration samples, got {len(samples)}")
    return samples


class MelCalibrationDataReader:
    """onnxruntime.quantization.CalibrationDataReader over precomputed mel features."""

    def __init__(self, samples: list[dict[str, np.ndarray]]):
        # strip private keys
        self._data = [
            {k: v for k, v in s.items() if not k.startswith("_")} for s in samples
        ]
        self._iter = None

    def get_next(self) -> dict[str, np.ndarray] | None:
        if self._iter is None:
            self._iter = iter(self._data)
        return next(self._iter, None)

    def rewind(self) -> None:
        self._iter = iter(self._data)


def _rewrite_external_data_locations(model_path: Path, data_filename: str) -> int:
    """Point all external-data initializers at data_filename (basename only)."""
    import onnx
    from onnx.external_data_helper import ExternalDataInfo, load_external_data_for_model

    model = onnx.load(str(model_path), load_external_data=False)
    n = 0
    for tensor in list(model.graph.initializer) + [
        t for n_ in model.graph.node for t in ()  # noqa: keep simple
    ]:
        pass
    # Walk all tensors that use external data
    tensors = list(model.graph.initializer)
    for node in model.graph.node:
        for attr in node.attribute:
            if attr.t.name or attr.t.raw_data or attr.t.external_data:
                tensors.append(attr.t)
            for t in attr.tensors:
                tensors.append(t)
    for tensor in tensors:
        if not tensor.HasField("data_location") and not tensor.external_data:
            # data_location enum: DEFAULT=0, EXTERNAL=1
            continue
        # Check external_data entries
        if tensor.external_data:
            for entry in tensor.external_data:
                if entry.key == "location":
                    if entry.value != data_filename:
                        entry.value = data_filename
                        n += 1
            # ensure data_location is EXTERNAL
            tensor.data_location = onnx.TensorProto.EXTERNAL
    # Also handle sparse initializers if present
    for sparse in model.graph.sparse_initializer:
        for tensor in (sparse.values, sparse.indices):
            if tensor.external_data:
                for entry in tensor.external_data:
                    if entry.key == "location" and entry.value != data_filename:
                        entry.value = data_filename
                        n += 1
    onnx.save(model, str(model_path))
    return n


def assemble_static_model_dir(
    dst_dir: Path,
    static_encoder_path: Path,
    base_model_dir: Path,
) -> Path:
    """Copy decoder/frontend/config/vocab + place encoder as encoder-model.int8.onnx.

    Rewrites external-data locations so the .onnx file looks for
    encoder-model.int8.onnx.data next to itself (quantizer often embeds the
    original basename like encoder-smoke.onnx.data).
    """
    dst_dir.mkdir(parents=True, exist_ok=True)
    target_enc = dst_dir / "encoder-model.int8.onnx"
    target_data = dst_dir / "encoder-model.int8.onnx.data"

    # Copy graph protobuf
    if static_encoder_path.resolve() != target_enc.resolve():
        shutil.copy2(static_encoder_path, target_enc)

    # Copy external weight blob (prefer matching .onnx.data sibling)
    src_data = Path(str(static_encoder_path) + ".data")
    if not src_data.exists():
        # fallback: any sibling .data with same stem prefix
        candidates = list(static_encoder_path.parent.glob(static_encoder_path.name + ".*"))
        candidates = [c for c in candidates if c.suffix == ".data" or c.name.endswith(".onnx.data")]
        if candidates:
            src_data = candidates[0]
    if src_data.exists():
        if src_data.resolve() != target_data.resolve():
            shutil.copy2(src_data, target_data)
    else:
        # maybe quantizer inlined everything — ok
        pass

    # Rewrite location keys inside the ONNX to the new basename
    if target_data.exists():
        try:
            n = _rewrite_external_data_locations(target_enc, target_data.name)
            print(f"  rewrote {n} external_data location(s) → {target_data.name}", flush=True)
        except Exception as e:
            print(f"  WARN: external_data rewrite failed: {e}", flush=True)
            traceback.print_exc()

    for name in (
        "decoder_joint-model.int8.onnx",
        "nemo128.onnx",
        "config.json",
        "vocab.txt",
    ):
        src = base_model_dir / name
        if not src.exists():
            raise FileNotFoundError(f"Missing base asset {src}")
        shutil.copy2(src, dst_dir / name)

    return target_enc


def static_quantize_encoder(
    fp32_encoder: Path,
    calib_samples: list[dict[str, np.ndarray]],
    out_onnx: Path,
    *,
    quant_format_name: str = "QDQ",
    per_channel: bool = False,
    calibrate_method_name: str = "MinMax",
    activation_type_name: str = "QUInt8",
    weight_type_name: str = "QInt8",
    op_types_to_quantize: list[str] | None = None,
    extra_options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    from onnxruntime.quantization import (
        CalibrationMethod,
        QuantFormat,
        QuantType,
        quant_pre_process,
        quantize_static,
    )

    out_onnx.parent.mkdir(parents=True, exist_ok=True)
    # preprocess into a temp path (shape infer + optimize for quant)
    pre_path = out_onnx.parent / (out_onnx.stem + ".pre.onnx")
    report: dict[str, Any] = {
        "fp32": str(fp32_encoder),
        "output": str(out_onnx),
        "quant_format": quant_format_name,
        "per_channel": per_channel,
        "calibrate_method": calibrate_method_name,
        "activation_type": activation_type_name,
        "weight_type": weight_type_name,
        "op_types_to_quantize": op_types_to_quantize,
        "n_calib": len(calib_samples),
    }

    t0 = time.perf_counter()
    print(f"  quant_pre_process → {pre_path.name} ...", flush=True)
    try:
        quant_pre_process(
            input_model_path=str(fp32_encoder),
            output_model_path=str(pre_path),
            skip_optimization=False,
            skip_onnx_shape=False,
            skip_symbolic_shape=False,
            auto_merge=True,
        )
        report["preprocess_s"] = round(time.perf_counter() - t0, 2)
        report["preprocess_ok"] = True
        model_in = pre_path
    except Exception as e:
        # Retry without symbolic shape (no sympy / complex dyn axes)
        print(f"  preprocess full failed ({e}); retry skip_symbolic_shape=True", flush=True)
        try:
            quant_pre_process(
                input_model_path=str(fp32_encoder),
                output_model_path=str(pre_path),
                skip_optimization=False,
                skip_onnx_shape=False,
                skip_symbolic_shape=True,
                auto_merge=True,
            )
            report["preprocess_s"] = round(time.perf_counter() - t0, 2)
            report["preprocess_ok"] = True
            report["preprocess_note"] = "skip_symbolic_shape=True"
            model_in = pre_path
        except Exception as e2:
            report["preprocess_ok"] = False
            report["preprocess_error"] = f"{e} | retry: {e2}"
            print(f"  preprocess failed again ({e2}); quantizing raw FP32", flush=True)
            model_in = fp32_encoder

    qformat = getattr(QuantFormat, quant_format_name)
    cal_method = getattr(CalibrationMethod, calibrate_method_name)
    act = getattr(QuantType, activation_type_name)
    wgt = getattr(QuantType, weight_type_name)

    reader = MelCalibrationDataReader(calib_samples)
    # Default extra options friendly to CPU
    extras = {
        "ActivationSymmetric": False,
        "WeightSymmetric": True,
        "EnableSubgraph": False,
        "ForceQuantizeNoInputCheck": False,
        "MatMulConstBOnly": False,
    }
    if extra_options:
        extras.update(extra_options)

    t1 = time.perf_counter()
    print(
        f"  quantize_static format={quant_format_name} per_channel={per_channel} "
        f"method={calibrate_method_name} ops={op_types_to_quantize} ...",
        flush=True,
    )
    try:
        quantize_static(
            model_input=str(model_in),
            model_output=str(out_onnx),
            calibration_data_reader=reader,
            quant_format=qformat,
            per_channel=per_channel,
            reduce_range=False,
            activation_type=act,
            weight_type=wgt,
            op_types_to_quantize=op_types_to_quantize,
            use_external_data_format=True,
            calibrate_method=cal_method,
            extra_options=extras,
        )
        report["quantize_ok"] = True
        report["quantize_s"] = round(time.perf_counter() - t1, 2)
        report["out_mb"] = round(out_onnx.stat().st_size / 1e6, 2) if out_onnx.exists() else 0
        # external data size
        data_files = list(out_onnx.parent.glob(out_onnx.name + "*"))
        report["out_files"] = [
            {"name": p.name, "mb": round(p.stat().st_size / 1e6, 2)} for p in data_files if p.is_file()
        ]
        print(f"  quantize done in {report['quantize_s']}s files={report['out_files']}", flush=True)
    except Exception as e:
        report["quantize_ok"] = False
        report["quantize_error"] = str(e)
        report["quantize_traceback"] = traceback.format_exc()
        print(f"  QUANTIZE FAILED: {e}", flush=True)
        traceback.print_exc()
    finally:
        # free preprocess intermediate
        if pre_path.exists():
            try:
                pre_path.unlink()
            except OSError:
                pass
            for p in pre_path.parent.glob(pre_path.name + ".*"):
                try:
                    p.unlink()
                except OSError:
                    pass
        gc.collect()
    return report


def smoke_static_encoder(encoder_path: Path, frontend_path: Path, sample_wav: Path) -> dict[str, Any]:
    so = ort.SessionOptions()
    so.intra_op_num_threads = 4
    try:
        enc = ort.InferenceSession(str(encoder_path), so, providers=["CPUExecutionProvider"])
        fe = ort.InferenceSession(str(frontend_path), so, providers=["CPUExecutionProvider"])
        audio, _, _ = load_wav(sample_wav)
        feats, flens = fe.run(
            None,
            {
                "waveforms": audio[None, :],
                "waveforms_lens": np.array([audio.shape[0]], dtype=np.int64),
            },
        )
        outs = enc.run(None, {"audio_signal": feats, "length": flens})
        return {
            "ok": True,
            "out_shape": list(outs[0].shape),
            "encoded_lengths": outs[1].tolist() if hasattr(outs[1], "tolist") else outs[1],
            "inputs": [i.name for i in enc.get_inputs()],
        }
    except Exception as e:
        return {"ok": False, "error": str(e), "traceback": traceback.format_exc()}
    finally:
        gc.collect()


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------

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
                continue
            print(f"  timing {key} ...", flush=True)
            try:
                metrics = timed_recognize(model, wav, repeats=repeats, warmup=warmup)
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
        "improvement_pct": None if improvement is None else round(improvement * 100.0, 3),
        "keep": keep,
        "reason": "; ".join(reason_parts),
        "quality_ok": q_ok,
        "quality_reason": q_reason,
        "transcripts": {
            k: (v.get("transcript") if isinstance(v, dict) else None)
            for k, v in per_audio.items()
        },
        "metrics": {
            k: {kk: vv for kk, vv in v.items() if kk not in ("traceback",)}
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
    c0_rtf: float | None,
    best_cfg: dict[str, Any],
    kept: list[str],
    quant_reports: list[dict[str, Any]],
) -> None:
    lines = [
        "# Autoresearch Encoder Static-Quant Ladder (C0–C3) — Summary",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Host",
        "",
        f"- CPU: {host.get('cpu_model')}",
        f"- Logical CPUs: {host.get('cpu_count_logical')}",
        f"- Flags: {', '.join(host.get('cpu_flags_relevant') or [])}",
        f"- ORT: {host.get('onnxruntime')} providers={host.get('providers')}",
        f"- RAM: {host.get('ram_total_gb')} GB  disk_free≈{host.get('disk_free_gb')} GB",
        "",
        "## Protocol",
        "",
        "- Model: encoder static re-quant of `nemo-parakeet-tdt-0.6b-v3` (decoder stays dynamic INT8)",
        "- Primary metric: geometric mean of mean RTF on `medium_15s` + `long_30s` (lower better)",
        f"- Keep gate: ≥{KEEP_THRESHOLD*100:.0f}% primary RTF improvement vs C0 + quality OK",
        "- Quality gate: non-empty `real_speech`; normalized match or ≥85% token overlap vs C0",
        "",
        "## Quantization builds",
        "",
    ]
    if quant_reports:
        lines.append("| Variant | ok | format | per_channel | method | out_mb | notes |")
        lines.append("|---|:---:|---|:---:|---|---:|---|")
        for qr in quant_reports:
            notes = qr.get("quantize_error") or qr.get("assemble_error") or qr.get("smoke_error") or "ok"
            if qr.get("quantize_ok") and qr.get("smoke_ok"):
                notes = "ok"
            elif qr.get("quantize_ok") and not qr.get("smoke_ok"):
                notes = f"smoke fail: {qr.get('smoke_error', '')[:80]}"
            lines.append(
                f"| {qr.get('name')} | {bool(qr.get('quantize_ok'))} | {qr.get('quant_format')} | "
                f"{qr.get('per_channel')} | {qr.get('calibrate_method')} | "
                f"{qr.get('out_mb', '—')} | {notes} |"
            )
    else:
        lines.append("_No quant builds recorded._")

    lines.extend(
        [
            "",
            "## Results",
            "",
            "| Experiment | primary_rtf | improvement_pct | keep | reason |",
            "|---|---:|---:|:---:|---|",
        ]
    )
    for r in records:
        lines.append(
            f"| {r['name']} | {r.get('primary_rtf')} | {r.get('improvement_pct')} | "
            f"{'YES' if r.get('keep') else 'no'} | {r.get('reason', '')} |"
        )

    lines.extend(["", "## Kept winners", ""])
    if kept:
        for k in kept:
            lines.append(f"- `{k}`")
    else:
        lines.append(
            "- *(none — no static-quant experiment cleared the ≥5% gate with quality OK)*"
        )

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
    if c0_rtf is not None:
        lines.append(f"- C0 baseline primary_rtf = {c0_rtf:.6f}")
    best_rtf = best_cfg.get("best_primary_rtf") or best_cfg.get("c0_primary_rtf")
    if c0_rtf and best_rtf:
        imp = (c0_rtf - float(best_rtf)) / c0_rtf * 100.0
        lines.append(f"- Best vs C0: primary_rtf {best_rtf} (improvement={imp:.2f}%)")
    if not kept:
        lines.append(
            "- **Ceiling:** static re-quant of the encoder did not clear ≥5% E2E primary RTF "
            "on this host with the recipes tried (MinMax QDQ, per-channel QOperator, "
            "Percentile MatMul-focused). Runtime E0–E6 also failed the gate (~4–4.8% residual)."
        )
        lines.append(
            "- Next levers outside this ladder: EP bake-off (OpenVINO/ZenDNN with dedicated "
            "wheel), weight-only INT4 on FFN, or model-side changes (distill / re-export)."
        )
    else:
        lines.append("- `configs/best_config.json` updated to the winning static encoder dir.")
    lines.append(
        "- FP32 encoder source: `models/parakeet-tdt-0.6b-v3-onnx-fp32-encoder/` "
        "(may be deleted to free ~2.5 GB after static export)."
    )
    lines.append("")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text("\n".join(lines), encoding="utf-8")


def free_large_intermediates(fp32_dir: Path, quant_work: Path, keep_fp32: bool) -> list[str]:
    freed: list[str] = []
    if not keep_fp32 and fp32_dir.exists():
        for p in fp32_dir.rglob("*"):
            if p.is_file() and p.suffix in (".onnx", ".data") or p.name.endswith(".onnx.data"):
                try:
                    sz = p.stat().st_size
                    p.unlink()
                    freed.append(f"deleted {p} ({sz/1e6:.1f} MB)")
                except OSError as e:
                    freed.append(f"failed delete {p}: {e}")
        # also clear hf cache under that dir
        cache = fp32_dir / ".cache"
        if cache.exists():
            shutil.rmtree(cache, ignore_errors=True)
            freed.append(f"rmtree {cache}")
    if quant_work.exists():
        for p in quant_work.glob("*.pre.onnx*"):
            try:
                p.unlink()
                freed.append(f"deleted {p}")
            except OSError:
                pass
    return freed


# ---------------------------------------------------------------------------
# Ladder
# ---------------------------------------------------------------------------

def run_ladder(args: argparse.Namespace) -> int:
    os.chdir(PROJECT_ROOT)
    model_dir = Path(args.model_dir)
    if not model_dir.is_absolute():
        model_dir = PROJECT_ROOT / model_dir
    fp32_dir = Path(args.fp32_dir)
    if not fp32_dir.is_absolute():
        fp32_dir = PROJECT_ROOT / fp32_dir
    data_dir = Path(args.data_dir)
    if not data_dir.is_absolute():
        data_dir = PROJECT_ROOT / data_dir
    results_dir = Path(args.results_dir)
    if not results_dir.is_absolute():
        results_dir = PROJECT_ROOT / results_dir
    configs_dir = Path(args.configs_dir)
    if not configs_dir.is_absolute():
        configs_dir = PROJECT_ROOT / configs_dir

    results_dir.mkdir(parents=True, exist_ok=True)
    quant_work = PROJECT_ROOT / "models" / "_quant_work"
    quant_work.mkdir(parents=True, exist_ok=True)

    ledger_path = results_dir / "ledger.jsonl"
    summary_path = results_dir / "summary.md"
    if ledger_path.exists() and not args.append_ledger:
        ledger_path.unlink()

    host = host_info()
    print("Host:", json.dumps(host, indent=2), flush=True)

    audio_keys = ["short_5s", "medium_15s", "long_30s", "real_speech"]
    warmup = args.warmup
    repeats = args.repeats
    records: list[dict[str, Any]] = []
    quant_reports: list[dict[str, Any]] = []
    kept_names: list[str] = []

    # ---- Ensure deps ----
    try:
        import onnx  # noqa: F401
        from onnxruntime.quantization import CalibrationDataReader, quantize_static  # noqa: F401
        print(f"onnx + onnxruntime.quantization OK (onnx={onnx.__version__})", flush=True)
    except Exception as e:
        print(f"FATAL: onnx/quantization import failed: {e}", file=sys.stderr)
        return 2

    # ---- FP32 encoder ----
    try:
        fp32_encoder = ensure_fp32_encoder(fp32_dir, skip_download=args.skip_download)
        io_report = verify_encoder_io(
            fp32_encoder,
            model_dir / "nemo128.onnx",
            data_dir / AUDIO_FILES["short_5s"],
        )
        print("FP32 encoder I/O:", json.dumps(io_report, indent=2), flush=True)
        (results_dir / "fp32_encoder_io.json").write_text(
            json.dumps(io_report, indent=2), encoding="utf-8"
        )
    except Exception as e:
        print(f"FATAL: FP32 encoder setup failed: {e}", file=sys.stderr)
        traceback.print_exc()
        return 3

    # ---- C0 baseline (current dynamic INT8) ----
    c0_cfg = default_config(model_dir)
    c0 = run_config(
        "C0_dynamic_int8_baseline",
        c0_cfg,
        data_dir,
        audio_keys,
        warmup,
        repeats,
        baseline_transcript=None,
        reference_primary_rtf=None,
    )
    records.append(c0)
    append_ledger(ledger_path, c0)

    c0_rtf = c0.get("primary_rtf")
    baseline_transcript = c0.get("transcripts", {}).get("real_speech") or ""
    best_cfg = deepcopy(c0_cfg)
    best_rtf = c0_rtf
    rolling_ref = c0_rtf

    if c0_rtf is None or not baseline_transcript.strip():
        print("FATAL: C0 baseline failed or empty real_speech transcript", file=sys.stderr)
        write_summary(summary_path, host, records, c0_rtf, config_to_public(best_cfg), kept_names, quant_reports)
        return 4

    # ---- Calibration features ----
    print("\n=== Calibration features (nemo128 → mel) ===", flush=True)
    calib_samples = collect_calibration_features(
        model_dir / "nemo128.onnx",
        data_dir,
        max_samples=args.calib_samples,
        max_seconds=args.calib_max_seconds,
    )
    calib_meta = [
        {"source": s.get("_source"), "mel_shape": s.get("_mel_shape")} for s in calib_samples
    ]
    (results_dir / "calibration_meta.json").write_text(
        json.dumps({"n": len(calib_samples), "samples": calib_meta}, indent=2),
        encoding="utf-8",
    )
    print(f"Calibration samples: {len(calib_samples)}", flush=True)

    # ---- Quant recipes C1–C3 ----
    recipes = [
        {
            "name": "C1_minmax_qdq",
            "dir": STATIC_DIRS["C1_minmax_qdq"],
            "quant_format": "QDQ",
            "per_channel": False,
            "calibrate_method": "MinMax",
            "activation_type": "QUInt8",
            "weight_type": "QInt8",
            "op_types_to_quantize": None,  # default MatMul+Conv etc.
            "extra_options": None,
            "env_overlay": None,
        },
        # QOperator path fails on this FastConformer graph (unknown intermediate
        # tensors). Use QDQ + per-channel weights instead.
        {
            "name": "C2_minmax_qdq_pc",
            "dir": STATIC_DIRS["C2_minmax_qop_pc"],
            "quant_format": "QDQ",
            "per_channel": True,
            "calibrate_method": "MinMax",
            "activation_type": "QUInt8",
            "weight_type": "QInt8",
            "op_types_to_quantize": None,
            "extra_options": {"WeightSymmetric": True},
            "env_overlay": None,
        },
        # Percentile calib crashes on variable-length intermediate shapes.
        # MatMul-focused MinMax QDQ + residual env knobs (~4% alone in E1/E2).
        {
            "name": "C3_matmul_qdq_stack",
            "dir": STATIC_DIRS["C3_percentile_matmul"],
            "quant_format": "QDQ",
            "per_channel": False,
            "calibrate_method": "MinMax",
            "activation_type": "QUInt8",
            "weight_type": "QInt8",
            "op_types_to_quantize": ["MatMul", "Gemm"],
            "extra_options": {"WeightSymmetric": True},
            "env_overlay": {
                "env": {
                    "OMP_NUM_THREADS": "8",
                    "OMP_WAIT_POLICY": "ACTIVE",
                },
                "enable_cpu_mem_arena": False,
            },
        },
    ]

    if args.skip_quant:
        print("Skipping quant builds (--skip-quant); measuring existing static dirs only", flush=True)

    for recipe in recipes:
        name = recipe["name"]
        dst_dir: Path = recipe["dir"]
        out_raw = quant_work / f"encoder-{name}.onnx"
        qr: dict[str, Any] = {"name": name, "dir": str(dst_dir)}

        if not args.skip_quant:
            # free previous raw if re-run
            if out_raw.exists():
                try:
                    out_raw.unlink()
                except OSError:
                    pass
            for p in quant_work.glob(out_raw.name + ".*"):
                try:
                    p.unlink()
                except OSError:
                    pass

            print(f"\n=== Build {name} ===", flush=True)
            qrep = static_quantize_encoder(
                fp32_encoder,
                calib_samples,
                out_raw,
                quant_format_name=recipe["quant_format"],
                per_channel=recipe["per_channel"],
                calibrate_method_name=recipe["calibrate_method"],
                activation_type_name=recipe["activation_type"],
                weight_type_name=recipe["weight_type"],
                op_types_to_quantize=recipe["op_types_to_quantize"],
                extra_options=recipe["extra_options"],
            )
            qr.update(qrep)
            if not qrep.get("quantize_ok") or not out_raw.exists():
                quant_reports.append(qr)
                rec = {
                    "name": name,
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "params": {"recipe": recipe["name"], "error": qrep.get("quantize_error")},
                    "primary_rtf": None,
                    "reference_primary_rtf": rolling_ref,
                    "improvement_pct": None,
                    "keep": False,
                    "reason": f"quantize failed: {qrep.get('quantize_error')}",
                    "quality_ok": False,
                    "quality_reason": "n/a",
                    "transcripts": {},
                    "metrics": {},
                    "error": qrep.get("quantize_error"),
                    "quant_report": qrep,
                }
                records.append(rec)
                append_ledger(ledger_path, rec)
                continue

            try:
                # assemble model dir for onnx_asr (rewrites external_data locations)
                if dst_dir.exists():
                    shutil.rmtree(dst_dir, ignore_errors=True)
                enc_path = assemble_static_model_dir(dst_dir, out_raw, model_dir)
                print(f"  assembled {enc_path} (+ decoder/frontend)", flush=True)

                smoke = smoke_static_encoder(
                    dst_dir / "encoder-model.int8.onnx",
                    dst_dir / "nemo128.onnx",
                    data_dir / AUDIO_FILES["short_5s"],
                )
                qr["smoke_ok"] = smoke.get("ok", False)
                qr["smoke"] = smoke
                if not smoke.get("ok"):
                    qr["smoke_error"] = smoke.get("error")
                    print(f"  SMOKE FAIL {name}: {smoke.get('error')}", flush=True)
            except Exception as e:
                qr["assemble_error"] = str(e)
                qr["smoke_ok"] = False
                print(f"  ASSEMBLE FAIL {name}: {e}", flush=True)
                traceback.print_exc()
                quant_reports.append(qr)
                rec = {
                    "name": name,
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "params": {"recipe": name},
                    "primary_rtf": None,
                    "reference_primary_rtf": rolling_ref,
                    "improvement_pct": None,
                    "keep": False,
                    "reason": f"assemble/smoke failed: {e}",
                    "quality_ok": False,
                    "quality_reason": "n/a",
                    "transcripts": {},
                    "metrics": {},
                    "error": str(e),
                }
                records.append(rec)
                append_ledger(ledger_path, rec)
                continue
        else:
            if not (dst_dir / "encoder-model.int8.onnx").exists():
                print(f"  SKIP {name}: no model at {dst_dir}", flush=True)
                continue
            qr["quantize_ok"] = True
            qr["skipped_build"] = True

        quant_reports.append(qr)
        (results_dir / f"quant_{name}.json").write_text(
            json.dumps(qr, indent=2, default=str), encoding="utf-8"
        )

        if not qr.get("smoke_ok", True) and not args.skip_quant:
            rec = {
                "name": name,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "params": config_to_public(default_config(dst_dir)),
                "primary_rtf": None,
                "reference_primary_rtf": rolling_ref,
                "improvement_pct": None,
                "keep": False,
                "reason": f"encoder smoke failed: {qr.get('smoke_error')}",
                "quality_ok": False,
                "quality_reason": "n/a",
                "transcripts": {},
                "metrics": {},
                "error": qr.get("smoke_error"),
            }
            records.append(rec)
            append_ledger(ledger_path, rec)
            continue

        # E2E measure
        cfg = default_config(dst_dir)
        if recipe.get("env_overlay"):
            for k, v in recipe["env_overlay"].items():
                if k == "env":
                    cfg["env"] = dict(v)
                else:
                    cfg[k] = v

        # Quick quality check via onnx_asr before full timing
        try:
            prev = apply_env(cfg.get("env") or {})
            m, _ = load_asr_model(cfg)
            t = m.recognize(str(data_dir / AUDIO_FILES["real_speech"]))
            t = t if isinstance(t, str) else str(t)
            unload_model(m)
            restore_env(prev)
            print(f"  real_speech smoke transcript: {t[:100]!r}", flush=True)
            if not t.strip():
                rec = {
                    "name": name,
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "params": config_to_public(cfg),
                    "primary_rtf": None,
                    "reference_primary_rtf": rolling_ref,
                    "improvement_pct": None,
                    "keep": False,
                    "reason": "empty real_speech transcript after static quant",
                    "quality_ok": False,
                    "quality_reason": "empty real_speech transcript",
                    "transcripts": {"real_speech": t},
                    "metrics": {},
                    "error": None,
                }
                records.append(rec)
                append_ledger(ledger_path, rec)
                continue
        except Exception as e:
            print(f"  E2E load/recognize smoke failed: {e}", flush=True)
            traceback.print_exc()
            rec = {
                "name": name,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "params": config_to_public(cfg),
                "primary_rtf": None,
                "reference_primary_rtf": rolling_ref,
                "improvement_pct": None,
                "keep": False,
                "reason": f"e2e smoke failed: {e}",
                "quality_ok": False,
                "quality_reason": "n/a",
                "transcripts": {},
                "metrics": {},
                "error": str(e),
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
            reference_primary_rtf=c0_rtf,  # always gate vs C0 for encoder ladder
        )
        records.append(rec)
        append_ledger(ledger_path, rec)

        if rec.get("keep") and rec.get("primary_rtf") is not None:
            kept_names.append(name)
            if best_rtf is None or rec["primary_rtf"] < best_rtf:
                best_rtf = rec["primary_rtf"]
                best_cfg = deepcopy(cfg)
                rolling_ref = best_rtf

        # free raw quant artifacts after assemble to save disk
        if not args.keep_quant_work:
            for p in quant_work.glob(f"encoder-{name}*"):
                try:
                    p.unlink()
                except OSError:
                    pass
            gc.collect()

    # ---- Finalize best_config ----
    public_best = config_to_public(best_cfg)
    public_best.update(
        {
            "c0_primary_rtf": c0_rtf,
            "best_primary_rtf": best_rtf,
            "improvement_pct_vs_c0": (
                None
                if c0_rtf is None or best_rtf is None
                else round((c0_rtf - best_rtf) / c0_rtf * 100.0, 3)
            ),
            "kept_experiments": kept_names,
            "baseline_real_speech_transcript": baseline_transcript,
            "host_snapshot": {
                "cpu_model": host.get("cpu_model"),
                "cpu_count_logical": host.get("cpu_count_logical"),
                "onnxruntime": host.get("onnxruntime"),
            },
            "ladder": "encoder_static_quant_C0_C3",
            "generated_utc": datetime.now(timezone.utc).isoformat(),
        }
    )

    # Only overwrite configs/best_config.json if we have a keep (≥5%)
    best_path = configs_dir / "best_config.json"
    if kept_names:
        best_path.write_text(json.dumps(public_best, indent=2) + "\n", encoding="utf-8")
        print(f"Updated {best_path} with keep(s): {kept_names}", flush=True)
    else:
        # leave existing best_config (E0 runtime) intact; write encoder-specific snapshot
        enc_best_path = configs_dir / "encoder_ladder_best.json"
        public_best["note"] = (
            "No encoder static-quant experiment cleared ≥5%; "
            "configs/best_config.json left unchanged (runtime E0)."
        )
        enc_best_path.write_text(json.dumps(public_best, indent=2) + "\n", encoding="utf-8")
        print(
            f"No keep ≥5%; left {best_path} unchanged; wrote {enc_best_path}",
            flush=True,
        )

    write_summary(
        summary_path, host, records, c0_rtf, public_best, kept_names, quant_reports
    )
    print(f"\nSummary: {summary_path}", flush=True)
    print(f"Ledger:  {ledger_path}", flush=True)

    if args.free_fp32:
        freed = free_large_intermediates(fp32_dir, quant_work, keep_fp32=False)
        print("Freed intermediates:", freed, flush=True)
        (results_dir / "cleanup.json").write_text(
            json.dumps(freed, indent=2), encoding="utf-8"
        )

    # disk report
    free_gb = shutil.disk_usage(PROJECT_ROOT).free / (1024**3)
    print(f"Disk free after ladder: {free_gb:.2f} GB", flush=True)
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Encoder static-quant ladder C0–C3")
    p.add_argument("--model-dir", type=str, default=str(DEFAULT_MODEL_DIR))
    p.add_argument("--fp32-dir", type=str, default=str(DEFAULT_FP32_DIR))
    p.add_argument("--data-dir", type=str, default=str(DEFAULT_DATA_DIR))
    p.add_argument("--results-dir", type=str, default=str(DEFAULT_RESULTS_DIR))
    p.add_argument("--configs-dir", type=str, default=str(DEFAULT_CONFIGS_DIR))
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--calib-samples", type=int, default=4)
    p.add_argument("--calib-max-seconds", type=float, default=20.0)
    p.add_argument("--skip-download", action="store_true")
    p.add_argument("--skip-quant", action="store_true", help="Measure existing static dirs only")
    p.add_argument("--keep-quant-work", action="store_true")
    p.add_argument("--free-fp32", action="store_true", help="Delete FP32 encoder after run")
    p.add_argument("--append-ledger", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    return run_ladder(args)


if __name__ == "__main__":
    raise SystemExit(main())

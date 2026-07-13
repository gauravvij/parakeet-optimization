#!/usr/bin/env python3
"""Quality + RTF assessment: baseline (dynamic INT8) vs best (static QDQ per-channel).

Transcribes all audio under data/ with both configs, computes normalized exact-match,
token F1 / similarity, WER/CER (jiwer), absolute WER on JFK reference, and primary
RTF (geo-mean of medium_15s + long_30s). Writes:

  results/quality_baseline_vs_best.json
  results/quality_baseline_vs_best.md

Usage (from project root, venv active):
  python scripts/quality_baseline_vs_best.py
  python scripts/quality_baseline_vs_best.py --warmup 1 --repeats 3
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import re
import statistics
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import soundfile as sf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASELINE = PROJECT_ROOT / "configs" / "baseline.json"
DEFAULT_BEST = PROJECT_ROOT / "configs" / "best_config.json"
DEFAULT_DATA = PROJECT_ROOT / "data"
DEFAULT_OUT_JSON = PROJECT_ROOT / "results" / "quality_baseline_vs_best.json"
DEFAULT_OUT_MD = PROJECT_ROOT / "results" / "quality_baseline_vs_best.md"
MODEL_ID = "nemo-parakeet-tdt-0.6b-v3"

PREFERRED_AUDIO = [
    "short_5s.wav",
    "medium_15s.wav",
    "long_30s.wav",
    "real_speech.wav",
    "jfk.flac",
]
PRIMARY_KEYS = ("medium_15s", "long_30s")

JFK_REFERENCE = (
    "And so, my fellow Americans, ask not what your country can do for you "
    "— ask what you can do for your country."
)

CONTRACTIONS = {
    "don't": "do not",
    "doesn't": "does not",
    "didn't": "did not",
    "won't": "will not",
    "can't": "cannot",
    "couldn't": "could not",
    "shouldn't": "should not",
    "wouldn't": "would not",
    "isn't": "is not",
    "aren't": "are not",
    "wasn't": "was not",
    "weren't": "were not",
    "haven't": "have not",
    "hasn't": "has not",
    "hadn't": "had not",
    "i'm": "i am",
    "you're": "you are",
    "we're": "we are",
    "they're": "they are",
    "it's": "it is",
    "that's": "that is",
    "there's": "there is",
    "here's": "here is",
    "who's": "who is",
    "what's": "what is",
    "let's": "let us",
    "i've": "i have",
    "you've": "you have",
    "we've": "we have",
    "they've": "they have",
    "i'll": "i will",
    "you'll": "you will",
    "we'll": "we will",
    "they'll": "they will",
    "i'd": "i would",
    "you'd": "you would",
    "we'd": "we would",
    "they'd": "they would",
}


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
    n = audio.shape[0] if hasattr(audio, "shape") else len(audio)
    return float(n / sr)


def ensure_wav_path(audio: Path) -> tuple[Path, bool]:
    """onnx_asr reads via wave module (WAV only). Convert flac/ogg/mp3 to temp wav."""
    if audio.suffix.lower() == ".wav":
        return audio, False
    tmp_dir = PROJECT_ROOT / ".tmp" / "quality_audio"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    out = tmp_dir / f"{audio.stem}_16k.wav"
    data, sr = sf.read(str(audio), always_2d=False)
    if getattr(data, "ndim", 1) > 1:
        data = data.mean(axis=1)
    if sr != 16000:
        import numpy as np

        n = int(len(data) * 16000 / sr)
        x_old = np.linspace(0.0, 1.0, num=len(data), endpoint=False)
        x_new = np.linspace(0.0, 1.0, num=n, endpoint=False)
        data = np.interp(x_new, x_old, data.astype(float)).astype("float32")
        sr = 16000
    else:
        data = data.astype("float32")
    sf.write(str(out), data, sr, subtype="PCM_16")
    return out, True


def recognize_once(model, audio: Path) -> str:
    wav_path, _ = ensure_wav_path(audio)
    out = model.recognize(str(wav_path))
    return out if isinstance(out, str) else str(out)


def discover_audio(data_dir: Path) -> list[tuple[str, Path]]:
    found: dict[str, Path] = {}
    for p in sorted(data_dir.iterdir()):
        if not p.is_file():
            continue
        if p.suffix.lower() not in {".wav", ".flac", ".ogg", ".mp3"}:
            continue
        found[p.stem] = p
    ordered: list[tuple[str, Path]] = []
    seen: set[str] = set()
    for name in PREFERRED_AUDIO:
        key = Path(name).stem
        if key in found:
            ordered.append((key, found[key]))
            seen.add(key)
    for key, path in sorted(found.items()):
        if key not in seen:
            ordered.append((key, path))
    return ordered


def normalize_text(text: str) -> str:
    t = (text or "").lower().strip()
    for src, dst in CONTRACTIONS.items():
        t = re.sub(rf"\b{re.escape(src)}\b", dst, t)
    t = t.replace("—", " ").replace("–", " ").replace("−", " ")
    t = t.replace("'", "'").replace("'", "'").replace("`", "'")
    t = re.sub(r"[^\w\s]", " ", t, flags=re.UNICODE)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def token_f1(ref: str, hyp: str) -> float:
    r = normalize_text(ref).split()
    h = normalize_text(hyp).split()
    if not r and not h:
        return 1.0
    if not r or not h:
        return 0.0
    rc, hc = Counter(r), Counter(h)
    overlap = sum((rc & hc).values())
    if overlap == 0:
        return 0.0
    prec = overlap / len(h)
    rec = overlap / len(r)
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


def similarity_ratio(ref: str, hyp: str) -> float:
    a = normalize_text(ref)
    b = normalize_text(hyp)
    if a == b:
        return 1.0
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def compute_wer_cer(ref: str, hyp: str) -> tuple[float, float]:
    from jiwer import cer, wer

    r = normalize_text(ref)
    h = normalize_text(hyp)
    if not r and not h:
        return 0.0, 0.0
    if not r:
        return 1.0, 1.0
    return float(wer(r, h)), float(cer(r, h))


def geo_mean(values: list[float]) -> float:
    vals = [v for v in values if v is not None and v > 0]
    if not vals:
        return float("nan")
    return math.exp(sum(math.log(v) for v in vals) / len(vals))


def benchmark_clip(model, audio: Path, warmup: int, repeats: int) -> dict[str, Any]:
    for _ in range(max(0, warmup)):
        _ = recognize_once(model, audio)
    lats: list[float] = []
    transcript = ""
    for _ in range(max(1, repeats)):
        t0 = time.perf_counter()
        transcript = recognize_once(model, audio)
        lats.append(time.perf_counter() - t0)
    dur = audio_duration_s(audio)
    mean_lat = statistics.mean(lats)
    rtf = mean_lat / dur if dur > 0 else float("inf")
    rtfx = dur / mean_lat if mean_lat > 0 else 0.0
    return {
        "audio_file": audio.name,
        "audio_duration_s": round(dur, 4),
        "warmup": warmup,
        "repeats": repeats,
        "latencies_s": [round(x, 6) for x in lats],
        "latency_mean_s": round(mean_lat, 6),
        "latency_std_s": round(statistics.pstdev(lats) if len(lats) > 1 else 0.0, 6),
        "rtf": round(rtf, 6),
        "rtfx": round(rtfx, 4),
        "transcript": transcript,
        "transcript_nonempty": bool(transcript and str(transcript).strip()),
    }


def evaluate_config(
    label: str,
    cfg_path: Path,
    audio_items: list[tuple[str, Path]],
    warmup: int,
    repeats: int,
) -> dict[str, Any]:
    cfg = load_config(cfg_path)
    print(f"\n=== Evaluating {label}: {cfg_path} ===", flush=True)
    prev = apply_env(cfg.get("env") or {})
    try:
        model, model_dir = load_model(cfg)
        print(f"Model dir: {model_dir}", flush=True)
        clips: dict[str, Any] = {}
        for key, path in audio_items:
            print(f"  [{label}] {key} ({path.name}) ...", flush=True)
            m = benchmark_clip(model, path, warmup=warmup, repeats=repeats)
            clips[key] = m
            preview = (m["transcript"] or "")[:100]
            print(
                f"    rtf={m['rtf']:.6f} nonempty={m['transcript_nonempty']} "
                f"tx={preview!r}",
                flush=True,
            )
        primary_rtfs = [clips[k]["rtf"] for k in PRIMARY_KEYS if k in clips]
        primary = geo_mean(primary_rtfs) if primary_rtfs else float("nan")
        return {
            "label": label,
            "config_path": str(cfg_path.relative_to(PROJECT_ROOT)),
            "model_path": str(Path(cfg.get("model_path", "")).as_posix()),
            "resolved_model_dir": str(model_dir),
            "session": {
                "intra_op_num_threads": cfg.get("intra_op_num_threads"),
                "inter_op_num_threads": cfg.get("inter_op_num_threads"),
                "graph_optimization_level": cfg.get("graph_optimization_level"),
                "enable_mem_pattern": cfg.get("enable_mem_pattern"),
                "enable_cpu_mem_arena": cfg.get("enable_cpu_mem_arena"),
                "execution_mode": cfg.get("execution_mode"),
                "env": cfg.get("env"),
                "provider": cfg.get("provider"),
                "quantization": cfg.get("quantization"),
            },
            "primary_rtf": round(primary, 6) if primary == primary else None,
            "primary_keys": list(PRIMARY_KEYS),
            "clips": clips,
        }
    finally:
        restore_env(prev)


def pairwise_metrics(baseline_tx: str, best_tx: str) -> dict[str, Any]:
    exact = normalize_text(baseline_tx) == normalize_text(best_tx)
    f1 = token_f1(baseline_tx, best_tx)
    sim = similarity_ratio(baseline_tx, best_tx)
    w, c = compute_wer_cer(baseline_tx, best_tx)
    return {
        "exact_match_norm": exact,
        "token_f1": round(f1, 6),
        "similarity": round(sim, 6),
        "wer_vs_baseline": round(w, 6),
        "cer_vs_baseline": round(c, 6),
        "baseline_norm": normalize_text(baseline_tx),
        "best_norm": normalize_text(best_tx),
    }


def absolute_jfk(transcript: str) -> dict[str, Any]:
    w, c = compute_wer_cer(JFK_REFERENCE, transcript)
    return {
        "reference": JFK_REFERENCE,
        "reference_norm": normalize_text(JFK_REFERENCE),
        "hypothesis": transcript,
        "hypothesis_norm": normalize_text(transcript),
        "exact_match_norm": normalize_text(JFK_REFERENCE) == normalize_text(transcript),
        "token_f1": round(token_f1(JFK_REFERENCE, transcript), 6),
        "similarity": round(similarity_ratio(JFK_REFERENCE, transcript), 6),
        "wer": round(w, 6),
        "cer": round(c, 6),
        "nonempty": bool(transcript and str(transcript).strip()),
    }


def decide_freeze(
    pairwise: dict[str, dict[str, Any]],
    jfk_abs: dict[str, dict[str, Any]],
    baseline_primary: float | None,
    best_primary: float | None,
) -> dict[str, Any]:
    reasons: list[str] = []
    pass_flags: dict[str, bool] = {}

    empty_real = False
    for label, rec in jfk_abs.items():
        if not rec.get("nonempty"):
            empty_real = True
            reasons.append(f"empty transcript for {label} on JFK-style clip")
    pass_flags["no_empty_real_speech"] = not empty_real

    wers = [v["wer_vs_baseline"] for v in pairwise.values()]
    exacts = [v["exact_match_norm"] for v in pairwise.values()]
    mean_wer = sum(wers) / len(wers) if wers else 1.0
    n_exact = sum(1 for e in exacts if e)
    majority_exact = n_exact >= max(1, (len(exacts) + 1) // 2)
    near_exact = sum(1 for v in pairwise.values() if v["wer_vs_baseline"] <= 0.05)
    majority_near = near_exact >= max(1, (len(pairwise) + 1) // 2)
    quality_ok = (mean_wer <= 0.05) or majority_exact or majority_near
    pass_flags["pairwise_quality"] = quality_ok
    if quality_ok:
        reasons.append(
            f"pairwise quality OK: mean_wer={mean_wer:.4f}, exact={n_exact}/{len(exacts)}, "
            f"near_exact(<=5% WER)={near_exact}/{len(pairwise)}"
        )
    else:
        reasons.append(
            f"pairwise quality FAIL: mean_wer={mean_wer:.4f}, exact={n_exact}/{len(exacts)}"
        )

    jfk_ok = True
    if "baseline" in jfk_abs and "best" in jfk_abs:
        bw = jfk_abs["baseline"]["wer"]
        ew = jfk_abs["best"]["wer"]
        if ew <= bw + 0.03 or (bw <= 0.05 and ew <= 0.05):
            jfk_ok = True
            reasons.append(f"absolute JFK WER OK: baseline={bw:.4f} best={ew:.4f}")
        else:
            jfk_ok = False
            reasons.append(f"absolute JFK WER FAIL: baseline={bw:.4f} best={ew:.4f}")
    pass_flags["absolute_jfk"] = jfk_ok

    rtf_ok = False
    improvement_pct = None
    if baseline_primary and best_primary and baseline_primary > 0:
        improvement_pct = (baseline_primary - best_primary) / baseline_primary * 100.0
        rtf_ok = improvement_pct >= 5.0
        if rtf_ok:
            reasons.append(
                f"primary RTF improvement OK: {improvement_pct:.3f}% "
                f"(baseline={baseline_primary:.6f} best={best_primary:.6f})"
            )
        else:
            reasons.append(
                f"primary RTF improvement FAIL: {improvement_pct:.3f}% < 5%"
            )
    else:
        reasons.append("primary RTF missing — cannot verify speed gate")
    pass_flags["rtf_improvement"] = rtf_ok

    freeze = all(pass_flags.values())
    recommendation = "FREEZE_PRODUCTION" if freeze else "DO_NOT_FREEZE"
    return {
        "recommendation": recommendation,
        "freeze": freeze,
        "pass_flags": pass_flags,
        "mean_pairwise_wer": round(mean_wer, 6),
        "n_exact_match": n_exact,
        "n_clips": len(pairwise),
        "improvement_pct_primary_rtf": (
            round(improvement_pct, 3) if improvement_pct is not None else None
        ),
        "reasons": reasons,
    }


def write_markdown(report: dict[str, Any], path: Path) -> None:
    b = report["baseline"]
    e = report["best"]
    dec = report["decision"]
    lines: list[str] = []
    lines.append(
        "# Quality assessment: baseline (dynamic INT8) vs best (static QDQ per-channel)"
    )
    lines.append("")
    lines.append(f"Generated: {report['generated_utc']}")
    lines.append("")
    lines.append("## Configs")
    lines.append("")
    lines.append("| Role | Config | Model path | Primary RTF | RTFx (1/RTF) |")
    lines.append("|------|--------|------------|------------:|-------------:|")
    for row in (b, e):
        pr = row.get("primary_rtf")
        rtfx = (1.0 / pr) if pr else None
        pr_s = f"{pr:.6f}" if pr is not None else "n/a"
        rtfx_s = f"{rtfx:.2f}" if rtfx is not None else "n/a"
        lines.append(
            f"| {row['label']} | `{row['config_path']}` | `{row['model_path']}` | "
            f"{pr_s} | {rtfx_s} |"
        )
    imp = dec.get("improvement_pct_primary_rtf")
    lines.append("")
    if imp is not None:
        lines.append(f"**Primary RTF improvement (best vs baseline):** {imp:.3f}%")
    else:
        lines.append("**Primary RTF improvement:** n/a")
    lines.append("")
    lines.append(
        "Primary metric = geometric mean of mean RTF on `medium_15s` + `long_30s`."
    )
    lines.append("")
    lines.append("## Decision")
    lines.append("")
    lines.append(f"**Recommendation: `{dec['recommendation']}`**")
    lines.append("")
    lines.append("| Gate | Pass |")
    lines.append("|------|:----:|")
    for k, v in dec["pass_flags"].items():
        lines.append(f"| {k} | {'YES' if v else 'NO'} |")
    lines.append("")
    lines.append("Reasons:")
    for r in dec["reasons"]:
        lines.append(f"- {r}")
    lines.append("")
    lines.append("## Pairwise quality (best vs baseline as reference)")
    lines.append("")
    lines.append(
        "| Clip | Exact (norm) | Token F1 | Similarity | WER | CER | "
        "Baseline nonempty | Best nonempty |"
    )
    lines.append(
        "|------|:------------:|---------:|-----------:|----:|----:"
        "|:-----------------:|:-------------:|"
    )
    for key, m in report["pairwise"].items():
        bc = b["clips"][key]
        ec = e["clips"][key]
        lines.append(
            f"| {key} | {'YES' if m['exact_match_norm'] else 'NO'} | "
            f"{m['token_f1']:.4f} | {m['similarity']:.4f} | "
            f"{m['wer_vs_baseline']:.4f} | {m['cer_vs_baseline']:.4f} | "
            f"{'YES' if bc['transcript_nonempty'] else 'NO'} | "
            f"{'YES' if ec['transcript_nonempty'] else 'NO'} |"
        )
    lines.append("")
    lines.append(
        f"**Mean pairwise WER:** {dec['mean_pairwise_wer']:.4f}  \n"
        f"**Exact matches:** {dec['n_exact_match']}/{dec['n_clips']}"
    )
    lines.append("")
    lines.append("## Absolute quality vs JFK reference")
    lines.append("")
    lines.append(f"Reference: _{JFK_REFERENCE}_")
    lines.append("")
    lines.append(
        "| Config | Clip | Exact | Token F1 | Similarity | WER | CER | Nonempty |"
    )
    lines.append(
        "|--------|------|:-----:|---------:|-----------:|----:|----:|:--------:|"
    )
    for label, rec in report["absolute_jfk"].items():
        clip = rec.get("clip", "real_speech/jfk")
        lines.append(
            f"| {label} | {clip} | {'YES' if rec['exact_match_norm'] else 'NO'} | "
            f"{rec['token_f1']:.4f} | {rec['similarity']:.4f} | "
            f"{rec['wer']:.4f} | {rec['cer']:.4f} | "
            f"{'YES' if rec['nonempty'] else 'NO'} |"
        )
    lines.append("")
    lines.append("## Per-clip RTF")
    lines.append("")
    lines.append(
        "| Clip | Dur (s) | Baseline RTF | Best RTF | Baseline RTFx | Best RTFx | Δ RTF % |"
    )
    lines.append(
        "|------|--------:|-------------:|---------:|--------------:|----------:|--------:|"
    )
    for key in b["clips"]:
        bc = b["clips"][key]
        ec = e["clips"][key]
        br, er = bc["rtf"], ec["rtf"]
        delta = ((br - er) / br * 100.0) if br else float("nan")
        lines.append(
            f"| {key} | {bc['audio_duration_s']:.2f} | {br:.6f} | {er:.6f} | "
            f"{bc['rtfx']:.2f} | {ec['rtfx']:.2f} | {delta:.2f} |"
        )
    lines.append("")
    lines.append("## Per-clip transcripts")
    lines.append("")
    for key in b["clips"]:
        lines.append(f"### {key}")
        lines.append("")
        lines.append(f"- **baseline:** {b['clips'][key]['transcript']!r}")
        lines.append(f"- **best:**     {e['clips'][key]['transcript']!r}")
        if key in report["pairwise"]:
            m = report["pairwise"][key]
            lines.append(f"- **norm baseline:** `{m['baseline_norm']}`")
            lines.append(f"- **norm best:**     `{m['best_norm']}`")
        lines.append("")
    lines.append("## Eval set notes")
    lines.append("")
    for n in report.get("eval_notes") or []:
        lines.append(f"- {n}")
    lines.append("")
    lines.append("## How to reproduce")
    lines.append("")
    lines.append("```bash")
    lines.append("cd /path/to/parakeet-optimization")
    lines.append("source venv/bin/activate")
    lines.append("python scripts/quality_baseline_vs_best.py --warmup 1 --repeats 3")
    lines.append("```")
    lines.append("")
    lines.append("## Freeze action")
    lines.append("")
    if dec["freeze"]:
        lines.append(
            "- Keep `configs/best_config.json` as **production default** "
            "(static QDQ per-channel encoder)."
        )
        lines.append(
            "- Keep `configs/baseline.json` as **Hub dynamic INT8 reference**."
        )
        lines.append("- Run production: `python scripts/apply_best_config.py`")
        lines.append(
            "- A/B baseline: "
            "`python scripts/apply_best_config.py --config configs/baseline.json`"
        )
    else:
        lines.append(
            "- **Do not freeze.** Leave `best_config.json` experimental; document warning."
        )
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Quality baseline vs best for Parakeet CPU")
    p.add_argument("--baseline", type=str, default=str(DEFAULT_BASELINE))
    p.add_argument("--best", type=str, default=str(DEFAULT_BEST))
    p.add_argument("--data-dir", type=str, default=str(DEFAULT_DATA))
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--out-json", type=str, default=str(DEFAULT_OUT_JSON))
    p.add_argument("--out-md", type=str, default=str(DEFAULT_OUT_MD))
    args = p.parse_args(argv)

    os.chdir(PROJECT_ROOT)
    baseline_path = Path(args.baseline)
    best_path = Path(args.best)
    data_dir = Path(args.data_dir)
    if not baseline_path.is_absolute():
        baseline_path = PROJECT_ROOT / baseline_path
    if not best_path.is_absolute():
        best_path = PROJECT_ROOT / best_path
    if not data_dir.is_absolute():
        data_dir = PROJECT_ROOT / data_dir

    if not baseline_path.exists():
        print(f"ERROR: baseline config missing: {baseline_path}", file=sys.stderr)
        return 1
    if not best_path.exists():
        print(f"ERROR: best config missing: {best_path}", file=sys.stderr)
        return 1
    if not data_dir.exists():
        print(f"ERROR: data dir missing: {data_dir}", file=sys.stderr)
        return 1

    audio_items = discover_audio(data_dir)
    if not audio_items:
        print("ERROR: no audio files found under data/", file=sys.stderr)
        return 1
    print("Audio set:", [(k, p.name) for k, p in audio_items], flush=True)

    eval_notes = [
        "Eval set is small and mostly English clean speech.",
        "short_5s / medium_15s / long_30s are typically looped variants of the same JFK phrase; "
        "real_speech.wav and jfk.flac are the natural JFK inaugural line.",
        "This is NOT a full multilingual WER suite; pairwise vs baseline + absolute JFK only.",
        "No extra public speech samples downloaded (disk/time careful; optional).",
    ]

    baseline = evaluate_config(
        "baseline", baseline_path, audio_items, args.warmup, args.repeats
    )
    best = evaluate_config(
        "best", best_path, audio_items, args.warmup, args.repeats
    )

    pairwise: dict[str, dict[str, Any]] = {}
    for key in baseline["clips"]:
        if key not in best["clips"]:
            continue
        pairwise[key] = pairwise_metrics(
            baseline["clips"][key]["transcript"],
            best["clips"][key]["transcript"],
        )
        print(
            f"pairwise[{key}]: exact={pairwise[key]['exact_match_norm']} "
            f"wer={pairwise[key]['wer_vs_baseline']:.4f} "
            f"f1={pairwise[key]['token_f1']:.4f}",
            flush=True,
        )

    absolute_jfk_map: dict[str, dict[str, Any]] = {}
    for label, pack in (("baseline", baseline), ("best", best)):
        clip_key = (
            "real_speech"
            if "real_speech" in pack["clips"]
            else ("jfk" if "jfk" in pack["clips"] else next(iter(pack["clips"])))
        )
        rec = absolute_jfk(pack["clips"][clip_key]["transcript"])
        rec["clip"] = clip_key
        absolute_jfk_map[label] = rec
        print(
            f"absolute_jfk[{label}/{clip_key}]: wer={rec['wer']:.4f} "
            f"exact={rec['exact_match_norm']} nonempty={rec['nonempty']}",
            flush=True,
        )

    absolute_jfk_per_clip: dict[str, dict[str, Any]] = {}
    for clip_key in ("real_speech", "jfk"):
        if clip_key not in baseline["clips"]:
            continue
        absolute_jfk_per_clip[clip_key] = {
            "baseline": absolute_jfk(baseline["clips"][clip_key]["transcript"]),
            "best": absolute_jfk(best["clips"][clip_key]["transcript"]),
        }

    decision = decide_freeze(
        pairwise,
        absolute_jfk_map,
        baseline.get("primary_rtf"),
        best.get("primary_rtf"),
    )
    for clip_key in ("real_speech", "jfk"):
        for pack in (baseline, best):
            if clip_key in pack["clips"] and not pack["clips"][clip_key]["transcript_nonempty"]:
                decision["pass_flags"]["no_empty_real_speech"] = False
                decision["freeze"] = False
                decision["recommendation"] = "DO_NOT_FREEZE"
                decision["reasons"].append(
                    f"empty transcript: {pack['label']}/{clip_key}"
                )

    report = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "host": {
            "cpu_model": _cpu_model(),
            "cpu_count_logical": os.cpu_count(),
        },
        "protocol": {
            "warmup": args.warmup,
            "repeats": args.repeats,
            "primary_keys": list(PRIMARY_KEYS),
            "normalization": (
                "lowercase, strip punctuation, collapse whitespace, "
                "expand common contractions"
            ),
            "jfk_reference": JFK_REFERENCE,
        },
        "eval_notes": eval_notes,
        "baseline": baseline,
        "best": best,
        "pairwise": pairwise,
        "absolute_jfk": absolute_jfk_map,
        "absolute_jfk_per_clip": absolute_jfk_per_clip,
        "decision": decision,
    }

    out_json = Path(args.out_json)
    out_md = Path(args.out_md)
    if not out_json.is_absolute():
        out_json = PROJECT_ROOT / out_json
    if not out_md.is_absolute():
        out_md = PROJECT_ROOT / out_md
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
        f.write("\n")
    write_markdown(report, out_md)

    print("\n=== DECISION ===", flush=True)
    print(json.dumps(decision, indent=2), flush=True)
    print(f"Wrote {out_json}", flush=True)
    print(f"Wrote {out_md}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

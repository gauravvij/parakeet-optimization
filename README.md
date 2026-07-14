# Parakeet TDT 0.6B v3 — CPU Optimization

Portable **CPU-only** optimization notes and tooling for
[`nvidia/parakeet-tdt-0.6b-v3`](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3)
via community INT8 ONNX + ONNX Runtime.

**Headline (AMD EPYC 9V74, 8 vCPU):** static QDQ per-channel **encoder** cut primary RTF by **~52%** vs Hub dynamic INT8 (~26× → ~54× real-time), with quality gates passing on a small English eval set.

> This is **standard ORT static quantization** applied carefully to the FastConformer encoder (QDQ MinMax; QOperator / Percentile failed on this graph) — not a new architecture. Contribution is the measured ladder, keep/discard gates, production freeze, and reproducible scripts.

**Repo:** [github.com/gauravvij/parakeet-optimization](https://github.com/gauravvij/parakeet-optimization)  
**Production weights:** [`gvij/parakeet-tdt-0.6b-v3-onnx-static-qdq-pc`](https://huggingface.co/gvij/parakeet-tdt-0.6b-v3-onnx-static-qdq-pc)

## Production default vs Hub default (frozen)

| | **Production default** | **Hub / dynamic reference** |
|--|------------------------|-----------------------------|
| Config | `configs/best_config.json` (alias `configs/production.json`) | `configs/baseline.json` |
| Model path | `models/parakeet-tdt-0.6b-v3-onnx-static-qdq-pc` | `models/parakeet-tdt-0.6b-v3-onnx` |
| Encoder quant | **Static QDQ MinMax per-channel** | Dynamic-activation INT8 (Hub) |
| Primary RTF* | **0.018432** | **0.038156** |
| RTFx (1/RTF) | **~54.3×** | **~26.2×** |
| vs Hub | **~52% lower RTF** | — |
| Quality (JFK abs WER) | **0.0** (exact norm match) | **0.0** (exact norm match) |
| Pairwise vs Hub | exact **3/5** clips; mean WER ~0.09 (looped synth tails differ) | reference |

\*Primary = geo-mean of mean RTF on `medium_15s` + `long_30s` (remeasured in quality assessment).  
Freeze decision + full tables: [`reports/production_default.md`](reports/production_default.md), [`results/quality_baseline_vs_best.md`](results/quality_baseline_vs_best.md).

### How to run production

```bash
source venv/bin/activate
python scripts/apply_best_config.py
python scripts/apply_best_config.py --audio data/real_speech.wav
python scripts/apply_best_config.py --benchmark --warmup 1 --repeats 3
# or explicit alias:
python scripts/apply_best_config.py --config configs/production.json
```

### Long audio (multi-minute) — avoid OOM

Default production is **full-file** `recognize`. Encoder activations scale with duration; multi-minute files (e.g. a ~30 min speech) can climb to **tens of GB RSS** and get killed. That is expected, not a bad config path.

Use **app-level chunking** to bound peak RAM (window + overlap + transcript merge — **not** true streaming / encoder cache):

```bash
python scripts/apply_best_config.py \
  --config configs/production.json \
  --audio /path/to/long_talk.wav \
  --chunk-window-s 30

# tighter RAM / more boundaries:
python scripts/apply_best_config.py --audio /path/to/long_talk.wav \
  --chunk-window-s 15 --chunk-overlap-s 2

# force full-file even if a config enables chunking:
python scripts/apply_best_config.py --audio data/real_speech.wav --no-chunk
```

| Flag | Meaning |
|------|---------|
| `--chunk-window-s SEC` | Enable chunking with this window (try **15–60**; start at **30**) |
| `--chunk-overlap-s SEC` | Overlap for stitch (default **2** when window is set) |
| `--no-chunk` | Full-file path (default for frozen production JSON) |

**Tradeoffs:** lower peak memory; may add boundary artifacts; RTF can be worse than full-file on short clips (E5 ladder did not keep chunking for speed). Frozen production RTF numbers assume full-file short/medium clips.

### How to run Hub baseline (A/B)

```bash
python scripts/apply_best_config.py --config configs/baseline.json --audio data/real_speech.wav
python scripts/apply_best_config.py --config configs/baseline.json --benchmark
```

### How to reproduce quality assessment

```bash
pip install -r requirements.txt   # includes jiwer
python scripts/quality_baseline_vs_best.py --warmup 1 --repeats 3
# → results/quality_baseline_vs_best.json + .md
```

### Caveats

- Eval set is **small** and mostly the same English JFK phrase (looped 5/15/30 s + natural clip) — **not** a full multilingual WER suite.
- Speed numbers are **host-specific** (AMD EPYC 9V74, 8 vCPU, ORT 1.27 CPU).
- Technique is **standard ORT static quant**, not a new architecture; Hub still ships dynamic INT8 by default.
- **Long audio OOM:** full-file path is not safe for multi-minute files — use `--chunk-window-s` (see above).

## Host used for published numbers

- AMD EPYC 9V74, 8 vCPU, AVX-512 + VNNI + BF16
- ~63 GB RAM, **no GPU**
- ONNX Runtime 1.27.0 CPU EP, `onnx-asr` 0.11.0
- Peak RSS ~1.5 GB (INT8)

Headline: Hub dynamic INT8 ~**26×** real-time on 30 s; **production static encoder ~54×** (primary RTF ~0.018). Encoder ≈ **98%** of stage time.

Full analysis: [`reports/parakeet_tdt_v3_cpu_optimization.md`](reports/parakeet_tdt_v3_cpu_optimization.md) · Production freeze: [`reports/production_default.md`](reports/production_default.md)

## Layout

```
parakeet-optimization/
├── scripts/
│   ├── profile_parakeet_cpu.py       # profiling harness
│   ├── autoresearch_cpu_opts.py      # runtime keep/discard ladder (E0–E6)
│   ├── autoresearch_encoder_opts.py  # encoder static-quant ladder (C0–C3)
│   ├── quality_baseline_vs_best.py   # quality + RTF: baseline vs production
│   └── apply_best_config.py          # run inference with best/production config
├── configs/
│   ├── baseline.json                 # Hub dynamic INT8 reference (A/B)
│   ├── best_config.json              # PRODUCTION default (static QDQ-pc)
│   └── production.json               # alias of best_config.json
├── models/                           # weights gitignored — see models/README.md
├── data/*.wav                        # 16 kHz mono samples
├── results/                          # ladder ledgers + quality report
├── reports/
│   ├── parakeet_tdt_v3_cpu_optimization.md
│   └── production_default.md
├── requirements.txt
└── LICENSE
```

## Quick start

```bash
git clone https://github.com/gauravvij/parakeet-optimization.git
cd parakeet-optimization
python3 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -U pip
pip install -r requirements.txt
```

### Download model weights

Weights are **not** in git (~640MB+). See [`models/README.md`](models/README.md).

**Production (static QDQ-pc encoder)** — required for default config:

```bash
source venv/bin/activate
python - <<'PY'
from pathlib import Path
from huggingface_hub import snapshot_download
snapshot_download(
    "gvij/parakeet-tdt-0.6b-v3-onnx-static-qdq-pc",
    local_dir="models/parakeet-tdt-0.6b-v3-onnx-static-qdq-pc",
)
print("OK")
PY
```

**Hub dynamic INT8 (A/B baseline)** — optional:

```bash
python - <<'PY'
from pathlib import Path
from huggingface_hub import hf_hub_download
repo = "istupakov/parakeet-tdt-0.6b-v3-onnx"
local = Path("models/parakeet-tdt-0.6b-v3-onnx")
local.mkdir(parents=True, exist_ok=True)
for f in [
    "config.json", "vocab.txt", "nemo128.onnx",
    "encoder-model.int8.onnx", "decoder_joint-model.int8.onnx",
]:
    print(hf_hub_download(repo, f, local_dir=str(local)))
PY
```

**Important:** `onnx-asr` globs like `encoder-model?int8.onnx` do not match Hub names
`encoder-model.int8.onnx` — download explicitly as above.

Do **not** install full NeMo/CUDA stacks on disk-constrained hosts; INT8 ONNX is sufficient.

### Transcribe

```bash
# Production (static encoder) — prefer --benchmark for fair RTF (excludes load)
python scripts/apply_best_config.py --config configs/production.json --audio data/real_speech.wav
python scripts/apply_best_config.py --config configs/production.json --audio data/real_speech.wav --benchmark

# A/B vs Hub dynamic INT8
python scripts/apply_best_config.py --config configs/baseline.json --audio data/real_speech.wav --benchmark

# Multi-minute files (bounds peak RAM; not the frozen RTF path)
python scripts/apply_best_config.py --config configs/production.json \
  --audio /path/to/long_talk.wav --chunk-window-s 30
```

**Note:** shell `time python scripts/apply_best_config.py ...` includes **model load** (~1.5–2s). Production can look similar or slower on short one-shot runs; warm RTF (via `--benchmark`) is the metric optimized here.

**Long audio:** without `--chunk-window-s`, the whole file is encoded in one shot and multi-minute audio can OOM. See [Long audio](#long-audio-multi-minute--avoid-oom).

### Sample audio

Place 16 kHz mono WAVs under `data/`:

| File | Role |
|------|------|
| `short_5s.wav` | ~5 s |
| `medium_15s.wav` | ~15 s |
| `long_30s.wav` | ~30 s |
| `real_speech.wav` | real speech sanity (e.g. Whisper JFK sample) |

Example (JFK → 16 kHz mono + length variants):

```bash
source venv/bin/activate
python - <<'PY'
from pathlib import Path
import urllib.request, numpy as np, soundfile as sf
data = Path("data"); data.mkdir(exist_ok=True)
urllib.request.urlretrieve(
    "https://github.com/openai/whisper/raw/main/tests/jfk.flac",
    data / "jfk.flac",
)
audio, sr = sf.read(data / "jfk.flac")
if audio.ndim > 1: audio = audio.mean(1)
if sr != 16000:
    n = int(len(audio) * 16000 / sr)
    audio = np.interp(np.linspace(0,1,n,endpoint=False),
                      np.linspace(0,1,len(audio),endpoint=False),
                      audio.astype(float)).astype(np.float32)
else:
    audio = audio.astype(np.float32)
base = audio
while len(base) < 35 * 16000:
    base = np.concatenate([base, audio])
for name, dur in [("short_5s.wav",5),("medium_15s.wav",15),("long_30s.wav",30)]:
    sf.write(data/name, base[:int(dur*16000)], 16000, subtype="PCM_16")
sf.write(data/"real_speech.wav", audio, 16000, subtype="PCM_16")
print("OK")
PY
```

## Run profiling

```bash
source venv/bin/activate
python scripts/profile_parakeet_cpu.py \
  --threads 1,2,4,8 \
  --audio short_5s,medium_15s,long_30s,real_speech \
  --warmup 1 --repeats 3 \
  --profile-threads 4 --profile-audio medium_15s
```

Useful flags:

| Flag | Default | Meaning |
|------|---------|---------|
| `--skip-baseline` | off | Only stage/operator profile |
| `--skip-operator` | off | Only baseline RTF sweep |
| `--threads` | `1,2,4,8` | ORT `intra_op_num_threads` list |
| `--profile-threads` | `4` | Threads for stage/ORT profile |
| `--warmup` / `--repeats` | `1` / `3` | Timing protocol |

## Outputs

| Path | Contents |
|------|----------|
| `results/baseline_metrics.json` | Per (audio × threads): latency, RTF, RTFx, peak RSS, transcript |
| `results/operator_profile.json` | Frontend / encoder / decoder % + top ORT ops |
| `results/ort_profiles/*.json` | Raw ORT profiling traces |
| `reports/parakeet_tdt_v3_cpu_optimization.md` | Full report + ranked optimizations |

## Quick smoke test

```bash
source venv/bin/activate
python - <<'PY'
import onnx_asr, onnxruntime as ort
so = ort.SessionOptions(); so.intra_op_num_threads = 4
m = onnx_asr.load_model(
    "nemo-parakeet-tdt-0.6b-v3",
    path="models/parakeet-tdt-0.6b-v3-onnx",
    quantization="int8",
    providers=["CPUExecutionProvider"],
    sess_options=so,
)
print(m.recognize("data/real_speech.wav"))
PY
```

Expected: non-empty English transcript of the JFK line.

## Autoresearch CPU optimization ladder

Portable keep/discard ladder over runtime knobs (no retrain). Primary metric =
geometric mean of mean **RTF** on `medium_15s` + `long_30s` (lower is better).
Keep only if improvement ≥ **5%** and `real_speech` quality does not regress.

### Run the ladder

```bash
cd /path/to/parakeet-optimization
source venv/bin/activate
python scripts/autoresearch_cpu_opts.py --warmup 1 --repeats 3
```

Optional flags:

| Flag | Meaning |
|------|---------|
| `--skip-openvino` | Do not attempt OpenVINO EP (default path skips cleanly if unavailable) |
| `--skip-chunking` | Skip app-level long-audio chunking (E5) |
| `--warmup` / `--repeats` | Timing protocol (defaults 1 / 3) |

Experiments:

| Step | What |
|------|------|
| E0 | Remeasure baseline (`ORT_ENABLE_ALL`, inter_op=1); confirm threads 4 vs 8 |
| E1 | Thread/env matrix (`intra_op` 2/4/8, `OMP_*`, optional affinity) |
| E2 | Session/memory opts (mem pattern, CPU arena, sequential vs parallel) |
| E3 | Offline ORT graph optimize → `models/parakeet-tdt-0.6b-v3-onnx-opt/` |
| E4 | Optional OpenVINO EP (skip if not installed / disk tight) |
| E5 | App-level chunk+concat for `long_30s` (not true streaming) |
| E6 | Stack kept winners and remeasure vs E0 |

Outputs: `results/autoresearch/ledger.jsonl`, `results/autoresearch/summary.md`,
`configs/baseline.json`, `configs/best_config.json`.

### Apply best config (any developer)

```bash
source venv/bin/activate
python scripts/apply_best_config.py
python scripts/apply_best_config.py --audio data/real_speech.wav
python scripts/apply_best_config.py --benchmark --warmup 1 --repeats 3
python scripts/apply_best_config.py --config configs/baseline.json --audio data/medium_15s.wav
```

`best_config.json` is config-driven (threads, env, provider, optional optimized
model dir / chunking). No host secrets. Other machines: install slim deps, place
INT8 model + audio, run autoresearch or apply an existing best_config as a starting point.

## Encoder static-quant ladder (C0–C3)

Runtime knobs (E0–E6) did **not** clear the ≥5% primary RTF gate on this host
(best residual ~4–4.8%). The encoder is ~98% of E2E and the Hub INT8 graph uses
**dynamic-activation** MatMuls (`DynamicQuantizeMatMul`). Re-quantizing the
**encoder only** from FP32 with offline activation scales (QDQ) is the portable
win.

### Run the encoder ladder

```bash
source venv/bin/activate
# needs: onnx (+ sympy) for onnxruntime.quantization; does not replace ORT wheel
python scripts/autoresearch_encoder_opts.py --warmup 1 --repeats 3
```

| Step | Recipe | Notes |
|------|--------|-------|
| C0 | Dynamic INT8 baseline | `models/parakeet-tdt-0.6b-v3-onnx` |
| C1 | QDQ MinMax (per-tensor) | full encoder static quant |
| C2 | QDQ MinMax **per-channel** weights | best on this host |
| C3 | MatMul/Gemm-only QDQ + residual env | OMP ACTIVE, arena off |

Primary metric and keep gate match the runtime ladder: geo-mean RTF of
`medium_15s` + `long_30s`, keep if ≥**5%** vs C0 and `real_speech` quality OK.

Outputs: `results/autoresearch_encoder/ledger.jsonl`,
`results/autoresearch_encoder/summary.md`, static packs under
`models/parakeet-tdt-0.6b-v3-onnx-static-*`, and `configs/best_config.json`
**only if** a keep wins.

### Results on this host (AMD EPYC 9V74, 8 vCPU)

| Exp | primary_rtf | vs C0 | keep |
|-----|------------:|------:|:----:|
| C0 dynamic INT8 | 0.038742 | — | ref |
| C1 QDQ MinMax | 0.017709 | **+54.3%** | YES |
| C2 QDQ per-channel | **0.017692** | **+54.3%** | YES (best) |
| C3 MatMul-only + env | 0.021249 | **+45.2%** | YES |

`best_config.json` points at `models/parakeet-tdt-0.6b-v3-onnx-static-qdq-pc`.
Decoder/frontend stay the original INT8 Hub assets; only the encoder is static.

```bash
python scripts/apply_best_config.py --audio data/real_speech.wav
# → non-empty JFK transcript via static encoder pack
```

FP32 quant source (`encoder-model.onnx` + `.data`, ~2.5 GB) can be removed after
export to free disk; re-download from `istupakov/parakeet-tdt-0.6b-v3-onnx` if
re-quantizing.

## Key findings (this machine)

- **RTFx** scales from ~7.5× (1 thread) to ~24–26× (8 threads) on 15–30 s audio with **dynamic** INT8; **static QDQ encoder** reaches ~**50–60×** on 15–30 s (primary RTF ~0.018).
- **Encoder ≈ 97–98%** of stage time; mel frontend and TDT decoder ~1% each (offline greedy).
- Encoder kernels (dynamic INT8): **`ConvInteger`**, **`DynamicQuantizeMatMul`**, **`MatMulIntegerToFloat`**. Static QDQ removes runtime activation quant on FFN/attn MatMuls → **~54% primary RTF cut**.
- Runtime-only levers (threads/env/session/offline ORT opt) plateaued **below 5%**; encoder static re-quant is the portable ≥5% win. OpenVINO EP not available on this ORT wheel.

## License notes

- NVIDIA Parakeet weights: check model card (typically CC-BY-4.0 for this family).
- ONNX community export: see Hub repo license.
- Sample JFK audio: public domain speech recording used via Whisper test assets.

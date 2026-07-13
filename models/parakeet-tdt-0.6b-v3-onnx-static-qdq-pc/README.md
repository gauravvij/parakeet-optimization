---
license: cc-by-4.0
library_name: onnx-asr
tags:
  - automatic-speech-recognition
  - speech-to-text
  - onnx
  - onnxruntime
  - int8
  - quantization
  - cpu
  - parakeet
  - nvidia
  - multilingual
language:
  - multilingual
  - en
base_model:
  - nvidia/parakeet-tdt-0.6b-v3
pipeline_tag: automatic-speech-recognition
---

# Parakeet TDT 0.6B v3 — Static QDQ Per-Channel Encoder (ONNX INT8)

CPU-optimized ONNX pack for [`nvidia/parakeet-tdt-0.6b-v3`](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) with a **static QDQ MinMax per-channel INT8 encoder**. Decoder/joint and frontend stay as Hub dynamic INT8.

This is **standard ONNX Runtime static quantization** applied carefully to the FastConformer encoder — not a new architecture. The contribution is a measured keep/discard ladder, quality gates, and a frozen production pack.

| Item | Value |
|------|--------|
| Hub repo | [`gvij/parakeet-tdt-0.6b-v3-onnx-static-qdq-pc`](https://huggingface.co/gvij/parakeet-tdt-0.6b-v3-onnx-static-qdq-pc) |
| Base model | [`nvidia/parakeet-tdt-0.6b-v3`](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) |
| Dynamic INT8 base pack | [`istupakov/parakeet-tdt-0.6b-v3-onnx`](https://huggingface.co/istupakov/parakeet-tdt-0.6b-v3-onnx) |
| Optimization notes | [github.com/gauravvij/parakeet-optimization](https://github.com/gauravvij/parakeet-optimization) |
| License (weights family) | **CC-BY-4.0** (NVIDIA Parakeet family; attribute NVIDIA) |

## Headline metrics (AMD EPYC 9V74, 8 vCPU)

Measured with ONNX Runtime **1.27.0** CPU EP, `onnx-asr` **0.11.0**, `intra_op=8`, `inter_op=1`.

Primary metric = geometric mean of mean RTF on `medium_15s` + `long_30s` (warm inference).

| Config | Encoder | Primary RTF | RTFx (1/RTF) | vs Hub dynamic INT8 |
|--------|---------|------------:|-------------:|--------------------:|
| Hub dynamic INT8 (`istupakov`) | Dynamic-activation INT8 | **0.038156** | ~26.2× | — |
| **This pack (production)** | **Static QDQ MinMax per-channel** | **0.018432** | ~**54.3×** | **~51.7% lower RTF** |

Absolute JFK WER (normalized) on natural speech: **0.0** for both this pack and Hub dynamic INT8 on the project eval clip.

Full tables: [production freeze report](https://github.com/gauravvij/parakeet-optimization/blob/main/reports/production_default.md) · [quality assessment](https://github.com/gauravvij/parakeet-optimization/blob/main/results/quality_baseline_vs_best.md).

## What changed vs Hub dynamic INT8

| Component | Source | Quantization |
|-----------|--------|--------------|
| **Encoder** | Re-quantized from FP32 encoder export | **Static QDQ**, MinMax calibration, **per-channel** weights |
| Decoder / joint | From `istupakov` pack | Dynamic INT8 (unchanged) |
| Frontend (`nemo128`) | From `istupakov` pack | As shipped |
| Vocab / config | From `istupakov` pack | Unchanged |

Ladder notes (project): QOperator and Percentile calibration failed on this FastConformer graph; **QDQ MinMax** worked. Per-channel (C2) was selected as production over per-tensor (C1) and MatMul-only (C3).

## Files

| File | Role | Approx. size |
|------|------|-------------:|
| `encoder-model.int8.onnx` | Static QDQ encoder graph | ~4.5 MB |
| `encoder-model.int8.onnx.data` | External encoder weights | ~620 MB |
| `decoder_joint-model.int8.onnx` | Decoder + joint INT8 | ~17 MB |
| `nemo128.onnx` | Mel frontend | ~0.1 MB |
| `vocab.txt` | Tokenizer vocabulary | — |
| `config.json` | `onnx-asr` model config (`nemo-conformer-tdt`) | — |
| `README.md` | This model card | — |

`config.json`:

```json
{
  "model_type": "nemo-conformer-tdt",
  "features_size": 128,
  "subsampling_factor": 8
}
```

## How to load with `onnx-asr`

```bash
pip install "onnx-asr[cpu,hub]==0.11.0" onnxruntime==1.27.0 soundfile huggingface_hub
```

```python
from pathlib import Path
from huggingface_hub import snapshot_download
import onnx_asr
import onnxruntime as ort

local = Path("parakeet-tdt-0.6b-v3-onnx-static-qdq-pc")
snapshot_download(
    "gvij/parakeet-tdt-0.6b-v3-onnx-static-qdq-pc",
    local_dir=str(local),
)

so = ort.SessionOptions()
so.intra_op_num_threads = 8
so.inter_op_num_threads = 1
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

model = onnx_asr.load_model(
    "nemo-parakeet-tdt-0.6b-v3",
    path=str(local),
    quantization="int8",
    providers=["CPUExecutionProvider"],
    sess_options=so,
)

# Prefer 16 kHz mono WAV
text = model.recognize("audio.wav")
print(text)
```

### With the optimization repo CLI

```bash
git clone https://github.com/gauravvij/parakeet-optimization.git
cd parakeet-optimization
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
# download this pack into models/parakeet-tdt-0.6b-v3-onnx-static-qdq-pc (see models/README.md)
python scripts/apply_best_config.py --config configs/production.json --audio data/real_speech.wav
python scripts/apply_best_config.py --config configs/production.json --benchmark --warmup 1 --repeats 3
```

## Intended use

- **Primary:** CPU batch / offline ASR with ONNX Runtime where encoder RTF dominates.
- **Good fit:** x86_64 CPUs with AVX-512 / VNNI (e.g. recent AMD EPYC, Intel Xeon).
- **Not a drop-in claim for every SKU:** absolute RTF is host-specific; relative gain vs dynamic INT8 should remain large when VNNI/AVX-512 is available.

## Quality caveats

1. Project eval set is **small** and mostly **English** clean speech (JFK phrase loops + natural clip) — **not** a full multilingual / multi-domain WER suite.
2. Pairwise vs Hub dynamic INT8: exact match on **3/5** clips; mean pairwise WER ~0.09 (looped synthetic tails can differ slightly). Natural JFK line: **exact** normalized match, WER **0.0**.
3. Static quant can change edge behavior on long/looped audio; validate on your domain before production.
4. One-shot CLI wall clock includes **model load** (~1.5–2 s); compare **warm RTF** for fair speed claims.

## How this pack was built (summary)

1. Start from community dynamic INT8 ONNX (`istupakov/parakeet-tdt-0.6b-v3-onnx`).
2. Obtain FP32 encoder; run ORT `quantize_static` with **QDQ**, **MinMax**, **per_channel=True**.
3. Keep Hub INT8 decoder/joint + frontend; assemble pack with `config.json` / `vocab.txt`.
4. Gate on ≥5% primary RTF improvement + non-empty / quality checks vs dynamic baseline.
5. Freeze as production default in [parakeet-optimization](https://github.com/gauravvij/parakeet-optimization).

Scripts: `scripts/autoresearch_encoder_opts.py`, `scripts/quality_baseline_vs_best.py`.

## License and attribution

- **Model weights** derive from NVIDIA Parakeet TDT 0.6B v3 and community ONNX exports. Treat the weight family as **CC-BY-4.0**: attribute **NVIDIA** (and community exporters as appropriate).
- **Static encoder quant + packaging** by [gvij](https://huggingface.co/gvij); methodology and code under MIT in the GitHub repo.
- Please cite / link:
  - [nvidia/parakeet-tdt-0.6b-v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3)
  - [istupakov/parakeet-tdt-0.6b-v3-onnx](https://huggingface.co/istupakov/parakeet-tdt-0.6b-v3-onnx)
  - [github.com/gauravvij/parakeet-optimization](https://github.com/gauravvij/parakeet-optimization)

```
@misc{parakeet-tdt-0.6b-v3,
  title  = {Canary-1B-v2 \& Parakeet-TDT-0.6B-v3: Efficient and High-Performance Multilingual Speech Recognition},
  author = {NVIDIA},
  year   = {2025},
  url    = {https://arxiv.org/abs/2509.14128}
}
```

## Disclaimer

Provided as-is for research and engineering. Validate latency and accuracy on your hardware and audio domain before production deployment.

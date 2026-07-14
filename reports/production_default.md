# Production default freeze — Parakeet TDT 0.6B v3 CPU

**Decision: `FREEZE_PRODUCTION`**  
**Date (UTC):** 2026-07-13  
**Host:** AMD EPYC 9V74, 8 vCPU, ORT 1.27.0 CPU EP

## What is frozen

| Role | Config | Model path | Description |
|------|--------|------------|-------------|
| **Production default** | `configs/best_config.json` (alias: `configs/production.json`) | `models/parakeet-tdt-0.6b-v3-onnx-static-qdq-pc` | Static QDQ MinMax **per-channel** encoder + Hub INT8 decoder/frontend |
| **Hub / A/B reference** | `configs/baseline.json` | `models/parakeet-tdt-0.6b-v3-onnx` | Shipped dynamic-activation INT8 pack (`istupakov`) |

Session knobs (both): `intra_op=8`, `inter_op=1`, `ORT_ENABLE_ALL`, mem pattern + CPU arena on, sequential, `OMP_NUM_THREADS=8`, `OMP_WAIT_POLICY=PASSIVE`.

## Quality assessment (remeasured)

Source: `results/quality_baseline_vs_best.json` / `.md`  
Script: `scripts/quality_baseline_vs_best.py`

### Primary speed (geo-mean RTF of medium_15s + long_30s)

| Config | Primary RTF | RTFx | vs baseline |
|--------|------------:|-----:|------------:|
| baseline (dynamic INT8) | **0.038156** | 26.21× | — |
| best / production (static QDQ-pc) | **0.018432** | 54.25× | **−51.7% RTF** |

Gate: ≥5% primary RTF improvement → **PASS** (measured ~52%; ladder originally ~54%).

### Quality gates

| Gate | Result | Evidence |
|------|:------:|----------|
| No empty real-speech transcripts | **PASS** | real_speech + jfk nonempty for both configs |
| Pairwise quality (best vs baseline) | **PASS** | exact match **3/5** clips; mean pairwise WER 0.087 (majority exact/near-exact) |
| Absolute JFK WER | **PASS** | baseline WER **0.0**, best WER **0.0** (exact normalized match) |
| Primary RTF ≥5% better | **PASS** | **51.693%** |

**Recommendation: freeze production default.**

### Pairwise detail (best vs baseline as reference)

| Clip | Exact (norm) | Token F1 | WER | CER |
|------|:------------:|---------:|----:|----:|
| short_5s | YES | 1.00 | 0.00 | 0.00 |
| medium_15s | NO | 0.93 | 0.11 | 0.08 |
| long_30s | NO | 0.80 | 0.33 | 0.33 |
| real_speech | YES | 1.00 | 0.00 | 0.00 |
| jfk | YES | 1.00 | 0.00 | 0.00 |

Notes on medium/long pairwise WER: those WAVs are **looped synthetic repeats** of the JFK phrase. Differences are mostly end-of-loop truncation / slight wording (`ask` vs `asking for the`), not empty or garbage transcripts. Natural speech (`real_speech`, `jfk`) is **exact** after normalization for both configs.

### Absolute JFK reference

Reference: *And so, my fellow Americans, ask not what your country can do for you — ask what you can do for your country.*

Both baseline and best: **WER = 0**, exact normalized match on `real_speech.wav`.

## How other developers use this

### Production (default)

```bash
cd /path/to/parakeet-optimization
source venv/bin/activate
python scripts/apply_best_config.py
python scripts/apply_best_config.py --audio data/real_speech.wav
python scripts/apply_best_config.py --benchmark --warmup 1 --repeats 3
# explicit production alias:
python scripts/apply_best_config.py --config configs/production.json
```

Requires local pack: `models/parakeet-tdt-0.6b-v3-onnx-static-qdq-pc/`  
(encoder static QDQ-pc + decoder/frontend INT8 from Hub).

### A/B vs Hub dynamic INT8

```bash
python scripts/apply_best_config.py --config configs/baseline.json --audio data/real_speech.wav
python scripts/apply_best_config.py --config configs/baseline.json --benchmark
```

### Reproduce quality assessment

```bash
pip install -r requirements.txt   # includes jiwer
python scripts/quality_baseline_vs_best.py --warmup 1 --repeats 3
# → results/quality_baseline_vs_best.json + .md
```

## Caveats

1. **Eval set is small** and mostly the same English JFK phrase (looped 5s/15s/30s + natural clip). Not a full multilingual / multi-domain WER suite.
2. **Speed is host-specific** (EPYC 9V74, 8 vCPU, ORT 1.27 CPU). Other CPUs may see different absolute RTF; relative static-vs-dynamic gain should still be large when VNNI/AVX-512 is present.
3. **Technique is standard ORT static quant (QDQ MinMax per-channel)** applied to the public FP32 encoder — not a novel architecture. Hub default remains dynamic INT8.
4. Runtime-only knobs (E0–E6) did not clear ≥5%; encoder static quant (C2) is the kept win.
5. Do **not** re-run full C0–C3 unless re-exporting; production pack is already on disk.
6. **Long audio / memory:** frozen production leaves `"chunking": null` (full-file `recognize`). Multi-minute files can OOM as encoder activations scale with duration. For long talks use CLI chunking, e.g.  
   `python scripts/apply_best_config.py --config configs/production.json --audio long.wav --chunk-window-s 30`  
   That is app-level window+concat (not true streaming) and is **not** the frozen primary-RTF path.

## Related artifacts

- Ladder: `results/autoresearch_encoder/summary.md`, `ledger.jsonl`
- Profiling: `reports/parakeet_tdt_v3_cpu_optimization.md`
- Quality: `results/quality_baseline_vs_best.md`
- Apply: `scripts/apply_best_config.py`

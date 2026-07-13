# Quality assessment: baseline (dynamic INT8) vs best (static QDQ per-channel)

Generated: 2026-07-13T16:18:26.095396+00:00

## Configs

| Role | Config | Model path | Primary RTF | RTFx (1/RTF) |
|------|--------|------------|------------:|-------------:|
| baseline | `configs/baseline.json` | `models/parakeet-tdt-0.6b-v3-onnx` | 0.038156 | 26.21 |
| best | `configs/best_config.json` | `models/parakeet-tdt-0.6b-v3-onnx-static-qdq-pc` | 0.018432 | 54.25 |

**Primary RTF improvement (best vs baseline):** 51.693%

Primary metric = geometric mean of mean RTF on `medium_15s` + `long_30s`.

## Decision

**Recommendation: `FREEZE_PRODUCTION`**

| Gate | Pass |
|------|:----:|
| no_empty_real_speech | YES |
| pairwise_quality | YES |
| absolute_jfk | YES |
| rtf_improvement | YES |

Reasons:
- pairwise quality OK: mean_wer=0.0869, exact=3/5, near_exact(<=5% WER)=3/5
- absolute JFK WER OK: baseline=0.0000 best=0.0000
- primary RTF improvement OK: 51.693% (baseline=0.038156 best=0.018432)

## Pairwise quality (best vs baseline as reference)

| Clip | Exact (norm) | Token F1 | Similarity | WER | CER | Baseline nonempty | Best nonempty |
|------|:------------:|---------:|-----------:|----:|----:|:-----------------:|:-------------:|
| short_5s | YES | 1.0000 | 1.0000 | 0.0000 | 0.0000 | YES | YES |
| medium_15s | NO | 0.9310 | 0.9609 | 0.1071 | 0.0815 | YES | YES |
| long_30s | NO | 0.8041 | 0.8009 | 0.3276 | 0.3321 | YES | YES |
| real_speech | YES | 1.0000 | 1.0000 | 0.0000 | 0.0000 | YES | YES |
| jfk | YES | 1.0000 | 1.0000 | 0.0000 | 0.0000 | YES | YES |

**Mean pairwise WER:** 0.0869  
**Exact matches:** 3/5

## Absolute quality vs JFK reference

Reference: _And so, my fellow Americans, ask not what your country can do for you — ask what you can do for your country._

| Config | Clip | Exact | Token F1 | Similarity | WER | CER | Nonempty |
|--------|------|:-----:|---------:|-----------:|----:|----:|:--------:|
| baseline | real_speech | YES | 1.0000 | 1.0000 | 0.0000 | 0.0000 | YES |
| best | real_speech | YES | 1.0000 | 1.0000 | 0.0000 | 0.0000 | YES |

## Per-clip RTF

| Clip | Dur (s) | Baseline RTF | Best RTF | Baseline RTFx | Best RTFx | Δ RTF % |
|------|--------:|-------------:|---------:|--------------:|----------:|--------:|
| short_5s | 5.00 | 0.050844 | 0.037204 | 19.67 | 26.88 | 26.83 |
| medium_15s | 15.00 | 0.038733 | 0.020680 | 25.82 | 48.36 | 46.61 |
| long_30s | 30.00 | 0.037588 | 0.016429 | 26.60 | 60.87 | 56.29 |
| real_speech | 11.00 | 0.043815 | 0.020662 | 22.82 | 48.40 | 52.84 |
| jfk | 11.00 | 0.045225 | 0.025548 | 22.11 | 39.14 | 43.51 |

## Per-clip transcripts

### short_5s

- **baseline:** 'And so, my fellow Americans, ask not'
- **best:**     'And so, my fellow Americans, ask not'
- **norm baseline:** `and so my fellow americans ask not`
- **norm best:**     `and so my fellow americans ask not`

### medium_15s

- **baseline:** 'And so, my fellow Americans, ask not what your country can do for you, ask what you can do for your country. And so, my fellow Americans, ask.'
- **best:**     'And so my fellow Americans, ask not what your country can do for you, ask what you can do for your country. And so my fellow Americans, asking for the'
- **norm baseline:** `and so my fellow americans ask not what your country can do for you ask what you can do for your country and so my fellow americans ask`
- **norm best:**     `and so my fellow americans ask not what your country can do for you ask what you can do for your country and so my fellow americans asking for the`

### long_30s

- **baseline:** 'And so, my fellow Americans, ask not what your country can do for you, ask what you can do for your country. And so, my fellow Americans, ask not what your country can do for you, ask what you can do for your country. And so, my fellow Americans, ask not what your country can do for you,'
- **best:**     'And so my fellow Americans, ask not what your country can do for you, ask what you can do for your country. And so my fellow Americans, ask not what your country can do for you, ask what you'
- **norm baseline:** `and so my fellow americans ask not what your country can do for you ask what you can do for your country and so my fellow americans ask not what your country can do for you ask what you can do for your country and so my fellow americans ask not what your country can do for you`
- **norm best:**     `and so my fellow americans ask not what your country can do for you ask what you can do for your country and so my fellow americans ask not what your country can do for you ask what you`

### real_speech

- **baseline:** 'And so, my fellow Americans, ask not what your country can do for you. Ask what you can do for your country.'
- **best:**     'And so, my fellow Americans, ask not what your country can do for you, ask what you can do for your country.'
- **norm baseline:** `and so my fellow americans ask not what your country can do for you ask what you can do for your country`
- **norm best:**     `and so my fellow americans ask not what your country can do for you ask what you can do for your country`

### jfk

- **baseline:** 'And so, my fellow Americans, ask not what your country can do for you. Ask what you can do for your country.'
- **best:**     'And so, my fellow Americans, ask not what your country can do for you, ask what you can do for your country.'
- **norm baseline:** `and so my fellow americans ask not what your country can do for you ask what you can do for your country`
- **norm best:**     `and so my fellow americans ask not what your country can do for you ask what you can do for your country`

## Eval set notes

- Eval set is small and mostly English clean speech.
- short_5s / medium_15s / long_30s are typically looped variants of the same JFK phrase; real_speech.wav and jfk.flac are the natural JFK inaugural line.
- This is NOT a full multilingual WER suite; pairwise vs baseline + absolute JFK only.
- No extra public speech samples downloaded (disk/time careful; optional).

## How to reproduce

```bash
cd /path/to/parakeet-optimization
source venv/bin/activate
python scripts/quality_baseline_vs_best.py --warmup 1 --repeats 3
```

## Freeze action

- Keep `configs/best_config.json` as **production default** (static QDQ per-channel encoder).
- Keep `configs/baseline.json` as **Hub dynamic INT8 reference**.
- Run production: `python scripts/apply_best_config.py`
- A/B baseline: `python scripts/apply_best_config.py --config configs/baseline.json`


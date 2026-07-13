# Parakeet TDT 0.6B v3 — CPU Inference Profiling & Optimization Report

**Model:** `nvidia/parakeet-tdt-0.6b-v3` via community ONNX export `istupakov/parakeet-tdt-0.6b-v3-onnx`  
**Runtime:** ONNX Runtime 1.27.0 CPU EP + `onnx-asr` 0.11.0 (INT8)  
**Host:** AMD EPYC 9V74, 8 vCPU, AVX-512 / VNNI / BF16, ~63 GB RAM, **no GPU**  
**Date:** 2026-07-13  
**Artifacts:** `results/baseline_metrics.json`, `results/operator_profile.json`, `scripts/profile_parakeet_cpu.py`

---

## 1. Executive summary

On this 8-core AMD EPYC host, **INT8 ONNX Parakeet TDT 0.6B v3** delivers:

| Threads | 5 s audio RTFx | 15 s audio RTFx | 30 s audio RTFx | Peak RSS |
|--------:|---------------:|----------------:|----------------:|---------:|
| 1 | 7.7× | 7.8× | 7.4× | ~1.4 GB |
| 2 | 13.1× | 13.4× | 13.0× | ~1.5 GB |
| 4 | 16.1× | 19.6× | 20.3× | ~1.5 GB |
| 8 | 17.3× | **24.5×** | **26.3×** | ~1.5 GB |

- **Successful real-speech transcription** (JFK sample):  
  *“And so, my fellow Americans, ask not what your country can do for you. Ask what you can do for your country.”*
- **Primary bottleneck: FastConformer encoder (~97–98% of stage time)** at 4 threads on 15 s audio.
- Mel frontend and TDT decoder are each ~1% of stage time in offline batch=1 greedy decode.
- Within the encoder, **INT8 convolutions (`ConvInteger`) and dynamic-quant MatMuls (`DynamicQuantizeMatMul` / `MatMulIntegerToFloat`)** dominate compute; attention softmax is a small fraction after 8× subsampling.
- Thread scaling is good 1→4 and still positive 4→8 for longer audio (encoder-bound GEMM/conv parallelizes); short clips show diminishing returns (overhead + serial decode).

**Bottom line for generic CPUs:** keep INT8 ONNX + ORT/oneDNN, maximize encoder MatMul/Conv efficiency (VNNI, packing, fusion), optionally add local/chunked attention and streaming; decoder is not the first lever for offline throughput but matters for streaming latency and very long audio with weak duration skipping.

---

## 2. System & software configuration

| Item | Value |
|------|-------|
| CPU | AMD EPYC 9V74 80-Core Processor (8 vCPU exposed) |
| ISA | AVX, AVX2, AVX-512F/DQ/BW/VL, **AVX512_VNNI**, **AVX512_BF16** |
| RAM | 62.8 GB total |
| GPU | None |
| OS | Linux 6.17 Azure (glibc 2.39) |
| Python | 3.12.3 (project venv) |
| onnxruntime | 1.27.0 (`CPUExecutionProvider`) |
| onnx-asr | 0.11.0 |
| Model files | `encoder-model.int8.onnx` (652 MB), `decoder_joint-model.int8.onnx` (18 MB), `nemo128.onnx`, `vocab.txt` |
| Quantization | Static/dynamic INT8 ONNX (producer `onnx.quantize`) |
| Session opts | `intra_op_num_threads` ∈ {1,2,4,8}, `inter_op_num_threads=1`, `ORT_ENABLE_ALL` |

**Note on model load:** `onnx-asr`’s HF pattern `encoder-model?int8.onnx` does **not** match Hub filenames `encoder-model.int8.onnx`. Files were downloaded explicitly into `models/parakeet-tdt-0.6b-v3-onnx/` and loaded with `quantization='int8'` + local `path=`.

---

## 3. Architecture under test

```
waveform 16 kHz mono
    │
    ▼
┌───────────────────┐
│ Frontend nemo128  │  STFT + 128-bin log-mel (ONNX)
└─────────┬─────────┘
          │ features [B, 128, T_mel]   (~100 frames/s)
          ▼
┌───────────────────┐
│ FastConformer enc │  8× conv subsampling → [B, 1024, T_enc]
│  (INT8 ONNX)      │  T_enc ≈ T_mel / 8  (≈12.5 Hz)
│  blocks: MHSA +   │  depthwise conv module + dual FFN (macaron)
│  DW-conv + FFN    │
└─────────┬─────────┘
          │ encoder states
          ▼
┌───────────────────┐
│ TDT decoder_joint │  Pred net: 2-layer LSTM (640-d)
│  (INT8 ONNX)      │  Joiner: FFN over enc⊕pred
│                   │  Outputs: token logits + duration logits
│                   │  Greedy: emit token, skip `duration` frames
└───────────────────┘
```

Measured shapes (15 s audio, 4 threads):

| Tensor | Shape |
|--------|-------|
| Mel features | `[1, 128, 1501]` |
| Encoder out | `[1, 1024, 188]` |
| Subsampling | 8× (1501 → 188) |
| Decoder LSTM state | `[2, 1, 640]` × 2 |
| Joiner output width | 8198 (token + duration heads) |

Literature context: FastConformer is designed for ~2.8× faster inference than classic Conformer (arXiv:2305.05084); TDT reduces decoder steps vs RNN-T via duration frame-skip (arXiv:2304.06795).

---

## 4. Baseline throughput (this host)

Methodology: 1 warmup + 3 timed `model.recognize()` runs per (audio × threads); wall clock via `perf_counter`; RTF = latency / audio_duration; RTFx = 1/RTF; peak RSS via `resource.ru_maxrss`.

### 4.1 Latency & real-time factors

| Audio | Dur (s) | Threads | Latency mean (s) | RTF | RTFx |
|-------|--------:|--------:|-----------------:|----:|-----:|
| short_5s | 5.0 | 1 | 0.647 | 0.1295 | 7.72 |
| short_5s | 5.0 | 2 | 0.382 | 0.0764 | 13.09 |
| short_5s | 5.0 | 4 | 0.310 | 0.0619 | 16.15 |
| short_5s | 5.0 | 8 | 0.288 | 0.0577 | 17.34 |
| medium_15s | 15.0 | 1 | 1.931 | 0.1287 | 7.77 |
| medium_15s | 15.0 | 2 | 1.122 | 0.0748 | 13.37 |
| medium_15s | 15.0 | 4 | 0.767 | 0.0511 | 19.56 |
| medium_15s | 15.0 | 8 | 0.612 | 0.0408 | **24.51** |
| long_30s | 30.0 | 1 | 4.032 | 0.1344 | 7.44 |
| long_30s | 30.0 | 2 | 2.317 | 0.0772 | 12.95 |
| long_30s | 30.0 | 4 | 1.480 | 0.0493 | 20.27 |
| long_30s | 30.0 | 8 | 1.142 | 0.0381 | **26.28** |
| real_speech (JFK) | 11.0 | 1 | 1.415 | 0.1286 | 7.78 |
| real_speech | 11.0 | 2 | 0.821 | 0.0746 | 13.40 |
| real_speech | 11.0 | 4 | 0.557 | 0.0506 | 19.76 |
| real_speech | 11.0 | 8 | 0.458 | 0.0417 | 24.00 |

All listed runs produced **non-empty** transcripts.

### 4.2 Thread scaling (30 s clip)

| Threads | Latency (s) | Speedup vs 1 thread | Efficiency |
|--------:|------------:|--------------------:|-----------:|
| 1 | 4.032 | 1.00× | 100% |
| 2 | 2.317 | 1.74× | 87% |
| 4 | 1.480 | 2.72× | 68% |
| 8 | 1.142 | 3.53× | 44% |

Interpretation: encoder MatMul/Conv parallelize well to 4 threads; 8 threads still help longer audio but efficiency drops (memory bandwidth, oversubscription on 8 vCPU, fixed serial decode overhead).

### 4.3 Memory

Peak RSS during profiling: **~1.1–1.5 GB** for INT8 path (acceptable for edge/server CPU). FP32 ONNX was not loaded (disk constraint; encoder FP32 + `.onnx.data` is multi-GB).

---

## 5. Stage breakdown

Instrumented via `onnx-asr` preprocessor + `_encode` + `_decoding` (same weights as E2E).

### 5.1 15 s audio, 4 threads (primary)

| Stage | Time (s) | Share |
|-------|---------:|------:|
| Frontend (mel / nemo128) | 0.0087 | **1.13%** |
| **Encoder (FastConformer INT8)** | **0.7517** | **97.75%** |
| Decoder (TDT greedy) | 0.0086 | **1.12%** |
| Sum | 0.7690 | 100% |

### 5.2 5 s audio, 4 threads

| Stage | Time (s) | Share |
|-------|---------:|------:|
| Frontend | 0.0074 | 2.56% |
| Encoder | 0.2779 | **96.33%** |
| Decoder | 0.0032 | 1.11% |

**Conclusion:** Offline batch=1 inference is **encoder-bound**. Optimizing MHSA/FFN/conv in the FastConformer stack is the highest-ROI path. Frontend STFT and TDT joiner/LSTM are secondary for throughput (but decoder seriality matters for streaming TTFT and pathological long utterances).

---

## 6. Operator-level hotspots (ORT profiling)

ORT chrome-trace profiles: `results/ort_profiles/{fe,enc,dec}_*.json`.  
Note: profile totals include `session_initialization` / `model_run` wrappers; **kernel-level** ranking below excludes pure framework overhead where possible.

### 6.1 Encoder — top compute kernels

| Op | Count (profile window) | Total ms (window) | Role in architecture |
|----|-----------------------:|------------------:|----------------------|
| **ConvInteger** | 154 | **572.6** | 8× pre-encode conv stack + depthwise/pointwise conv modules in FastConformer blocks |
| **DynamicQuantizeMatMul** | 242 | **489.0** | FFN linear1/linear2, attention `linear_out`, pre-encode projection |
| **MatMulIntegerToFloat** | 192 | **145.9** | Attention Q/K/V and positional projections (INT8 weights → float accum) |
| LayerNormalization | 192 | 47.1 | Pre-norm around MHSA / conv / FFN |
| MatMul (FP) | 144 | 34.1 | Attention score / context GEMMs (`self_attn/MatMul*`) |
| Cast / Where / Transpose | many | ~55 combined | Quant dequant edges, masking, layout |

**Architecture → kernel map (encoder):**

| FastConformer block | Dominant ORT ops |
|---------------------|------------------|
| 8× conv subsampling (`pre_encode/conv/*`) | `ConvInteger` |
| Multi-head self-attention (Q/K/V/out, pos) | `MatMulIntegerToFloat`, FP `MatMul`, `Softmax` (small) |
| Depthwise-separable conv module | `ConvInteger` |
| Macaron FFN (linear1/linear2 ×2) | `DynamicQuantizeMatMul` |
| Pre-norm | `LayerNormalization` |

Softmax attention is **not** the top kernel after 8× downsample (T≈188 for 15 s): compute is dominated by **wide INT8 GEMMs and convs** at d_model=1024.

### 6.2 Decoder (64 profiled steps) — top kernels

| Op | Role |
|----|------|
| **DynamicQuantizeLSTM** | Prediction network (2×640 LSTM) |
| **DynamicQuantizeMatMul** | Joiner: `joint/enc`, `joint/pred`, `joint_net` |
| Split / Concat / Squeeze / Transpose | LSTM state packing |

TDT duration head enables frame skipping → far fewer decoder steps than classic RNN-T; measured decoder stage ~1% confirms this for the JFK-derived clips.

### 6.3 Frontend

| Op | Role |
|----|------|
| **STFT** | Dominant mel frontend cost |
| ReduceSumSquare / MatMul / Slice | Power, mel filterbank projection, framing |

Frontend is ~1–3% of E2E; still worth fusing if building a custom pipeline, but not the first optimization target.

---

## 7. Ranked CPU optimization opportunities

Ranked by **expected speedup × portability to generic CPUs × implementation cost**. All apply without a GPU.

### 1) Keep / harden INT8 + ensure VNNI/oneDNN codepaths (already baseline)

- **What:** Ship INT8 ONNX (current path). Verify ORT is built with oneDNN / DNNL and that `DynamicQuantizeMatMul` / `ConvInteger` hit **AVX-512 VNNI** on capable CPUs (this EPYC has VNNI).
- **Why:** Encoder is ~98% INT8 MatMul/Conv. VNNI can substantially beat pure AVX2 INT8.
- **Expected:** 1.2–2× vs naive INT8 or FP32 on VNNI CPUs; large RAM/disk win vs FP32 (~1.5 GB vs multi-GB).
- **Risk:** Accuracy regression on rare languages/noise — validate WER on target domain.
- **Portability:** High (ORT CPU EP everywhere; graceful fallback without VNNI).

### 2) Graph / kernel fusion and layout optimization in ORT

- **What:** `ORT_ENABLE_ALL` (on), plus optional ORT transformer passes: fuse LayerNorm+MatMul, reduce Cast/Transpose around quant nodes, enable memory pattern / arena (on). Consider exporting with **channels-last-friendly** weights and fewer explicit Transposes in attention.
- **Why:** Profile shows many Cast/Transpose/Where nodes around quant edges; each burns bandwidth on CPU.
- **Expected:** 10–25% encoder latency reduction with low effort.
- **Risk:** Low if numerical checks pass.
- **Portability:** High.

### 3) Attention structure: local / chunked / limited-context MHSA

- **What:** Replace global self-attention with **chunked or local attention** (e.g. 8–30 s windows, or relative local windows) at export or training time; or use efficient attention kernels (blocked, online softmax) in a custom ORT contrib op.
- **Why:** Score MatMuls are O(T²) per layer. At T=188 (15 s) they are moderate; at multi-minute audio without chunking they dominate and blow memory. Softmax is small now but QK/AV GEMMs grow.
- **Expected:** Near-linear scaling for long audio; 1.3–2× on long-form vs global attention; enables streaming.
- **Risk:** WER impact on long-range dependencies; needs re-export or fine-tune.
- **Portability:** High for chunking (pure graph change); custom kernels need per-ISA work.

### 4) Depthwise / pointwise conv kernel tuning (FastConformer conv module + 8× frontend)

- **What:** Ensure depthwise convs use efficient CPU im2col-free kernels; fuse DW+PW where possible; consider **Winograd or direct** conv for small kernels used in FastConformer; pack INT8 conv weights for VNNI.
- **Why:** `ConvInteger` was the single largest kernel family in the encoder profile (~573 ms in the profile window).
- **Expected:** 15–40% of conv time; ~5–15% E2E if conv is ~half of encoder compute.
- **Risk:** Medium engineering cost; accuracy neutral if numerically equivalent.
- **Portability:** Medium–high via oneDNN/OpenVINO; custom ASM is ISA-specific.

### 5) FFN GEMM packing, cache blocking, and optional structured sparsity

- **What:** Large FFN MatMuls (`feed_forward*/linear*`) benefit from better weight packing, multi-thread cache blocking, and optionally **2:4 / structured sparsity** or INT4 weight-only quantization with INT8 activations.
- **Why:** `DynamicQuantizeMatMul` is the second largest encoder kernel family; d_model=1024 FFNs are bandwidth + compute heavy.
- **Expected:** 10–30% on FFN-heavy layers; INT4 WoQ can cut memory bandwidth further (WER tradeoff).
- **Risk:** INT4 needs calibration; sparsity needs tool support in ORT.
- **Portability:** Packing via oneDNN is portable; INT4 less universal.

### 6) TDT decode loop: state caching, batching blanks, multi-utterance batching

- **What:** (a) Avoid Python-side per-step overhead by a single ORT custom op for greedy TDT; (b) cache pred-net state; (c) batch multiple utterances; (d) early exit / larger duration skips when confident.
- **Why:** Decoder is ~1% offline here, but each step is a small LSTM+MatMul with poor GPU/CPU occupancy; streaming and high-QPS servers feel this. Profile shows `DynamicQuantizeLSTM` + joiner MatMuls per step.
- **Expected:** Large win for **streaming latency** and multi-stream throughput; modest offline RTFx gain (~1–5%) unless decode share grows.
- **Risk:** Low for engineering fusion; correctness must match blank/duration rules.
- **Portability:** High (algorithmic).

### 7) Alternative CPU runtimes: OpenVINO / ONNX Runtime + DNNL EP tuning / IPEX

- **What:** Re-run the same INT8 ONNX under **OpenVINO** (strong on Intel, improving on AVX-512 AMD) or ensure ORT uses **oneDNN** fully; compare thread affinity (`OMP_NUM_THREADS`, `KMP_AFFINITY` / `GOMP_CPU_AFFINITY`).
- **Why:** Same graph, different kernel library can move ConvInteger/MatMul by tens of percent.
- **Expected:** 1.1–1.5× depending on CPU vendor.
- **Risk:** Extra dependency; validate numerics.
- **Portability:** OpenVINO best on Intel; ORT+oneDNN more universal.

### 8) Streaming / chunked inference with encoder state reuse

- **What:** Process audio in fixed chunks (e.g. 1–2 s) with limited left context; reuse encoder cache where architecture allows; emit partial TDT hypotheses.
- **Why:** Caps attention cost, reduces peak RSS, enables live captions; aligns with local attention.
- **Expected:** Lower latency to first partial; RTFx similar or better on long audio; better UX.
- **Risk:** Boundary artifacts; needs careful chunk overlap.
- **Portability:** High.

### 9) Architecture-level distill / smaller encoder (if product allows)

- **What:** Distill to fewer FastConformer layers, smaller d_model (e.g. 512–768), or hybrid CTC for first-pass + TDT rescoring.
- **Why:** Encoder is 98% of time — fewer layers almost linearly reduce cost.
- **Expected:** 1.5–3× with WER tradeoff; CTC first-pass can be much faster for keyword/voice-command.
- **Risk:** Requires training data and quality acceptance.
- **Portability:** High once re-exported to ONNX INT8.

### 10) Frontend STFT fusion / kissfft / Intel MKL DFT

- **What:** Replace generic ONNX STFT with optimized FFT (MKL, FFTW, pocketfft) fused into mel.
- **Why:** STFT dominates frontend; frontend is only ~1–3% E2E today.
- **Expected:** <3% E2E — do last.
- **Portability:** High.

---

## 8. Recommended action plan (practical order)

1. **Ship current INT8 ORT path** with `intra_op_num_threads` tuned per SKU (here: **4–8** for throughput; **2–4** if co-locating other work).
2. **Confirm VNNI/oneDNN** is active in production ORT builds; pin thread affinity.
3. **ORT graph cleanup** (fusion, fewer Cast/Transpose) on the exported encoder.
4. **Chunked / local attention export** for long-form and streaming products.
5. **Conv + FFN kernel pressure** via OpenVINO or ORT EP bake-off on target CPUs.
6. **Fuse TDT greedy loop** if streaming QPS becomes the limiter.
7. Only then consider **distill / INT4 / sparse** if still short of SLA.

---

## 9. Acceptance checklist (this project)

| Criterion | Status |
|-----------|--------|
| Non-empty transcription of real speech | **PASS** (JFK quote) |
| RTF/RTFx for ≥2 audio lengths | **PASS** (5s, 15s, 30s, 11s) |
| RTF/RTFx for ≥2 thread settings | **PASS** (1, 2, 4, 8) |
| Stage/operator breakdown ≥70% attributed | **PASS** (encoder ~97%) |
| ≥5 concrete CPU optimization opportunities | **PASS** (10 listed, ranked) |
| Reproducible script + JSON metrics | **PASS** |

---

## 10. How to reproduce

See project `README.md`. Quick path:

```bash
cd /path/to/parakeet-optimization
source venv/bin/activate
python scripts/profile_parakeet_cpu.py \
  --threads 1,2,4,8 \
  --audio short_5s,medium_15s,long_30s,real_speech \
  --warmup 1 --repeats 3
```

Outputs:

- `results/baseline_metrics.json`
- `results/operator_profile.json`
- `results/ort_profiles/*.json`

---

## 11. References

- Recasens et al., *Fast Conformer with Linearly Scalable Attention*, arXiv:2305.05084  
- Xu et al., *Efficient Sequence Transduction by Jointly Predicting Tokens and Durations* (TDT), arXiv:2304.06795  
- NVIDIA Parakeet TDT 0.6B v3 model card (Hugging Face / NGC)  
- ONNX Runtime CPU EP / oneDNN documentation  
- Community ONNX export: `istupakov/parakeet-tdt-0.6b-v3-onnx` + `onnx-asr`

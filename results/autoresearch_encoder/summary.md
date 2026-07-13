# Autoresearch Encoder Static-Quant Ladder (C0–C3) — Summary

Generated: 2026-07-13T15:17:10.798085+00:00

## Host

- CPU: AMD EPYC 9V74 80-Core Processor
- Logical CPUs: 8
- Flags: avx, avx2, avx512_bf16, avx512_vnni, avx512bw, avx512dq, avx512f, avx512vl
- ORT: 1.27.0 providers=['AzureExecutionProvider', 'CPUExecutionProvider']
- RAM: 62.79 GB  disk_free≈9.49 GB

## Protocol

- Model: encoder static re-quant of `nemo-parakeet-tdt-0.6b-v3` (decoder stays dynamic INT8)
- Primary metric: geometric mean of mean RTF on `medium_15s` + `long_30s` (lower better)
- Keep gate: ≥5% primary RTF improvement vs C0 + quality OK
- Quality gate: non-empty `real_speech`; normalized match or ≥85% token overlap vs C0

## Quantization builds

| Variant | ok | format | per_channel | method | out_mb | notes |
|---|:---:|---|:---:|---|---:|---|
| C1_minmax_qdq | True | QDQ | False | MinMax | 2.4 | ok |
| C2_minmax_qdq_pc | True | QDQ | True | MinMax | 4.76 | ok |
| C3_matmul_qdq_stack | True | QDQ | False | MinMax | 1.64 | ok |

## Results

| Experiment | primary_rtf | improvement_pct | keep | reason |
|---|---:|---:|:---:|---|
| C0_dynamic_int8_baseline | 0.038742 | None | YES | baseline reference |
| C1_minmax_qdq | 0.017709 | 54.29 | YES | primary RTF improved 54.29% (≥5%); quality: exact normalized match |
| C2_minmax_qdq_pc | 0.017692 | 54.334 | YES | primary RTF improved 54.33% (≥5%); quality: exact normalized match |
| C3_matmul_qdq_stack | 0.021249 | 45.154 | YES | primary RTF improved 45.15% (≥5%); quality: exact normalized match |

## Kept winners

- `C1_minmax_qdq`
- `C2_minmax_qdq_pc`
- `C3_matmul_qdq_stack`

## Best config

```json
{
  "model_path": "models/parakeet-tdt-0.6b-v3-onnx-static-qdq-pc",
  "quantization": "int8",
  "provider": "CPUExecutionProvider",
  "provider_options": null,
  "intra_op_num_threads": 8,
  "inter_op_num_threads": 1,
  "graph_optimization_level": "ORT_ENABLE_ALL",
  "enable_mem_pattern": true,
  "enable_cpu_mem_arena": true,
  "execution_mode": "ORT_SEQUENTIAL",
  "env": {
    "OMP_NUM_THREADS": "8",
    "OMP_WAIT_POLICY": "PASSIVE"
  },
  "chunking": null,
  "optimized_model_dir": null,
  "model_id": "nemo-parakeet-tdt-0.6b-v3",
  "c0_primary_rtf": 0.038742,
  "best_primary_rtf": 0.017692,
  "improvement_pct_vs_c0": 54.334,
  "kept_experiments": [
    "C1_minmax_qdq",
    "C2_minmax_qdq_pc",
    "C3_matmul_qdq_stack"
  ],
  "baseline_real_speech_transcript": "And so, my fellow Americans, ask not what your country can do for you. Ask what you can do for your country.",
  "host_snapshot": {
    "cpu_model": "AMD EPYC 9V74 80-Core Processor",
    "cpu_count_logical": 8,
    "onnxruntime": "1.27.0"
  },
  "ladder": "encoder_static_quant_C0_C3",
  "generated_utc": "2026-07-13T15:17:10.797619+00:00"
}
```

## Notes

- C0 baseline primary_rtf = 0.038742
- Best vs C0: primary_rtf 0.017692 (improvement=54.33%)
- `configs/best_config.json` updated to the winning static encoder dir.
- FP32 encoder source: `models/parakeet-tdt-0.6b-v3-onnx-fp32-encoder/` (may be deleted to free ~2.5 GB after static export).

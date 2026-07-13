# Autoresearch CPU Optimization Ladder — Summary

Generated: 2026-07-13T13:02:11.936351+00:00

## Host

- CPU: AMD EPYC 9V74 80-Core Processor
- Logical CPUs: 8
- Flags: avx, avx2, avx512_bf16, avx512_vnni, avx512bw, avx512dq, avx512f, avx512vl
- ORT: 1.27.0 providers=['AzureExecutionProvider', 'CPUExecutionProvider']
- RAM: 62.79 GB

## Protocol

- Model: INT8 ONNX `nemo-parakeet-tdt-0.6b-v3` via onnx-asr
- Primary metric: geometric mean of mean RTF on `medium_15s` + `long_30s` (lower better)
- Keep gate: ≥5% primary RTF improvement vs rolling best (E0 for first keeps; stack vs E0)
- Quality gate: non-empty `real_speech` transcript; normalized match or ≥85% token overlap vs E0

## Results

| Experiment | primary_rtf | improvement_pct | keep | reason |
|---|---:|---:|:---:|---|
| E0_baseline_threads8 | 0.040145 | None | YES | baseline reference |
| E0_confirm_threads4 | 0.048232 | -20.144 | no | improvement -20.14% < 5%; quality: exact normalized match |
| E1_intra2_omp2_passive | 0.076045 | -89.426 | no | improvement -89.43% < 5%; quality: exact normalized match |
| E1_intra4_omp4_passive | 0.049415 | -23.091 | no | improvement -23.09% < 5%; quality: exact normalized match |
| E1_intra8_omp8_active | 0.038421 | 4.294 | no | improvement 4.29% < 5%; quality: exact normalized match |
| E1_intra8_omp8_passive | 0.040145 | 0.0 | no | duplicate of rolling best; skipped remeasure |
| E1_intra8_omp8_active_kmp_affinity | 0.038315 | 4.56 | no | improvement 4.56% < 5%; quality: exact normalized match |
| E2_mem_pattern_off | 0.038422 | 4.291 | no | improvement 4.29% < 5%; quality: exact normalized match |
| E2_cpu_arena_off | 0.038475 | 4.161 | no | improvement 4.16% < 5%; quality: exact normalized match |
| E2_mem_pattern_off_arena_off | 0.039712 | 1.079 | no | improvement 1.08% < 5%; quality: exact normalized match |
| E2_execution_parallel | 0.066285 | -65.114 | no | improvement -65.11% < 5%; quality: exact normalized match |
| E3_optimized_onnx_dir | 0.038547 | 3.981 | no | improvement 3.98% < 5%; quality: exact normalized match |
| E4_openvino_cpu | None | None | no | skipped: OpenVINO EP not in providers=['AzureExecutionProvider', 'CPUExecutionProvider']; skipping optional install to avoid replacing existing onnxruntime CPU wheel |
| E5_chunk_w12_o1.0 | 0.042855 | -6.751 | no | E5 gate vs E0 long_30s: long_30s_imp=-20.27% (need ≥5%), quality_ok=True, primary_imp=-6.751 |
| E5_chunk_w15_o1.5 | 0.044751 | -11.473 | no | E5 gate vs E0 long_30s: long_30s_imp=-25.11% (need ≥5%), quality_ok=True, primary_imp=-11.473 |
| E5_chunk_w10_o0.5 | 0.046581 | -16.031 | no | E5 gate vs E0 long_30s: long_30s_imp=-34.54% (need ≥5%), quality_ok=True, primary_imp=-16.031 |
| E6_stacked_best | 0.038218 | 4.801 | YES | no individual keeps; stacked equals E0 baseline |

## Kept winners

- *(none — no experiment cleared the ≥5% gate with quality OK)*

## Best config

```json
{
  "model_path": "/home/azureuser/latest_llm_eval/models/parakeet-tdt-0.6b-v3-onnx",
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
  "e0_primary_rtf": 0.040145,
  "best_primary_rtf": 0.038218,
  "improvement_pct_vs_e0": 4.801,
  "kept_experiments": [],
  "baseline_real_speech_transcript": "And so, my fellow Americans, ask not what your country can do for you. Ask what you can do for your country.",
  "host_snapshot": {
    "cpu_model": "AMD EPYC 9V74 80-Core Processor",
    "cpu_count_logical": 8,
    "onnxruntime": "1.27.0"
  },
  "model_id": "nemo-parakeet-tdt-0.6b-v3",
  "generated_utc": "2026-07-13T13:02:11.936059+00:00"
}
```

## Notes

- E6 stacked best vs E0: primary_rtf 0.038218 (E0=0.040145, improvement=4.80%)
- **No experiment cleared the hard ≥5% keep gate.** Several E1/E2/E3 candidates landed ~3–4.5% (noise / residual) and were discarded.
- Confirmed: **8 threads >> 4 threads** on this 8-vCPU host (~20% worse primary RTF at 4).
- OpenVINO EP unavailable (`AzureExecutionProvider` + `CPUExecutionProvider` only); skipped without replacing the ORT wheel.
- Offline ORT optimized ONNX written to `models/parakeet-tdt-0.6b-v3-onnx-opt/` (~1.8–4% primary, below gate).
- App-level chunk+concat **hurt** long_30s RTF (extra session overhead; no encoder cache) — discarded.
- `configs/best_config.json` ships the E0 baseline (intra=8, inter=1, ORT_ENABLE_ALL, OMP PASSIVE) as the portable best known config on this host.
- Phase-2 for ≥5% would need EP bake-off with a dedicated OpenVINO/ORT build, INT4 re-quant, or architecture/export changes — outside this quick runtime ladder.

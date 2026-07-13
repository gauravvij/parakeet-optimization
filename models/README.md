# Model packs (not stored in git)

ONNX weights are **gitignored** (~640MB–850MB per pack). Download them after clone.

## Required for production inference

**Static QDQ per-channel encoder pack** (frozen production default):

| Item | Value |
|------|--------|
| Local path | `models/parakeet-tdt-0.6b-v3-onnx-static-qdq-pc/` |
| Hugging Face | [`gvij/parakeet-tdt-0.6b-v3-onnx-static-qdq-pc`](https://huggingface.co/gvij/parakeet-tdt-0.6b-v3-onnx-static-qdq-pc) |
| Config | `configs/production.json` / `configs/best_config.json` |

```bash
source venv/bin/activate
python - <<'PY'
from pathlib import Path
from huggingface_hub import snapshot_download
local = Path("models/parakeet-tdt-0.6b-v3-onnx-static-qdq-pc")
snapshot_download(
    "gvij/parakeet-tdt-0.6b-v3-onnx-static-qdq-pc",
    local_dir=str(local),
)
print("OK", local)
PY
```

Files expected:

- `encoder-model.int8.onnx` + `encoder-model.int8.onnx.data` (static QDQ encoder)
- `decoder_joint-model.int8.onnx`, `nemo128.onnx`, `vocab.txt`, `config.json`

## Hub dynamic INT8 (A/B baseline)

| Item | Value |
|------|--------|
| Local path | `models/parakeet-tdt-0.6b-v3-onnx/` |
| Hugging Face | [`istupakov/parakeet-tdt-0.6b-v3-onnx`](https://huggingface.co/istupakov/parakeet-tdt-0.6b-v3-onnx) |
| Config | `configs/baseline.json` |

```bash
source venv/bin/activate
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

**Note:** `onnx-asr` globs like `encoder-model?int8.onnx` do not match Hub names
`encoder-model.int8.onnx` — download explicitly as above.

## Optional packs (reproduce encoder ladder)

Built by `scripts/autoresearch_encoder_opts.py` (needs FP32 encoder from Hub):

| Path | Ladder step |
|------|-------------|
| `models/parakeet-tdt-0.6b-v3-onnx-static-minmax/` | C1 QDQ MinMax per-tensor |
| `models/parakeet-tdt-0.6b-v3-onnx-static-qdq-pc/` | C2 best / production |
| `models/parakeet-tdt-0.6b-v3-onnx-static-matmul/` | C3 MatMul-only QDQ |
| `models/parakeet-tdt-0.6b-v3-onnx-opt/` | E3 offline ORT optimize (dynamic) |

## Attribution

- Base model: [nvidia/parakeet-tdt-0.6b-v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3)
- Dynamic INT8 ONNX: [istupakov/parakeet-tdt-0.6b-v3-onnx](https://huggingface.co/istupakov/parakeet-tdt-0.6b-v3-onnx)
- Static encoder quant: standard ONNX Runtime QDQ (this project)

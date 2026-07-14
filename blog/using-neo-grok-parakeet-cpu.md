![Header: CPU ASR optimization case study](assets/header.png)

# I used Neo + Grok 4.5 to optimize Parakeet on CPU (and mostly got out of the way)

**Companion technical record:** [parakeet-cpu-optimization-case-study.md](parakeet-cpu-optimization-case-study.md)  
**Repo:** [gauravvij/parakeet-optimization](https://github.com/gauravvij/parakeet-optimization)  
**Production pack:** [gvij/parakeet-tdt-0.6b-v3-onnx-static-qdq-pc](https://huggingface.co/gvij/parakeet-tdt-0.6b-v3-onnx-static-qdq-pc)

This is the personal version. The other post is the lab notebook: metrics, ladders, dead ends, replicate commands. This one is about **how I actually ran the work** with Neo in VS Code and **Grok 4.5** as the model through BYOK, what I said, what Neo did without me micromanaging, and where I still had to step in as a human.

If you only want numbers, read the technical post. If you want to know whether this style of agent workflow is real for multi-day ML optimization, stay here.

---

## What I wanted

I care about **CPU ASR that is actually usable**: not a GPU demo, not a blog claim of "we tried some flags." Target was NVIDIA **Parakeet TDT 0.6B v3** in the public **INT8 ONNX** packaging, on a real host (AMD EPYC 9V74, 8 vCPU), with a clear baseline (the community dynamic INT8 pack) and a production freeze I would trust enough to publish.

I did **not** want to hand-write every experiment script, babysit every ORT session option, or re-discover quant format failures by hand. I wanted an agent that could:

1. Profile first  
2. Define a metric and a keep gate  
3. Run experiments, discard noise, keep only real wins  
4. Hit failures, fix them, continue  
5. Ship code + weights + docs  

That is a long loop. It is also the kind of loop where a strong model plus an agent that can **write and run code** is more useful than a chat that only suggests commands for me to paste.

---

## Setup I used

- **Neo** in VS Code as the engineering agent (plan, implement, run, read errors, iterate)  
- **Grok 4.5 (xAI)** as the LLM via Neo’s **BYOK** path  
- Project workspace on the EPYC box, later my Mac for portability checks  
- GitHub + Hugging Face credentials already available so publish was not a separate "please paste a token" drama  

I treated Neo less like autocomplete and more like a junior-to-senior pair who owns the sandbox: scripts, venv, ledgers, reports. My job was goals, constraints, product judgment, and Mac measurements.

---

## How I talked to it (high level, not a 40-step recipe)

I did not start with "write `quantize_static` with these exact args." Early prompts looked more like:

> Profile Parakeet TDT 0.6B v3 INT8 ONNX on this CPU. Find stage and operator bottlenecks. Then run a disciplined optimization ladder with a fixed primary RTF metric and a ≥5% keep gate. Do not ship folklore knobs. Freeze something only if quality holds.

Later I narrowed:

> Runtime knobs did not clear the gate. Go after the encoder quant path. Prefer approaches that stay portable in ONNX Runtime.

And product follow-ups after the freeze:

> Long multi-minute audio OOMs with full-file production. Add CLI chunking without changing the frozen RTF default.  
> Document Apple Silicon results so people do not treat EPYC RTFx as universal.  
> Make dual-pack download obvious for A/B and quality scripts.

The pattern that worked: **goal + metric + gate + constraints + "do not redo known dead ends."** When I over-specified implementation, I was doing Neo’s job. When I under-specified the gate, I risked pretty charts with no decision rule.

---

## What Neo did on its own (the interesting part)

This is the list I would not have wanted to grind through manually in one sitting.

### 1. Built the measurement spine before "optimizing"

Neo wrote profiling and config-driven inference paths instead of guessing. Stage split on medium audio: **encoder ~98%** of time. Operator profile pointed at INT8 conv / dynamic quant MatMul paths. That single fact killed a lot of bad ideas (decoder micro-tuning, random session folklore) before they became a week of thrash.

### 2. Ran a runtime ladder and accepted a negative result

E0–E6 covered threads, OMP, arena/mem, offline ORT optimize, OpenVINO (skipped cleanly: not in providers), app-level chunk+concat, and a stacked residual.

**Nothing cleared a honest ≥5% primary RTF keep gate.** Best residual stack was still under the line. Chunking for speed **hurt** long audio RTF.

A human team often rationalizes a 3–4% win into a blog section. Neo’s keep/discard ledger made the negative result legible. That is a feature. Shipping "we set OMP_ACTIVE and felt faster" would have been worse than shipping nothing from that ladder.

### 3. Pivoted to static quant and found the real win

Aligned with the profile: static **QDQ** on the encoder (MinMax, per-tensor then per-channel), decoder/frontend left as Hub dynamic INT8. **C2 per-channel** won; later freeze remeasure was about **51.7% lower primary RTF** vs Hub dynamic INT8 on EPYC (~**2.07×**). Same packs on my Mac for a long English file still showed a clear relative win (~**1.42×** with chunking), which is the portability story I actually care about as a product person.

### 4. Hit ugly failures and recovered without me debugging every stack trace

Things Neo worked through that are easy to underestimate if you only read the happy path:

- Hub INT8 filename vs `onnx-asr` glob mismatch → explicit `hf_hub_download`  
- QOperator static quant failing on FastConformer intermediates → **QDQ**  
- Percentile calib blowing up on variable-length mels → **MinMax**  
- FLAC not loading through `wave` → temp 16 kHz WAV  
- Multi-GB FP32 intermediates and "do not put 600MB ONNX in git" discipline  
- Sandbox `rm` policy → pathlib unlink from project cwd  

I did not sit and pair-program each of those. I saw outcomes, ledgers, and pack paths. That is the difference between "AI wrote a sketch" and "AI ran a research loop."

### 5. Froze production and published

Quality script, freeze docs, `production.json` / `best_config.json`, README, model card, **GitHub** for code/results, **Hugging Face** for the static pack only. Weights out of git. Dual-pack clarity so A/B is reproducible.

Later, when I hit multi-minute OOM on full-file production, Neo added **CLI chunking** as an escape hatch without redefining the frozen short-clip RTF path. That matched how I think about product: memory path and speed path are not the same knob.

---

## What I still did as the human

Agents do not replace product taste. My lane:

| I owned | Why |
|---------|-----|
| Goal and success criteria | "Faster on CPU, publishable, honest gates" not "try quant" |
| Keep gate (≥5%) and freeze judgment | Prevents shipping noise |
| Mac long-audio A/B | Portability is a claim only if I measure it on my machine |
| Publish destinations and naming | `gauravvij/…`, `gvij/…`, what is production vs baseline |
| Scope cuts | Do not install OpenVINO if it replaces a working ORT wheel; do not re-run dead ladders |
| Narrative and audience | Technical record vs this personal post; no marketing fluff |
| "Is 3–4× realistic on the same 0.6B graph?" | Ceiling talk: encoder-bound, runtime ladder exhausted, next levers are smaller model / other backends |

I also had to clarify for myself mid-project: **production = static pack**, **baseline = Hub dynamic INT8 ONNX**, not "run NeMo every time." Obvious in hindsight; easy to muddle when three model IDs are in play.

---

## What surprised me

1. **The big win was not clever threading.** It was encoder static quant after a profile that said the encoder was everything.  
2. **Negative results were as valuable as keeps.** E5 chunking-as-speed and the whole E-ladder under 5% saved me from a false optimization story.  
3. **Agency shows up in recovery, not in the first green run.** Anyone can draft a quant script. Recovering from QOperator / Percentile / Hub glob failures and still landing a loadable pack is the bar.  
4. **Docs are product.** Dual-pack download, long-audio CLI, dual-host tables: without those, the HF pack is a file dump and the Mac measurement is a private anecdote.  
5. **BYOK model choice mattered for the long loop.** I used Grok 4.5 for this run. I am not claiming every model would plan and repair the same way; I am saying this combination completed a multi-phase ML optimization project with real artifacts, not a toy notebook.

---

## What I would do differently next time

- Start with a slightly larger quality set if I want public WER claims (the freeze set is a regression gate, not LibriSpeech).  
- Decide earlier whether OpenVINO / ANE is in scope as a **separate** backend track so it does not collide with the portable ORT story.  
- Keep a one-page "human decisions log" from day one (I reconstructed some of this from ledgers and chat).  
- Ask for the personal write-up and the technical record as two docs from the start (you are reading the late addition).

---

## If you want to try the same workflow

1. Clone the repo and read the technical case study for metrics and commands.  
2. Open the project in VS Code with Neo, point BYOK at a strong model (this project used **Grok 4.5**).  
3. Give a **gated goal**, not a script outline. Example shape:

> Using this repo, do X. Primary metric = geo-mean RTF of medium+long as already defined. Keep only if ≥5% and quality gate Y. Do not re-run dead ends in the ledgers / failed approaches. Report keep/discard with artifacts.

4. Stay available for product calls: freeze or not, publish or not, Mac check, what "done" means.

Prompt sketches for extensions (OpenVINO bake-off, real WER suite, INT4 probe, ANE, smaller model, multi-stream) live in the technical post. Steal those if you want Neo to continue from this codebase instead of rediscovering E5.

---

## Bottom line (personal)

I used Neo with Grok 4.5 to run a full CPU optimization loop on Parakeet TDT 0.6B v3: profile, runtime ladder (mostly discard), static quant ladder (real win), quality freeze, publish, then long-audio and dual-host product polish. Neo owned the implement-run-fix cycle and the ugly quant/export failures. I owned the goal, the honesty of the gates, Mac numbers, and what was allowed to ship.

Result I am willing to stand behind: on EPYC, about **half the primary RTF** vs Hub dynamic INT8 (~**2.07×**); on my Mac long audio, still a solid relative win (~**1.42×**). Not a new architecture. Not a 4× fairy tale on the same graph. A measured, published pack with a paper trail.

Technical depth and charts: [parakeet-cpu-optimization-case-study.md](parakeet-cpu-optimization-case-study.md).  
Code and weights: links at the top.

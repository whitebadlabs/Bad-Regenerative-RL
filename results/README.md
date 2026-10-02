# Raw results

Numbers only; no frames, clips or reference images.

| File | Source | Notes |
|---|---|---|
| `spark-gb10-runs.jsonl` | DGX Spark (GB10), H3 fp8 `fl2va`, Turbo 8-step LoRA unless `lora: false` | Wall clock with a **1 s poll** (no `exec_s`), so every value is rounded up to the next poll; see the 21.04-21.05 s chunk rows. `m-*` = resolution/steps matrix, `len-*-warm` = length sweep for the cost model, `same-c*` / `switch-c*` = chunk chains, `single-158-8` = single-clip control, `smoke-512-56-8` = first (cold) run, 48.1 s, excluded from all analysis. |
| `spark-gb10-chains.jsonl` | same | Chain summaries (time to first frame, total). |
| `spark-gb10-drift.jsonl` | same | DINOv2-base CLS cosine of each frame vs the reference image, with the full per-frame `series`. |
| `maxq-fasth3-timings.jsonl` | 2x RTX PRO 6000 Max-Q, FastH3 4-step, 512x512, 56 frames | `exec_s` from ComfyUI timestamps. `warm-*` are cold-start runs; `scale1-*` = 8 clips on one card; `scale2-*` = the same 8 clips split across two cards; `char-*` = 20 clips of a 2D anime-style character across two cards; batch rows give wall time per batch. Different reference and prompts from the Workstation runs; see findings/fasth3-benchmark.md section 5. |

The RTX PRO 6000 Workstation numbers in `findings/` were recorded in write-ups
rather than kept as raw files.

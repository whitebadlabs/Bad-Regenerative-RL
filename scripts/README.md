# Scripts

Cleaned versions of the harness behind [`findings/`](../findings/). They drive a
local ComfyUI server that has the MiniMax H3 nodes and weights installed; they do
not download models.

| Script | What it does |
|---|---|
| `comfy_bench.py` | Builds the H3 image-to-video graph (optional LoRA/adapter, sigma shift, sampler, scheduler, first/last frame), submits it, polls `/history` finely and records wall time **and** ComfyUI's own execution time. Appends one JSON row per run. |
| `chunk_chain.py` | Generates N chained chunks, each conditioned on the last decoded frame of the previous one. Records time to first frame and total time. Prompts from a JSON list (`prompts/example_chain.json`). |
| `dino_drift.py` | DINOv2 CLS cosine of every frame against a reference image, the run's own first frame, or another run frame-by-frame (fidelity vs a 20-step gold run). |
| `fit_cost_model.py` | Least-squares fit of `time = fixed + per_frame x frames`. |

```bash
pip install -r requirements.txt

# one clip, FastH3 4-step
python comfy_bench.py idle-fasth3 --reference subject.png \
    --prompt "One <subject> matching the first frame, standing relaxed. Locked camera, no zoom." \
    --lora fasth3_dense.safetensors --steps 4 --length 56 --out runs.jsonl

# 4-chunk chain
python chunk_chain.py switch --prompts prompts/example_chain.json \
    --reference subject.png --comfy-input-dir $COMFY/input --comfy-output-dir $COMFY/output \
    --lora fasth3_dense.safetensors --steps 4 --length 39

# drift of each chunk against the reference
for i in 0 1 2 3; do
  python dino_drift.py --frames-dir $COMFY/output/bench --prefix switch-c$i \
      --anchor $COMFY/input/subject.png --out drift.jsonl
done

# cost model from the archived DGX Spark sweep
python fit_cost_model.py ../results/spark-gb10-runs.jsonl --name-prefix len- --steps 8
# -> fixed_s 3.0, per_frame_s 0.4636
```

Notes

- Weight filenames default to the ComfyUI repackaged H3 files we used; override
  with `--unet/--clip/--vae/--lora`. The Turbo LoRA and the FastH3
  `dense-datafree` adapter both load through `LoraLoaderModelOnly`.
- Change `--seed` between repeated runs, and discard the first (cold) run.
- Differences from the original scripts: hard-coded paths and the reference
  filename removed, poll interval lowered from 1 s to 0.15 s, execution-time
  capture added (the original DGX Spark runs in `results/` predate it and are
  wall-clock only), prompts moved to a file.
- Not yet re-run end to end in this cleaned form. The cost-model fit has been
  checked against the archived data.

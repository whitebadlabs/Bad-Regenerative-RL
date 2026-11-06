# Diffusion RL fast-fail scripts

Write-up: [../../findings/diffusion-rl-fastfail.md](../../findings/diffusion-rl-fastfail.md).

- `gate3.sh`: official Flow-GRPO (`yifan123/flow_grpo@879042c`), `general_ocr_sd3_1gpu`, one GPU, eval and checkpoints pushed out of reach. Expects `$FASTFAIL_ROOT` with `venv/` (torch 2.8.0+cu128 for SM120, the repo's other pins, headless OpenCV), `src/flow_grpo` and an HF cache in `hf/` holding SD3.5-M (gated).
- `gate4_probe.py OUTDIR`: one prompt, seeds 42/1/2. Samples a 10-step SDE trajectory in FP32 with Flow-GRPO's `pipeline_with_logprob`, then re-scores every transition with `sde_step_with_logprob` with the transformer in FP32, BF16 and FP16. Writes `gate4_results.json` and `gate4_raw.npz`. Run from `$FASTFAIL_ROOT` with the same venv.

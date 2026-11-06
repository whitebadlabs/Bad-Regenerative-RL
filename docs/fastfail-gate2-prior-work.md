# Gate 2: is the diffusion-RL rollout/trainer log-prob mismatch question still open?

Date: 2026-11-05. Desk research only; nothing was executed from downloaded repos. Public sources only.

## 1. Classification: PARTLY DONE

Paper question: how much do per-step log-probs from the rollout engine (vLLM-Omni) differ from the trainer's (PyTorch/diffusers) across precision, kernel backend and GPU, and does that difference change GRPO training outcomes?

| Half | Status | Decisive source |
|---|---|---|
| Measure the mismatch on vLLM-Omni vs diffusers/FSDP, across attention backend, trainer precision and two GPU types | Largely DONE, as a public GitHub RFC with merged tooling, not as a paper | verl-omni #280, PR #291 |
| Show that the difference changes training outcomes (mismatch as the manipulated variable, reward as the outcome) | OPEN. I found no controlled diffusion experiment | Negative result, see section 5 |

The attractive headline is not available. "Matched backends sit at the bf16 noise floor" and "no kernel work is needed" are the maintainers' own stated findings. A paper presenting a measurement sweep as new would be scooped by #280 and would have to cite it.

What is left:

1. **Outcome effect.** A controlled reward-curve comparison. This is the real gap.
2. **Same pipeline across GPUs.** #280 ran SD3.5 only on RTX 4090, and Qwen-Image/BAGEL only on RTX PRO 6000. A later comment adds tiny Qwen-Image-Edit on L20. No pipeline was compared across GPU types, so GPU and pipeline are confounded.
3. **DGX Spark / SM121.** I found no measurement.
4. **Rollout and trainer on different GPU types.** Not tested in #280 (my reading: its runs each name one GPU type). DigenRL has a heterogeneous testbed, but a text search of the paper found no mismatch measurement.
5. **Precision arms.** #280 varied the trainer forward (bf16 vs fp32). I found no vLLM-Omni fp16 arm or NVFP4 rollout arm measuring per-step log-prob difference.
6. **Statistics.** #280 says "single seed per arm" and each arm is 3 training steps x 64 samples.

Scoop risk: verl-omni's open Q4 roadmap (#700) lists "0-diff train rollout ... for diffusion models [assignee]" and "Determinstic diffusion RL", both unchecked. I found no PR for the diffusion item. If it lands, it removes mismatch as a problem but also provides a clean control arm (0-diff vs native) for the outcome question.

### A design constraint for any outcome experiment

In default verl-omni operation the rollout/trainer difference does not enter the PPO ratio directly:

- `bypass_mode` defaults to false (`_generated_diffusion_trainer.yaml:666`), and `calculate_log_probs` defaults to false (`workers/config/diffusion/rollout.py:197`).
- `old_log_probs` is a trainer replay; the ratio compares the current policy against that replay.
- Bypass mode sets `old_log_probs := rollout_log_probs` (`rollout_correction.py:65`). Only then does the difference sit inside the ratio.
- Official `flow_grpo` has no replay stage: `ratio = torch.exp(log_prob - sample["log_probs"][:, j])` (`scripts/train_sd3.py:891`), with both sides from the same code stack.
- `docs/contributing/common_pitfalls.md` supports this: a stored-latent precision bug showed "`actor/ratio_mean` consistently below `1.0` (e.g. `0.99996`)" at step 1 and was "Most visible with rollout correction (`bypass_mode=True`)".

So a default-mode A/B would mostly test the behaviour-policy shift, not ratio corruption. Bypass mode is the arm where an effect is most plausible. This is my reading of the code and docs, not a conclusion any source states.

Also note scale: `log_prob` is mean-reduced over latent dimensions (`sd3_sde_with_logprob.py:89`), so these magnitudes are not comparable to per-token LLM values. That matters when comparing against the LLM section below.

## 2. Verified papers and repositories

| Method | Paper | Repo | License | Pinned commit / last push |
|---|---|---|---|---|
| Flow-GRPO (NeurIPS 2025) | [arXiv 2505.05470](https://arxiv.org/abs/2505.05470) | [yifan123/flow_grpo](https://github.com/yifan123/flow_grpo) | MIT | `879042cf5707f8b90daa98d147d7deac2317c5da` (2026-05-07) |
| DanceGRPO | [arXiv 2505.07818](https://arxiv.org/abs/2505.07818) | [XueZeyue/DanceGRPO](https://github.com/XueZeyue/DanceGRPO) | Apache-2.0 | `15cc71d53cc2e6e18a68ee607d5fb6ba9a99e344` (2025-10-16) |
| MixGRPO | [arXiv 2507.21802](https://arxiv.org/abs/2507.21802) | [Tencent-Hunyuan/MixGRPO](https://github.com/Tencent-Hunyuan/MixGRPO) | Custom "License Terms of MixGRPO" (`License.txt`; GitHub API: NOASSERTION) | `19946facaefa33eca1b565ec87c6fc7cca14e339` (2026-07-01) |
| GRPO-Guard | [arXiv 2510.22319](https://arxiv.org/abs/2510.22319) | In `flow_grpo` (`scripts/train_sd3_GRPO_Guard.py`) | MIT (repo) | as Flow-GRPO |
| VeRL-Omni | No arXiv paper found (the arXiv API query for "verl-omni" returned only DigenRL; LeanGRPO cites the repo URL) | [verl-project/verl-omni](https://github.com/verl-project/verl-omni), [docs](https://verl-omni.readthedocs.io/en/latest/) | Apache-2.0 | `9bcb5af0b81f69c17d63b2af0cbc1ec16cf1cd3f` (2026-11-05); pins `vllm==0.30.0`, vllm-omni `12e9280010afc41ce5c6a1f46900fc329cf08377` |

The plan's names were all correct. The canonical repo is `verl-project/verl-omni`.

## 3. Gate 3 prerequisites

### Flow-GRPO official code (`yifan123/flow_grpo` @ `879042cf`)

**This repo contains no vLLM-Omni rollout.** Rollout and training are the same diffusers stack, so it cannot supply the engine-vs-trainer comparison alone. The README changelog points to verl-omni for that path. Gate 3 likely needs verl-omni (below), with `flow_grpo` as a reference implementation.

| Item | Finding |
|---|---|
| Smallest supported config | `config/grpo.py:general_ocr_sd3_1gpu` (line 444): SD3.5-Medium, LoRA (`use_lora=True` inherited), resolution 512, `num_steps=10`, `guidance_scale=4.5`, `train_batch_size=8`, `num_image_per_prompt=8`, reward `ocr`, `clip_range=1e-4` (inherited from `config/base.py`) |
| Precision | fp16: `config.mixed_precision = "fp16"` in `config/base.py`, and `mixed_precision: fp16` in `scripts/accelerate_configs/multi_gpu.yaml`. verl-omni's recipe is bf16, so the two stacks differ |
| Entry command | Quoted from `scripts/single_node/grpo.sh`: `accelerate launch --config_file scripts/accelerate_configs/multi_gpu.yaml --num_processes=1 --main_process_port 29501 scripts/train_sd3.py --config config/grpo.py:general_ocr_sd3_1gpu` |
| Install | README: `conda create -n flow_grpo python=3.10.16`, `pip install -e .`. `setup.py` pins `torch==2.6.0`, `torchvision==0.21.0`, `transformers==4.40.0`, `accelerate==1.4.0`, `diffusers==0.33.1`, `peft==0.10.0`, `deepspeed==0.16.4`, `bitsandbytes==0.45.3`; `xformers` unpinned. `config/grpo.py` does `import imp`, removed in Python 3.12, so stay on 3.10 or 3.11 |
| Policy weights | `stabilityai/stable-diffusion-3.5-medium`: **gated** (HF API `gated: auto`, `license:other`, sha `b940f670f0eda2d07fbb75229e779da1ad11eb80`). Needs an HF token and license acceptance |
| Reward weights | OCR: PaddleOCR (`pip install paddlepaddle-gpu==2.6.2 paddleocr==2.9.1 python-Levenshtein`). PickScore: `yuvalkirstain/PickScore_v1` + `laion/CLIP-ViT-H-14-laion2B-s32B-b79K`, both `gated: False`. Only a 4-GPU SD3 PickScore config exists (`pickscore_sd3_4gpu`), so a 1-GPU PickScore run needs a config edit |
| Memory, official | No claim for SD3. The only stated figure in the README is for BAGEL: "at least 8 x 80GB GPUs" for full-parameter mode |
| Memory, community reports (not verified) | [#25](https://github.com/yifan123/flow_grpo/issues/25): 1 GPU OOM on A100 with an older config; reducing `train_batch_size` from 12 to 6 brought it to 55GB. [#172](https://github.com/yifan123/flow_grpo/issues/172): reporter says configs assume 80GB A100s; OCR and PickScore ran on 24GB GPUs with gradient checkpointing and DeepSpeed off |
| Time, maintainer report | [#96](https://github.com/yifan123/flow_grpo/issues/96): OCR reaches 90% at 1k steps, "16 hours on 24 A-series GPUs or 8 hours on 24 H-series GPUs" |

### Known constraints on SM120/SM121

| Constraint | Evidence | Status |
|---|---|---|
| `torch==2.6.0` pin likely fails on SM120 | a PyTorch maintainer on the [PyTorch forum](https://discuss.pytorch.org/t/pytorch-support-for-sm120/216099): first prebuilt wheel he tested listing `sm_120` was `2.7.0.dev20250131+cu128`. The torch 2.6.0 CUDA wheels I saw for cp310 are cu126 | Inferred. I did not download the 2.6.0 wheel to read its arch list. Expect to override the pin, then re-check the other pins |
| SM121 (GB10) | Not mentioned in that forum thread. No public SM121 result for this stack was found | Unverified |
| aarch64 (Spark) torch 2.6.0 | `whl/cu126/torch/` lists `torch-2.6.0+cu126-cp310-cp310-linux_aarch64.whl`, so a wheel exists | The wheel exists; its kernels for GB10 are unverified |
| OCR reward on Spark | `paddlepaddle-gpu==2.6.2` on PyPI: only `manylinux1_x86_64` and `win_amd64` wheels. `flow_grpo/ocr.py` defaults `use_gpu=False`, but the pinned GPU package still has no aarch64 wheel | Blocks the documented install on aarch64. Options: a different reward, or a CPU paddle build (not checked) |
| Other aarch64 pins | `bitsandbytes==0.45.3`: `manylinux_2_24_x86_64` and `win_amd64` only. `deepspeed==0.16.4`: sdist and `win_amd64` only | Install likely needs editing on Spark. The `train.use_8bit_adam` default is False, so bitsandbytes may be avoidable |
| FLASH_ATTN / `flash-attn` | Commented out in `setup.py`; imported only under `flow_grpo/bagel/`. Not needed for SD3 | Clear for SD3 |
| DeepSpeed | [#172](https://github.com/yifan123/flow_grpo/issues/172): "I highly discourage using Deepspeed, as it seems to destabilize training" (one user, no controlled test) | Avoid. The 1-GPU command uses `multi_gpu.yaml`, not DeepSpeed |
| Batch-size dependence | README FAQ: "SD3 has this issue, which is why I ensure that the batch size for training is the same as that used for data collection" | Keep rollout and replay batch shape controlled |
| No SM120 reports in the repo | GitHub issue search `repo:yifan123/flow_grpo sm_120 OR RTX CUDA capability`: 0 results | Absence of reports, not evidence it works |

### verl-omni path (the one that has vLLM-Omni) @ `9bcb5af`

| Item | Finding |
|---|---|
| Requirements | `docs/start/install.md`: Python >= 3.11, CUDA >= 13.0, driver 580+ natively (535+ on datacenter GPUs via cuda-compat). `pyproject.toml`: `vllm==0.30.0`, vllm-omni @ `12e9280`, `diffusers>=0.41.0` |
| Smallest documented real recipe | `examples/flowgrpo_trainer/sd35/run_sd35_medium_ocr_lora_v1.sh`: SD3.5-Medium LoRA rank 32, 384x384, 10 steps, `guidance_scale=1.0`, `sde_type="cps"`, `sde_window_size=3`, `clip_ratio=1e-5`, `model_dtype=bfloat16`, 100 steps |
| GPUs / memory | `docs/start/models.md`: "3×GPU (2 actor+rollout, 1 reward)". No per-GPU memory figure found for this recipe |
| Entry | `python3 -m verl_omni.trainer.main_diffusion_v1 ...` (via the script above) |
| Weights | SD3.5-Medium (gated), `Qwen/Qwen2.5-VL-3B-Instruct` as the OCR judge (gating not checked) |
| Metric switch | `actor_rollout_ref.rollout.calculate_log_probs=True` logs `rollout_corr/logprob_abs_diff_mean`, `_max`, and `rollout_corr/logprob_abs_diff/ts_{t}` (`docs/algo/rollout_correction.md`), with bypass off |
| Bypass arm | `algorithm.rollout_correction.bypass_mode=True` |
| Attention arms | `ATTN_BACKEND=native`, `ROLLOUT_ATTN_BACKEND=TORCH_SDPA` by default; `FA3=1` selects `_flash_3_varlen_hub` and `FLASH_ATTN_3_HUB`. #280 says of its Qwen/BAGEL runs: "ran on sm_120 (forces the native/TORCH_SDPA pairing...)". The reason is not stated, so the backend axis may collapse on Blackwell. Needs an empirical check |
| Cheap smoke test | `tests/special_e2e/build_sd3_tiny_random.py` builds a random-weight SD3 checkpoint, avoiding the gated download for plumbing checks |
| Arm64 | I found no aarch64 install path for the CUDA stack in `docs/start` or `docker`. Whether vLLM-Omni runs on GB10 is unverified |

**Two settings differ between stacks.** The verl-omni v1 recipe uses `guidance_scale=1.0` and `clip_ratio=1e-5`; the `flow_grpo` 1-GPU config uses guidance 4.5 and `clip_range=1e-4`. #280 found CFG the strongest amplifier, and its SD3.5 baseline mean|d| of 2.24e-5 is above 1e-5, so these choices materially affect the result.

## 4. verl / verl-omni status (quotes)

**Roadmap Q3, [#97](https://github.com/verl-project/verl-omni/issues/97) (closed 2026-10-13), "Rollout-Train consistency":** unchecked at close: "Improve VeOmni/DiffusersFSDP trainer and vLLM-Omni rollout consistency (attention kernel, fused ops, mixed precision config etc)"; "0-diff train rollout for true on-policy RL". Checked: "Attention kernel alignment: support `kernel` package backend in vllm-omni". A 2026-07-10 comment states the exact plan under test here: "first measure the rollout vs trainer-replay log-prob mismatch on SD3.5-Medium FlowGRPO across attention-backend / precision configs".

**[#280](https://github.com/verl-project/verl-omni/issues/280) "[RFC] Rollout-train log-prob consistency: measurements + default monitoring"** (2026-07-17, open). Definition: "`d = old_log_probs (trainer replay) − rollout_log_probs (vLLM-Omni)`". Setup: "verl-omni `01c87ee` + vllm-omni `fe478a95`", "3 training steps × 64 samples per arm, seed 42".

| Pipeline / arm | Hardware | mean\|d\| | p50 | p99 |
|---|---|---|---|---|
| SD3.5-M baseline (native/TORCH_SDPA, bf16, cps) | 3×RTX 4090 | 2.24e-5 | 1.44e-5 | 1.30e-4 |
| SD3.5-M `sde_type=sde` | 3×RTX 4090 | 3.60e-5 | 1.42e-5 | 2.60e-4 |
| SD3.5-M late window | 3×RTX 4090 | 6.13e-5 | 5.84e-5 | 2.06e-4 |
| SD3.5-M CFG 4.5 | 3×RTX 4090 | 4.71e-4 | 2.90e-4 | 2.69e-3 |
| Qwen-Image 20B, true CFG 4.0 + sde | 4×RTX PRO 6000 96GB | 5.89e-5 | 3.55e-5 | 2.95e-4 |
| BAGEL as found (schedule bug) | 4×RTX PRO 6000 96GB | 5.03e-3 | 5.00e-3 | 5.56e-3 |
| BAGEL after fix | 4×RTX PRO 6000 96GB | 3.20e-5 | 2.65e-5 | 1.16e-4 |

Quotes:
- "Three more arms with FA3 hub attention, fp32 trainer forward, replay micro-batch 4 are all within single-seed noise of a0." (Their numbers are not published.)
- "Matched backends sit at the bf16 noise floor on all three pipelines ... attention pairing, trainer precision, and replay batch shape don't move it".
- "CFG amplifies the mismatch pair-dependently: 21× on SD3.5 at g=4.5 (median past the default `clip_ratio=1e-4`)".
- "d is timestep- and sde_type-dependent (7–34× growth across the window".
- "Non-goals: kernel/op-level alignment and default changes".
- "VeOmni remains unmeasured".
- Comment by 0z5a, 2026-09-15, on 50 tiny Qwen-Image-Edit steps on one L20: maximum 4.4994e-4, "tiny-model observed values, not a universal numerical tolerance or model-quality claim."

**Merged:** [PR #291](https://github.com/verl-project/verl-omni/pull/291) (2026-07-22) logs the metric per global batch when `calculate_log_probs=true` and bypass is off; its body also notes "recipes shipping `clip_ratio`/`kl_mask_threshold` at 1e-5 sit below the noise floor". [PR #279](https://github.com/verl-project/verl-omni/pull/279) (2026-07-17) fixed the BAGEL schedule. [PR #93](https://github.com/verl-project/verl-omni/pull/93) (2026-05-27) added rollout correction, "only validated in a small set of training (flowgrpo+ocr)".

**In progress:**
- [#700](https://github.com/verl-project/verl-omni/issues/700) Q4 roadmap (open): "0-diff train rollout for true on-policy RL ... for omni-modal models (AR) #637 ... for diffusion models [assignee]"; "Determinstic diffusion RL"; "Improve VeOmni trainer and vLLM-Omni rollout consistency".
- [#637](https://github.com/verl-project/verl-omni/issues/637) (open): 0-diff for the Qwen3-Omni thinker, an autoregressive LLM-like path rather than diffusion. It reports 0/8192 mismatches on a 4-layer truncated model with an in-process prototype. Follow-ups: verl#8025 (closed), verl#8026 (open), vllm-omni#8186 (state not checked).
- [PR #784](https://github.com/verl-project/verl-omni/pull/784) (open): L4 convergence gate fails if "any post-warmup step > 1e-4" on the consistency metric. It assumes the threshold matters; it does not demonstrate it.
- [#484](https://github.com/verl-project/verl-omni/issues/484) (closed): train-side rollouter, and [#203](https://github.com/verl-project/verl-omni/issues/203) (open): determinism. `docs/algo/deterministic_post_training.md` marks "Diffusion sampling determinism" as future.

**Repo docs:** `docs/algo/rollout_correction.md` (Experimental): "With matched attention backends the gap is bf16 noise, mean |Δlogp| in the 1e-5 range; guidance scale and later denoising steps amplify it."

## 5. Adversarial search for the DONE paper

### Closest prior work on the outcome half (none is a controlled test)

| Source | What it shows | Why it does not close the gap |
|---|---|---|
| verl-omni #279/#280 | BAGEL bug gave `d=+5.0e-3`: "every sample ~500x outside the clip band". Replay crashed at HEAD | Numbers on the ratio only. No before/after reward curve in the issue. It is a schedule bug, not numerics |
| [flow_grpo #172](https://github.com/yifan123/flow_grpo/issues/172) | Reward flat with DeepSpeed; recovered with it off. User's hypothesis: "DeepSpeed may be introducing additional numerical errors" | One user, hypothesis only, and DeepSpeed changes more than numerics |
| `flow_grpo` README FAQ | "use fp16 ... smaller log-probability errors"; "Training only on high-noise steps yields better results in this case" | Observation with no published ablation |
| verl-omni [#561](https://github.com/verl-project/verl-omni/issues/561) | v1 trainer collapse | Reporter suspected seed collision (not solved); notes "recomputed old log probs are bitwise-equal". Not attributed to mismatch |
| verl-omni [#188](https://github.com/verl-project/verl-omni/issues/188) | Flow-DPPO reward gap; maintainers plan deterministic rollout | No log-prob numbers on the issue page |
| verl-omni [PR #766](https://github.com/verl-project/verl-omni/pull/766) (open) | Convergence guide routes numerical mismatch to other docs; the visible page has no controlled mismatch test | Guide body not read in full |
| [Sol-RL, arXiv 2604.06916](https://arxiv.org/abs/2604.06916) (NVIDIA/HKU/MIT) | Fig. 3b: "Directly integrating FP4 rollout in RL pipeline leads to severe instability and performance degradation compared to the BF16 baseline." | **Nearest paper.** Real outcome evidence, but for FP4 quantized rollout, with the mechanism framed via off-policy distribution shift and cited LLM work [14][15]. Text search found no per-step log-prob difference measure. Not bf16 engine-vs-trainer. Worth citing as motivation |

### Same words, different problem

- [LC-GRPO, 2608.05600](https://arxiv.org/abs/2608.05600): "train-inference gap" means SDE rollout vs ODE inference discretization, not backend numerics.
- [λ-Controlled GRPO, 2609.22041](https://arxiv.org/abs/2609.22041): ratio instability attributed to path variance from the sampler kernel (abstract).
- [GRPO-Guard, 2510.22319](https://arxiv.org/abs/2510.22319) and [Flow-DPPO, 2606.11025](https://arxiv.org/abs/2606.11025): ratio bias and noise as algorithmic properties.
- [LeanGRPO, 2609.03528](https://arxiv.org/abs/2609.03528): assumes "rollout and update use the same backend"; its numerics concern gradient equivalence. It says VeRL-Omni's rollout graph cannot be consumed directly by the training backend.
- [DigenRL, 2606.24369](https://arxiv.org/abs/2606.24369): uses per-step log-probs on vLLM-Omni + FSDP, covers staleness and policy-version consistency; text search found no numeric mismatch measurement.
- [OP-GRPO, 2604.04142](https://arxiv.org/abs/2604.04142): off-policy reuse efficiency (seen in search summaries only, not read in full).
- arXiv hits for "training-inference mismatch" AND "diffusion" (27, 12 listed) are by title mostly diffusion language models or unrelated. SOAR (2604.12617) and Diffusion-DRF (2601.04153) matched "Flow-GRPO" AND "mismatch"; I judged them by title and did not read them.
- [celve/miles PR #27](https://github.com/celve/miles/pull/27): Flow-GRPO SD3.5 port on SGLang; lists a `log_prob_mean_abs_diff` parity metric but reports no values. Shows a third engine exists.

### LLM-side analogues (kept separate: not diffusion)

| Source | Finding |
|---|---|
| Yao et al., [On the Rollout-Training Mismatch in Modern RL Systems](https://opt-ml.org/papers/2025/paper116.pdf) (NeurIPS 2025 workshop) | vLLM vs FSDP token probabilities differ; proposes truncated importance sampling |
| [VeXact, 2605.14220](https://arxiv.org/abs/2605.14220) | "small token-level numerical disagreements can independently cause training collapse"; this is the outcome design the diffusion side lacks |
| [Relax, 2604.11554](https://arxiv.org/abs/2604.11554) | bf16 to fp16: difference 0.0122 to 0.0016, 7.7× (via search summary: earlier positive reward, similar final reward) |
| [Defeating the Mismatch via FP16, 2510.26788](https://arxiv.org/abs/2510.26788) | FP16 vs BF16 stability (search summary) |
| [bf16 Rounding, 2610.41073](https://arxiv.org/abs/2610.41073) | Hidden master-vs-served mismatch; abstract does not mention diffusion |
| [Deterministic Inference across TP sizes, 2511.17826](https://arxiv.org/abs/2511.17826), [QaRL, 2604.07853](https://arxiv.org/abs/2604.07853), [Mirage, 2606.29526](https://arxiv.org/abs/2606.29526) | Seen in search results; not read |
| vLLM [RFC #48305](https://github.com/vllm-project/vllm/issues/48305), [RFC #42259](https://github.com/vllm-project/vllm/issues/42259); verl PR 7688 (as referenced by #637); ServiceNow [blog](https://huggingface.co/blog/ServiceNow-AI/correctness-before-corrections) | Engine-level work on LLM consistency |

**Transfer caution:** diffusion log-probs here are mean-pooled and diffusion recipes use clip ratios of 1e-4 to 1e-5, so LLM collapse results may or may not carry over. That uncertainty is part of what is still open.

## 6. Searches run

**WebSearch (14):**
1. Flow-GRPO training flow matching models via reinforcement learning arXiv
2. DanceGRPO unleashing GRPO on visual generation arXiv github
3. MixGRPO deterministic stochastic sampling diffusion GRPO arXiv
4. verl-omni diffusion reinforcement learning vLLM-Omni rollout Flow-GRPO
5. rollout engine vs trainer log-prob mismatch diffusion reinforcement learning GRPO per-step log probability vLLM-Omni diffusers (extended)
6. vLLM logprobs mismatch trainer rollout RL importance ratio inference engine numerical discrepancy paper 2026 (extended)
7. diffusion GRPO "training-inference mismatch" OR "rollout-training mismatch" flow matching log-likelihood importance ratio numerical precision arXiv 2026 (extended)
8. does floating point mismatch between sampler and trainer affect policy gradient for diffusion models image generation reinforcement learning bf16 log-prob noise floor clip ratio (extended)
9. Flow-GRPO log-prob reproducibility attention kernel backend GPU architecture discrepancy old_log_prob recompute vs rollout, diffusion RL stability "bypass" importance sampling correction
10. arXiv 2026 video generation RL DanceGRPO FlowGRPO disaggregated rollout inference engine consistency on-policy zero-diff log probability
11. PyTorch 2.6 "sm_120" not compatible with current PyTorch installation RTX PRO 6000 Blackwell which version supports
12. Flow-GRPO SD3.5 "bf16" vs "fp16" training outcome log-prob error reward curve ablation, rollout replay mismatch effect on convergence GRPO diffusion arXiv study (extended)
13. "logprob_abs_diff" OR "rollout_probs_diff" diffusion FlowGRPO rollout trainer mismatch clip_ratio noise floor reward effect
14. zero-mismatch rollout engine diffusion RL image generation training collapse numerical disagreement Flow-GRPO true on-policy bitwise log-probs 2026 (extended)

**arXiv API (total results):**
- `all:"verl-omni"`: 1 (DigenRL)
- `all:"rollout" AND all:"log-prob" AND all:"diffusion" AND all:"mismatch"`: 0
- `all:"training-inference mismatch" AND all:"diffusion"`: 27
- `all:"rollout-training mismatch" AND all:"flow matching"`: 0
- `all:"Flow-GRPO" AND all:"mismatch"`: 2
- `all:"FlowGRPO" AND all:"numerical"`: 0
- `all:"diffusion reinforcement learning" AND all:"precision" AND all:"log-probability"`: 0

**GitHub issue/PR search (unauthenticated API):**
- `repo:verl-project/verl-omni rollout_corr logprob`: 9
- `repo:verl-project/verl-omni "log-prob" consistency`: 40 (top 15 titles reviewed)
- `repo:verl-project/verl-omni bypass mode log probs`: 33 (top 15 titles reviewed)
- `repo:verl-project/verl-omni convergence mismatch rollout trainer`: first attempt rate-limited, retry 34 (top 10 reviewed)
- `repo:verl-project/verl-omni rollout correction bypass diffusion convergence is:pr`: 2
- `repo:verl-project/verl-omni 0-diff diffusion`: 46 (top 10 reviewed, no diffusion 0-diff PR by title)
- `repo:verl-project/verl 0-diff log-probs`: 32 (top 15 reviewed)
- `repo:vllm-project/vllm-omni log_probs rollout trainer consistency`: rate-limited, not retried
- `repo:vllm-project/vllm-omni log_probs FlowGRPO consistency`: 0
- `repo:yifan123/flow_grpo ratio log_prob`: 6
- `repo:yifan123/flow_grpo memory OOM GPU`: 10
- `repo:yifan123/flow_grpo sm_120 OR RTX CUDA capability`: 0

Weakness: GitHub search covers titles and bodies, not code in other forks. I did not search DanceGRPO or MixGRPO issues, Chinese-language venues, WeChat groups or OpenReview. Search-engine summaries can miss recent preprints; the arXiv API uses literal phrases, so a paper with different wording could be missed.

## 7. Unverified

- Actual peak memory of `general_ocr_sd3_1gpu`, and of the verl-omni SD3.5 recipe per GPU.
- Whether torch 2.6.0 wheels really lack `sm_120` (inferred from the forum and wheel lines; not read from the wheel).
- Whether any torch/vLLM-Omni build supports SM121 for this stack; whether vLLM-Omni installs on aarch64.
- Whether a torch >= 2.7 override coexists with `diffusers==0.33.1`, `transformers==4.40.0`, `peft==0.10.0`.
- Why #280 says sm_120 "forces" native/TORCH_SDPA, and whether that is still true at the current pin.
- Values for the #280 FA3, fp32 and micro-batch arms (only described as within noise).
- Whether #280's rollout and trainer ever ran on different GPU types (it does not say so).
- #280 numbers were not reproduced; they are self-reported and AI-assisted per the author ("Analysis AI-assisted; all numbers hardware-validated").
- Reward impact of BAGEL #279: not found.
- Sol-RL figures were read as text only, not visually. DigenRL, Sol-RL and LeanGRPO absence claims come from text search, plus one fetch-tool summary for LeanGRPO that covered 100k of 112k characters.
- `Qwen/Qwen2.5-VL-3B-Instruct` gating; vllm-omni#8186 state; verl#7688 not read directly.
- PR #766 guide body, and full text of OP-GRPO, SOAR and Diffusion-DRF.
- DanceGRPO and MixGRPO repos: metadata and license only; contents not read.

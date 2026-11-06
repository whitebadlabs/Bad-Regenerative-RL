# Gate 1: does vLLM-Omni expose per-step diffusion log-probs?

Desk research only, 2026-11-05. No repo code was executed and nothing was posted.

## Classification: SMALL PATCH

Stock vLLM-Omni does not give per-step mean, sigma, noise or log-prob for any standard image or video model.
The gap is small, and it does not require engine work:

- **Output side: already in the engine.** `trajectory_log_probs` travels from the pipeline to `OmniRequestOutput`.
- **Producer side: missing.** Except for BAGEL's log-prob slot, no stock pipeline fills it. BAGEL also needs a scheduler supplied by a subclass.
- **Zero-engine-patch route: documented and used.** `custom_pipeline_args.pipeline_class` lets a user subclass a pipeline and swap the scheduler. Two public RL frameworks do this. UniRL's Qwen-Image subclass is 111 lines, plus a 260-line shared SDE scheduler and a 206-line shared interception helper.

If the patch goes in the engine instead, the exact site is:

- **File and function:** `vllm_omni/diffusion/distributed/cfg_parallel.py`, `CFGParallelMixin.scheduler_step` (line 563)
- **Why:** its last line, `return sched.step(noise_pred, t, latents, **step_kwargs)[0]`, drops everything except `prev_sample`.
- **Second site:** `QwenImagePipeline.step_scheduler` (`pipeline_qwen_image.py:974`) passes no `generator`, so an injected stochastic scheduler would not be seeded from the request.

The < ~100 line figure is my estimate for the capture hook alone. Nothing was built or run. RFC #8182 item 3 proposes this same change.

### Read this first: the paper's measurement already exists

This is outside Gate 1, but it bears on whether the paper is novel.

When the rollout returns log-probs and bypass mode is off, verl-omni already logs the paper's quantity once per global batch as `rollout_corr/logprob_abs_diff_mean`, `_max`, and per timestep as `logprob_abs_diff/ts_{t}`. Its docs state a magnitude:

> "With matched attention backends the gap is bf16 noise, mean |Δlogp| in the 1e-5 range; guidance scale and later denoising steps amplify it."
> (`docs/algo/rollout_correction.md`, verl-omni@9bcb5af)

vLLM-Omni PR #7907 (open) reports rollout/replay mean differences by mode: image edit 5.64e-4 / 1.86e-4, video edit 4.07e-5 / 2.48e-5, image-to-video 1.71e-5 / 1.77e-5. The PR itself says these runs predate its active-token correction and do not validate the current objective.

One standard web search found no paper dedicated to this measurement. That is weak evidence, not a literature review. The closest hit was arXiv 2605.14220 (VeXact), which covers LLM token log-probs, not diffusion; I saw it only through a search summary.

**Normalization caveat:** verl-omni's log-prob is averaged over all non-batch dimensions (`log_prob.mean(dim=tuple(range(1, log_prob.ndim)))`, `flow_match_sde.py`). So 1e-5 is a per-element figure. A joint Gaussian log-prob sums over elements, which by my arithmetic scales the gap by the latent's element count. How much that matters to a PPO ratio is an open question and possibly the paper's real angle. Note that the `cps` branch also omits the Gaussian normalizer and the variance term.

## Pinned sources

| Repo | Commit | Commit date (UTC) |
| --- | --- | --- |
| [vllm-project/vllm-omni](https://github.com/vllm-project/vllm-omni/tree/04ddf21049664c0c4e91dfbff1a97e4db7a9d90c) | `04ddf21049664c0c4e91dfbff1a97e4db7a9d90c` | 2026-11-05T15:23:05Z |
| [verl-project/verl-omni](https://github.com/verl-project/verl-omni/tree/9bcb5af0b81f69c17d63b2af0cbc1ec16cf1cd3f) | `9bcb5af0b81f69c17d63b2af0cbc1ec16cf1cd3f` | 2026-11-05T14:42:12Z |
| [Tencent-Hunyuan/UniRL](https://github.com/Tencent-Hunyuan/UniRL/tree/0d07b80b952a8ea3ae6c4d2df3a42a3628cec096) | `0d07b80b952a8ea3ae6c4d2df3a42a3628cec096` | 2026-11-02T05:56:10Z |
| huggingface/diffusers (one file) | tag `v0.40.0`, the version pinned in `requirements/common.txt:15` | not recorded |

Paths below are in vLLM-Omni at the pinned commit unless marked otherwise.

The frameworks do not track this commit. verl-omni pins vLLM-Omni at `12e9280010afc41ce5c6a1f46900fc329cf08377` (`pyproject.toml:54`), and UniRL pins `vllm-omni==0.28.0` (`pyproject.toml:87`). So their adapters are not shown to work against `04ddf21`.

## Q1. Supported diffusion, image and video models

Source: `docs/models/supported_models.md` lines 22-138 (117 rows, which also cover TTS, ASR and AR models). The registry is `vllm_omni/diffusion/registry.py`.

| Group | Models |
| --- | --- |
| Image | Qwen-Image, Qwen-Image-2512, Qwen-Image-Edit, -Edit-2509, -Edit-2511, -Layered, Qwen-Image-2.1; JoyAI-Image-Edit; GLM-Image; Z-Image; Krea 2; MOSS-Image-Base; Ovis-Image; LongCat-Image and -Edit; Boogu-Image and Turbo; SDXL; SD3.5; FLUX.1-dev, -schnell, -Kontext; FLUX.2-dev, -klein; OmniGen2; NextStep-1.1; ERNIE-Image; HiDream-O1-Image; Anima |
| Image, MoT or multi-stage | BAGEL; HunyuanImage3.0; SenseNova-U1, U1.5, U1-A3B; Ming-flash-omni-2.0 imagegen; MammothModa2-Preview |
| Video | Wan2.1 and 2.2 T2V, TI2V, I2V, S2V, VACE; LTX-2, 2.3, 2.5 (one-stage, two-stage, distilled); LingBot-Video; SANA-Video T2V and I2V; SeedVR2; MiniMax H3, FastH3 variants, Open-H3-Regenerate-2K; TaoMate-H3; MAGI-2; Helios; LongCat-Video-Avatar; HunyuanVideo-1.5 T2V and I2V; Kandinsky 6; Lance; Cosmos3 (also T2I) |
| World model | LingBot-World, ABot-World, SANA-WM, Hy-Nomas |
| Robotics policy | InternVLA-A1, Alpamayo 2 Super, GR00T N1.7, DreamZero, Pi0, Pi0.5 |
| Audio diffusion | Stable-Audio-Open, Stable-Audio-3, AuK, OmniVoice |
| DMD2 pipelines (registry only, not table rows) | `QwenImageDMD2Pipeline`, `FluxDMD2Pipeline`, `WanT2VDMD2Pipeline`, `WanI2VDMD2Pipeline`, `LTX2T2VDMD2Pipeline`, `LTX2I2VDMD2Pipeline` (`registry.py:136-348`) |

The grouping is mine, and the audio row is my judgement of which models are diffusion-based. Sam3 and Video Depth Anything run through the diffusion stage but are not generative, so I left them out.

## Q2. Stochastic sampling: yes for some, but not user-selectable on the main models

**There is no request or CLI flag that turns on SDE for stock Qwen-Image, FLUX, SD3, Wan or LTX-2.** The scheduler is hardcoded:

```python
# vllm_omni/diffusion/models/qwen_image/pipeline_qwen_image.py:316
self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
    model, subfolder="scheduler", local_files_only=local_files_only
)
```

Scheduler injection is proposed, not landed. RFC [#6269](https://github.com/vllm-project/vllm-omni/issues/6269) is open, and PR [#6277](https://github.com/vllm-project/vllm-omni/pull/6277) is open and unmerged. A grep for `build_pipeline_scheduler`, `register_scheduler` and `vllm_omni.schedulers` across `vllm_omni`, `docs`, `pyproject.toml` and `setup.py` finds nothing at this commit. The RFC states the motivation directly:

> "These algorithms need the denoise loop to run an **SDE variant** of the flow-matching scheduler — one that injects noise at chosen steps and computes per-step log-probabilities of the sampled latents."

The `eta` field on the sampling params (`vllm_omni/inputs/data.py`, `eta: float = 0.0`) is not evidence of SDE support. My grep for readers of it was noisy (it matched `theta=`), so whether any stock pipeline reads it is unverified. UniRL's subclass does read it.

### Stochastic code paths that exist

| Path | How it is enabled | Evidence | Selectable? |
| --- | --- | --- | --- |
| DMD2 pipelines (Qwen, FLUX, Wan, LTX-2) | `model_index.json` has `dmd2_config.solver: "sde"` | `dmd2/mixin.py:35` `stochastic_sampling=(self.dmd2_config.solver == "sde")`; `dmd2/config.py:11,31` | Per checkpoint, not per request; default is `"ode"` |
| Cosmos3 distilled | Forced on when the checkpoint is distilled | `cosmos3/pipeline_cosmos3.py:1178-1181`; passes `generator` at `:1639` | No |
| Stable Audio 3, `pingpong` | `extra_args["sampler_type"]`, audio only | `stable_audio_3/pipeline_stable_audio_3.py:341`; `sampling.py:489` `torch.randn`, then re-noise at `t_next` | **Yes, the only user-selectable one found** |
| FlowMatch Euler with `stochastic_sampling` in `scheduler_config.json` | Possible by editing a local checkpoint config | diffusers v0.40.0 `scheduling_flow_match_euler_discrete.py:506-509` | Inferred from code, not run |

The DMD2 and FlowMatch paths share one transition:

```python
# diffusers v0.40.0 scheduling_flow_match_euler_discrete.py:506-509
# the vLLM-Omni copy matches at vllm_omni/diffusion/models/schedulers/scheduling_flow_match_euler_discrete.py:513-516
if self.config.stochastic_sampling:
    x0 = sample - current_sigma * model_output
    noise = randn_tensor(sample.shape, generator=generator, device=sample.device, dtype=sample.dtype)
    prev_sample = (1.0 - next_sigma) * x0 + next_sigma * noise
```

By my reading this is a Gaussian with mean `(1 - σ_next)·x̂0` and std `σ_next`, so a log-prob is well defined. Its std is 0 at a final `σ_next = 0`, where the log-prob is undefined. It is a full re-noising step, **not** the Flow-GRPO SDE, where noise scales with `noise_level` and the mean has a drift correction. See `verl_omni/pipelines/schedulers/flow_match_sde.py` in verl-omni and [arXiv 2505.05470](https://arxiv.org/abs/2505.05470).

### Lookalikes that do not count

- **Stable Audio 3 `dpmpp-3m-sde` (the default):** a grep of `sampling.py` for `randn` and noise assignment hits only line 489, inside `pingpong`. The name alone says SDE; I found no noise draw in it.
- **MiniMax H3 `MiniMaxH3EulerAncestralEta0SchedulerAdapter`:** eta=0, and RFC #8182 says it is "called from nowhere".
- **Hy-Nomas:** rejects `stochastic_sampling` (`hy_nomas/pipeline_hy_nomas.py:258`).
- **HunyuanImage3:** sets `stochastic_sampling=False` (`hunyuan_image3/pipeline_hunyuan_image3.py:527`).
- **Helios `step_dmd`:** re-noises from a passed `dmd_noisy_tensor` (`helios/scheduling_helios.py:725`). I did not check whether that tensor is fresh per step, so it is not counted.

## Q3. Per-step mean, sigma and noise

### What a user can get

| Item | Stock status | Evidence |
| --- | --- | --- |
| Per-step latents and timesteps | **BAGEL only**, with `return_trajectory_latents=True` | `inputs/data.py:326`; a grep finds the flag used only in `bagel/` and `inputs/data.py` |
| Per-step log-prob | **BAGEL only**, and only if a subclass supplies a scheduler whose output has `.log_prob`. `self.scheduler` defaults to `None` | `bagel/pipeline_bagel.py:388,1422,1431`; `bagel_transformer.py:1955-1956` |
| Log-prob output transport | Present | `diffusion/data.py:1890`; `diffusion/ipc.py:477-488`; `output_formatter.py:309,394`; `outputs/__init__.py:202` |
| Predicted mean | **No field exists** | Payload has only `latents`, `timesteps`, `log_probs`, `decoded` (`outputs/output_metadata.py:47-52`) |
| Step variance or sigma | **No field exists** | Same |
| Sampled noise | **No field exists**; derivable as `(x_next - mean) / std` if mean and std are captured | Same; derivation is mine |

The payload is explicitly intended for this use:

```python
# vllm_omni/outputs/output_metadata.py:47-52
class DiffusionTrajectoryPayload(TypedDict, total=False):
    # Optional denoising trajectory payload for RL/debug consumers.
    latents: torch.Tensor | dict[str, Any] | None
    timesteps: torch.Tensor | dict[str, Any] | None
    log_probs: torch.Tensor | dict[str, Any] | None
    decoded: list[PIL.Image.Image] | None
```

A custom scheduler can compute all of these. verl-omni's returns `prev_sample, log_prob, prev_sample_mean, std_dev_t` (`flow_match_sde.py:142`). The stock engine path keeps only element 0.

### Hooks and RL-oriented APIs

| Mechanism | Finding | Evidence |
| --- | --- | --- |
| `custom_pipeline_args.pipeline_class` | **The usable hook today.** Documented, with a CLI flag | `docs/features/custom_pipeline.md:25,55,95`; `entrypoints/cli/serve.py:752`; `diffusion/data.py:1050` |
| Step execution protocol | Splits a request into `prepare_encode`, `denoise_step`, `step_scheduler`, `post_decode`. A subclass sees `noise_pred` at each step | `diffusion/models/interface.py:88-122`; `docs/user_guide/diffusion/execution_modes.md:71-126`. Capability-based, not universal |
| `callback_on_step_end` | Called in some loops (FLUX.2, FLUX.2-klein, FLUX Kontext, Z-Image, ERNIE-Image, HunyuanImage3), but a grep finds no reference in `inputs/`, `entrypoints/`, `diffusion/data.py` or `diffusion/worker/`, so it is not reachable from a request | e.g. `flux2/pipeline_flux2.py:1144-1148`. Runs after the step, so `noise_pred` is available only through `locals()` |
| `ModelHook` | Intercepts module forwards, so it could see velocity but not the scheduler's noise. Used for caching and parallelism | `diffusion/hooks/base.py` |
| Stable Audio 3 sampler `callback` | Sampler accepts one, but the pipeline call passes none | `sampling.py:486`; `pipeline_stable_audio_3.py:456-462` |
| Weight transfer | RL-oriented and real. Live policy updates into the rollout engine, validated for unquantized Qwen-Image over `ipc` | `docs/user_guide/weight_transfer.md`; `docs/design/feature/weight_transfer.md` |
| Initial noise | `latents` param exists, but RFC #8182 says only HunyuanVideo-1.5 honors it (at the RFC's older commit) | `inputs/data.py:258` |

### RFCs and PRs

| Item | State | Relevance |
| --- | --- | --- |
| [RFC #8182](https://github.com/vllm-project/vllm-omni/issues/8182) | Open | **Best single summary.** Lists the four things RL rollouts must intercept. Item 3 is shared log-prob capture |
| [RFC #6269](https://github.com/vllm-project/vllm-omni/issues/6269), [PR #6277](https://github.com/vllm-project/vllm-omni/pull/6277) | Open | Scheduler registry, `scheduler` and `scheduler_kwargs` config |
| [PR #7907](https://github.com/vllm-project/vllm-omni/pull/7907) | Open | Lance replay for FlowGRPO, with rollout/replay log-prob numbers |
| [PR #9535](https://github.com/vllm-project/vllm-omni/pull/9535) | Open | HunyuanImage3 RL rollout, adds an `adding_rl_rollout_model.md` guide |
| [PR #2483](https://github.com/vllm-project/vllm-omni/pull/2483) | Merged 2026-04-08 | BAGEL trajectory recording for FlowGRPO |
| [PR #7551](https://github.com/vllm-project/vllm-omni/pull/7551) | Closed, not merged | Would have threaded `generator` into Qwen-Image `scheduler.step` |
| [#5512](https://github.com/vllm-project/vllm-omni/issues/5512), [#3435](https://github.com/vllm-project/vllm-omni/issues/3435), [#4864](https://github.com/vllm-project/vllm-omni/issues/4864) | Open | Batch invariance and deterministic rollout. Directly affect measurement design |
| [#778](https://github.com/vllm-project/vllm-omni/issues/778), [#3747](https://github.com/vllm-project/vllm-omni/issues/3747) | Open | General RL and world-model RFCs. Titles only, bodies not read |

## Q4. Justification by model route

| Route | Class | Reason |
| --- | --- | --- |
| Any stock model, no code | **NO** | No log-prob, mean, sigma or noise output, and no SDE flag on the main models |
| BAGEL plus a small subclass | **SMALL PATCH** | Loop, stacking and transport exist; supply `self.scheduler`. Only `log_prob` is forwarded, so mean and std need recording inside the scheduler |
| Qwen-Image or SD3.5 by subclass | **SMALL PATCH, no engine change** | UniRL's `RLQwenImagePipeline` (111 lines) and SD3 equivalent (119 lines) swap the scheduler at init and drain its recording into `trajectory_*`. The scheduler records log-probs itself, so `[0]` in `scheduler_step` does not block it |
| Upstream shared capture | **SMALL PATCH** | `CFGParallelMixin.scheduler_step`, per RFC #8182 item 3. An open comment there notes the capture owner differs under CFG-parallel and pipeline-parallel, which is the part that could exceed the estimate |
| MiniMax H3 | **More than small** | RFC #8182: the step is inline in two places, and UniRL carries "a ~130-line copy of the loop body" |

verl-omni's Qwen-Image adapter is 997 lines, but it also handles tokens, LoRA, SDE windows and more, so it overstates the minimum. UniRL's is the better size reference. Its subclass also sets `supports_request_batch = False`, so it does not exercise batched rollouts.

**For this paper, use a subclass rather than an engine patch.** It keeps the engine unmodified, which is itself what a study of the rollout engine's numerics should want.

## Measurement traps found along the way

- **Precision is part of the setup.** verl-omni computes the rollout scheduler step in fp32 while the transformer can run bf16 (`vllm_omni_rollout_adapter.py:567-584`). So a measured gap depends on where the casts sit, and comparing against a trainer needs both stacks recorded.
- **Batch composition changes outputs.** A comment on #5512 reports co-batched tiny Qwen-Image output differing from solo output in about 40 percent of elements. That is one contributor's report, not verified here.
- **Fresh-process determinism is unresolved.** See the Helios evidence on #4864 and the 1e-2 end-to-end difference without `cudnn.deterministic` reported on #5512.
- **Attention backend matters.** The verl-omni doc says values orders of magnitude above 1e-5 indicate "a schedule or backend mismatch".

## What remains unverified

- **Nothing was executed.** Every claim comes from reading source, docs and issues. No log-prob was computed.
- **Size estimate:** the < ~100 line hook is my estimate, not a built patch.
- **Framework compatibility:** verl-omni and UniRL adapters are pinned to older vLLM-Omni versions and may not run at `04ddf21`.
- **The 1e-5 claim:** it is a doc statement with no linked data, and I did not reproduce it. PR #7907's numbers are the author's own and partly superseded.
- **Editing `scheduler_config.json`:** that `stochastic_sampling: true` reaches stock Qwen-Image is inferred from `from_pretrained` plus the diffusers code. Noise would be unseeded in that path, since `pipeline_qwen_image.py:985-991` and `qwen_image/cfg_parallel.py:124` pass no `generator`.
- **Generator threading elsewhere:** a multi-line grep across pipelines' `scheduler_step_maybe_with_cfg` calls found `generator=` only in Cosmos3. I did not audit pipelines that call `scheduler.step` directly (HunyuanImage3 does pass one).
- **DMD2 checkpoints:** I did not check whether any published checkpoint actually sets `solver: "sde"`. The tests do (`tests/diffusion/models/dmd2/test_dmd2_scheduler.py:100-120`).
- **Readers of `eta`:** see Q2.
- **GitHub search:** the `return_trajectory_latents` issue search hit the unauthenticated rate limit and was not retried. Other searches returned capped pages, so relevant threads may be missing. Comment threads were read only up to the first 6 comments, except #8182.
- **Slack:** not searchable from public sources, so any discussion there is unseen.
- **Prior art:** one web search only. A real literature check is still needed.

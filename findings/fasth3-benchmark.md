# FastH3 vs Turbo vs 20-step reference on MiniMax H3

Measured 26-27 September 2026. Author: Carmen Lok.

**Summary.** On MiniMax H3's image-conditioned path, the FastVideo FastH3 4-step
`dense-datafree` adapter is **1.65x faster than the H3 Turbo 8-step LoRA and closer
to a 20-step reference** (DINOv2 similarity 0.9907 vs 0.9603). It is also far more
stable over a long single clip. Generation cost is linear in frame count on every
GPU we tried. Even so, the best time to first frame we could reach with any
bidirectional configuration is **~4.7 s**, against a sub-second target, which is
why the research moves to causal models.

---

## 1. Setup

| | |
|---|---|
| Base model | MiniMax H3, `minimax_h3_fl2va_pruned_fp8_scaled` (fp8, first/last-frame-to-audio-video path) |
| Text encoder / VAE | Qwen3-VL-32B int8 text encoder for H3; H3 video VAE fp16 |
| Runtime | ComfyUI, `MiniMaxH3ImageToVideo` node, `SamplerCustomAdvanced`, `BasicGuider` (no CFG) |
| Conditioning | `first_frame` = one reference image of a corgi test subject, background keyed to black, centred on a 512x512 canvas |
| Default sampler | `res_multistep` / `simple` scheduler |
| Default clip | 512x512, 56 frames (2.33 s at 24 fps) |
| GPUs | NVIDIA DGX Spark (GB10); rented RTX PRO 6000 Blackwell Workstation Edition (96 GB); rented 2x RTX PRO 6000 Blackwell Max-Q |

### The three configurations

Same base checkpoint in every case. Only the adapter and step count differ.

| Name | Adapter | Steps | Mechanism |
|---|---|---|---|
| gold | none | 20 | Plain diffusion. A reference point, not a different model. |
| turbo | `minimax_h3_fl2v_turbo_8step` LoRA | 8 | Step-distillation LoRA trained for 8 steps. |
| fasth3 | FastVideo FastH3 4-step Preview, `dense-datafree` variant | 4 | DMD2 distillation (frozen teacher plus learned critic), trained without video data. |

`dense-datafree` is the variant without VSA sparse attention, so it needs no
`fastvideo-kernel` or `flash-attn-4` build and loads with a stock
`LoraLoaderModelOnly` (its keys are in ComfyUI's native `.diff` / `.diff_b` patch
format). Its model card says the FL2VA and Ref2VA paths were **not** distilled
(text-to-video only). Empirically it works well on the image-conditioned path
anyway (sections 3 and 4).

---

## 2. Two different similarity numbers

This write-up uses two scores. Do not mix them.

- **Fidelity** = DINOv2-base CLS-token cosine similarity between a configuration's
  output and the 20-step gold output for the same prompt and seed, averaged over 5
  sampled frames. It answers "how close to the slow reference is the fast
  version?"
- **Identity** = DINOv2-base CLS cosine between each output frame and the original
  **reference image**, reported as mean / worst frame. It answers "does it still
  look like the reference?", with the large caveats in
  [identity-drift.md](identity-drift.md).

---

## 3. Speed and fidelity (idle prompt)

RTX PRO 6000 Workstation, 512x512, 56 frames, identical prompt and seed. `exec` is
ComfyUI's own execution time (see section 9).

| Config | exec | Fidelity vs gold |
|---|---|---|
| gold, 20 steps | 18.86 s | 1.0000 |
| **fasth3, 4 steps** | **5.42 s** | **0.9907** |
| fasth3, 8 steps | 8.89 s | 0.9872 |
| turbo, 6 steps | 7.21 s | 0.9648 |
| turbo, 4 steps | 5.46 s | 0.9634 |
| turbo, 8 steps (previous default) | 8.93 s | 0.9603 |

- FastH3 at 4 steps beats Turbo at every step count, in roughly half the time of
  Turbo at 8 (8.93 / 5.42 = 1.65x).
- Running FastH3 at 8 steps gains nothing; it is distilled for 4.
- Checked by eye, not just by metric: Turbo renders the subject with its mouth
  closed, while FastH3 reproduces gold's open-mouth expression. It follows the
  reference trajectory rather than producing a blander frame.

**Distillation beats truncation.** Four properly distilled steps beat eight
truncated ones on both speed and fidelity.

---

## 4. Motion prompts

Eight motion cues (sit, walk, speak, rollover, playdead, bow, spin, wave), the
prompt wording used in the testbed, seed 4242, 56 frames. **Identity** score
(mean / worst frame) against the reference image.

| Cue | FastH3 4-step | Turbo 8-step |
|---|---|---|
| sit | **0.952 / 0.932** | 0.932 / 0.894 |
| walk | **0.962 / 0.946** | 0.948 / 0.920 |
| speak | **0.951 / 0.902** | 0.940 / 0.847 |
| rollover | **0.833 / 0.723** | 0.788 / 0.582 |
| playdead | 0.879 / 0.815 | 0.878 / 0.812 |
| bow | 0.904 / 0.796 | 0.910 / 0.812 |
| spin | 0.814 / 0.617 | 0.803 / 0.682 |
| wave | 0.877 / 0.815 | **0.926 / 0.886** |

- FastH3 wins 4, ties 3, loses 1. Image conditioning survives real motion even
  though the adapter was not distilled on this path.
- `wave` is the one real regression; not yet reviewed by eye.
- `spin` scores worst for both. That is the metric, not the generator: the dog
  faces away mid-turn and a photographic embedding cannot separate a pose change
  from an identity change.
- Speed on motion prompts: 6.0-6.5 s (FastH3) vs 10.8-11.4 s (Turbo), a steady
  1.75x.

---

## 5. Cost model per GPU

At a fixed step count and resolution, wall time is almost exactly linear in
frame count:

| GPU | Fit | Measured with |
|---|---|---|
| RTX PRO 6000 Blackwell Workstation | `elapsed = 1.27 s + 0.166 s x frames` | ComfyUI execution timestamps |
| DGX Spark (GB10) | `elapsed = 3.00 s + 0.464 s x frames` | wall clock, 1 s poll (see section 9) |

Both fits are Turbo 8-step at 512x512. The DGX Spark fit comes from a warm length
sweep:

| Frames | 22 | 39 | 56 | 90 | 124 |
|---|---|---|---|---|---|
| DGX Spark elapsed (s) | 14.05 | 21.04 | 28.06 | 44.11 | 61.21 |

Consequences:

- **Fixed overhead is small** (1.3 s on the PRO 6000, 3 s on DGX Spark), so longer
  clips are slightly *more* efficient per second of video. This corrected our
  earlier assumption that attention would make cost quadratic in clip length;
  at these lengths the linear terms dominate.
- **Resolution is linear in pixels too**: 768x768 took 2.3x the time of 512x512
  for 2.25x the pixels (DGX Spark, 56 frames, 8 steps).
- **DGX Spark vs PRO 6000 is 2.8x on marginal cost**, not the ~6.5x a memory
  bandwidth ratio (~273 GB/s vs ~1790 GB/s) predicted.

### PRO 6000 Workstation vs DGX Spark, same graphs

Turbo LoRA unless stated; wall-clock elapsed.

| Config | DGX Spark GB10 | PRO 6000 WS | Speedup |
|---|---|---|---|
| 512², 56f, 4 steps | 16.04 s | 7.01 s | 2.29x |
| 512², 56f, 8 steps | 28.06 s | 9.16 s | 3.06x |
| 512², 56f, 20 steps, no LoRA | 56.11 s | 20.02 s | 2.80x |
| 512², 124f, 8 steps | 61.21 s | 24.16 s | 2.53x |
| 512², 158f, 8 steps | 83.16 s | 28.26 s | 2.94x |
| 768², 56f, 8 steps | 65.12 s | 23.02 s | 2.83x |
| 768², 124f, 8 steps | 170.34 s | 56.04 s | 3.04x |

(These are wall-clock numbers, so they sit slightly above the `exec` times in
section 3 for the same configuration, e.g. 9.16 s vs 8.93 s for Turbo 8-step.)

### RTX PRO 6000 Max-Q (2 cards)

FastH3 4-step, 512x512, 56 frames. No linear fit was made on this card.

| Measurement | Value |
|---|---|
| Fastest single clip | 7.00 s exec |
| Typical clip in a sequential 8-clip batch on one card | 8.2-9.8 s exec; batch 71.03 s wall = 8.88 s/clip |
| Same 8 clips split across 2 cards | 35.70 s wall = 4.46 s/clip, **1.99x** |
| 20-clip batch across 2 cards | 87.5 s wall = 4.38 s/clip |
| First (cold) run per card | 28-42 s, excluded |

Caveat: the Max-Q runs used a different reference image (a 2D anime-style
character) and motion prompts, while the 5.42 s Workstation figure is the corgi
idle prompt. The like-for-like comparison is Workstation motion prompts (6.0-6.5 s)
vs Max-Q 8.2-9.8 s, which is a larger gap than the 29% one gets from comparing the
single fastest Max-Q run to the Workstation idle run, and larger than the 12-15%
in NVIDIA's published specs. We have not isolated why (power limit and thermals
are the obvious suspects).

**Two cards scale throughput, not latency.** ComfyUI has no tensor parallelism,
the cards talk over PCIe rather than NVLink, and FastH3 needs ~53 GB, which fits
on one card. Two cards give two independent pipelines: 1.99x clips per minute,
no change in time to first frame.

---

## 6. Base-model step count and an unreproduced figure

20 steps, no LoRA, 124 frames (5.17 s of video), PRO 6000 Workstation, warm:

| Resolution | Time |
|---|---|
| 480x480 | 39.03 s |
| 832x480 | 71.06 s |
| 832x480, with Turbo LoRA (at 20 steps) | 77.06 s |

An external figure of ~15 s for a 5 s, 480p, 20-step H3 clip on the same GPU class
did not reproduce: the most generous reading (480x480) is 2.6x slower, and
widescreen 480p 4.7x slower. Working backwards from the cost model, 15 s for ~5 s
of video corresponds to roughly 4-8 steps, not 20. Every mechanism we could think
of to explain the gap on our side (audio generation, fp8 flags, sigma shift,
hidden upscaling; section 8) was tested and ruled out. The likely explanation is a
different configuration (frame count, steps, checkpoint or pipeline), which is the
general lesson: **latency claims are meaningless without frames, steps,
resolution and pipeline attached.**

For context, earlier runs on a different serving stack (SGLang, with DiT, VAE and
text-encoder CPU offload enabled and audio generation on) took 269 s at 50 steps
and 71.6 s at 12 steps for a ~4.5 s 864x768 clip. The Turbo LoRA failed to load in
three configurations on that stack but loaded and ran without issue in ComfyUI on
smaller hardware. **The serving stack decides what loads, not the hardware.**

---

## 7. Chunking vs one long clip

A "chunk chain" generates N short clips, conditioning each chunk's first frame on
the last decoded frame of the previous one. It is the cheapest way to get earlier
first frames from a bidirectional model, and a crude stand-in for autoregression.

PRO 6000 Workstation, 4 x 39-frame chunks vs one 158-frame clip:

| Run | Drift (identity drop) | Time to first frame | Total exec |
|---|---|---|---|
| fasth3, 4 x 39f, same prompt | 0.100 | 4.82 s | 18.70 s |
| fasth3, 4 x 39f, switching prompts | 0.109 | 4.67 s | 19.19 s |
| turbo, 4 x 39f, same prompt | 0.106 | 10.00 s | 31.02 s |
| turbo, 4 x 39f, switching prompts | 0.104 | 7.29 s | 28.34 s |
| **fasth3, one 158f clip** | **0.010** | 19.64 s | 19.64 s |
| turbo, one 158f clip | 0.071 | 30.20 s | 30.20 s |

- **FastH3 does not fix chunk drift.** It stays near 0.10 with either adapter. The
  boundary (a VAE decode/encode round-trip plus loss of temporal context) is the
  cause, not the sampler.
- **FastH3 makes a single long clip nearly drift-free**: 0.010 over 158 frames,
  7x better than Turbo.
- At equal wall clock (19.6 s vs 18.7 s) the single clip has 10x less drift.

With the Turbo adapter, the latency win of chunking was 3.95x time-to-first-frame
for 1.2% more total compute on DGX Spark, and 3.53x for 13% more on the PRO 6000 (the
DGX Spark single-clip run was unusually slow, which flattered it). Call it **~3.5x
earlier first frame for 1-13% more compute, paid for with ~10x the drift.**

### Time to first frame, best achievable

| Path | TTFF |
|---|---|
| FastH3, 39-frame chunk | **4.7 s** |
| FastH3, 56-frame clip | 5.4 s |
| Turbo, 39-frame chunk | 7.3 s |
| Target | < 1 s |

FastH3 is a 1.65x improvement against a ~5x gap. No combination of resolution,
steps, chunk size or GPU in this class closes it.

---

## 8. Settings measured to make no difference

Recorded so they are not re-tested.

| Hypothesis | Result |
|---|---|
| Audio generation is expensive; generate video only | **~1% of tokens** (audio latent `[1,32,2,207]` vs video `37x30x30`). No video-only node exists; all H3 nodes build a joint audio-video latent. Not worth patching. |
| `--fast fp8_matrix_mult` unlocks Blackwell fp8 | **No effect.** 8.67 s baseline vs 8.84 s with the flag vs 8.91 s with all four `--fast` flags, 3 reps each; between-rep spread (0.30 s) exceeds the difference. The checkpoint carries its own quantisation metadata and already uses mixed-precision ops. |
| Sigma shift was missing from the graph | **No effect.** `MiniMaxH3SigmaShift` defaults to `shift_video=12.0`, matching the model default. Shift 6 / 12 / 20 differ by 0.0004 similarity; 20-step with and without the node are identical (1.0000). |
| The node silently upscales to a 768 short edge | **Not applied** on the image-to-video path. 480x480 renders at 480x480. |
| Alternative samplers | `euler` 30% slower for equal quality; `dpmpp_2m` matches `res_multistep`. |
| 8 steps instead of 4 with FastH3 | Slower and slightly *lower* fidelity (0.9872 vs 0.9907). |
| 768² instead of 512² | Cost scales with pixels (2.3x); little visible gain for a single subject on black. |
| A second identical GPU for latency | 1.99x throughput, zero latency gain (section 5). |

One open question rather than a negative result: the H3 node's `length` sits on a
grid of 17k+5 frames, and its tooltip gives the trained range as roughly 124-362
frames. Our 56-frame default is below that range, which may cost quality
independently of everything above. Not yet tested.

### Small-model probe

FastWan-QAD-1.3B (published: a 5 s 480p clip in 1.78 s on an RTX 5090) could not
be run: it is text-to-video only (no identity conditioning) and its
`fastvideo-kernel` wheel does not build on sm_120. As a proxy, stock
Wan2.1-T2V-1.3B through plain diffusers in bf16 on one Max-Q, 81 frames at 832x480:

| Steps | Time (2 runs) |
|---|---|
| 50 | 163.96 s / 166.31 s |
| 20 | 69.18 s / 69.40 s |
| 4 (extrapolated) | ~14 s |

About 3.1-3.4 s per step. A 1.3B model on an unoptimised stack is no faster per
second of video than a ~33B model on an optimised one. We attribute the ~8x gap
to the published FastWan number to its NVFP4 weights plus SageAttention3 (our
inference, not verified). **The inference stack matters more than parameter count**,
and a smaller model buys nothing if its kernels do not build on your GPU.

---

## 9. Measurement pitfalls

Each of these produced a confident wrong answer at least once.

1. **Polling quantises timings.** A harness that sleeps 1 s between polls of
   ComfyUI's `/history` endpoint rounds every run up to the next poll. Two
   different configurations both read exactly 20.09 s. The raw DGX Spark data in
   [`results/`](../results/) shows it too: every 39-frame chunk reads 21.04-21.05 s.
   Poll at ~0.15 s *and* read ComfyUI's own `execution_start` /
   `execution_success` timestamps from the history messages. The DGX Spark cost model
   above was fit on 1 s-polled wall clock and should be read as ±1 s per point.
2. **ComfyUI's execution cache.** Re-submitting an identical graph returns in
   ~1.00 s and looks like a dramatic speedup. Vary the seed between runs.
3. **Cold first run.** The first generation after loading takes 28-42 s on a
   card that then does the same clip in ~7-9 s. Always discard a warm-up run.
4. **Comparing across GPUs with different inputs.** Different reference images or
   prompts change per-clip time by up to ~40% on the same card. Compare like with
   like (section 5, Max-Q caveat).
5. **Download integrity.** A parallel range downloader reassembled parts with a
   shell glob, which sorts `p0, p1, p10, ..., p2`. The file had the correct size
   and scrambled bytes. Verify model downloads by SHA-256, never by size.
6. **Unattributed latency claims.** See section 6. Always record frames, steps,
   resolution, checkpoint, pipeline and whether the time is exec or wall clock.

---

## 10. Limitations

- One subject per class (one corgi, one 2D anime-style character), one seed per
  configuration in most rows. Treat differences under ~0.01 similarity as noise.
- DINOv2 similarity is a proxy with known failure modes; see
  [identity-drift.md](identity-drift.md).
- No VSA (sparse attention) results: the kernels did not build on our GPUs.
- Workstation absolutes come from rented single-GPU instances; DGX Spark and Max-Q
  numbers come from different harness versions (section 9).

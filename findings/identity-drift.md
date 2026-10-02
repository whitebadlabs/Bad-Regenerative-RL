# Identity drift: what a photographic embedding does and does not measure

Measured 26-27 September 2026. Author: Carmen Lok.

**Summary.** We scored every generated frame against the reference image with
DINOv2-base CLS cosine similarity. A single bidirectional clip barely drifts
(0.007-0.010 over 158 frames). Chaining four short clips drifts 6-10x more, and
the loss compounds at chunk boundaries. The result reproduces exactly across two
GPUs. But on inspection, what drifted was mostly **framing and pose, not
identity**, and on a 2D anime-style character the same metric falls hardest on
pose and occlusion changes. A photographic embedding conflates identity, pose and
framing, and there is nothing better off the shelf for non-human subjects. That is
the verifier gap this repo is about.

---

## 1. Metric

- Model: `facebook/dinov2-base`, CLS token of the last hidden state, L2-normalised.
- Score: cosine similarity of each generated frame to the **reference image** (the
  single conditioning image, not frame 0 of the output).
- Reported per run: first frame, last frame, min, mean, and **drop** = first - last.
- Script: [`scripts/dino_drift.py`](../scripts/dino_drift.py).

This is a generic self-supervised image embedding. It was chosen because it
exists, runs anywhere, and has no face-specific assumptions. It was not designed
for identity.

## 2. Setup

- Generator: MiniMax H3 fp8, image-conditioned (`fl2va`) path, ComfyUI, 512x512,
  24 fps. Turbo 8-step LoRA unless stated; FastH3 4-step for the adapter
  comparison.
- Subject: a corgi test subject (photograph), background keyed to black.
- **Single clip**: one 124- or 158-frame generation from the reference.
- **Chunk chain**: 4 x 39-frame chunks (156 frames, 6.5 s). Chunk *k*+1 is
  conditioned on the last decoded frame of chunk *k*. Seeds increment per chunk.
  - *same prompt*: an idle prompt repeated four times.
  - *switching prompts*: idle -> sit -> wave a paw -> stand up and hop.
- Prompts ask explicitly for a locked camera, full body in frame, no zoom.

## 3. Results, DGX Spark (Turbo 8-step)

### Totals

| Run | Start | End | Total drop |
|---|---|---|---|
| Single clip, 124f | 0.9811 | 0.9732 | 0.008 |
| Single clip, 158f | 0.9811 | 0.9742 | 0.007 |
| 4 chunks, same prompt | 0.9823 | 0.9210 | **0.061** |
| 4 chunks, switching prompts | 0.9816 | 0.9083 | **0.073** |

"Total drop" for a chain is the first frame of chunk 0 minus the last frame of
chunk 3.

### Per chunk

| Chunk | Same: first / last / min | Same: within-chunk drop | Switch: first / last / min | Switch: within-chunk drop |
|---|---|---|---|---|
| 0 | 0.9823 / 0.9783 / 0.9656 | 0.0040 | 0.9816 / 0.9747 / 0.9686 | 0.0069 |
| 1 | 0.9740 / 0.9538 / 0.9362 | 0.0202 | 0.9687 / 0.9041 / 0.8976 | **0.0645** |
| 2 | 0.9478 / 0.9338 / 0.9287 | 0.0140 | 0.8881 / 0.8905 / 0.8437 | -0.0024 |
| 3 | 0.9264 / 0.9210 / 0.8988 | 0.0054 | 0.8770 / 0.9083 / 0.8475 | -0.0313 |

What this shows, stated carefully:

- **A single bidirectional clip barely drifts** over 6.6 s. Bidirectional attention
  holds the subject across its whole context.
- **Chunking drifts ~9x more** than a single clip of the same length *with the same
  prompt*. Chunking itself is the cause, not prompt switching.
- **Prompt switching adds roughly 20%** on top. The largest single-chunk loss
  (0.0645) lands on the first switch.
- **The compounding happens at boundaries.** Each chunk starts below where the
  previous one ended (boundary losses 0.004-0.016). *Within* a chunk the score is
  not monotone: in the switching chain, chunks 2 and 3 end *higher* than they
  start.
- Rate: ~0.0094 per second of video with a fixed prompt, ~0.0113 while switching.
  Naively extrapolated, 30 s of continuous chained animation would lose 0.28-0.34.
  (We would not trust that extrapolation; see section 5.)

## 4. It reproduces across hardware

Same graphs, prompts and seeds on an RTX PRO 6000 Blackwell Workstation:

| Run | DGX Spark total drop | PRO 6000 total drop |
|---|---|---|
| Single clip, 158f | 0.007 | 0.007 |
| 4 chunks, same prompt | 0.061 | 0.057 |
| 4 chunks, switching prompts | 0.073 | 0.074 |

The largest single-chunk loss again lands on the first prompt switch (0.0786 on
the PRO 6000, 0.0645 on DGX Spark). Drift is a property of the generation procedure,
not of the GPU, so drift experiments can run on cheap hardware.

### With the FastH3 adapter (PRO 6000)

| Run | Start | End | Drop |
|---|---|---|---|
| fasth3, 4 x 39f, same prompt | 0.9802 | 0.8803 | 0.100 |
| fasth3, 4 x 39f, switching | 0.9804 | 0.8714 | 0.109 |
| turbo, 4 x 39f, same prompt | 0.9779 | 0.8715 | 0.106 |
| turbo, 4 x 39f, switching | 0.9778 | 0.8735 | 0.104 |
| **fasth3, one 158f clip** | 0.9786 | 0.9688 | **0.010** |
| turbo, one 158f clip | 0.9793 | 0.9081 | 0.071 |

The adapter changes single-clip stability a lot (7x) and chain drift not at all.
(The Turbo chain drops here, ~0.10, are larger than in the cross-hardware table
above, ~0.06-0.07. These runs came from a later harness and the exact prompt and
seed differences were not recorded, so compare within a table, not across.)

## 5. The caveat that is the research question

Looking at the frames contradicts the naive reading of the numbers. After four
chunks the corgi is **plainly the same corgi**: correct colouring, correct
markings, same accessory. What changed is:

- **Framing**: the virtual camera zooms in, despite "no zoom" in every prompt.
- **Pose / viewpoint**: the subject rotates from side profile toward a frontal
  three-quarter view.
- **Cropping**: the bottom of the body leaves the frame.

So the metric detected real drift and then could not say *what* drifted. The same
signature shows up elsewhere:

- **`spin` on the corgi** is the worst-scoring motion cue for both adapters (worst
  frame 0.617-0.682) because the dog faces away mid-turn. Nothing about the dog
  changed.
- **A 2D anime-style character**, FastH3 4-step, 20 clips of 56 frames with
  everyday actions, scored against its reference image:

  | | Value |
  |---|---|
  | Mean of per-clip means | 0.813 |
  | Median clip | 0.834 |
  | Range of clip means | 0.638 - 0.888 |
  | Clips with mean >= 0.80 | 15 / 20 |
  | Mean of per-clip worst frames | 0.718 (lowest 0.410) |

  For comparison the corgi's motion cues sit at 0.81-0.96 and its idle clips at
  ~0.97. The lowest-scoring character clips are a bow, a full turn-around, and a
  hand-over-face gesture: pose and occlusion changes again, the same signature as
  `spin`. Whether identity was actually lost in those clips has not been judged
  systematically by eye.

Two conclusions follow.

1. **A photographic embedding conflates identity, pose and framing.** Used as a
   reward, it would push a generator toward holding the reference pose and camera
   still, which is the opposite of what a motion prompt asks for. Used as a
   re-anchoring trigger, it would fire on every turn and bow.
2. **"Non-human" is not one category.** The drawn character starts ~0.1-0.15 lower
   than the photographed dog on the same metric and the same generator. Either the
   embedding transfers poorly to non-photographic images, or drawn characters are
   genuinely harder to hold, or both. The current metric cannot tell these apart.

There is also a practical point for any application that pins subject size and
position (the desktop-pet testbed does): **framing drift is a visible failure on
its own**, independent of identity. It may deserve a separate score rather than
being folded into identity.

## 6. What we have not measured

- Any causal autoregressive model. Chunk chains are a proxy for compounding error,
  not a substitute.
- More than one subject per class, more than one seed per condition.
- Whether a re-anchoring step (re-conditioning on the original reference instead
  of the previous chunk's last frame) removes boundary drift.
- Human identity judgements to calibrate the metric against.
- Any embedding other than DINOv2-base.

These are the inputs to [../docs/verifier-design.md](../docs/verifier-design.md).

## Raw data

Per-run DGX Spark numbers, including the per-frame similarity series, are in
[`results/`](../results/). Generated frames and reference images are not
published.

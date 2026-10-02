# Related work

Scope: work we have actually read or run so far, and only the claims we took from
it. This is a working list, not a survey.

> **TODO: add proper citations** (authors, venue/arXiv id, year, links) for every
> entry before any of this is cited externally. Figures below are as reported by
> the respective model cards, project pages or repositories at the time we read
> them (September 2026) unless marked as our own measurement; re-check them
> against the sources when adding citations.

---

## Causal autoregressive / streaming video generation

### LongLive (NVIDIA)

Causal autoregressive video generation with a KV cache, so frames are emitted as
they are produced and the prompt can change mid-stream.

- The LongLive research page reports 20.7 FPS on one H100 (we did not note
  which model size or release this refers to).
- **LongLive 2.0 5B**, built on Wan2.2, supports text-to-video and
  image-to-video at 1280x720:

  | Variant | FPS on one H100 | VBench |
  |---|---|---|
  | LongLive-2.0-5B | 24.8 | 85.06 |
  | LongLive-2.0-5B-NVFP4-2Step | 45.7 | 83.14 |

  NVFP4 weights are 19.4 GB vs 35.4 GB in BF16.
- **Prompt switching uses KV-recache** (our understanding, to verify: on a prompt
  change the cached keys and values are recomputed under the new prompt rather
  than restarting generation). As far as we have read, it is documented for *scene* consistency; whether it
  preserves a specific *subject's* identity across many switches from one
  reference image is not addressed. That is exactly our research question 3.
- Licensing as we read it: the GitHub code is Apache-2.0; the 2.0 weights on
  Hugging Face are under the NVIDIA Open Model License; the older LongLive-1.3B
  weights are CC-BY-NC-SA. Check the current terms before building on them.

Relevance: the most concrete open candidate for the causal generator in roadmap
step 2. Our case is easier than its benchmark (one subject on a plain background
at ~512 px, not a full 720p scene).

TODO: citation; confirm which FPS figure belongs to which release.

### Self-Forcing

Named in our notes as one of the distillation methods used to recover the quality
that causal (autoregressive) video models lose relative to bidirectional ones.
The general point we take from it: going causal costs temporal coherence, and
distillation from a bidirectional teacher is how current work claws it back.

TODO: read in full; citation; summarise the training procedure and how exposure
bias is handled.

### Causal Forcing

Named alongside Self-Forcing as a distillation approach for causal video
generation. Listed as a candidate recipe for turning Wan 2.2 into a causal base.

TODO: read in full; citation; summarise.

### KV-recache

See LongLive above. The mechanism a streaming approach depends on: a prompt
switch updates the cache rather than restarting generation. Our chunk-chain
experiments are a crude, cache-free analogue (each chunk restarts from the
previous chunk's last decoded frame), and they show the largest identity loss at
the first prompt switch ([findings/identity-drift.md](../findings/identity-drift.md)).

TODO: identify the original source of the term and any analysis of identity
preservation under recache.

---

## Few-step and fast inference

### FastH3 (FastVideo)

`FastVideo/FastVideo-FastH3-4-step-Preview-v1-*`: distils MiniMax H3 from 49
transformer forwards to 4 using data-free DMD2 (frozen teacher plus learned
critic), optionally paired with ~90% sparse video attention (VSA).

- Reported up to 14.38x end-to-end on a B200: a 15 s 1344x768 clip from 678.7 s to
  47.2 s. Most of that is the step reduction (49 -> 4); a user already running
  fewer steps realises much less.
- Model card: FL2VA and Ref2VA (the image-conditioned paths) were **not**
  distilled.
- **Our measurement:** the `dense-datafree` (non-VSA) variant works on H3's
  image-conditioned path anyway and beats the H3 Turbo 8-step LoRA on speed and
  fidelity ([findings/fasth3-benchmark.md](../findings/fasth3-benchmark.md)).
- The VSA variant needs `fastvideo-kernel` and `flash-attn-4` builds that we could
  not get to build on our GPUs (sm_120 / aarch64).

TODO: citation for FastVideo, DMD2 and VSA.

### FastWan-QAD-1.3B (FastVideo)

`FastVideo/FastWan-QAD-1.3B`: a distilled, low-precision 1.3B Wan-family model
(TODO: confirm what "QAD" stands for and the exact recipe).

- Reported: a 5 s 480p clip in 1.78 s on an RTX 5090, i.e. faster than realtime.
- Text-to-video only, so no identity conditioning from a reference image.
- Its stack uses NVFP4 and SageAttention3; `fastvideo-kernel` did not build on
  sm_120 for us.
- **Our measurement (proxy):** stock Wan2.1-T2V-1.3B via plain diffusers in bf16
  takes ~164 s at 50 steps for 81 frames at 832x480 on one RTX PRO 6000 Max-Q, so
  the published speed is almost entirely the inference stack, not the parameter
  count.

TODO: citation; method description.

### DMD2

Distribution-matching distillation, the method behind FastH3: a frozen teacher
plus a learned critic, here applied without video data.

TODO: citation; note how a DMD-style critic relates to a learned reward model.

---

## Base models

### MiniMax H3

~33B-parameter bidirectional audio-video diffusion model. All three ComfyUI H3
nodes build a joint audio-video latent; audio is ~1% of tokens. The
image-conditioned path (`fl2va`, first/last frame to audio-video) is what our
measurements use. A Turbo 8-step step-distillation LoRA is available for it. Weights
are under the MiniMax H3 Community License.

TODO: citation / model card link; licence summary.

### Wan 2.1 / Wan 2.2

Open video diffusion models. Wan2.1-T2V-1.3B is our small-model proxy. Wan 2.2 is
the base of LongLive 2.0 and our candidate base for Self-Forcing / Causal Forcing
distillation. The publisher's Wan2.2-TI2V-5B documentation reports a 5 s 720p clip
in under nine minutes on a consumer GPU without specific optimisation.

TODO: citations.

### LTX

Named only as another bidirectional video model with the same structural problem
(time to first frame equals time to whole clip). Not measured.

TODO: citation.

---

## Identity and similarity metrics

### DINOv2

Self-supervised vision transformer. We use the `dinov2-base` CLS token, cosine
similarity, as our only identity/fidelity metric so far, and document how it
conflates identity with pose and framing
([findings/identity-drift.md](../findings/identity-drift.md)).

TODO: citation.

### Face-embedding identity metrics

The standard identity measure for human subjects in video-generation work. Not
applicable to animals or drawn characters, which is the gap this repo addresses.

TODO: cite the face-ID metrics used in recent identity-preserving video papers
and any existing animal re-identification or character re-identification work;
this is the most important missing section.

---

## To read next

- RL / reward fine-tuning for video diffusion (reward-weighted regression,
  preference optimisation, direct reward backprop), with attention to long-horizon
  and per-block rewards.
- Animal re-identification literature (individual-level ID for pets and wildlife).
- Character / anime re-identification and style-robust embeddings.
- VBench and other video benchmarks' subject-consistency dimensions, and how they
  are computed.

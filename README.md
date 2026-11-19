# Bad-Regenerative-RL

Reinforcement learning and reward models for generative video, focused on
**verifiable identity rewards for non-human subjects** (pets, animals, cartoon and
anime-style characters) in **causal autoregressive / streaming** video generation,
with **few-step distilled inference** as the serving substrate.

**Goal:** achieve low-latency (~1 second) streaming video generation that preserves subject identity for non-human and non-photographic subjects

---

## The problem

Give a generative video model one reference image of a subject and a stream of
short instructions ("sit", "wave", "turn around"). Make it produce video that
starts within about a second of each instruction, can change instruction
mid-stream, and keeps looking like **the same subject** the whole time.

The first testbed is a realtime desktop pet: a spoken command becomes a short
video of the pet doing it. It is a convenient testbed because it is small (one
subject on a plain background, ~512 px), has a hard latency target, and makes
identity failure obvious to anyone watching. It is not the end goal; the end goal
is a general, checkable notion of "this is still the same subject" that can be
used as a reward.

### Why bidirectional video models cannot stream

Most strong open video diffusion models (MiniMax H3, LTX, Wan in its standard
form) are **bidirectional**: every frame attends to every other frame and the
whole clip is denoised together. Frame 1 does not exist until the last denoising
step of the whole clip finishes, so **time to first frame equals time to the
whole clip**, and the prompt cannot change mid-clip.

We measured how far tuning can push this (see
[findings/fasth3-benchmark.md](findings/fasth3-benchmark.md)). With the best
few-step adapter we found (FastH3, 4 steps) on an RTX PRO 6000 Blackwell, a
56-frame 512x512 clip takes 5.4 s and the shortest sensible chunk (39 frames)
4.7 s to first frame. The target is under 1 s. Resolution, CFG, sigma shift,
sampler, fp8 flags and more GPUs of the same class each moved the number
little or not at all. The remaining gap is a model-class problem, not a settings
problem.

### Why causal models drift

Causal autoregressive video models (e.g. LongLive) generate frames or small
blocks in order with a KV cache, so frames leave as they are produced and prompts
can switch mid-stream (LongLive uses KV-recache for this). Going causal costs
temporal coherence; recovering it is what distillation methods such as
Self-Forcing and Causal Forcing target. The deeper problem is that each block is
conditioned on the model's own earlier output, which already carries error.
Errors compound, and the subject drifts. Bidirectional attention was what held
the subject together.

We can already see this compounding without a causal model: chaining short
bidirectional chunks, each conditioned on the last frame of the previous one, is
a crude stand-in for autoregression. A single 158-frame clip loses 0.007-0.010
of DINOv2 similarity to the reference; four chained 39-frame chunks covering the
same length lose 0.06-0.11 (see [findings/identity-drift.md](findings/identity-drift.md)).

### The non-human identity verifier gap

The obvious fix is a reward: score each generated block against the reference
and either **re-anchor** at inference time when the score drops, or **train**
against the score with RL. That needs a verifier that answers "is this the same
subject?"

For people this exists: face-embedding models are the standard identity metric
in the literature. For a corgi, a red panda or a drawn character there is no
equivalent off the shelf. The general-purpose photographic embedding we used
(DINOv2 CLS cosine) does detect drift, but on inspection it **conflates identity,
pose and framing**: a dog turning away, a camera zoom or a cropped body all lower
the score as much as an actual change of subject would. On a 2D anime-style
character the same metric sits lower (mean 0.81 over 20 clips, vs 0.81-0.96 across
the corgi's eight motion cues and ~0.97 for its idle clips)
and its worst scores land on pose and occlusion changes, not identity loss.

A reward built on that metric would teach a model to hold the camera and the
pose still, not to preserve the subject. That is the gap this repo is about.

---

## Research questions

1. **Verifier.** Can we build an identity score for non-human and
   non-photographic subjects that is invariant to pose, viewpoint, framing and
   expression but sensitive to changes in the subject itself (markings, colour,
   proportions, species, costume)? How do we validate it without a face-ID-style
   labelled dataset?
2. **Generality.** Does one verifier hold across a photographed animal, a
   different species and a drawn character, or is "non-human" several problems?
3. **Re-anchoring.** In a causal model, does a verifier-triggered intervention
   (re-injecting the reference, recaching, or rolling back a block) hold identity
   over long streams and repeated prompt switches, and at what latency cost?
4. **RL.** Used as a reward for fine-tuning a causal or few-step generator, does
   the verifier improve long-horizon identity without collapsing motion (the
   "hold still and score well" failure)? What does reward hacking look like here?
5. **Framing as a separate signal.** For applications that pin subject scale and
   position, is framing stability a reward of its own, and should it be scored
   separately from identity?
6. **Serving substrate.** How much of the realtime budget can few-step distilled
   models (FastH3-style DMD2 adapters, FastWan-style quantised small models)
   recover, and what does it cost per streamed second on a given GPU?

---

## What is in this repo

| Path | What |
|---|---|
| [findings/fasth3-benchmark.md](findings/fasth3-benchmark.md) | FastH3 4-step vs Turbo 8-step vs 20-step reference on MiniMax H3: speed, fidelity, per-frame cost models per GPU, settings measured to make no difference, measurement pitfalls. |
| [findings/identity-drift.md](findings/identity-drift.md) | DINOv2 identity scores, single clip vs chunk chains, cross-hardware reproduction, and why the photographic embedding conflates identity with pose and framing. |
| [docs/verifier-design.md](docs/verifier-design.md) | **Proposal.** Candidate designs for a non-human identity verifier and how it plugs into re-anchoring and RL. Nothing in it is measured yet. |
| [docs/related-work.md](docs/related-work.md) | Causal AR video and fast-inference work referenced so far, limited to what we have actually read. Citations still TODO. |
| [scripts/](scripts/) | Cleaned benchmark harness: ComfyUI runner, chunk chainer, DINOv2 drift scorer, cost-model fit. |
| [results/](results/) | Raw per-run numbers from the DGX Spark and RTX PRO 6000 Max-Q runs (no media). |

No generated media, reference images or weights are committed; see `.gitignore`.

---

## Roadmap

Ordered; each step's output decides whether the next is worth doing.

**0. Baseline (done, September 2026).** Few-step bidirectional generation measured
on DGX Spark (GB10), RTX PRO 6000 Workstation and 2x RTX PRO 6000 Max-Q.
Single-clip vs chunked drift measured with DINOv2 and reproduced across hardware.
Conclusion: bidirectional generation is ~5x off the sub-second target; streaming
needs a causal model; the identity metric is the weak link.

**1. Verifier v0 and an evaluation set.**
- Build a small, licence-clean evaluation set: several non-human subjects
  (photographed animals of more than one species, original drawn characters),
  each with clips that vary pose, viewpoint and framing *without* changing
  identity, plus deliberate identity perturbations (recolour, swapped markings,
  different individual of the same breed, different character in the same style).
- Score DINOv2, CLIP-style and other off-the-shelf embeddings against it to
  quantify the pose/framing confound we observed.
- Try the candidate designs in [docs/verifier-design.md](docs/verifier-design.md),
  starting with the cheapest (multi-view anchor sets, pose/framing-factored
  scoring).

**2. Causal model in the loop.**
- Stand up an open causal AR video model (LongLive-class) on a single
  workstation GPU, at ~512 px with one subject.
- Measure time to first frame, sustained FPS, and identity over long streams
  with repeated prompt switches, using the verifier from step 1.
- Licence check on the weights before building on them.

**3. Verifier-triggered re-anchoring (inference only).** Threshold the verifier
per block; on a drop, re-inject the reference or recache. Measure identity
held vs latency added.

**4. Verifier as an RL reward.** Fine-tune a causal or few-step generator
against the verifier (with motion / prompt-adherence terms to prevent
"stand still" hacking). Compare to re-anchoring alone.

**5. Generalise.** Repeat across subject classes; report where one verifier is
enough and where it is not.

---

## Reproducing the existing numbers

The benchmark scripts drive a local ComfyUI instance running MiniMax H3 (fp8
`fl2va` checkpoint) with either the H3 Turbo 8-step LoRA or the FastVideo FastH3
4-step `dense-datafree` adapter. See [scripts/README.md](scripts/README.md). You
need your own reference image; ours is not published.

## Licence

Code and text: Apache-2.0, see [LICENSE](LICENSE). Model weights referenced here
are under their own licences, which you must check separately.

Data and results in this repository: CC-BY-4.0 (reuse freely, with attribution).

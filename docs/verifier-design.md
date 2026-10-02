# Non-human identity verifier: candidate designs

> **PROPOSAL. Nothing in this document has been built or measured.** It lays out
> options and how we intend to test them. Measured results live in
> [`findings/`](../findings/).

Author: Carmen Lok · draft, October 2026

---

## 1. What the verifier has to do

Input: a reference (one image, or a small set) of a subject, and a generated
frame or short block of frames. Output: a score that is

- **high** when the block shows the same subject, in *any* pose, viewpoint,
  expression, framing or lighting the prompt asked for;
- **low** when the subject itself has changed: different colouring or markings,
  proportions, species or breed, a different individual of the same breed,
  a different character drawn in the same style, lost or swapped costume elements;
- **cheap** enough to run on every block of a stream (target: a few ms per frame
  on the same GPU as the generator) and **differentiable or at least
  low-variance** enough to use as an RL reward;
- **valid across subject classes**: photographed animals of several species and
  non-photographic drawn characters, from a single reference each.

The baseline we have, DINOv2-base CLS cosine, fails the first requirement: it
drops on turns, bows, zooms and crops as much as on identity change
([findings/identity-drift.md](../findings/identity-drift.md)). Face-embedding
models, the standard identity metric for people, do not apply.

## 2. Two separate scores, not one

Before choosing a model, separate what we want to reward:

| Score | Question | Why separate |
|---|---|---|
| **Identity** `s_id` | Is this the same subject? | The thing we actually want to preserve. Must be pose- and framing-invariant. |
| **Framing** `s_frame` | Is the subject at the expected scale and position, fully in frame? | Applications that pin subject size and placement treat framing drift as a failure on its own. Easy to measure directly (segmentation mask -> bbox, area, centroid). |
| **Prompt adherence / motion** `s_motion` | Did the requested action happen? | Without it, any identity reward is maximised by standing still in the reference pose. |

A reward is then a weighted combination, with the weights a per-application
choice. The rest of this document is mostly about `s_id`.

## 3. Candidate designs for `s_id`

Ordered roughly from cheapest to most ambitious.

### A. Multi-view anchor set (no training)

Score against a *set* of anchors instead of one reference: the reference plus
frames from the subject's own earlier, verified-good clips (or from a single
long, low-drift bidirectional clip, which we measured to hold at ~0.99 over 158
frames). Use max or top-k mean similarity over anchors.

- *Hypothesis:* covers more poses, so a turn or bow matches a turned or bowing
  anchor instead of being punished against a side-profile reference.
- *Risk:* anchors already carry generator error; a bad anchor set legitimises
  drift. Mitigate by only admitting anchors whose own score clears a strict bar.
- *Cost:* zero training; k extra dot products per frame.

### B. Pose- and framing-normalised scoring

Remove the nuisance factors before embedding:

1. Segment the subject (foreground mask) and crop/rescale to a canonical box,
   removing zoom and position.
2. Optionally estimate coarse viewpoint (front / side / back / three-quarter) and
   compare only against anchors of the same viewpoint bucket (combines with A).
3. Report the framing parameters separately as `s_frame`.

- *Hypothesis:* most of the drift we measured in chunk chains was zoom, crop and
  rotation; normalising scale and position removes the first two directly.
- *Risk:* animal and character viewpoint estimators are themselves out of
  distribution for drawn subjects.

### C. Attribute-factored identity

Score explicit, mostly pose-invariant attributes rather than a global embedding:

- colour palette of the subject region (histogram distance in a perceptual
  colour space);
- distinctive markings and accessories (local patch matching, e.g. dense DINOv2
  patch features matched between reference and frame);
- silhouette proportions normalised by pose family (head-to-body ratio, leg
  count, ear shape);
- for drawn characters: line style, costume elements, hair/fur colour regions.

- *Hypothesis:* the attributes a human uses to say "same dog" or "same character"
  are local and mostly survive pose change; a global CLS token mixes them with
  layout.
- *Risk:* hand-chosen attributes miss things and are easy to game in RL.

### D. Contrastive fine-tune of a general encoder

Fine-tune DINOv2 / a CLIP-style encoder (or a small head on top of frozen
features) so that *same subject, different pose/view/framing* pairs are close and
*different subject, similar pose* pairs are far.

- Positives: frames of one subject across a long clip, across clips, and across
  augmentations (crop, scale, flip, colour-preserving perspective). Generated
  multi-view renders of the same subject are a cheap source, if we trust the
  generator on short horizons.
- Hard negatives: the same subject with targeted edits (recolour, removed or
  swapped markings and accessories), other individuals of the same breed, other
  characters in the same art style.
- Train across photographed animals *and* drawn characters from the start.

- *Hypothesis:* the confound is learnable away with a few thousand curated pairs.
- *Risk:* generator-made positives can teach the verifier the generator's own
  drift as "same subject". Positives must be checked or come from real footage
  and original artwork. Licence-clean data for drawn characters is a real
  constraint: we will not train on third-party copyrighted characters.

### E. VLM judge as a weak label, not a reward

Ask a vision-language model "same subject? yes/no, and what differs?" on
reference-frame pairs.

- *Use:* labelling and auditing (building the evaluation set for A-D, catching
  verifier failures), not the per-block reward. Too slow and too noisy for
  per-frame RL, and easy to hack.

## 4. How we will validate a verifier

Build a small evaluation set first, before any training:

- **Subjects:** at least two photographed animal species and two original drawn
  characters (licence-clean, created or commissioned for this purpose).
- **Positives:** the same subject across pose, viewpoint, framing and expression
  changes (real footage for animals; artist-made turnarounds for characters).
- **Negatives:** controlled identity perturbations (section 3D) at graded
  severity.
- **Metrics:** ROC-AUC same vs different; and, critically, **score drop under
  pose/framing change alone** (should be near zero) vs **score drop under identity
  change** (should be large). The ratio of the two is the headline number;
  DINOv2-base is the baseline to beat.
- **Human check:** a blind same/different judgement on a sample, to calibrate
  thresholds.

## 5. Plugging the verifier in

### 5.1 Inference-time re-anchoring (no training)

In a causal AR generator producing blocks `b_1, b_2, ...`:

```
for each block b_t:
    s = s_id(b_t, anchors)            # plus s_frame if the application pins framing
    if s < tau_warn:  re-inject the reference image as conditioning for b_{t+1}
    if s < tau_fail:  discard b_t, roll back the KV cache to b_{t-1}, recache with
                      the reference (and the current prompt), regenerate b_t
```

- Measures: identity held over long streams and repeated prompt switches; added
  latency per intervention; intervention rate.
- The chunk-chain proxy suggests where to look: most loss happens at boundaries
  and at the first prompt switch, so those are the points to gate.
- Requires the verifier to be fast: it sits on the latency-critical path.

### 5.2 Verifier as an RL reward

Fine-tune a causal (or few-step bidirectional) generator with a per-block
reward

```
r_t = w_id * s_id(b_t) + w_frame * s_frame(b_t) + w_motion * s_motion(b_t, prompt_t)
      - w_kl * KL(policy || reference model)
```

- Candidate optimisers: reward-weighted or preference-based fine-tuning on
  sampled rollouts, or direct reward backpropagation through the few final
  denoising steps if `s_id` is differentiable. To be chosen after a literature
  pass (see [related-work.md](related-work.md), TODO).
- Long horizons: score at every block so credit is local, not only at the end of
  a stream.
- **Reward hacking to expect:** freezing pose and camera to match the reference;
  ignoring motion prompts; over-saturating distinctive colours. The motion term
  and a held-out human-judged set are the guard rails.
- Compare against 5.1 alone: if re-anchoring is enough at acceptable latency, RL
  may not be worth its cost.

### 5.3 Data generation for the verifier itself

The few-step bidirectional path (FastH3-style, ~5 s per 2.3 s clip on one
workstation GPU) can cheaply produce many short, low-drift clips of a subject
from one reference. Single long clips held DINOv2 similarity within 0.010 over
158 frames, so they are candidate positives for design A and D, with the risk
noted in 3D.

## 6. Open questions

- Is one verifier enough across photographed animals and drawn characters, or
  do we need a class-conditional head?
- How small can the reference set be? One image is the testbed's constraint; is a
  generated multi-view anchor set from that one image good enough?
- Does a verifier trained on short clips generalise to drift over minutes?
- Where should the threshold `tau` sit relative to the latency cost of a
  re-anchor?

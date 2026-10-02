#!/usr/bin/env python3
"""Identity drift with DINOv2: cosine similarity of every frame to an anchor.

Embeds frames with the DINOv2 CLS token (L2-normalised) and scores each against
either an external anchor image (the generation's reference, the normal case) or
the run's own first frame. Writes one row per run with first / last / min / mean,
the first-minus-last drop and the full per-frame series.

Caveat: this embedding conflates identity with pose and framing. Read
findings/identity-drift.md before using the number as an identity score.

Usage:
    python dino_drift.py --frames-dir out/bench --prefix same-c0 \
        --anchor reference.png --out drift.jsonl
    # score against the 20-step gold output instead (fidelity):
    python dino_drift.py --frames-dir out/bench --prefix fasth3-4 \
        --anchor-prefix gold-20 --sample 5 --out fidelity.jsonl
"""
import argparse
import json
import re
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel


def load(model_name, device):
    proc = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).eval().to(device)
    return proc, model


@torch.no_grad()
def embed(proc, model, paths, device, bs=16):
    vecs = []
    for i in range(0, len(paths), bs):
        ims = [Image.open(q).convert("RGB") for q in paths[i:i + bs]]
        t = proc(images=ims, return_tensors="pt").to(device)
        v = model(**t).last_hidden_state[:, 0]  # CLS token
        vecs.append(torch.nn.functional.normalize(v, dim=-1).cpu())
    return torch.cat(vecs)


def frames(frames_dir: Path, prefix: str):
    fs = list(frames_dir.glob(prefix + "_*.png"))
    return sorted(fs, key=lambda q: int(re.search(r"_(\d+)_", q.name).group(1)))


def sample(paths, k):
    if not k or k >= len(paths):
        return paths
    idx = [round(i * (len(paths) - 1) / (k - 1)) for i in range(k)] if k > 1 else [0]
    return [paths[i] for i in idx]


def main():
    a = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument("--frames-dir", type=Path, required=True)
    a.add_argument("--prefix", required=True, help="SaveImage filename prefix of the run")
    anc = a.add_mutually_exclusive_group()
    anc.add_argument("--anchor", type=Path, help="external anchor image (reference)")
    anc.add_argument("--anchor-prefix",
                     help="another run to compare frame-by-frame against (e.g. gold)")
    a.add_argument("--sample", type=int, default=0,
                   help="score only k evenly spaced frames (0 = all)")
    a.add_argument("--label")
    a.add_argument("--model", default="facebook/dinov2-base")
    a.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a.add_argument("--out", type=Path, default=Path("drift.jsonl"))
    g = a.parse_args()

    fs = sample(frames(g.frames_dir, g.prefix), g.sample)
    if not fs:
        raise SystemExit("no frames for " + g.prefix)
    proc, model = load(g.model, g.device)
    E = embed(proc, model, fs, g.device)

    if g.anchor_prefix:
        ref = sample(frames(g.frames_dir, g.anchor_prefix), g.sample)
        if len(ref) != len(fs):
            raise SystemExit("anchor run has a different frame count")
        R = embed(proc, model, ref, g.device)
        sims = (E * R).sum(-1).tolist()
        anchor = "run:" + g.anchor_prefix
    elif g.anchor:
        sims = (E @ embed(proc, model, [g.anchor], g.device)[0]).tolist()
        anchor = "external"
    else:
        sims = (E @ E[0]).tolist()
        anchor = "frame0"

    row = dict(prefix=g.prefix, label=g.label or g.prefix, n=len(fs), anchor=anchor,
               model=g.model, first=round(sims[0], 4), last=round(sims[-1], 4),
               min=round(min(sims), 4), mean=round(sum(sims) / len(sims), 4),
               drop_first_to_last=round(sims[0] - sims[-1], 4),
               series=[round(s, 4) for s in sims])
    with open(g.out, "a") as f:
        f.write(json.dumps(row) + "\n")
    print(json.dumps({k: v for k, v in row.items() if k != "series"}), flush=True)


if __name__ == "__main__":
    main()

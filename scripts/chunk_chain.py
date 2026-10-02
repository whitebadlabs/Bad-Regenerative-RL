#!/usr/bin/env python3
"""Chained chunk generation: a crude stand-in for autoregressive generation with a
bidirectional model.

Chunk 0 is conditioned on the reference image. Each later chunk is conditioned on
the last *decoded* frame of the previous chunk (so every boundary is a VAE
round-trip). Seeds increment per chunk. Records time to first frame (end of chunk
0) and total time, and writes one summary row per chain.

Prompts come from a JSON file: a list of strings, one per chunk. See
prompts/example_chain.json.

Usage:
    python chunk_chain.py same-prompt \
        --prompts prompts/example_chain.json --reference subject.png \
        --comfy-input-dir /path/to/ComfyUI/input \
        --comfy-output-dir /path/to/ComfyUI/output \
        --lora fasth3_dense.safetensors --steps 4 --length 39
"""
import argparse
import json
import re
import shutil
from pathlib import Path

from comfy_bench import run


def frames_for(output_dir: Path, prefix: str):
    """Frames SaveImage wrote for `prefix`, in frame order."""
    fs = list(output_dir.glob(prefix + "_*.png"))
    return sorted(fs, key=lambda q: int(re.search(r"_(\d+)_", q.name).group(1)))


def chain(tag, prompts, *, reference, input_dir: Path, output_dir: Path,
          comfy="http://127.0.0.1:8188", steps=8, width=512, height=512, length=39,
          seed=874330031, lora=None, out_runs=None, out_chains=None,
          output_prefix="bench/"):
    rows, cur = [], reference
    for i, prompt in enumerate(prompts):
        name = "%s-c%d" % (tag, i)
        row, _ = run(name, prompt, reference=cur, comfy=comfy, seed=seed + i,
                     steps=steps, width=width, height=height, length=length,
                     lora=lora, out=out_runs, output_prefix=output_prefix)
        if row["error"]:
            raise RuntimeError("chunk %d failed: %s" % (i, row["error"]))
        rows.append(row)
        fs = frames_for(output_dir / output_prefix.rstrip("/"), name)
        if not fs:
            raise RuntimeError("no frames found for " + name)
        handoff = "%s-handoff-%d.png" % (tag, i)
        shutil.copyfile(fs[-1], input_dir / handoff)
        cur = handoff

    def t(r):
        return r["exec_s"] if r["exec_s"] is not None else r["elapsed"]

    summary = dict(tag=tag, chunks=len(prompts), chunk_frames=length,
                   total_frames=length * len(prompts),
                   total_seconds_video=round(length * len(prompts) / 24.0, 3),
                   time_to_first_frame=round(t(rows[0]), 3),
                   total_time=round(sum(t(r) for r in rows), 3),
                   per_chunk=[t(r) for r in rows], lora=lora or "none", steps=steps)
    if out_chains:
        with open(out_chains, "a") as f:
            f.write(json.dumps(summary) + "\n")
    print(json.dumps(summary), flush=True)
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("tag")
    p.add_argument("--prompts", type=Path, required=True, help="JSON list of prompts")
    p.add_argument("--reference", required=True, help="filename in ComfyUI's input dir")
    p.add_argument("--comfy-input-dir", type=Path, required=True)
    p.add_argument("--comfy-output-dir", type=Path, required=True)
    p.add_argument("--comfy-url", default="http://127.0.0.1:8188")
    p.add_argument("--steps", type=int, default=8)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--length", type=int, default=39)
    p.add_argument("--seed", type=int, default=874330031)
    p.add_argument("--lora", default=None)
    p.add_argument("--out-runs", type=Path, default=Path("runs.jsonl"))
    p.add_argument("--out-chains", type=Path, default=Path("chains.jsonl"))
    a = p.parse_args()
    prompts = json.loads(a.prompts.read_text())
    chain(a.tag, prompts, reference=a.reference, input_dir=a.comfy_input_dir,
          output_dir=a.comfy_output_dir, comfy=a.comfy_url, steps=a.steps,
          width=a.width, height=a.height, length=a.length, seed=a.seed,
          lora=a.lora, out_runs=a.out_runs, out_chains=a.out_chains)


if __name__ == "__main__":
    main()

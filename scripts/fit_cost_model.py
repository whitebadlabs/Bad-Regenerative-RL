#!/usr/bin/env python3
"""Fit the linear cost model `time = fixed + per_frame * frames` to benchmark rows.

Reads a runs JSONL (from comfy_bench.py, or results/*.jsonl), keeps rows matching
the filters, and does an ordinary least-squares fit of time against frame count.
Uses `exec_s` when present (preferred), else `elapsed`.

Usage:
    python fit_cost_model.py ../results/spark-gb10-runs.jsonl --name-prefix len- --steps 8
"""
import argparse
import json
from pathlib import Path


def fit(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    c = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    f = my - c * mx
    ss_res = sum((y - (f + c * x)) ** 2 for x, y in zip(xs, ys))
    ss_tot = sum((y - my) ** 2 for y in ys)
    return f, c, 1 - ss_res / ss_tot if ss_tot else 1.0


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("runs", type=Path)
    p.add_argument("--name-prefix", default="")
    p.add_argument("--steps", type=int)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--height", type=int, default=512)
    a = p.parse_args()

    rows = [json.loads(l) for l in a.runs.read_text().splitlines() if l.strip()]
    rows = [r for r in rows if not r.get("error")
            and r["name"].startswith(a.name_prefix)
            and (a.steps is None or r.get("steps") == a.steps)
            and r.get("width", r.get("w")) == a.width
            and r.get("height", r.get("h")) == a.height]
    if len(rows) < 2:
        raise SystemExit("need at least two matching rows")
    xs = [r["length"] for r in rows]
    ys = [r["exec_s"] if r.get("exec_s") is not None else r["elapsed"] for r in rows]
    f, c, r2 = fit(xs, ys)
    print(json.dumps(dict(model="time = fixed + per_frame * frames",
                          fixed_s=round(f, 2), per_frame_s=round(c, 4), r2=round(r2, 5),
                          n=len(xs), points=sorted(zip(xs, ys)))))


if __name__ == "__main__":
    main()

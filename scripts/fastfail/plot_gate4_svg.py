#!/usr/bin/env python3
"""Draw the Gate 4 figure (per-step |d log p| vs FP32, log scale) as a stdlib-only SVG.

The repo ignores *.png, so the figure lives at docs/figures/diffusion-rl-fastfail-gate4.svg.
  python3 scripts/fastfail/plot_gate4_svg.py results/diffusion-rl-fastfail-gate4.json docs/figures/diffusion-rl-fastfail-gate4.svg
"""
import json, math, sys

src, out = sys.argv[1], sys.argv[2]
d = json.load(open(src))
s = d['settings']
W, H, L, R, T, B = 900, 540, 95, 20, 50, 70
pw, ph = W - L - R, H - T - B
steps = s['steps']
lo, hi = -7.2, -1.8                      # log10 range of the y axis


def x(i):
    return L + pw * (i + 0.5) / steps


def y(v):
    v = max(v, 10 ** lo)
    return T + ph * (hi - math.log10(v)) / (hi - lo)


parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="Helvetica, Arial, sans-serif" font-size="13">',
         f'<rect width="{W}" height="{H}" fill="#fff"/>',
         f'<text x="{L + pw / 2}" y="30" text-anchor="middle" font-size="17">SD3.5-M, Flow-GRPO SDE step, {steps} steps, '
         f'CFG {s["cfg"]}, noise {s["noise_level"]}</text>']
for e in range(-7, -1):                  # gridlines and decade labels
    yy = y(10 ** e)
    parts.append(f'<line x1="{L}" x2="{L + pw}" y1="{yy:.1f}" y2="{yy:.1f}" stroke="#ddd"/>')
    parts.append(f'<text x="{L - 8}" y="{yy + 4:.1f}" text-anchor="end">1e{e}</text>')
for i in range(0, steps, 2):
    parts.append(f'<line x1="{x(i):.1f}" x2="{x(i):.1f}" y1="{T}" y2="{T + ph}" stroke="#eee"/>')
    parts.append(f'<text x="{x(i):.1f}" y="{T + ph + 18}" text-anchor="middle">{i}</text>')
parts.append(f'<rect x="{L}" y="{T}" width="{pw}" height="{ph}" fill="none" stroke="#333"/>')
parts.append(f'<text x="{L + pw / 2}" y="{H - 22}" text-anchor="middle">denoising step (0 = noisiest)</text>')
parts.append(f'<text transform="translate(24 {T + ph / 2}) rotate(-90)" text-anchor="middle">'
             '|Δ log p| vs FP32, nats per latent element</text>')

series = []
for seed in [k for k in d if k.startswith('seed')]:
    for prec, color, dash in (('bf16', '#1f77b4', ''), ('fp16', '#d62728', '6 4')):
        main = seed == 'seed42'
        series.append((f'{prec.upper()} {seed}', d[seed][prec]['abs_dlogp_per_element'], color, dash,
                       2.2 if main else 1.2, 1.0 if main else 0.55, prec == 'bf16'))
for name, vals, color, dash, w, op, dots in series:
    pts = ' '.join(f'{x(i):.1f},{y(v):.1f}' for i, v in enumerate(vals))
    da = f' stroke-dasharray="{dash}"' if dash else ''
    parts.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="{w}" opacity="{op}"{da}/>')
    if dots:
        parts += [f'<circle cx="{x(i):.1f}" cy="{y(v):.1f}" r="3" fill="{color}" opacity="{op}"/>' for i, v in enumerate(vals)]
for k, (name, _, color, dash, w, op, _d) in enumerate(series):  # legend, two columns
    lx, ly = L + 12 + (k % 2) * 150, T + 16 + (k // 2) * 18
    da = f' stroke-dasharray="{dash}"' if dash else ''
    parts.append(f'<line x1="{lx}" x2="{lx + 26}" y1="{ly}" y2="{ly}" stroke="{color}" stroke-width="{w}" opacity="{op}"{da}/>')
    parts.append(f'<text x="{lx + 32}" y="{ly + 4}" font-size="11">{name}</text>')
parts.append('</svg>')
open(out, 'w').write('\n'.join(parts) + '\n')
print('wrote', out)

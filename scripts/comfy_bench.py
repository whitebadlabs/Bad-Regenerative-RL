#!/usr/bin/env python3
"""Run one MiniMax H3 image-to-video generation through a local ComfyUI server and
record its timing.

Builds the same graph used for every measurement in findings/fasth3-benchmark.md:
UNETLoader -> (optional LoRA / adapter) -> (optional sigma shift) -> BasicGuider
-> SamplerCustomAdvanced -> VAEDecode -> SaveImage, conditioned on a first frame
(and optionally a last frame) through MiniMaxH3ImageToVideo.

Timing: polls /history every --poll seconds (default 0.15) and records both the
client-side wall clock and ComfyUI's own execution time (span of the timestamps
in the history status messages). Prefer `exec_s`. A coarse poll interval
quantises wall-clock results; see findings/fasth3-benchmark.md section 9.

Usage:
    python comfy_bench.py idle-fasth3 \
        --reference subject.png --prompt-file prompts/idle.txt \
        --lora fasth3_dense.safetensors --steps 4 --length 56 --out runs.jsonl

The reference (and --last) must already be in ComfyUI's input directory. Change
--seed between repeated runs: ComfyUI caches identical graphs and returns in ~1 s.
"""
import argparse
import json
import time
import urllib.request
import uuid
from pathlib import Path

DEFAULT_UNET = "minimax_h3_fl2va_pruned_fp8_scaled.safetensors"
DEFAULT_CLIP = "qwen3vl_32b_minimax_h3_int8_convrot.safetensors"
DEFAULT_VAE = "minimax_h3_video_vae_fp16.safetensors"


def _post(base, path, body):
    req = urllib.request.Request(base + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.load(resp)


def _get(base, path):
    with urllib.request.urlopen(base + path, timeout=60) as resp:
        return json.load(resp)


def build_graph(prompt, seed, steps, width, height, length, first, last, prefix,
                unet=DEFAULT_UNET, clip=DEFAULT_CLIP, vae=DEFAULT_VAE,
                lora=None, lora_strength=1.0, shift_video=None, shift_audio=3.0,
                sampler="res_multistep", scheduler="simple"):
    def node(cls, **inputs):
        return {"class_type": cls, "inputs": inputs}

    g = {
        "1": node("UNETLoader", unet_name=unet, weight_dtype="default"),
        "2": node("CLIPLoader", clip_name=clip, type="minimax", device="default"),
        "3": node("VAELoader", vae_name=vae),
        "4": node("LoadImage", image=first),
        "5": node("MiniMaxH3ImageToVideo", clip=["2", 0], vae=["3", 0], prompt=prompt,
                  width=width, height=height, length=length, first_frame=["4", 0]),
        "8": node("RandomNoise", noise_seed=seed),
        "9": node("KSamplerSelect", sampler_name=sampler),
        "11": node("SamplerCustomAdvanced", noise=["8", 0], guider=["7", 0],
                   sampler=["9", 0], sigmas=["10", 0], latent_image=["5", 1]),
        "12": node("VAEDecode", samples=["11", 0], vae=["3", 0]),
        "14": node("SaveImage", images=["12", 0], filename_prefix=prefix),
    }
    model = ["1", 0]
    if lora:
        g["6"] = node("LoraLoaderModelOnly", model=model, lora_name=lora,
                      strength_model=lora_strength)
        model = ["6", 0]
    if shift_video is not None:
        g["16"] = node("MiniMaxH3SigmaShift", model=model, shift_video=shift_video,
                       shift_audio=shift_audio)
        model = ["16", 0]
    g["7"] = node("BasicGuider", model=model, conditioning=["5", 0])
    g["10"] = node("BasicScheduler", model=model, scheduler=scheduler, steps=steps,
                   denoise=1.0)
    if last:
        g["15"] = node("LoadImage", image=last)
        g["5"]["inputs"]["last_frame"] = ["15", 0]
    return g


def run(name, prompt, *, reference, comfy="http://127.0.0.1:8188", seed=874330031,
        steps=8, width=512, height=512, length=56, last=None, lora=None,
        lora_strength=1.0, shift_video=None, sampler="res_multistep",
        scheduler="simple", unet=DEFAULT_UNET, clip=DEFAULT_CLIP, vae=DEFAULT_VAE,
        out=None, poll=0.15, timeout=1800, fps=24.0, output_prefix="bench/"):
    """Submit one graph, wait for it, append a result row to `out`, return (row, files)."""
    g = build_graph(prompt, seed, steps, width, height, length, reference, last,
                    output_prefix + name, unet, clip, vae, lora, lora_strength,
                    shift_video, 3.0, sampler, scheduler)
    t0 = time.time()
    pid = _post(comfy, "/prompt", {"prompt": g, "client_id": str(uuid.uuid4())})["prompt_id"]
    files, err, exec_s = [], None, None
    while True:
        time.sleep(poll)
        hist = _get(comfy, "/history/" + pid)
        if pid in hist:
            status = hist[pid].get("status", {})
            if status.get("status_str") == "error":
                err = json.dumps(status)[:400]
            stamps = [m[1].get("timestamp") for m in status.get("messages", [])
                      if isinstance(m, (list, tuple)) and len(m) > 1
                      and isinstance(m[1], dict) and m[1].get("timestamp")]
            if len(stamps) >= 2:
                exec_s = round((max(stamps) - min(stamps)) / 1000.0, 3)
            for o in hist[pid].get("outputs", {}).values():
                files += [i["filename"] for i in o.get("images", [])]
            break
        if time.time() - t0 > timeout:
            err = "timeout"
            break
    seconds_of_video = round(length / fps, 3)
    row = dict(name=name, elapsed=round(time.time() - t0, 2), exec_s=exec_s,
               steps=steps, width=width, height=height, length=length,
               lora=lora or "none", lora_strength=lora_strength,
               shift_video=shift_video, sampler=sampler, scheduler=scheduler,
               seed=seed, last_frame_cond=bool(last), frames_out=len(files),
               seconds_of_video=seconds_of_video, error=err)
    t = exec_s if exec_s is not None else row["elapsed"]
    row["sec_per_sec_video"] = round(t / seconds_of_video, 2) if not err else None
    if out:
        with open(out, "a") as f:
            f.write(json.dumps(row) + "\n")
    print(json.dumps(row), flush=True)
    return row, files


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("name", help="run name; also the output filename prefix")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--prompt")
    src.add_argument("--prompt-file", type=Path)
    p.add_argument("--reference", required=True,
                   help="first-frame image, by filename in ComfyUI's input dir")
    p.add_argument("--last", help="optional last-frame image (first-last-frame conditioning)")
    p.add_argument("--comfy-url", default="http://127.0.0.1:8188")
    p.add_argument("--steps", type=int, default=8)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--length", type=int, default=56, help="frames (H3 grid: 17k+5)")
    p.add_argument("--seed", type=int, default=874330031)
    p.add_argument("--lora", default=None,
                   help="LoRA/adapter filename, e.g. the Turbo 8-step LoRA or the "
                        "FastH3 dense adapter; omit for the base model")
    p.add_argument("--lora-strength", type=float, default=1.0)
    p.add_argument("--sigma-shift", type=float, default=None,
                   help="add MiniMaxH3SigmaShift with this video shift (default: omit node)")
    p.add_argument("--sampler", default="res_multistep")
    p.add_argument("--scheduler", default="simple")
    p.add_argument("--unet", default=DEFAULT_UNET)
    p.add_argument("--clip", default=DEFAULT_CLIP)
    p.add_argument("--vae", default=DEFAULT_VAE)
    p.add_argument("--poll", type=float, default=0.15, help="history poll interval, s")
    p.add_argument("--out", type=Path, default=Path("runs.jsonl"))
    a = p.parse_args()
    prompt = a.prompt if a.prompt is not None else a.prompt_file.read_text().strip()
    run(a.name, prompt, reference=a.reference, comfy=a.comfy_url, seed=a.seed,
        steps=a.steps, width=a.width, height=a.height, length=a.length, last=a.last,
        lora=a.lora, lora_strength=a.lora_strength, shift_video=a.sigma_shift,
        sampler=a.sampler, scheduler=a.scheduler, unet=a.unet, clip=a.clip,
        vae=a.vae, out=a.out, poll=a.poll)


if __name__ == "__main__":
    main()

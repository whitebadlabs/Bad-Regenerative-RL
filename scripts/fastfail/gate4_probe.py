"""Gate 4: minimal mismatch probe (diffusion RL fast-fail).

One prompt, fixed seed, fixed noise. Sample one SDE trajectory with Flow-GRPO's own sampler in FP32, then score
each sampled transition x_t -> x_{t-1} with Flow-GRPO's own sde_step_with_logprob, with the transformer in FP32 and
in BF16 (plan arms), plus FP16 (the repo's default mixed_precision, extra arm). Same GPU, same inputs.

Reports per-step |delta log-prob| in the repo's unit (mean over latent elements) and as the sum over the latent
(mean x element count). Raw arrays saved. No interpretation, by design.
"""
import json
import os
import sys
import time

import numpy as np
import torch

R = os.environ.get('FASTFAIL_ROOT', '.')  # dir holding src/flow_grpo
sys.path.insert(0, f'{R}/src/flow_grpo')
from diffusers import StableDiffusion3Pipeline  # noqa: E402
from flow_grpo.diffusers_patch.sd3_pipeline_with_logprob import pipeline_with_logprob  # noqa: E402
from flow_grpo.diffusers_patch.sd3_sde_with_logprob import sde_step_with_logprob  # noqa: E402

OUT = sys.argv[1]
SEEDS = [42, 1, 2]           # 42 is the primary result; 1 and 2 are a robustness check
STEPS, CFG, NOISE, RES = 10, 4.5, 0.7, 512   # general_ocr_sd3_1gpu training settings
MODEL = 'stabilityai/stable-diffusion-3.5-medium'
dev = torch.device('cuda')
torch.backends.cuda.matmul.allow_tf32 = False   # true FP32 reference
torch.backends.cudnn.allow_tf32 = False

prompt = open(f'{R}/src/flow_grpo/dataset/ocr/test.txt').readline().strip()
pipe = StableDiffusion3Pipeline.from_pretrained(MODEL, torch_dtype=torch.float32).to(dev)
pipe.set_progress_bar_config(disable=True)
with torch.no_grad():
    pe, npe, ppe, nppe = pipe.encode_prompt(prompt, prompt, prompt, device=dev, do_classifier_free_guidance=True,
                                            max_sequence_length=128)
t32 = pipe.transformer
arms = {'fp32': (t32, torch.float32)}
arms['bf16'] = (StableDiffusion3Pipeline.from_pretrained(MODEL, torch_dtype=torch.bfloat16, text_encoder=None,
                text_encoder_2=None, text_encoder_3=None, vae=None, tokenizer=None, tokenizer_2=None,
                tokenizer_3=None).transformer.to(dev), torch.bfloat16)
arms['fp16'] = (StableDiffusion3Pipeline.from_pretrained(MODEL, torch_dtype=torch.float16, text_encoder=None,
                text_encoder_2=None, text_encoder_3=None, vae=None, tokenizer=None, tokenizer_2=None,
                tokenizer_3=None).transformer.to(dev), torch.float16)
for t, _ in arms.values():
    t.eval()


def score(transformer, dtype, latents, timesteps, j):
    """compute_log_prob() from scripts/train_sd3.py with train.cfg, for one step, transformer in `dtype`."""
    x = latents[j]
    with torch.no_grad():
        out = transformer(hidden_states=torch.cat([x] * 2).to(dtype),
                          timestep=torch.cat([timesteps[j:j + 1]] * 2),
                          encoder_hidden_states=torch.cat([npe, pe]).to(dtype),
                          pooled_projections=torch.cat([nppe, ppe]).to(dtype), return_dict=False)[0]
        u, c = out.float().chunk(2)
        pred = u + CFG * (c - u)
        _, lp, mean, std = sde_step_with_logprob(pipe.scheduler, pred, timesteps[j:j + 1], x.float(),
                                                 prev_sample=latents[j + 1].float(), noise_level=NOISE)
    return lp.double().item(), mean.double().cpu().numpy(), std.double().item()


results, raw = {'prompt': prompt, 'settings': dict(steps=STEPS, cfg=CFG, noise_level=NOISE, resolution=RES,
                model=MODEL, tf32=False, reference='fp32 sampling, fp32 text embeddings')}, {}
for seed in SEEDS:
    g = torch.Generator(device=dev).manual_seed(seed)
    t0 = time.time()
    with torch.no_grad():
        _, lat, lp_sampler = pipeline_with_logprob(pipe, prompt_embeds=pe, pooled_prompt_embeds=ppe,
                                                   negative_prompt_embeds=npe, negative_pooled_prompt_embeds=nppe,
                                                   num_inference_steps=STEPS, guidance_scale=CFG, output_type='pt',
                                                   height=RES, width=RES, noise_level=NOISE, generator=g)
    ts = pipe.scheduler.timesteps.to(dev)
    n_el = lat[0][0].numel()
    per = {a: [] for a in arms}
    means = {a: [] for a in arms}
    for j in range(STEPS):
        for a, (t, dt) in arms.items():
            lp, mean, std = score(t, dt, lat, ts, j)
            per[a].append(lp)
            means[a].append(mean)
    sampler = np.array([x.double().item() for x in lp_sampler])
    r = {'n_latent_elements': n_el, 'sample_secs': round(time.time() - t0, 1),
         'sampler_logprob_fp32': sampler.tolist(),
         'recomputed_fp32_minus_sampler_abs': np.abs(np.array(per['fp32']) - sampler).tolist()}
    for a in ('bf16', 'fp16'):
        d = np.abs(np.array(per[a]) - np.array(per['fp32']))
        r[a] = {'abs_dlogp_per_element': d.tolist(), 'max_per_element': float(d.max()),
                'abs_dlogp_sum_over_latent': (d * n_el).tolist(), 'max_sum_over_latent': float(d.max() * n_el),
                'max_abs_dmean_per_step': [float(np.abs(means[a][j] - means['fp32'][j]).max()) for j in range(STEPS)]}
    r['logprob_per_element'] = {a: per[a] for a in arms}
    results[f'seed{seed}'] = r
    raw.update({f'seed{seed}_latents': torch.stack([x.float() for x in lat]).cpu().numpy(),
                f'seed{seed}_timesteps': ts.float().cpu().numpy(), f'seed{seed}_sampler_logprob': sampler,
                **{f'seed{seed}_logprob_{a}': np.array(per[a]) for a in arms},
                **{f'seed{seed}_mean_{a}': np.stack(means[a]) for a in arms}})
    print(f'seed {seed}: max |dlogp| per element bf16 {r["bf16"]["max_per_element"]:.3e} fp16 {r["fp16"]["max_per_element"]:.3e}', flush=True)

json.dump(results, open(f'{OUT}/gate4_results.json', 'w'), indent=1)
np.savez_compressed(f'{OUT}/gate4_raw.npz', **raw)
print('saved', OUT)

"""
Ablation Study — Systematic Comparison
=========================================
Compares 4 configurations:
  1. Baseline (original ModelScope T2V)
  2. +Enhancement A (temporal smoothing only)
  3. +Enhancement B (semantic enhancement only)
  4. +A+B Combined (both enhancements)

Metrics: CLIP-SIM, Temporal LPIPS, SSIM, PSNR
"""

import os
import torch
import numpy as np
from PIL import Image
import imageio
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler

from prompts import ALL_PROMPTS
from enhancement_temporal import (
    apply_temporal_smoothing_postprocess,
    interpolate_frames_linear,
)
from enhancement_semantic import augment_prompt, detect_element


def load_pipeline():
    pipe = DiffusionPipeline.from_pretrained(
        "damo-vilab/text-to-video-ms-1.7b",
        torch_dtype=torch.float16, variant="fp16",
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()
    pipe.enable_vae_slicing()
    return pipe


def generate_base(pipe, prompt, guidance=7.5, seed=42):
    generator = torch.Generator(device='cpu').manual_seed(seed)
    output = pipe(
        prompt=prompt, num_frames=16, height=256, width=256,
        num_inference_steps=25, guidance_scale=guidance, generator=generator,
    )
    frames = output.frames[0]
    pil = []
    for f in frames:
        if isinstance(f, np.ndarray):
            if f.dtype in (np.float32, np.float64):
                f = (f * 255).clip(0, 255).astype(np.uint8)
            pil.append(Image.fromarray(f))
        else:
            pil.append(f)
    return pil


def save_video(frames, path, fps=8):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    arrays = [np.array(f) for f in frames]
    imageio.mimsave(path, arrays, fps=fps, codec="libx264")


# ── Metric computation ──
def compute_metrics(frames, prompt, clip_sim_fn, lpips_fn):
    """Compute all metrics for a set of frames."""
    clip_score, _ = clip_sim_fn.compute(frames, prompt)
    lpips_score, _ = lpips_fn.compute(frames)

    from skimage.metrics import structural_similarity, peak_signal_noise_ratio
    ssim_vals, psnr_vals = [], []
    for i in range(len(frames) - 1):
        f1, f2 = np.array(frames[i]), np.array(frames[i+1])
        ssim_vals.append(structural_similarity(f1, f2, channel_axis=2, data_range=255))
        psnr_vals.append(peak_signal_noise_ratio(f1, f2, data_range=255))

    return {
        "clip_sim": clip_score,
        "temporal_lpips": lpips_score,
        "ssim": np.mean(ssim_vals),
        "psnr": np.mean(psnr_vals),
    }


def run_ablation(pipe, output_dir="/content/outputs/ablation"):
    """Run the full ablation study."""
    from evaluate_metrics import CLIPSimilarity, TemporalLPIPS

    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_sim = CLIPSimilarity(device)
    temp_lpips = TemporalLPIPS(device)

    configs = {
        "baseline": {"use_aug": False, "guidance": 7.5, "smooth": False},
        "enh_A": {"use_aug": False, "guidance": 7.5, "smooth": True},
        "enh_B": {"use_aug": True, "guidance": 8.0, "smooth": False},
        "combined": {"use_aug": True, "guidance": 8.0, "smooth": True},
    }

    # Use a subset of prompts for ablation (1 per element + simple/complex)
    ablation_prompts = {
        "fire": [ALL_PROMPTS["fire"][0]],
        "water": [ALL_PROMPTS["water"][0]],
        "earth": [ALL_PROMPTS["earth"][0]],
        "wind": [ALL_PROMPTS["wind"][0]],
        "simple": [ALL_PROMPTS["simple"][0]],
        "complex": [ALL_PROMPTS["complex"][0]],
    }

    results = {cfg: {} for cfg in configs}

    for cfg_name, cfg in configs.items():
        print(f"\n{'='*60}")
        print(f"CONFIG: {cfg_name.upper()}")
        print(f"  Augmentation: {cfg['use_aug']}, Guidance: {cfg['guidance']}, Smoothing: {cfg['smooth']}")
        print(f"{'='*60}")

        for cat, prompts in ablation_prompts.items():
            cat_metrics = []
            for i, prompt in enumerate(prompts):
                print(f"  [{cat}] {prompt[:50]}...")

                # Apply prompt augmentation if enabled
                gen_prompt = augment_prompt(prompt, cat) if cfg["use_aug"] else prompt

                # Generate
                frames = generate_base(pipe, gen_prompt, cfg["guidance"])

                # Apply temporal smoothing if enabled
                if cfg["smooth"]:
                    frames = apply_temporal_smoothing_postprocess(frames, kernel_size=3)

                # Save video
                vid_dir = os.path.join(output_dir, cfg_name, cat)
                vid_path = os.path.join(vid_dir, f"{cat}_{i+1:02d}.mp4")
                save_video(frames, vid_path)

                # Compute metrics (use ORIGINAL prompt for CLIP-SIM)
                m = compute_metrics(frames, prompt, clip_sim, temp_lpips)
                cat_metrics.append(m)
                print(f"    CLIP={m['clip_sim']:.4f} LPIPS={m['temporal_lpips']:.4f} "
                      f"SSIM={m['ssim']:.4f} PSNR={m['psnr']:.2f}")

            if cat_metrics:
                avg = {k: np.mean([m[k] for m in cat_metrics]) for k in cat_metrics[0]}
                results[cfg_name][cat] = avg

    # Print summary table
    print("\n" + "="*80)
    print("ABLATION STUDY RESULTS")
    print("="*80)
    print(f"{'Config':<12} {'Category':<10} {'CLIP-SIM':>10} {'T-LPIPS':>10} {'SSIM':>8} {'PSNR':>8}")
    print("-"*60)
    for cfg_name in configs:
        for cat in ablation_prompts:
            if cat in results[cfg_name]:
                m = results[cfg_name][cat]
                print(f"{cfg_name:<12} {cat:<10} {m['clip_sim']:>10.4f} "
                      f"{m['temporal_lpips']:>10.4f} {m['ssim']:>8.4f} {m['psnr']:>8.2f}")
        print("-"*60)

    return results


if __name__ == "__main__":
    pipe = load_pipeline()
    results = run_ablation(pipe)

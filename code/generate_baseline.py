"""
Baseline Video Generation using ModelScope T2V
================================================
Generates videos for all prompt categories using the
damo-vilab/text-to-video-ms-1.7b model via HuggingFace Diffusers.
"""

import os
import torch
import imageio
import numpy as np
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from prompts import ALL_PROMPTS


def load_pipeline(device="cuda", dtype=torch.float16):
    """Load the ModelScope T2V pipeline."""
    pipe = DiffusionPipeline.from_pretrained(
        "damo-vilab/text-to-video-ms-1.7b",
        torch_dtype=dtype,
        variant="fp16",
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config
    )
    pipe.enable_model_cpu_offload()  # saves GPU memory
    pipe.enable_vae_slicing()
    return pipe


def generate_video(pipe, prompt, num_frames=16, height=256, width=256,
                   num_inference_steps=25, guidance_scale=7.5, seed=42):
    """Generate a single video from a text prompt."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    output = pipe(
        prompt=prompt,
        num_frames=num_frames,
        height=height,
        width=width,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=generator,
    )
    return output.frames[0]  # list of PIL images


def save_video(frames, path, fps=8):
    """Save frames as an MP4 video."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    video_array = [np.array(f) for f in frames]
    imageio.mimsave(path, video_array, fps=fps, codec="libx264")
    print(f"  Saved: {path}")


def main():
    output_dir = os.path.join(os.path.dirname(__file__), "..", "outputs", "baseline")
    os.makedirs(output_dir, exist_ok=True)

    print("Loading ModelScope T2V pipeline...")
    pipe = load_pipeline()

    for category, prompts in ALL_PROMPTS.items():
        cat_dir = os.path.join(output_dir, category)
        os.makedirs(cat_dir, exist_ok=True)
        print(f"\n=== Category: {category.upper()} ===")

        for i, prompt in enumerate(prompts):
            print(f"  [{i+1}/{len(prompts)}] {prompt[:60]}...")
            frames = generate_video(pipe, prompt)
            video_path = os.path.join(cat_dir, f"{category}_{i+1:02d}.mp4")
            save_video(frames, video_path)

            # Also save individual frames for analysis
            frame_dir = os.path.join(cat_dir, f"{category}_{i+1:02d}_frames")
            os.makedirs(frame_dir, exist_ok=True)
            for j, frame in enumerate(frames):
                frame.save(os.path.join(frame_dir, f"frame_{j:03d}.png"))

    print("\nDone! All videos saved to:", output_dir)


if __name__ == "__main__":
    main()

"""
End-to-End Pipeline: Video Generation + Enhancements + TTS
============================================================
Generates the final output: enhanced video with narration.
"""

import os
import torch
import numpy as np
import imageio
from PIL import Image
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler

from enhancement_temporal import (
    generate_with_temporal_enhancement,
    apply_temporal_smoothing_postprocess,
    interpolate_frames_linear,
)
from enhancement_semantic import (
    augment_prompt, detect_element,
    generate_with_semantic_enhancement,
)
from tts_module import generate_narrated_video


def load_pipeline(device="cuda", dtype=torch.float16):
    """Load ModelScope T2V pipeline."""
    pipe = DiffusionPipeline.from_pretrained(
        "damo-vilab/text-to-video-ms-1.7b",
        torch_dtype=dtype, variant="fp16",
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()
    pipe.enable_vae_slicing()
    return pipe


def save_video(frames, path, fps=8):
    """Save frames as MP4."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    arrays = []
    for f in frames:
        arr = np.array(f) if isinstance(f, Image.Image) else f
        if arr.dtype in (np.float32, np.float64):
            arr = (arr * 255).clip(0, 255).astype(np.uint8)
        arrays.append(arr)
    imageio.mimsave(path, arrays, fps=fps, codec="libx264")


def save_frames(frames, frame_dir):
    """Save individual frames as PNGs."""
    os.makedirs(frame_dir, exist_ok=True)
    for j, f in enumerate(frames):
        if isinstance(f, np.ndarray):
            if f.dtype in (np.float32, np.float64):
                f = (f * 255).clip(0, 255).astype(np.uint8)
            f = Image.fromarray(f)
        f.save(os.path.join(frame_dir, f"frame_{j:03d}.png"))


def generate_combined(pipe, prompt, element=None, num_frames=24, seed=42):
    """
    Generate video with BOTH enhancements combined:
    1. Semantic: prompt augmentation + higher guidance
    2. Temporal: post-processing smoothing + interpolation
    """
    if element is None:
        element = detect_element(prompt)

    # Enhancement B: semantic
    enhanced_prompt = augment_prompt(prompt, element)
    effective_guidance = 8.0  # (12+4)/2

    generator = torch.Generator(device='cpu').manual_seed(seed)
    output = pipe(
        prompt=enhanced_prompt,
        num_frames=num_frames,
        height=256, width=256,
        num_inference_steps=25,
        guidance_scale=effective_guidance,
        generator=generator,
    )

    frames = output.frames[0]
    pil_frames = []
    for f in frames:
        if isinstance(f, np.ndarray):
            if f.dtype in (np.float32, np.float64):
                f = (f * 255).clip(0, 255).astype(np.uint8)
            pil_frames.append(Image.fromarray(f))
        else:
            pil_frames.append(f)

    # Enhancement A: temporal smoothing + interpolation
    smoothed = apply_temporal_smoothing_postprocess(pil_frames, kernel_size=3)
    final = interpolate_frames_linear(smoothed, factor=2)
    return final, enhanced_prompt


def main():
    output_dir = os.path.join(os.path.dirname(__file__), "..", "outputs", "final")
    os.makedirs(output_dir, exist_ok=True)

    # Hero prompts — one per element, chosen for visual impact
    hero_prompts = {
        "fire": "A volcanic eruption with glowing lava streams flowing down a mountainside",
        "water": "Ocean waves crashing dramatically on rocky shores at golden sunset",
        "earth": "Crystals growing rapidly from the ground inside a dark glowing cave",
        "wind": "A powerful tornado forming over an open golden wheat field",
    }

    print("Loading ModelScope T2V pipeline...")
    pipe = load_pipeline()

    for element, prompt in hero_prompts.items():
        print(f"\n{'='*60}")
        print(f"GENERATING: {element.upper()}")
        print(f"Prompt: {prompt}")
        print(f"{'='*60}")

        # Generate with combined enhancements (24 frames → interpolated to 47)
        frames, aug_prompt = generate_combined(pipe, prompt, element,
                                                num_frames=24, seed=42)
        print(f"  Frames: {len(frames)} (after interpolation)")
        print(f"  Duration: {len(frames)/16:.1f}s at 16fps")

        # Save video
        video_path = os.path.join(output_dir, f"{element}_enhanced.mp4")
        save_video(frames, video_path, fps=16)

        # Save frames
        frame_dir = os.path.join(output_dir, f"{element}_enhanced_frames")
        save_frames(frames, frame_dir)

        # Add TTS narration
        narrated_path = os.path.join(output_dir, f"{element}_final.mp4")
        generate_narrated_video(video_path, prompt, narrated_path, element)

    print(f"\n{'='*60}")
    print("ALL FINAL VIDEOS GENERATED!")
    print(f"Output: {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

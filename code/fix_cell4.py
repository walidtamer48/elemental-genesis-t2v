# Cell 4: Generate ALL videos (FIXED)
# Copy-paste this entire cell to replace Cell 4 in Colab

import os
import numpy as np
import imageio
from PIL import Image

OUTPUT_DIR = '/content/outputs/baseline'

def generate_and_save(pipe, prompt, save_path, num_frames=16, seed=42):
    generator = torch.Generator(device='cpu').manual_seed(seed)
    output = pipe(
        prompt=prompt,
        num_frames=num_frames,
        height=256, width=256,
        num_inference_steps=25,
        guidance_scale=7.5,
        generator=generator,
    )
    frames = output.frames[0]  # list of numpy arrays or PIL images

    # Save video - convert to uint8 if needed
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    video_array = []
    for f in frames:
        if isinstance(f, np.ndarray):
            if f.dtype == np.float32 or f.dtype == np.float64:
                f = (f * 255).clip(0, 255).astype(np.uint8)
            video_array.append(f)
        else:
            video_array.append(np.array(f))
    imageio.mimsave(save_path, video_array, fps=8, codec='libx264')

    # Save individual frames
    frame_dir = save_path.replace('.mp4', '_frames')
    os.makedirs(frame_dir, exist_ok=True)
    for j, f in enumerate(frames):
        if isinstance(f, np.ndarray):
            if f.dtype == np.float32 or f.dtype == np.float64:
                f = (f * 255).clip(0, 255).astype(np.uint8)
            img = Image.fromarray(f)
        else:
            img = f
        img.save(os.path.join(frame_dir, f'frame_{j:03d}.png'))

    # Return as PIL images for later evaluation
    pil_frames = []
    for f in frames:
        if isinstance(f, np.ndarray):
            if f.dtype == np.float32 or f.dtype == np.float64:
                f = (f * 255).clip(0, 255).astype(np.uint8)
            pil_frames.append(Image.fromarray(f))
        else:
            pil_frames.append(f)
    return pil_frames

all_frames = {}
for category, prompts in ALL_PROMPTS.items():
    print(f'\n=== {category.upper()} ===')
    all_frames[category] = []
    for i, prompt in enumerate(prompts):
        print(f'  [{i+1}/{len(prompts)}] {prompt[:60]}...')
        path = os.path.join(OUTPUT_DIR, category, f'{category}_{i+1:02d}.mp4')
        frames = generate_and_save(pipe, prompt, path)
        all_frames[category].append(frames)
        print(f'    Saved: {path}')

print('\n All videos generated!')

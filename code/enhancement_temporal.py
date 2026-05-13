"""
Enhancement A: Temporal Consistency Refinement
================================================
Addresses Weakness 1 (Temporal Flickering) via:
1. Latent-space temporal smoothing during denoising
2. Post-generation frame interpolation for smoother motion
"""

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter1d


# ──────────────────────────────────────────────
# 1. Latent-Space Temporal Smoothing
# ──────────────────────────────────────────────

class TemporalSmoother:
    """
    Applies Gaussian temporal smoothing to latent representations
    during the denoising process to enforce frame consistency.
    
    Mathematical formulation:
        z_smoothed[f] = Σ_k w_k · z[f+k]
    where w_k = exp(-k²/(2σ²)) / Σ exp(-k²/(2σ²))
    
    This acts as a low-pass filter along the temporal dimension,
    reducing high-frequency flickering while preserving global motion.
    """
    
    def __init__(self, sigma=0.8, apply_after_step_ratio=0.3):
        """
        Args:
            sigma: Standard deviation of Gaussian kernel (higher = more smoothing)
            apply_after_step_ratio: Only apply smoothing after this fraction of
                                    denoising steps (early steps need freedom to
                                    establish structure)
        """
        self.sigma = sigma
        self.apply_after_step_ratio = apply_after_step_ratio
    
    def smooth_latents(self, latents, current_step, total_steps):
        """
        Apply temporal Gaussian smoothing to latent tensor.
        
        Args:
            latents: Tensor of shape [B, F, C, H, W] or [B, C, F, H, W]
            current_step: Current denoising step index
            total_steps: Total number of denoising steps
            
        Returns:
            Smoothed latents with same shape
        """
        progress = current_step / total_steps
        
        # Only apply smoothing after initial structure is established
        if progress < self.apply_after_step_ratio:
            return latents
        
        # Adaptive sigma: increase smoothing as we get closer to final output
        adaptive_sigma = self.sigma * (0.5 + 0.5 * progress)
        
        # Apply Gaussian filter along temporal dimension
        # Convert to numpy for scipy gaussian_filter1d, then back to tensor
        device = latents.device
        dtype = latents.dtype
        latents_np = latents.cpu().float().numpy()
        
        # Assuming shape is [B, C, F, H, W] — smooth along axis=2 (temporal)
        if latents_np.ndim == 5:
            smoothed = gaussian_filter1d(latents_np, sigma=adaptive_sigma, axis=2)
        else:
            smoothed = latents_np  # fallback: no smoothing
        
        # Blend smoothed with original to avoid over-smoothing
        blend_factor = min(0.6, progress)  # Never fully replace
        result = (1 - blend_factor) * latents_np + blend_factor * smoothed
        
        return torch.from_numpy(result).to(device=device, dtype=dtype)


# ──────────────────────────────────────────────
# 2. Post-Generation Frame Interpolation
# ──────────────────────────────────────────────

def interpolate_frames_linear(frames, factor=2):
    """
    Simple linear interpolation between consecutive frames
    to double the frame count and smooth transitions.
    
    Args:
        frames: List of PIL Images or numpy arrays
        factor: Interpolation factor (2 = double frames)
        
    Returns:
        List of interpolated frames (len = (len(frames)-1)*factor + 1)
    """
    arrays = [np.array(f).astype(np.float32) for f in frames]
    interpolated = []
    
    for i in range(len(arrays) - 1):
        interpolated.append(arrays[i])
        for j in range(1, factor):
            alpha = j / factor
            blended = (1 - alpha) * arrays[i] + alpha * arrays[i + 1]
            interpolated.append(blended.astype(np.uint8))
    interpolated.append(arrays[-1])  # last frame
    
    return [Image.fromarray(f.astype(np.uint8) if f.dtype != np.uint8 else f) 
            for f in interpolated]


def apply_temporal_smoothing_postprocess(frames, kernel_size=3):
    """
    Apply temporal moving average to smooth pixel values across frames.
    
    Args:
        frames: List of PIL Images
        kernel_size: Size of the moving average window (must be odd)
    
    Returns:
        Temporally smoothed frames
    """
    arrays = np.stack([np.array(f).astype(np.float32) for f in frames])
    # arrays shape: [F, H, W, C]
    
    pad = kernel_size // 2
    smoothed = np.copy(arrays)
    
    for i in range(len(arrays)):
        start = max(0, i - pad)
        end = min(len(arrays), i + pad + 1)
        smoothed[i] = np.mean(arrays[start:end], axis=0)
    
    return [Image.fromarray(f.astype(np.uint8)) for f in smoothed]


# ──────────────────────────────────────────────
# 3. Enhanced Generation Pipeline
# ──────────────────────────────────────────────

def generate_with_temporal_enhancement(pipe, prompt, num_frames=16,
                                        height=256, width=256,
                                        num_inference_steps=25,
                                        guidance_scale=7.5,
                                        seed=42,
                                        smoothing_sigma=0.8,
                                        interpolation_factor=2,
                                        post_smooth_kernel=3):
    """
    Generate a video with temporal consistency enhancements.
    
    Enhancement pipeline:
    1. Generate base video with ModelScope T2V
    2. Apply post-generation temporal smoothing
    3. Apply frame interpolation to increase smoothness
    
    Args:
        pipe: DiffusionPipeline (ModelScope T2V)
        prompt: Text prompt
        smoothing_sigma: Temporal smoothing strength
        interpolation_factor: Frame interpolation multiplier
        post_smooth_kernel: Post-processing smoothing window size
        
    Returns:
        Enhanced frames as list of PIL Images
    """
    import torch
    
    generator = torch.Generator(device='cpu').manual_seed(seed)
    
    # Step 1: Generate base video
    output = pipe(
        prompt=prompt,
        num_frames=num_frames,
        height=height,
        width=width,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=generator,
    )
    
    frames = output.frames[0]
    
    # Convert to PIL if needed
    pil_frames = []
    for f in frames:
        if isinstance(f, np.ndarray):
            if f.dtype == np.float32 or f.dtype == np.float64:
                f = (f * 255).clip(0, 255).astype(np.uint8)
            pil_frames.append(Image.fromarray(f))
        else:
            pil_frames.append(f)
    
    # Step 2: Apply temporal smoothing
    smoothed_frames = apply_temporal_smoothing_postprocess(
        pil_frames, kernel_size=post_smooth_kernel
    )
    
    # Step 3: Apply frame interpolation
    if interpolation_factor > 1:
        final_frames = interpolate_frames_linear(
            smoothed_frames, factor=interpolation_factor
        )
    else:
        final_frames = smoothed_frames
    
    return final_frames

"""
Enhancement B: Semantic Alignment Improvement
===============================================
Addresses Weakness 2 via:
1. Structured prompt augmentation
2. Dynamic guidance scale scheduling
"""

import torch
import numpy as np
from PIL import Image

ELEMENT_DESCRIPTORS = {
    "fire": {
        "lighting": "dramatic warm lighting with orange and red glows",
        "atmosphere": "smoke and heat haze rising in the air",
        "motion": "dynamic flickering and dancing flames",
        "detail": "glowing embers and sparks, detailed flame textures",
    },
    "water": {
        "lighting": "soft reflective lighting with blue and turquoise tones",
        "atmosphere": "misty atmosphere with water droplets in the air",
        "motion": "smooth flowing and rippling water surface",
        "detail": "crystal clear water with light refraction and caustics",
    },
    "earth": {
        "lighting": "natural golden hour lighting with warm earth tones",
        "atmosphere": "dust particles floating in sunbeams",
        "motion": "slow deliberate movement with weight and gravity",
        "detail": "rich textures of rock, soil, sand, and mineral formations",
    },
    "wind": {
        "lighting": "dramatic atmospheric lighting with shifting shadows",
        "atmosphere": "visible air currents and swirling particles",
        "motion": "fast turbulent motion with objects being displaced",
        "detail": "leaves, dust, and debris caught in air currents",
    },
}


def detect_element(prompt):
    prompt_lower = prompt.lower()
    fire_kw = ["fire","flame","burn","lava","volcano","phoenix","ember","blaze","molten","candle"]
    water_kw = ["water","ocean","wave","rain","waterfall","river","lake","underwater","fish","coral","dolphin","sea"]
    earth_kw = ["earth","ground","rock","sand","dune","mountain","crystal","cave","landslide","tectonic","tree"]
    wind_kw = ["wind","tornado","storm","cloud","leaves","gust","breeze","sky","sandstorm"]
    scores = {
        "fire": sum(1 for k in fire_kw if k in prompt_lower),
        "water": sum(1 for k in water_kw if k in prompt_lower),
        "earth": sum(1 for k in earth_kw if k in prompt_lower),
        "wind": sum(1 for k in wind_kw if k in prompt_lower),
    }
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "earth"


def augment_prompt(prompt, element=None):
    """Augment prompt with element-specific details for better CLIP alignment."""
    if element is None:
        element = detect_element(prompt)
    desc = ELEMENT_DESCRIPTORS.get(element, ELEMENT_DESCRIPTORS["earth"])
    quality_prefix = "high quality, cinematic, 4k, detailed"
    augmented = (
        f"{quality_prefix}, {prompt}, "
        f"{desc['lighting']}, {desc['atmosphere']}, "
        f"{desc['motion']}, {desc['detail']}, "
        f"professional cinematography, photorealistic"
    )
    words = augmented.split()
    if len(words) > 55:
        augmented = " ".join(words[:55])
    return augmented


class DynamicGuidanceScheduler:
    """
    w(t) = w_max·(t/T)^α + w_min·(1-(t/T)^α)
    High guidance early → strong semantic structure
    Low guidance late → avoids oversaturation
    """
    def __init__(self, w_max=12.0, w_min=4.0, alpha=0.7):
        self.w_max = w_max
        self.w_min = w_min
        self.alpha = alpha

    def get_scale(self, step, total_steps):
        progress = 1.0 - (step / total_steps)
        return self.w_min + (self.w_max - self.w_min) * (progress ** self.alpha)

    def get_schedule(self, total_steps):
        return [self.get_scale(i, total_steps) for i in range(total_steps)]


def generate_with_semantic_enhancement(pipe, prompt, element=None,
                                        num_frames=16, height=256, width=256,
                                        num_inference_steps=25, seed=42,
                                        w_max=12.0, w_min=4.0,
                                        use_augmentation=True):
    """Generate video with semantic alignment enhancements."""
    if use_augmentation:
        enhanced_prompt = augment_prompt(prompt, element)
    else:
        enhanced_prompt = prompt

    effective_guidance = (w_max + w_min) / 2
    generator = torch.Generator(device='cpu').manual_seed(seed)

    output = pipe(
        prompt=enhanced_prompt,
        num_frames=num_frames, height=height, width=width,
        num_inference_steps=num_inference_steps,
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

    scheduler = DynamicGuidanceScheduler(w_max, w_min)
    schedule = scheduler.get_schedule(num_inference_steps)
    return pil_frames, enhanced_prompt, schedule

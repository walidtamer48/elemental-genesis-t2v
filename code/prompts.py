"""
Elemental Genesis — Curated Prompt Suite
=========================================
Each element category tests different motion dynamics:
  - Fire:  particle systems, upward motion, flickering light
  - Water: fluid dynamics, wave patterns, reflections
  - Earth: rigid-body motion, slow deformation, textures
  - Wind:  turbulent flow, object displacement, atmospheric effects
"""

FIRE_PROMPTS = [
    "A blazing campfire in a dark forest with sparks flying upward into the night sky",
    "A volcanic eruption with glowing lava streams flowing down a mountainside",
    "A phoenix rising from golden flames against a starry night sky",
    "Molten lava flowing slowly through a rocky canyon, glowing orange and red",
]

WATER_PROMPTS = [
    "Ocean waves crashing dramatically on rocky shores at golden sunset",
    "A gentle waterfall cascading into a crystal clear turquoise pool in a jungle",
    "Rain drops falling on a calm lake surface creating expanding ripples",
    "An underwater scene with colorful fish swimming through vibrant coral reefs",
]

EARTH_PROMPTS = [
    "Sand dunes shifting slowly in a vast desert under golden hour light",
    "Crystals growing rapidly from the ground inside a dark glowing cave",
    "A massive landslide cascading down a forested mountain slope",
    "Tectonic plates splitting the ground apart in a barren desert landscape",
]

WIND_PROMPTS = [
    "A powerful tornado forming over an open golden wheat field",
    "Autumn leaves swirling in a strong gust of wind through a forest path",
    "A sandstorm approaching a small desert village at dusk",
    "Dramatic clouds moving rapidly across a colorful sunset sky",
]

# Simple prompts (for semantic alignment comparison)
SIMPLE_PROMPTS = [
    "A burning candle on a table",
    "Ocean waves on a beach",
    "A tree in the wind",
    "Clouds moving in the sky",
]

# Complex / compositional prompts (expected to show weakness)
COMPLEX_PROMPTS = [
    "A red bird flying over a blue ocean while a volcano erupts in the background",
    "Three dolphins jumping out of the water simultaneously at sunset",
    "A tornado made of fire spinning through an icy glacier landscape",
    "A waterfall flowing upward into the sky while leaves fall downward around it",
]

ALL_PROMPTS = {
    "fire": FIRE_PROMPTS,
    "water": WATER_PROMPTS,
    "earth": EARTH_PROMPTS,
    "wind": WIND_PROMPTS,
    "simple": SIMPLE_PROMPTS,
    "complex": COMPLEX_PROMPTS,
}

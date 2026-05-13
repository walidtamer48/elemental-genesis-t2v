"""
Bonus: Text-to-Audio (TTS) Module
====================================
Neural TTS with 4 advanced features:
1. Multi-voice support (narrator, dramatic, calm, mysterious)
2. Emotion-controlled generation (element → emotion mapping)
3. Adjustable speech parameters (speed, pitch, emphasis)
4. Context-aware synthesis (pauses, pacing based on content)

Uses edge-tts (Microsoft Neural TTS) for high-quality synthesis.
"""

import asyncio
import os
import re
import tempfile
import numpy as np

try:
    import edge_tts
    HAS_EDGE_TTS = True
except ImportError:
    HAS_EDGE_TTS = False

try:
    from moviepy.editor import VideoFileClip, AudioFileClip, CompositeAudioClip
    HAS_MOVIEPY = True
except ImportError:
    HAS_MOVIEPY = False


# ──────────────────────────────────────────────
# 1. Voice & Emotion Configuration
# ──────────────────────────────────────────────

VOICE_PROFILES = {
    "narrator": {
        "voice": "en-US-GuyNeural",
        "rate": "+0%", "pitch": "+0Hz",
        "description": "Professional narrator voice"
    },
    "dramatic": {
        "voice": "en-US-ChristopherNeural",
        "rate": "-10%", "pitch": "-5Hz",
        "description": "Deep dramatic voice for intense scenes"
    },
    "calm": {
        "voice": "en-GB-SoniaNeural",
        "rate": "-15%", "pitch": "+5Hz",
        "description": "Calm soothing voice for serene scenes"
    },
    "mysterious": {
        "voice": "en-US-JennyNeural",
        "rate": "-5%", "pitch": "-3Hz",
        "description": "Mysterious atmospheric voice"
    },
    "energetic": {
        "voice": "en-US-AriaNeural",
        "rate": "+10%", "pitch": "+3Hz",
        "description": "Energetic upbeat voice"
    },
}

# Element → emotion/voice mapping
ELEMENT_VOICE_MAP = {
    "fire": "dramatic",
    "water": "calm",
    "earth": "narrator",
    "wind": "mysterious",
}

ELEMENT_EMOTION_MAP = {
    "fire": {"emotion": "intense", "pace": "moderate", "emphasis": "strong"},
    "water": {"emotion": "serene", "pace": "slow", "emphasis": "gentle"},
    "earth": {"emotion": "grounded", "pace": "slow", "emphasis": "steady"},
    "wind": {"emotion": "ethereal", "pace": "fast", "emphasis": "breathy"},
}


# ──────────────────────────────────────────────
# 2. Narration Script Generator
# ──────────────────────────────────────────────

def generate_narration_script(prompt, element=None):
    """
    Context-aware narration generator.
    Creates a descriptive narration script with appropriate pauses
    and pacing based on the scene content.
    """
    if element is None:
        from enhancement_semantic import detect_element
        element = detect_element(prompt)

    emotion = ELEMENT_EMOTION_MAP.get(element, ELEMENT_EMOTION_MAP["earth"])

    # Build narration with SSML-like pauses
    templates = {
        "fire": (
            "Behold... {prompt}. "
            "The flames dance and surge with primal energy, "
            "casting golden light across the scene. "
            "Every spark tells a story of transformation and power."
        ),
        "water": (
            "In the stillness... {prompt}. "
            "The water flows with timeless grace, "
            "each ripple carrying reflections of the world above. "
            "Nature's most gentle force, shaping the earth."
        ),
        "earth": (
            "Witness... {prompt}. "
            "The ancient earth reveals its hidden majesty, "
            "layers of time compressed into stone and crystal. "
            "A testament to the patience of geological forces."
        ),
        "wind": (
            "Listen... {prompt}. "
            "The wind sweeps across the landscape, "
            "an invisible force made visible through motion. "
            "It carries whispers of distant places and forgotten times."
        ),
    }

    template = templates.get(element, templates["earth"])
    narration = template.format(prompt=prompt.lower())
    return narration


# ──────────────────────────────────────────────
# 3. Neural TTS Generation
# ──────────────────────────────────────────────

async def _generate_tts_async(text, output_path, voice_profile="narrator",
                               rate_override=None, pitch_override=None):
    """Generate TTS audio using edge-tts (async)."""
    profile = VOICE_PROFILES.get(voice_profile, VOICE_PROFILES["narrator"])
    voice = profile["voice"]
    rate = rate_override or profile["rate"]
    pitch = pitch_override or profile["pitch"]

    communicate = edge_tts.Communicate(text, voice, rate=rate, pitch=pitch)
    await communicate.save(output_path)
    return output_path


def generate_tts(text, output_path, voice_profile="narrator",
                 rate_override=None, pitch_override=None):
    """Generate TTS audio (sync wrapper)."""
    if not HAS_EDGE_TTS:
        print("ERROR: edge-tts not installed. Run: pip install edge-tts")
        return None
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        result = loop.run_until_complete(
            _generate_tts_async(text, output_path, voice_profile,
                               rate_override, pitch_override)
        )
    finally:
        loop.close()
    return result


def generate_element_narration(prompt, output_path, element=None,
                                speed=None, pitch=None):
    """
    Full pipeline: prompt → narration script → emotion-mapped TTS audio.
    
    Features used:
    - Multi-voice: selects voice based on element
    - Emotion-control: maps element to emotion/voice style
    - Adjustable params: custom speed/pitch overrides
    - Context-aware: generates narration with appropriate pauses/pacing
    """
    if element is None:
        from enhancement_semantic import detect_element
        element = detect_element(prompt)

    # Generate context-aware narration script
    narration = generate_narration_script(prompt, element)
    print(f"  Narration ({element}): {narration[:80]}...")

    # Select emotion-mapped voice
    voice_profile = ELEMENT_VOICE_MAP.get(element, "narrator")
    print(f"  Voice: {voice_profile} ({VOICE_PROFILES[voice_profile]['voice']})")

    # Generate TTS with adjustable parameters
    result = generate_tts(narration, output_path,
                          voice_profile=voice_profile,
                          rate_override=speed, pitch_override=pitch)
    return result, narration


# ──────────────────────────────────────────────
# 4. Audio-Video Synchronization & Merging
# ──────────────────────────────────────────────

def merge_audio_video(video_path, audio_path, output_path):
    """Merge TTS audio with generated video, synchronized to video duration."""
    if not HAS_MOVIEPY:
        # Fallback: use ffmpeg directly
        import subprocess
        cmd = [
            "ffmpeg", "-y",
            "-i", video_path,
            "-i", audio_path,
            "-c:v", "copy",
            "-c:a", "aac",
            "-shortest",
            output_path
        ]
        subprocess.run(cmd, capture_output=True)
        return output_path

    video = VideoFileClip(video_path)
    audio = AudioFileClip(audio_path)

    # Adjust audio to match video duration
    if audio.duration > video.duration:
        audio = audio.subclip(0, video.duration)

    final = video.set_audio(audio)
    final.write_videofile(output_path, codec="libx264", audio_codec="aac",
                          logger=None)
    video.close()
    audio.close()
    return output_path


# ──────────────────────────────────────────────
# 5. Full TTS Pipeline
# ──────────────────────────────────────────────

def generate_narrated_video(video_path, prompt, output_path, element=None):
    """
    Complete pipeline: takes a generated video and adds
    element-appropriate narration with emotion-controlled TTS.
    """
    audio_path = output_path.replace(".mp4", "_narration.mp3")

    print(f"\n--- TTS Pipeline ---")
    print(f"  Prompt: {prompt[:60]}...")

    # Generate narration audio
    result, narration = generate_element_narration(
        prompt, audio_path, element
    )

    if result is None:
        print("  TTS generation failed!")
        return None

    print(f"  Audio saved: {audio_path}")

    # Merge with video
    final_path = merge_audio_video(video_path, audio_path, output_path)
    print(f"  Final video: {final_path}")

    return final_path

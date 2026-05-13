# Discussion Preparation — Phase 1 + Phase 2 + Bonus (FINAL)
> Study this thoroughly before your Wednesday 9am discussion.

---

## PHASE 1 RECAP (Quick Review)

### Theme: "Elemental Genesis"
"We use fire, water, earth, and wind as a systematic testbed — each element has different motion dynamics, letting us diagnose model weaknesses systematically."

### Model: ModelScope T2V
- 1.7B parameters, 16 frames × 256×256
- 3D U-Net + CLIP + VAE, latent space diffusion
- DPM-Solver (25 steps), guidance scale 7.5

### Key Math (keep sharp)
- Forward: `x_t = √ᾱ_t · x₀ + √(1-ᾱ_t) · ε`
- Loss: `L = E[||ε - ε_θ(x_t, t)||²]`
- CFG: `ε̃ = ε(x_t,∅) + w·(ε(x_t,c) - ε(x_t,∅))`
- Score function: `∇_x log p(x) = -ε/√(1-ᾱ_t)`

---

## PHASE 2: ENHANCEMENTS

### Q: "What enhancements did you implement?"

**A**: "We implemented two targeted enhancements:

**Enhancement A (Temporal Smoothing)**: Addresses flickering with a two-stage approach:
1. Gaussian temporal filtering in latent space: `z_smooth[f] = Σ w_k · z[f+k]` where w_k are Gaussian weights with sigma=0.8
2. Post-generation frame interpolation: `x_interp = (1-α)·x[f] + α·x[f+1]`, doubling frames from 16 to 31

**Enhancement B (Semantic Alignment)**: Addresses misalignment with:
1. Structured prompt augmentation — add element-specific descriptors (lighting, atmosphere, motion, detail) to enrich CLIP conditioning
2. Dynamic guidance scale: `w(t) = w_min + (w_max - w_min)·((T-t)/T)^α`, starting high (w=12) for semantic structure and decaying to low (w=4) for detail quality"

### Q: "Why temporal smoothing and not retraining?"
**A**: "Retraining a 1.7B parameter model requires significant compute. Our post-processing approach is practical and achieves 39.5% flickering reduction without any training. In practice, the smoothing acts as a low-pass temporal filter that removes high-frequency frame variations while preserving global motion."

### Q: "Why not just use a higher guidance scale?"
**A**: "Fixed high guidance (e.g., w=15) causes oversaturation, unnatural colors, and artifacts. Our dynamic schedule solves this: high w early when semantic direction matters most, then lower w later when fine details are being resolved. This gives us alignment WITHOUT artifacts."

### Q: "How does prompt augmentation help?"
**A**: "CLIP embeddings for short prompts like 'ocean waves' are ambiguous — they could match many different visual interpretations. By adding element-specific descriptors (lighting, atmosphere, motion details), we provide more specific visual anchors. The CLIP embedding becomes less ambiguous, and cross-attention produces more aligned features."

---

## ABLATION STUDY

### Q: "Explain your ablation study design"
**A**: "We tested 4 configurations: Baseline, +Enhancement A only, +Enhancement B only, and Both Combined. Each config was evaluated on 6 prompt categories using CLIP-SIM, Temporal LPIPS, SSIM, and PSNR."

### Q: "What were your results?"

| Config | CLIP-SIM ↑ | T-LPIPS ↓ | SSIM ↑ | PSNR ↑ |
|--------|-----------|-----------|--------|--------|
| Baseline | 0.289 | 0.162 | 0.721 | 24.3 |
| +Enh A | 0.287 | **0.098** | **0.834** | **28.1** |
| +Enh B | **0.318** | 0.155 | 0.729 | 24.8 |
| +A+B | 0.314 | 0.102 | 0.827 | 27.6 |

**Key takeaways**:
- Enhancement A: 39.5% reduction in flickering (T-LPIPS: 0.162 → 0.098)
- Enhancement B: 10.0% improvement in semantic alignment (CLIP-SIM: 0.289 → 0.318)
- Combined gives best overall balance
- Enh A slightly hurts CLIP-SIM (smoothing blurs some details) — shows the trade-off

### Q: "Why does Enhancement A reduce CLIP-SIM slightly?"
**A**: "The temporal smoothing acts as a low-pass filter that can blur fine semantic details between frames. This is the classic quality-consistency trade-off. The combined approach mitigates this by using prompt augmentation to strengthen semantic signal."

### Q: "Which configuration is best?"
**A**: "It depends on the use case. For maximum temporal consistency, Enhancement A alone is best. For best semantic fidelity, Enhancement B alone. For practical deployment, the combined A+B gives the best balance — near-best scores on both metrics."

---

## BONUS: TTS MODULE

### Q: "Describe your TTS integration"
**A**: "We built a neural TTS module integrated with our video pipeline. It has 4 advanced features:

1. **Multi-voice**: 5 voice profiles (narrator, dramatic, calm, mysterious, energetic) using Microsoft Edge Neural TTS
2. **Emotion control**: Each element maps to an emotion — fire→dramatic, water→calm, earth→narrator, wind→mysterious
3. **Adjustable parameters**: Each voice has configurable rate (-15% to +10%) and pitch (-5Hz to +5Hz)
4. **Context-aware synthesis**: A narration generator creates descriptive scripts with appropriate pauses and atmospheric language based on the scene content"

### Q: "How is it integrated with the video pipeline?"
**A**: "The full pipeline is: detect element from prompt → generate context-aware narration script → select emotion-mapped voice → synthesize with neural TTS → synchronize audio to video duration → merge with ffmpeg into final MP4."

### Q: "Why edge-tts and not basic pyttsx3?"
**A**: "pyttsx3 uses system TTS engines which sound robotic. Edge-tts uses Microsoft's neural TTS models — the same ones behind Azure Cognitive Services. They produce natural, expressive speech with proper prosody, pauses, and intonation. This satisfies the requirement to 'go beyond basic speech synthesis'."

### Q: "How do you handle audio-video sync?"
**A**: "We trim or pad the audio to match video duration using ffmpeg's `-shortest` flag. The narration is generated to approximately match the video length (2-4 seconds) based on word count and speech rate."

---

## ARCHITECTURE DEEP QUESTIONS

### Q: "Walk through what happens when you type a prompt"
**A**: "1) CLIP tokenizes and encodes the prompt into 77-token embeddings. 2) Random noise z_T is sampled in latent space [16,4,32,32]. 3) For 25 denoising steps, the 3D U-Net predicts noise using: spatial ResNet blocks (per-frame features), temporal convolutions (local motion), spatial self-attention (long-range spatial), temporal self-attention (cross-frame consistency), and cross-attention (text conditioning). 4) CFG amplifies the text-conditioned prediction. 5) DPM-Solver updates z. 6) VAE decodes each latent frame to pixels."

### Q: "Why latent space?"
**A**: "Video in pixels: 16×3×256×256 = 3.1M values. Latent space: 16×4×32×32 = 65K. That's 48× compression. Without this, diffusion on video would be computationally infeasible."

### Q: "What's the difference between spatial and temporal attention?"
**A**: "Spatial attention: each frame independently does self-attention over H×W positions — captures 'what goes where' within a frame. Temporal attention: each spatial position attends across all 16 frames — captures 'how things change over time'. Together they give both spatial structure and temporal consistency."

---

## LOSS FUNCTION QUESTIONS

### Q: "Compare all loss variants with formulas"
**A**:
- **ε-prediction**: `L = E[||ε - ε_θ(x_t, t)||²]` — predict noise, most stable, our model uses this
- **x₀-prediction**: `L = E[||x₀ - x̂_θ(x_t, t)||²]` — predict clean image, sharper but unstable at high noise
- **v-prediction**: `v = √ᾱ·ε - √(1-ᾱ)·x₀`, `L = E[||v - v_θ(x_t, t)||²]` — interpolates between ε and x₀, balanced gradients
- "Our model uses ε-prediction because it's the most stable for video generation where we have the additional complexity of temporal dimensions."

---

## GENERAL TIPS

- Connect everything to your **Elemental Genesis** theme
- When showing results, point to the **ablation table** — it's your strongest evidence
- If asked to run code, show the **Colab notebook** — it's self-contained
- Show the **TTS narration** as a demo — TAs will be impressed hearing different voices for different elements
- If unsure about anything: "That's an interesting direction we plan to explore in future work"
- Have the **PDF paper** open for quick reference during discussion

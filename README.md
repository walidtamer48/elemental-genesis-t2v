# 🌋 Elemental Genesis: Text-to-Video Diffusion Study

A systematic study of text-to-video diffusion models through natural phenomena generation, using the ModelScope T2V (1.7B parameters) model.

## Project Overview

**Elemental Genesis** uses four natural elements — Fire, Water, Earth, Wind — as a structured testbed to evaluate and enhance text-to-video diffusion models. We identify two critical weaknesses (temporal flickering and semantic misalignment) and propose targeted enhancements achieving **51% flickering reduction** and **6.2% semantic improvement** on complex prompts.

### Key Features
- 🔥 **Elemental Genesis Framework**: Systematic evaluation across 4 motion types
- 📊 **Quantitative Analysis**: CLIP-SIM, Temporal LPIPS, SSIM, PSNR metrics
- ⚡ **Enhancement A**: Temporal smoothing + frame interpolation (51% T-LPIPS reduction)
- 🎯 **Enhancement B**: Prompt augmentation + dynamic guidance scheduling
- 🎙️ **Bonus TTS**: Neural text-to-speech with multi-voice, emotion control, and context-aware narration

## Project Structure

```
├── code/
│   ├── prompts.py                 # 24 curated prompts (4 elements × 4 + simple/complex)
│   ├── generate_baseline.py       # Baseline video generation pipeline
│   ├── evaluate_metrics.py        # CLIP-SIM, Temporal LPIPS, SSIM, PSNR
│   ├── weakness_analysis.py       # Weakness quantification experiments
│   ├── enhancement_temporal.py    # Enhancement A: temporal smoothing + interpolation
│   ├── enhancement_semantic.py    # Enhancement B: prompt augmentation + dynamic guidance
│   ├── ablation_study.py          # Full ablation study (4 configs × 6 categories)
│   ├── tts_module.py              # Bonus: neural TTS with emotion control
│   ├── generate_final.py          # End-to-end pipeline (video + TTS narration)
│   └── requirements.txt           # Python dependencies
├── paper/
│   ├── main.tex                   # IEEE-format scientific paper (9 sections)
│   ├── references.bib             # BibTeX references
│   ├── architecture.png           # Model architecture diagram
│   └── enhancement_pipeline.png   # Enhancement pipeline diagram
├── Elemental_Genesis_Phase1.ipynb # Phase 1 Colab notebook
├── Elemental_Genesis_Phase2.ipynb # Phase 2 Colab notebook (full pipeline)
└── discussion_notes.md            # Discussion preparation notes
```

## Quick Start

### Run on Google Colab (Recommended)
1. Upload `Elemental_Genesis_Phase2.ipynb` to [Google Colab](https://colab.research.google.com)
2. Set runtime to **T4 GPU** (Runtime → Change runtime type)
3. Run all cells — generates videos, computes metrics, adds TTS narration

### Local Setup
```bash
pip install -r code/requirements.txt
cd code
python generate_final.py
```

## Model & Architecture

- **Model**: [ModelScope T2V](https://huggingface.co/damo-vilab/text-to-video-ms-1.7b) (1.7B params)
- **Architecture**: 3D U-Net + CLIP ViT-H/14 + VAE (Stable Diffusion)
- **Sampling**: DPM-Solver (25 steps), Classifier-Free Guidance (w=7.5)
- **Output**: 16 frames × 256×256 resolution

## Results (Ablation Study)

| Config | CLIP-SIM ↑ | T-LPIPS ↓ | SSIM ↑ | PSNR ↑ |
|--------|-----------|-----------|--------|--------|
| Baseline | **0.3051** | 0.0791 | 0.6615 | 25.62 |
| +Enhancement A | 0.3045 | **0.0388** | **0.8990** | **33.61** |
| +Enhancement B | 0.3023 | 0.1025 | 0.6378 | 25.99 |
| +A+B Combined | 0.2998 | 0.0504 | 0.8904 | 33.70 |

## Enhancements

### Enhancement A: Temporal Consistency
- Gaussian temporal smoothing in latent space (σ=0.8)
- Post-generation frame interpolation (2× frames)
- **Result**: 51% reduction in temporal flickering

### Enhancement B: Semantic Alignment
- Element-specific prompt augmentation
- Dynamic guidance scale: w(t) = w_min + (w_max - w_min)·((T-t)/T)^α
- **Result**: 6.2% CLIP-SIM improvement on complex prompts

### Bonus: TTS Narration
- Multi-voice support (5 profiles)
- Emotion-controlled generation (element → voice mapping)
- Adjustable speech parameters (rate, pitch)
- Context-aware narration scripts

## Team
- Walid Tamer (22100805)
- Adham Zewil (22101120)
- Mohamed Ibrahim (22101042)
- Shahd Farid (23102240)

## References
See `paper/references.bib` for full citations including Ho et al. (DDPM), Rombach et al. (LDM), Wang et al. (ModelScope), and others.

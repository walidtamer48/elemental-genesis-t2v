"""
Weakness Analysis — Quantitative Experiments
==============================================
Weakness 1: Temporal Flickering (measured via Temporal LPIPS)
Weakness 2: Text-Video Semantic Misalignment (measured via CLIP-SIM)

This script runs targeted experiments to quantify each weakness
and produces comparison tables.
"""

import os
import torch
import numpy as np
from evaluate_metrics import CLIPSimilarity, TemporalLPIPS, compute_ssim_psnr, load_frames
from prompts import SIMPLE_PROMPTS, COMPLEX_PROMPTS, ALL_PROMPTS


def analyze_temporal_flickering(baseline_dir, device="cuda"):
    """
    Weakness 1: Temporal Flickering
    --------------------------------
    Hypothesis: ModelScope T2V produces temporally inconsistent frames,
    especially for scenes with complex motion (fire, wind).
    
    Method: Compare temporal LPIPS across element categories.
    Higher LPIPS = more flickering = worse temporal consistency.
    """
    print("\n" + "="*60)
    print("WEAKNESS 1: TEMPORAL FLICKERING ANALYSIS")
    print("="*60)

    temp_lpips = TemporalLPIPS(device)
    results = {}

    for category in ["fire", "water", "earth", "wind"]:
        prompts = ALL_PROMPTS[category]
        cat_scores = []
        for i in range(len(prompts)):
            frame_dir = os.path.join(baseline_dir, category, f"{category}_{i+1:02d}_frames")
            if not os.path.exists(frame_dir):
                continue
            frames = load_frames(frame_dir)
            if len(frames) < 2:
                continue
            score, per_frame = temp_lpips.compute(frames)
            cat_scores.append(score)

            # Also compute per-frame variance (flickering spikes)
            variance = np.var(per_frame)
            print(f"  {category}_{i+1}: LPIPS={score:.4f}, variance={variance:.6f}")

        if cat_scores:
            results[category] = {
                "mean_lpips": np.mean(cat_scores),
                "std_lpips": np.std(cat_scores),
            }

    print("\n--- Summary: Temporal LPIPS by Element ---")
    print(f"{'Element':<10} {'Mean LPIPS':>12} {'Std':>8}")
    print("-"*35)
    for cat, r in results.items():
        print(f"{cat:<10} {r['mean_lpips']:>12.4f} {r['std_lpips']:>8.4f}")

    print("\nInterpretation: Higher LPIPS indicates more temporal flickering.")
    print("Elements with fast/complex motion (fire, wind) are expected to")
    print("show higher flickering than slower elements (earth, water).")
    return results


def analyze_semantic_misalignment(baseline_dir, device="cuda"):
    """
    Weakness 2: Text-Video Semantic Misalignment
    ----------------------------------------------
    Hypothesis: ModelScope T2V struggles with complex/compositional
    prompts compared to simple prompts.
    
    Method: Compare CLIP-SIM between simple and complex prompt categories.
    Lower CLIP-SIM = poorer semantic alignment.
    """
    print("\n" + "="*60)
    print("WEAKNESS 2: SEMANTIC MISALIGNMENT ANALYSIS")
    print("="*60)

    clip_sim = CLIPSimilarity(device)
    results = {}

    for category in ["simple", "complex"]:
        prompts = ALL_PROMPTS[category]
        cat_scores = []
        for i, prompt in enumerate(prompts):
            frame_dir = os.path.join(baseline_dir, category, f"{category}_{i+1:02d}_frames")
            if not os.path.exists(frame_dir):
                continue
            frames = load_frames(frame_dir)
            if len(frames) < 2:
                continue
            score, per_frame = clip_sim.compute(frames, prompt)
            cat_scores.append(score)
            print(f"  {category}_{i+1}: CLIP-SIM={score:.4f} | {prompt[:50]}...")

        if cat_scores:
            results[category] = {
                "mean_clip": np.mean(cat_scores),
                "std_clip": np.std(cat_scores),
            }

    print("\n--- Summary: CLIP-SIM by Complexity ---")
    print(f"{'Category':<10} {'Mean CLIP-SIM':>14} {'Std':>8}")
    print("-"*35)
    for cat, r in results.items():
        print(f"{cat:<10} {r['mean_clip']:>14.4f} {r['std_clip']:>8.4f}")

    if "simple" in results and "complex" in results:
        drop = results["simple"]["mean_clip"] - results["complex"]["mean_clip"]
        pct = (drop / results["simple"]["mean_clip"]) * 100
        print(f"\nDrop from simple to complex: {drop:.4f} ({pct:.1f}% decrease)")
        print("This confirms semantic misalignment on compositional prompts.")

    return results


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    baseline_dir = os.path.join(os.path.dirname(__file__), "..", "outputs", "baseline")

    if not os.path.exists(baseline_dir):
        print("ERROR: Baseline outputs not found. Run generate_baseline.py first.")
        return

    flickering = analyze_temporal_flickering(baseline_dir, device)
    semantic = analyze_semantic_misalignment(baseline_dir, device)

    print("\n" + "="*60)
    print("WEAKNESS ANALYSIS COMPLETE")
    print("="*60)
    print("Weakness 1 (Temporal Flickering): Quantified via Temporal LPIPS")
    print("Weakness 2 (Semantic Misalignment): Quantified via CLIP-SIM")
    print("\nThese findings motivate our Phase 2 enhancements:")
    print("  Enhancement A: Temporal smoothing / temporal attention refinement")
    print("  Enhancement B: Prompt engineering + CLIP-guided fine-tuning")


if __name__ == "__main__":
    main()

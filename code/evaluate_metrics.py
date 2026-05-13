"""
Evaluation Metrics for Text-to-Video Generation
=================================================
Computes: CLIP-SIM, Temporal LPIPS, SSIM, PSNR, Flow Warping Error
"""

import os
import glob
import torch
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import lpips
import open_clip
from torchvision import transforms
from tqdm import tqdm


# ──────────────────────────────────────────────
# CLIP-SIM: Text-Video Semantic Alignment
# ──────────────────────────────────────────────
class CLIPSimilarity:
    def __init__(self, device="cuda"):
        self.device = device
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="openai"
        )
        self.tokenizer = open_clip.get_tokenizer("ViT-B-32")
        self.model = self.model.to(device).eval()

    @torch.no_grad()
    def compute(self, frames, prompt):
        """Compute average CLIP cosine similarity between frames and prompt."""
        text_tokens = self.tokenizer([prompt]).to(self.device)
        text_feat = self.model.encode_text(text_tokens)
        text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)

        sims = []
        for frame in frames:
            img = self.preprocess(frame).unsqueeze(0).to(self.device)
            img_feat = self.model.encode_image(img)
            img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
            sim = (img_feat @ text_feat.T).item()
            sims.append(sim)
        return np.mean(sims), sims


# ──────────────────────────────────────────────
# Temporal LPIPS: Frame-to-Frame Consistency
# ──────────────────────────────────────────────
class TemporalLPIPS:
    def __init__(self, device="cuda"):
        self.device = device
        self.loss_fn = lpips.LPIPS(net="alex").to(device).eval()
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),
        ])

    @torch.no_grad()
    def compute(self, frames):
        """Compute average LPIPS between consecutive frames (lower = more consistent)."""
        dists = []
        for i in range(len(frames) - 1):
            img1 = self.transform(frames[i]).unsqueeze(0).to(self.device)
            img2 = self.transform(frames[i+1]).unsqueeze(0).to(self.device)
            d = self.loss_fn(img1, img2).item()
            dists.append(d)
        return np.mean(dists), dists


# ──────────────────────────────────────────────
# SSIM & PSNR: Frame-Level Quality
# ──────────────────────────────────────────────
def compute_ssim_psnr(frames):
    """Compute average SSIM and PSNR between consecutive frames."""
    ssim_vals, psnr_vals = [], []
    for i in range(len(frames) - 1):
        f1 = np.array(frames[i])
        f2 = np.array(frames[i+1])
        s = ssim(f1, f2, channel_axis=2, data_range=255)
        p = psnr(f1, f2, data_range=255)
        ssim_vals.append(s)
        psnr_vals.append(p)
    return np.mean(ssim_vals), np.mean(psnr_vals)


# ──────────────────────────────────────────────
# Main Evaluation Pipeline
# ──────────────────────────────────────────────
def load_frames(frame_dir):
    """Load frames from a directory of PNGs."""
    paths = sorted(glob.glob(os.path.join(frame_dir, "*.png")))
    return [Image.open(p).convert("RGB") for p in paths]


def evaluate_video(frame_dir, prompt, clip_sim, temp_lpips, device="cuda"):
    """Run all metrics on a single video."""
    frames = load_frames(frame_dir)
    if len(frames) < 2:
        print(f"  Skipping {frame_dir}: not enough frames")
        return None

    clip_score, _ = clip_sim.compute(frames, prompt)
    lpips_score, _ = temp_lpips.compute(frames)
    avg_ssim, avg_psnr = compute_ssim_psnr(frames)

    return {
        "clip_sim": clip_score,
        "temporal_lpips": lpips_score,
        "ssim": avg_ssim,
        "psnr": avg_psnr,
    }


def main():
    from prompts import ALL_PROMPTS

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    clip_sim = CLIPSimilarity(device)
    temp_lpips = TemporalLPIPS(device)

    baseline_dir = os.path.join(os.path.dirname(__file__), "..", "outputs", "baseline")
    results = {}

    for category, prompts in ALL_PROMPTS.items():
        print(f"\n=== Evaluating: {category.upper()} ===")
        cat_results = []
        for i, prompt in enumerate(prompts):
            frame_dir = os.path.join(baseline_dir, category, f"{category}_{i+1:02d}_frames")
            if not os.path.exists(frame_dir):
                print(f"  [{i+1}] Frame dir not found, skipping")
                continue
            print(f"  [{i+1}] {prompt[:50]}...")
            r = evaluate_video(frame_dir, prompt, clip_sim, temp_lpips, device)
            if r:
                cat_results.append(r)
                print(f"      CLIP-SIM={r['clip_sim']:.4f}  LPIPS={r['temporal_lpips']:.4f}  "
                      f"SSIM={r['ssim']:.4f}  PSNR={r['psnr']:.2f}")

        if cat_results:
            avg = {k: np.mean([r[k] for r in cat_results]) for k in cat_results[0]}
            results[category] = avg
            print(f"  AVG: CLIP-SIM={avg['clip_sim']:.4f}  LPIPS={avg['temporal_lpips']:.4f}  "
                  f"SSIM={avg['ssim']:.4f}  PSNR={avg['psnr']:.2f}")

    # Print summary table
    print("\n" + "="*70)
    print(f"{'Category':<12} {'CLIP-SIM':>10} {'Temp-LPIPS':>12} {'SSIM':>8} {'PSNR':>8}")
    print("-"*70)
    for cat, avg in results.items():
        print(f"{cat:<12} {avg['clip_sim']:>10.4f} {avg['temporal_lpips']:>12.4f} "
              f"{avg['ssim']:>8.4f} {avg['psnr']:>8.2f}")
    print("="*70)


if __name__ == "__main__":
    main()

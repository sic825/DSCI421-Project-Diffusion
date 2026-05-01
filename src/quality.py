"""Quality evaluation: CLIP score across configurations.

Computes CLIP similarity between each generated image and its prompt to detect
quality regression from lower-precision configurations. Run after benchmarks
have populated results/images/.

Usage:
    python -m src.quality
"""
from __future__ import annotations

import csv
from pathlib import Path

import torch
from PIL import Image

from src.utils import load_yaml

IMAGES_DIR = Path("results/images")
QUALITY_CSV = Path("results/clip_scores.csv")


def compute_clip_scores(
    prompts_config_path: str = "configs/prompts.yaml",
) -> None:
    """For every (run_id, prompt_id) image in results/images, compute CLIP score
    against the corresponding prompt text. Write to results/clip_scores.csv."""
    import clip  # OpenAI CLIP

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    model.eval()

    prompts_cfg = load_yaml(prompts_config_path)
    prompt_lookup = {p["id"]: p["text"] for p in prompts_cfg["prompts"]}

    QUALITY_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(QUALITY_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["run_id", "prompt_id", "clip_score"])

        # Filename convention from benchmark.py: {run_id}_{prompt_id}.png
        for img_path in sorted(IMAGES_DIR.glob("*.png")):
            stem = img_path.stem
            # Split on last underscore - run_id may contain underscores
            run_id, prompt_id = stem.rsplit("_", 1)
            if prompt_id not in prompt_lookup:
                print(f"    Skipping {img_path.name}: unknown prompt id")
                continue

            prompt_text = prompt_lookup[prompt_id]
            image = preprocess(Image.open(img_path)).unsqueeze(0).to(device)
            text = clip.tokenize([prompt_text]).to(device)

            with torch.no_grad():
                img_feat = model.encode_image(image)
                txt_feat = model.encode_text(text)
                img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
                txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)
                score = (img_feat @ txt_feat.T).item()

            writer.writerow([run_id, prompt_id, round(score, 4)])
            print(f"    {run_id} / {prompt_id}: {score:.4f}")

    print(f"\n    Wrote {QUALITY_CSV}")


if __name__ == "__main__":
    compute_clip_scores()

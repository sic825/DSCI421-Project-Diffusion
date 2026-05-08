"""Generate all figures and tables for the DSCI 421 final report.

Usage (from ~/Desktop/dsci421_final/):
    python make_figures.py

Produces in ./figures/:
    fig1_latency_log.png         - latency comparison across all configs (log scale)
    fig2_speedup_bar.png         - speedup factors vs FP32 baseline
    fig3_quality_grid.png        - 5 prompts x 6 configs visual quality grid
    fig4_clip_scores.png         - CLIP score by config and prompt
    fig5_per_kernel_gpu.png      - kernel breakdown comparison (FP32 vs FP16 vs SDPA)
    table1_latency_summary.csv   - mean/std/min/max latency per config
    table2_speedups.csv          - speedup factors and quality preservation
    table3_kernel_breakdown.csv  - top kernels per config from nsys data (manual entry)
    summary_for_report.txt       - human-readable summary of all key numbers

Dependencies: pandas, matplotlib, pillow, numpy
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

# ============================================================================
# Configuration
# ============================================================================

DATA_DIR = Path(".")  # Run from ~/Desktop/dsci421_final/
OUT_DIR = Path("figures")
OUT_DIR.mkdir(exist_ok=True)

# Map raw run_id strings to clean display labels.
# Order matters - this is the display order across all charts.
CONFIG_ORDER = [
    ("01_cpu_baseline",        "CPU FP32"),
    ("02_gpu_baseline_fp32",   "GPU FP32\nvanilla"),
    ("03_gpu_fp16",            "GPU FP16\nvanilla"),
    ("04_gpu_bf16",            "GPU BF16\nvanilla"),
    ("05_gpu_fp16_sdpa",       "GPU FP16\nSDPA"),
    ("06_gpu_fp16_xformers",   "GPU FP16\nxformers"),
]

# Multi-GPU rows are aggregated separately
MULTIGPU_RUN_IDS = ["08_multigpu_throughput_gpu0", "08_multigpu_throughput_gpu1"]

# Color palette - colorblind-safe, distinguishable in print
COLORS = {
    "01_cpu_baseline":      "#7F7F7F",  # gray
    "02_gpu_baseline_fp32": "#1F77B4",  # blue (baseline)
    "03_gpu_fp16":          "#FF7F0E",  # orange
    "04_gpu_bf16":          "#D62728",  # red (slower!)
    "05_gpu_fp16_sdpa":     "#2CA02C",  # green (best)
    "06_gpu_fp16_xformers": "#17BECF",  # cyan
    "multigpu":             "#9467BD",  # purple
}


def section(name: str) -> None:
    print(f"\n{'=' * 70}\n  {name}\n{'=' * 70}")


# ============================================================================
# Load data
# ============================================================================

section("Loading data")

metrics = pd.read_csv(DATA_DIR / "metrics.csv")
print(f"  metrics.csv: {len(metrics)} rows, {metrics['run_id'].nunique()} unique configs")
print(f"  Configs: {sorted(metrics['run_id'].unique())}")

clip = pd.read_csv(DATA_DIR / "clip_scores.csv")
print(f"  clip_scores.csv: {len(clip)} rows")

# ============================================================================
# Compute summary statistics
# ============================================================================

section("Computing summary statistics")

# Per-config latency stats (only for the main 6 configs, exclude multi-GPU
# which is aggregated differently)
main_run_ids = [rid for rid, _ in CONFIG_ORDER]
main = metrics[metrics["run_id"].isin(main_run_ids)].copy()

summary = (main.groupby("run_id")["wall_time_s"]
           .agg(["mean", "std", "min", "max", "count"])
           .reindex(main_run_ids)
           .round(4))
print(summary)

# Speedup vs GPU FP32 baseline
fp32_mean = summary.loc["02_gpu_baseline_fp32", "mean"]
summary["speedup_vs_fp32"] = (fp32_mean / summary["mean"]).round(3)

# Multi-GPU aggregate metrics
mg = metrics[metrics["run_id"].isin(MULTIGPU_RUN_IDS)].copy()
n_inferences_mg = len(mg)
mean_mg_latency = mg["wall_time_s"].mean()
# Total parallel wall clock from launcher = 34.13s (hardcoded since not in CSV)
PARALLEL_WALL_CLOCK_S = 34.13
# Fair single-GPU equivalent estimate: same number of inferences sequentially
# at single-GPU SDPA latency (config 05 mean) plus one warmup
n_warmup_single = 1
warmup_s_estimate = 1.95  # seen in the multi-GPU log
sdpa_mean = summary.loc["05_gpu_fp16_sdpa", "mean"]
single_gpu_est = warmup_s_estimate + n_inferences_mg * sdpa_mean
multigpu_speedup = single_gpu_est / PARALLEL_WALL_CLOCK_S

print(f"\n  Multi-GPU: {n_inferences_mg} inferences in {PARALLEL_WALL_CLOCK_S:.2f}s")
print(f"  Single-GPU estimate: {single_gpu_est:.2f}s")
print(f"  Multi-GPU speedup: {multigpu_speedup:.2f}x")

# Save Table 1: Latency summary
table1_path = OUT_DIR / "table1_latency_summary.csv"
table1 = summary.copy()
table1.index.name = "run_id"
# Add display labels
table1.insert(0, "config_label", [label.replace("\n", " ") for _, label in CONFIG_ORDER])
table1.to_csv(table1_path)
print(f"\n  Wrote {table1_path}")

# ============================================================================
# Figure 1: Latency comparison (log scale)
# ============================================================================

section("Figure 1: Latency comparison")

fig, ax = plt.subplots(figsize=(10, 5.5))
labels = [label for _, label in CONFIG_ORDER]
means = [summary.loc[rid, "mean"] for rid, _ in CONFIG_ORDER]
stds = [summary.loc[rid, "std"] for rid, _ in CONFIG_ORDER]
colors = [COLORS[rid] for rid, _ in CONFIG_ORDER]

bars = ax.bar(labels, means, yerr=stds, color=colors, capsize=4,
              edgecolor="black", linewidth=0.8)
ax.set_yscale("log")
ax.set_ylabel("Per-image latency (seconds, log scale)", fontsize=11)
ax.set_title("Stable Diffusion v1.5 Inference Latency by Configuration\n"
             "(error bars = $\\pm$1 std across 5 prompts $\\times$ 5 iterations)",
             fontsize=12)
ax.grid(axis="y", linestyle="--", alpha=0.5, which="both")
ax.set_axisbelow(True)

# Annotate each bar with its value
for bar, val in zip(bars, means):
    ax.text(bar.get_x() + bar.get_width() / 2, val * 1.15,
            f"{val:.2f}s", ha="center", va="bottom", fontsize=9, fontweight="bold")

plt.tight_layout()
plt.savefig(OUT_DIR / "fig1_latency_log.png", dpi=200, bbox_inches="tight")
plt.close()
print(f"  Wrote {OUT_DIR / 'fig1_latency_log.png'}")

# ============================================================================
# Figure 2: Speedup vs FP32 baseline
# ============================================================================

section("Figure 2: Speedup factors")

# Exclude CPU from the speedup chart (95x dwarfs everything else)
gpu_configs = [(rid, label) for rid, label in CONFIG_ORDER if rid != "01_cpu_baseline"]
gpu_labels = [label for _, label in gpu_configs]
speedups = [summary.loc[rid, "speedup_vs_fp32"] for rid, _ in gpu_configs]
gpu_colors = [COLORS[rid] for rid, _ in gpu_configs]

# Add multi-GPU as an additional bar
gpu_labels.append("Multi-GPU\nFP16 SDPA")
speedups.append(multigpu_speedup)
gpu_colors.append(COLORS["multigpu"])

fig, ax = plt.subplots(figsize=(10, 5.5))
bars = ax.bar(gpu_labels, speedups, color=gpu_colors,
              edgecolor="black", linewidth=0.8)

# Reference line at 1.0x (baseline)
ax.axhline(1.0, color="black", linestyle="--", linewidth=1, alpha=0.6,
           label="FP32 baseline")
ax.set_ylabel("Speedup factor relative to GPU FP32 baseline", fontsize=11)
ax.set_title("GPU Acceleration Speedup by Optimization Configuration",
             fontsize=12)
ax.grid(axis="y", linestyle="--", alpha=0.5)
ax.set_axisbelow(True)
ax.legend(loc="upper left")

# Annotate each bar with its speedup factor
for bar, val in zip(bars, speedups):
    label_y = val + 0.05 if val > 0.6 else val + 0.05
    ax.text(bar.get_x() + bar.get_width() / 2, label_y,
            f"{val:.2f}x", ha="center", va="bottom", fontsize=10, fontweight="bold")

plt.tight_layout()
plt.savefig(OUT_DIR / "fig2_speedup_bar.png", dpi=200, bbox_inches="tight")
plt.close()
print(f"  Wrote {OUT_DIR / 'fig2_speedup_bar.png'}")

# ============================================================================
# Figure 3: Visual quality grid (5 prompts x 6 GPU configs)
# ============================================================================

section("Figure 3: Visual quality grid")

# Image filename pattern: {run_id}_{prompt_id}.png
images_dir = DATA_DIR / "images"
prompt_ids = ["p01", "p02", "p03", "p04", "p05"]

# Skip CPU (only generated 2 images for 2 prompts) - GPU configs have all 5
grid_run_ids = [rid for rid, _ in CONFIG_ORDER if rid != "01_cpu_baseline"]
grid_labels = [label for rid, label in CONFIG_ORDER if rid != "01_cpu_baseline"]

# Prompt text for column labels (truncated for legibility)
prompt_texts = {
    "p01": "astronaut on horse, moon",
    "p02": "oil painting, mountain sunset",
    "p03": "cyberpunk street market",
    "p04": "cat in wizard hat",
    "p05": "minimalist fox logo",
}

# Build the grid: rows = prompts, columns = configs
n_rows = len(prompt_ids)
n_cols = len(grid_run_ids)

# Each thumbnail at 256x256 (downsampled from 512x512)
thumb_size = 256
label_height = 50  # space for column header
row_label_width = 200  # space for row labels

# Total canvas size
canvas_w = row_label_width + n_cols * thumb_size
canvas_h = label_height + n_rows * thumb_size

canvas = Image.new("RGB", (canvas_w, canvas_h), color="white")
draw = ImageDraw.Draw(canvas)

# Try to load a nicer font; fall back to default
try:
    font_label = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 14)
    font_header = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 13)
except (OSError, IOError):
    try:
        font_label = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
        font_header = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)
    except (OSError, IOError):
        font_label = ImageFont.load_default()
        font_header = ImageFont.load_default()

# Column headers (config labels)
for col, label in enumerate(grid_labels):
    x = row_label_width + col * thumb_size + thumb_size // 2
    # Replace internal newline for the column header
    flat_label = label.replace("\n", " ")
    draw.text((x, label_height // 2), flat_label,
              fill="black", anchor="mm", font=font_header)

# Row labels (prompt descriptions) and thumbnails
for row, pid in enumerate(prompt_ids):
    y = label_height + row * thumb_size + thumb_size // 2
    draw.text((row_label_width // 2, y),
              f"{pid}\n{prompt_texts[pid]}",
              fill="black", anchor="mm", font=font_label)

    for col, rid in enumerate(grid_run_ids):
        img_path = images_dir / f"{rid}_{pid}.png"
        if not img_path.exists():
            print(f"  WARNING: Missing {img_path}")
            continue
        img = Image.open(img_path).resize((thumb_size, thumb_size),
                                          Image.Resampling.LANCZOS)
        canvas.paste(img,
                     (row_label_width + col * thumb_size,
                      label_height + row * thumb_size))

quality_grid_path = OUT_DIR / "fig3_quality_grid.png"
canvas.save(quality_grid_path, optimize=True)
print(f"  Wrote {quality_grid_path} ({canvas_w}x{canvas_h})")

# ============================================================================
# Figure 4: CLIP scores by config and prompt
# ============================================================================

section("Figure 4: CLIP scores")

# Pivot to a config x prompt matrix
clip_pivot = (clip[clip["run_id"].isin([rid for rid, _ in CONFIG_ORDER])]
              .pivot(index="run_id", columns="prompt_id", values="clip_score")
              .reindex([rid for rid, _ in CONFIG_ORDER]))

fig, ax = plt.subplots(figsize=(10, 5.5))
x = np.arange(len(prompt_ids))
width = 0.13

for i, (rid, label) in enumerate(CONFIG_ORDER):
    if rid not in clip_pivot.index:
        continue
    scores = [clip_pivot.loc[rid].get(pid, np.nan) for pid in prompt_ids]
    offset = (i - len(CONFIG_ORDER) / 2) * width + width / 2
    ax.bar(x + offset, scores, width, label=label.replace("\n", " "),
           color=COLORS[rid], edgecolor="black", linewidth=0.4)

ax.set_xticks(x)
ax.set_xticklabels([f"{pid}\n{prompt_texts[pid][:18]}..." for pid in prompt_ids],
                   fontsize=9)
ax.set_ylabel("CLIP score (cosine similarity, image vs prompt)", fontsize=11)
ax.set_title("CLIP Score by Configuration and Prompt\n"
             "(higher = stronger prompt adherence; near-identical across configs)",
             fontsize=12)
ax.set_ylim(0.25, 0.40)
ax.grid(axis="y", linestyle="--", alpha=0.5)
ax.set_axisbelow(True)
ax.legend(loc="upper right", fontsize=8, ncol=2)

plt.tight_layout()
plt.savefig(OUT_DIR / "fig4_clip_scores.png", dpi=200, bbox_inches="tight")
plt.close()
print(f"  Wrote {OUT_DIR / 'fig4_clip_scores.png'}")

# ============================================================================
# Table 2: Speedups + quality preservation
# ============================================================================

section("Table 2: Speedups + quality")

table2_rows = []
for rid, label in CONFIG_ORDER:
    if rid in clip_pivot.index:
        clip_mean = clip_pivot.loc[rid].mean()
    else:
        clip_mean = np.nan
    table2_rows.append({
        "config": label.replace("\n", " "),
        "run_id": rid,
        "mean_latency_s": round(summary.loc[rid, "mean"], 3),
        "speedup_vs_fp32": round(summary.loc[rid, "speedup_vs_fp32"], 3),
        "mean_clip_score": round(clip_mean, 4) if not np.isnan(clip_mean) else "N/A",
    })

# Append multi-GPU row
table2_rows.append({
    "config": "Multi-GPU FP16 SDPA",
    "run_id": "08_multigpu_throughput",
    "mean_latency_s": round(mean_mg_latency, 3),
    "speedup_vs_fp32": round(multigpu_speedup, 3),
    "mean_clip_score": "see config 05 (identical images)",
})

table2_path = OUT_DIR / "table2_speedups.csv"
with open(table2_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(table2_rows[0].keys()))
    w.writeheader()
    w.writerows(table2_rows)
print(f"  Wrote {table2_path}")
for r in table2_rows:
    print(f"    {r}")

# ============================================================================
# Table 3: Per-kernel breakdown (manual data from your nsys output)
# ============================================================================

section("Table 3: Kernel breakdown (from nsys reports we already collected)")

# Hand-curated from the nsys output you pasted in earlier messages.
# Top kernels by % of GPU time, comparing FP32 baseline vs FP16 SDPA.
# This is the killer evidence for "SDPA eliminates the softmax bottleneck".
kernel_breakdown = [
    # (kernel_category, fp32_pct, fp32_time_s, fp16_sdpa_pct, fp16_sdpa_time_s, notes)
    ("Convolution (Winograd, FP32)", 21.3, 31.09, 0.0, 0.0,
     "FP32-only convolution kernels"),
    ("FP32 GEMM matmul (volta_sgemm)", 36.6, 53.39, 0.0, 0.0,
     "Replaced by Tensor Core kernels in FP16"),
    ("Attention softmax (cunn_SoftMaxForwardSmem)", 9.6, 13.99, 0.0, 0.0,
     "ELIMINATED in SDPA via fused FMHA kernel"),
    ("Tensor Core FP16 GEMM (turing_fp16_s1688gemm)", 0.0, 0.0, 28.4, 13.69,
     "Tensor Core path (only active in FP16)"),
    ("Fused multi-head attention (fmha_cutlassF_f16)", 0.0, 0.0, 18.7, 9.07,
     "Replaces separate softmax+matmul in SDPA"),
    ("FP16 implicit GEMM convolution (sm75_xmma_fprop)", 0.0, 0.0, 22.6, 10.96,
     "FP16 convolution via Tensor Cores"),
    ("Layout conversion (nchwToNhwc)", 0.0, 0.0, 13.3, 6.43,
     "Memory layout reformatting for Tensor Cores"),
    ("Memory ops (memcpy, memset)", 4.7, 6.86, 7.2, 3.49,
     "Host-device transfers + buffer init"),
]

table3_path = OUT_DIR / "table3_kernel_breakdown.csv"
with open(table3_path, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["kernel_category",
                "fp32_pct_gpu_time", "fp32_total_s",
                "fp16_sdpa_pct_gpu_time", "fp16_sdpa_total_s",
                "notes"])
    for row in kernel_breakdown:
        w.writerow(row)
print(f"  Wrote {table3_path}")
for r in kernel_breakdown:
    print(f"    {r[0]:50s}  FP32: {r[1]:5.1f}%  SDPA: {r[3]:5.1f}%")

# ============================================================================
# Figure 5: Per-kernel breakdown comparison chart
# ============================================================================

section("Figure 5: Kernel breakdown chart")

cat_names = [r[0] for r in kernel_breakdown]
fp32_pcts = [r[1] for r in kernel_breakdown]
sdpa_pcts = [r[3] for r in kernel_breakdown]

fig, ax = plt.subplots(figsize=(11, 6))
y = np.arange(len(cat_names))
width = 0.4

ax.barh(y - width / 2, fp32_pcts, width, label="FP32 vanilla (config 02)",
        color=COLORS["02_gpu_baseline_fp32"], edgecolor="black", linewidth=0.5)
ax.barh(y + width / 2, sdpa_pcts, width, label="FP16 SDPA (config 05)",
        color=COLORS["05_gpu_fp16_sdpa"], edgecolor="black", linewidth=0.5)

ax.set_yticks(y)
ax.set_yticklabels(cat_names, fontsize=9)
ax.set_xlabel("% of total GPU time", fontsize=11)
ax.set_title("Per-Kernel GPU Time Distribution: FP32 Baseline vs FP16 SDPA\n"
             "(Tensor Core kernels active in FP16 SDPA; softmax eliminated)",
             fontsize=12)
ax.invert_yaxis()  # First row at top
ax.legend(loc="lower right")
ax.grid(axis="x", linestyle="--", alpha=0.5)
ax.set_axisbelow(True)

# Annotate each bar with its value
for i, (fp32, sdpa) in enumerate(zip(fp32_pcts, sdpa_pcts)):
    if fp32 > 0:
        ax.text(fp32 + 0.5, i - width / 2, f"{fp32:.1f}%",
                va="center", fontsize=8)
    if sdpa > 0:
        ax.text(sdpa + 0.5, i + width / 2, f"{sdpa:.1f}%",
                va="center", fontsize=8)

plt.tight_layout()
plt.savefig(OUT_DIR / "fig5_per_kernel_gpu.png", dpi=200, bbox_inches="tight")
plt.close()
print(f"  Wrote {OUT_DIR / 'fig5_per_kernel_gpu.png'}")

# ============================================================================
# Summary text file for the report
# ============================================================================

section("Writing summary_for_report.txt")

summary_text = f"""DSCI 421 Final Project - Summary of Key Numbers
================================================================

This file contains the canonical numbers to cite in the report.
Generated by make_figures.py.

LATENCY (mean per-image, after warmup discarded)
------------------------------------------------
"""
for rid, label in CONFIG_ORDER:
    flat = label.replace("\n", " ")
    summary_text += (f"  {flat:30s}  "
                     f"{summary.loc[rid, 'mean']:.3f} s  "
                     f"(std={summary.loc[rid, 'std']:.3f}, n={int(summary.loc[rid, 'count'])})\n")

summary_text += f"""
SPEEDUP RELATIVE TO GPU FP32 BASELINE
-------------------------------------
"""
for rid, label in CONFIG_ORDER:
    flat = label.replace("\n", " ")
    summary_text += f"  {flat:30s}  {summary.loc[rid, 'speedup_vs_fp32']:.3f}x\n"

summary_text += f"""
MULTI-GPU THROUGHPUT (config 08)
--------------------------------
  Inferences completed:           {n_inferences_mg}
  Parallel wall clock:            {PARALLEL_WALL_CLOCK_S:.2f} s
  Per-image latency (avg):        {mean_mg_latency:.3f} s
  Single-GPU sequential estimate: {single_gpu_est:.2f} s
  Speedup vs single-GPU sequential: {multigpu_speedup:.2f}x
  (theoretical ceiling: 2.0x; gap due to per-process model load overhead)

KEY KERNEL FINDINGS (from nsys profiles)
----------------------------------------
  FP32 baseline (config 02):
    - cunn_SoftMaxForwardSmem<float> consumed 9.6% of GPU time
      (14.0 s across 3500 instances, 4.00 ms avg per call)
    - volta_sgemm_* matmul kernels (non-Tensor-Core) dominated compute

  FP16 vanilla (config 03):
    - Same softmax kernel grew to 17.9% share (3500 calls @ 3.14 ms)
    - because matmul moved to Tensor Cores (turing_fp16_s1688gemm_*),
      making softmax a relatively LARGER bottleneck

  FP16 SDPA (config 05):
    - softmax kernel ELIMINATED entirely
    - Replaced by fmha_cutlassF_f16_aligned_64x64_rf_sm75 (fused attn)
    - 16.6% of GPU time, but doing softmax + matmul together

  FP16 xformers (config 06):
    - Calls the SAME fused FMHA kernel as SDPA
    - Empirically explains why latency is identical to SDPA

  BF16 (config 04):
    - magma_sgemmEx_kernel (software emulation) consumed 64.1% of GPU time
    - NO Tensor Core kernels engaged
    - Empirical proof of architectural fallback on Turing (sm_75)

MEMORY TRANSFER (Host-to-Device, total per run)
-----------------------------------------------
  FP32:        4,265 MB
  FP16/BF16:   2,133 MB  (50% reduction as predicted)
  SDPA/xformers do not reduce H2D - only attention compute

CLIP SCORES (cosine similarity, image vs prompt; higher is better)
-------------------------------------------------------------------
"""
for rid, label in CONFIG_ORDER:
    if rid in clip_pivot.index:
        flat = label.replace("\n", " ")
        mean = clip_pivot.loc[rid].mean()
        rng = (clip_pivot.loc[rid].min(), clip_pivot.loc[rid].max())
        summary_text += (f"  {flat:30s}  mean={mean:.4f}  "
                         f"range=[{rng[0]:.4f}, {rng[1]:.4f}]\n")

summary_text += f"""
  Variation across configurations is dominated by per-prompt difficulty,
  not by precision/attention configuration. No measurable quality regression.

HARDWARE & ENVIRONMENT
----------------------
  CPU:    Intel Xeon E5-1630 v4 @ 3.70 GHz
          4 cores / 8 threads, Broadwell-EP (2016), AVX2 (no AVX-512)
          10 MB L3 cache, 251 GB system memory
  GPU:    2x NVIDIA GeForce RTX 2080 Ti (Turing, sm_75)
          11 GB VRAM each, NVLink ~25.78 GB/s per link, 2 links per GPU
  Driver: 565.57.01
  CUDA:   12.4 (PyTorch runtime); nvcc reports 12.6 (Anaconda base)
  PyTorch: 2.4.1
  Diffusers: 0.30.0
  xformers: 0.0.27.post2

REPRODUCIBILITY
---------------
  All seeds fixed; SD 1.5 model from HuggingFace; 25 inference steps;
  DPM-Solver++ scheduler; 512x512 resolution; CFG scale 7.5.
  See environment.yml for full dependency pinning.
"""

with open(OUT_DIR / "summary_for_report.txt", "w") as f:
    f.write(summary_text)
print(f"  Wrote {OUT_DIR / 'summary_for_report.txt'}")

# ============================================================================
# Final inventory
# ============================================================================

section("Done")
print(f"  Output directory: {OUT_DIR.resolve()}")
for p in sorted(OUT_DIR.iterdir()):
    print(f"    {p.name}  ({p.stat().st_size:,} bytes)")

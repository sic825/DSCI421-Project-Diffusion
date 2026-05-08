# Profiling GPU Acceleration Techniques for Stable Diffusion v1.5 Inference

DSCI 421 Final Project, Spring 2026
**Authors:** Vincent Caruso, Simon Chen, Matt Olson
**Affiliation:** Department of Data Science, Lehigh University

This project profiles inference performance of Stable Diffusion v1.5 across three optimization axes — numerical precision, attention backend, and dual-GPU throughput — on NVIDIA RTX 2080 Ti hardware. We measure latency, kernel-level performance via NVIDIA Nsight Systems, and output quality via CLIP scores across a fixed prompt and seed set.

## Headline results

| Configuration | Latency (s/img) | Speedup vs FP32 | CLIP score |
|---|---|---|---|
| CPU FP32 | 173.92 | 0.030× | 0.358 |
| GPU FP32 vanilla (baseline) | 5.21 | 1.00× | 0.337 |
| GPU FP16 vanilla | 2.29 | 2.28× | 0.334 |
| GPU BF16 vanilla | 9.66 | 0.54× | 0.334 |
| GPU FP16 SDPA | 1.83 | 2.85× | 0.336 |
| GPU FP16 xformers | 1.85 | 2.82× | 0.336 |
| Dual-GPU FP16 SDPA | 1.73 | 4.58× throughput | 0.336 |

Key findings:
- FP16 with PyTorch SDPA achieves a 2.85× per-image latency speedup over FP32 by engaging Tensor Cores and eliminating the standalone softmax kernel via fused FMHA.
- BF16 degrades performance by 1.86× because the 2080 Ti (Turing, sm_75) lacks native BF16 Tensor Core support and falls back to MAGMA software emulation.
- Dual-GPU achieves a 4.58× throughput improvement over the FP32 baseline by running independent inference processes on each GPU.
- No measurable quality regression across any GPU configuration (CLIP score range 0.334–0.337).

## Repository structure

```
.
├── README.md                          # This file
├── environment.yml                    # Conda environment specification
├── make_figures.py                    # Generates all report figures from results/
├── configs/
│   ├── prompts.yaml                   # Fixed prompt set with seeds
│   └── runs/
│       ├── 01_cpu_baseline.yaml
│       ├── 02_gpu_baseline_fp32.yaml
│       ├── 03_gpu_fp16.yaml
│       ├── 04_gpu_bf16.yaml
│       ├── 05_gpu_fp16_sdpa.yaml
│       ├── 06_gpu_fp16_xformers.yaml
│       └── 08_multigpu_throughput.yaml
├── src/
│   ├── benchmark.py                   # Main benchmark harness with NVTX ranges
│   ├── pipeline.py                    # SD pipeline factory
│   ├── quality.py                     # CLIP score evaluation
│   └── utils.py                       # Seeding, env capture, config loading
├── scripts/
│   ├── run_all.sh                     # Sequential sweep across all single-GPU configs
│   ├── profile_nsys.sh                # Wraps benchmark with nsys profiling
│   └── run_multigpu_throughput.sh     # Two-process dual-GPU launcher
├── notebooks/
│   ├── 01_environment_check.ipynb     # Verify CUDA, GPU detection, model load
│   ├── 03_results_analysis.ipynb      # Post-hoc analysis of metrics.csv
│   └── int8_smoke_test.ipynb          # Exploratory INT8 quantization test
├── results/
│   ├── metrics.csv                    # Per-iteration timing data
│   ├── clip_scores.csv                # CLIP scores per (config, prompt)
│   ├── system_info.txt                # CPU/GPU/NVLink hardware specifications
│   ├── images/                        # Generated images (32 PNGs)
│   ├── multigpu_logs/                 # Stdout/stderr from dual-GPU run
│   └── env_snapshots/                 # Per-run environment captures
├── figures/                           # Charts and visual quality grid
└── report/                            # LaTeX source and final PDF
```

## Reproducing the results

### Hardware

Tested on the Lehigh ECE HPC cluster (hpc05.ece.lehigh.edu):
- 2× NVIDIA GeForce RTX 2080 Ti (Turing, sm_75), 11 GB VRAM each
- NVLink connection (~25.78 GB/s per link, 2 links per GPU)
- Intel Xeon E5-1630 v4 @ 3.70 GHz, 4 cores / 8 threads (Broadwell-EP)
- 251 GB system memory
- NVIDIA driver 565.57.01, CUDA runtime 12.4

### Software environment

```bash
# Create the conda environment
conda env create -f environment.yml
conda activate sd-profiling

# Verify the environment - opens notebooks/01_environment_check.ipynb
jupyter notebook notebooks/01_environment_check.ipynb
```

Key dependencies (full list in `environment.yml`):
- PyTorch 2.4.1 + CUDA 12.4
- diffusers 0.30.0
- transformers 4.44.0
- xformers 0.0.27.post2
- OpenAI CLIP (from GitHub source, not PyPI)

### Running the benchmark sweep

```bash
# Single-GPU sweep (configs 01-06, sequential)
bash scripts/run_all.sh

# Dual-GPU throughput experiment
bash scripts/run_multigpu_throughput.sh

# Per-config nsys profiling (run individually for each config)
bash scripts/profile_nsys.sh configs/runs/02_gpu_baseline_fp32.yaml
bash scripts/profile_nsys.sh configs/runs/03_gpu_fp16.yaml
bash scripts/profile_nsys.sh configs/runs/05_gpu_fp16_sdpa.yaml
# (etc. for other configs)
```

Results append to `results/metrics.csv`. Each run captures wall-clock latency for 5 prompts × 5 timed iterations after 3 warmup runs are discarded.

### Generating figures and tables

```bash
python make_figures.py
```

This produces all figures referenced in the report (`figures/fig1_latency_log.png`, etc.) plus three CSV tables and a `summary_for_report.txt` with all canonical numbers.

## Profiling artifacts (not in this repo)

The Nsight Systems `.nsys-rep` files are too large for GitHub (~60 MB each, ~300 MB total). They can be regenerated via `scripts/profile_nsys.sh`, or accessed at:

**[Lehigh Google Drive link](https://drive.google.com/drive/folders/1y4KKh3W2abIXzyqYn1PiQxkfAwPfzh_y?usp=sharing)**

Available reports:
- `02_gpu_baseline_fp32.nsys-rep` (FP32 vanilla, 67 MB)
- `03_gpu_fp16.nsys-rep` (FP16 vanilla, 61 MB)
- `04_gpu_bf16.nsys-rep` (BF16, 65 MB)
- `05_gpu_fp16_sdpa.nsys-rep` (FP16 SDPA, 54 MB)
- `06_gpu_fp16_xformers.nsys-rep` (FP16 xformers, 58 MB)

Open with `nsys-ui <filename>.nsys-rep` (Nsight Systems UI, available cross-platform).

## Methodology notes

**Timing.** Latency measurements use `time.perf_counter()` with explicit `torch.cuda.synchronize()` before and after each measurement to ensure GPU work has completed. Three warmup iterations per config are discarded.

**Reproducibility.** Each run captures full environment metadata (`results/env_snapshots/`) including driver version, CUDA version, PyTorch version, NVLink status, and config parameters. All seeds are fixed across configurations.

**Quality.** Output quality is evaluated using OpenAI's CLIP ViT-B/32 to compute cosine similarity between each generated image and its source prompt. CLIP scores are reported in `results/clip_scores.csv`.

**Multi-GPU implementation.** Dual-GPU is implemented as two independent Python processes pinned via `CUDA_VISIBLE_DEVICES`, each processing a disjoint subset of prompts. This is throughput parallelism (not data-parallel via NCCL); per-image latency cannot be reduced because SD's denoising loop is sequential within a single image.

**INT8 (out of scope).** An exploratory INT8 test using bitsandbytes 0.43.3 is in `notebooks/int8_smoke_test.ipynb`. Result: 2.4× slower than FP16 SDPA because bitsandbytes only quantizes Linear layers (~31% of UNet parameters), and dequantization overhead at INT8/FP16 boundaries dominates. INT8 was excluded from the main sweep based on these findings.

## Repository state

This repo represents the final submission state as of May 8, 2026. The git history shows the development process; the latest commit on `main` is the submission version.

For the final report PDF and the demo video, see the `report/` directory and the project submission on Course Site.

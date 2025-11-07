---
title: Instances for ML
---

The below table provides general recommendations for selecting AWS instances based on dataset size, computational needs, and cost considerations.

#### Genearl Notes:
- **Minimum RAM** should be at least 1.5X dataset size unless using batch processing (common in deep learning).
- The **m5** and **c5** instances are optimized for CPU-heavy tasks, such as preprocessing, feature engineering, and model training without GPUs.
- **GPU choices** depend on the task (T4 for cost-effective DL, V100/A100 for high performance).
- The **g4dn** instances are cost-effective GPU options, suitable for moderate-scale deep learning tasks.
- The **p3** instances offer high-performance GPU processing, best suited for large deep learning models requiring fast training times.
- **Free Tier Eligibility**: Some smaller instance types, such as `ml.t3.medium`, may be eligible for the AWS Free Tier, which provides limited hours of usage per month. Free Tier eligibility can vary, so check [AWS Free Tier details](https://aws.amazon.com/free/) before launching instances to avoid unexpected costs.


| **Dataset Size** | **Recommended Instance Type** | **vCPU** | **Memory (GiB)** | **GPU** | **Price per Hour (USD)** | **Suitable Tasks** | **Max Model Size (Approx.)** |
|------------------:|------------------------------|----------|------------------|---------|--------------------------|------------------------------------------------------|------------------------------|
| < 1 GB | `ml.t3.medium` | 2 | 4 | None | $0.04 | Lightweight preprocessing or small regression/classification models (< 50 M params) | Up to 100 M params |
| < 1 GB | `ml.m5.large` | 2 | 8 | None | $0.10 | Regression, feature engineering, small CNNs or tree ensembles (50 M–200 M params) | Up to 500 M params |
| < 1 GB | `g4dn.xlarge` (T4 GPU) | 4 | 16 | 1 × T4 (16 GB VRAM) | $0.75 | Cost-effective GPU training for compact DL models (500 M–3 B params – e.g., BERT-base, CLIP-ViT-B) | Up to 3 B params |
| < 1 GB | `p3.2xlarge` (V100 GPU) | 8 | 61 | 1 × V100 (16 GB VRAM) | **$3.83** | High-performance GPU jobs; fine-tuning or inference for 3 B–7 B models (e.g., LLaMA-2-7B, Mistral-7B) | Up to 7 B params |
| 10 GB | `ml.c5.2xlarge` | 8 | 16 | None | $0.34 | CPU-heavy processing, tabular/ensemble models (< 200 M params) | Up to 500 M params |
| 10 GB | `ml.m5.2xlarge` | 8 | 32 | None | $0.38 | Preprocessing, feature engineering, boosting or linear models (200 M–500 M params) | Up to 1 B params |
| 10 GB | `g4dn.2xlarge` (T4 GPU) | 8 | 32 | 1 × T4 | $0.94 | Moderate-scale DL; training/inference for 1 B–4 B models (e.g., BERT-large, ViT-L/16) | Up to 3 B–4 B params |
| 10 GB | `p3.2xlarge` (V100 GPU) | 8 | 61 | 1 × V100 | **$3.83** | Deep learning workloads; 3 B–7 B models (e.g., Mistral-7B, Gemma-7B) | Up to 7 B params |
| 50 GB | `ml.c5.4xlarge` | 16 | 64 | None | $0.77 | Large-scale CPU modeling or preprocessing (< 500 M params) | Up to 1 B params |
| 50 GB | `ml.m5.4xlarge` | 16 | 64 | None | $0.77 | Feature engineering, classic ML on wide data (200 M–500 M params) | Up to 1 B params |
| 50 GB | `g4dn.4xlarge` (T4 GPU) | 16 | 64 | 1 × T4 | $1.48 | Moderate DL models (1 B–4 B params); efficient for batch inference | Up to 4 B params |
| 100 GB | `g4dn.8xlarge` (T4 GPU) | 32 | 128 | 1 × T4 | **$2.76** | Large-scale DL training (2 B–5 B params; e.g., LLaMA-2-3B–5B) with mixed precision | Up to 5 B params |
| 100 GB | `p3.8xlarge` (V100 GPU) | 32 | 244 | 4 × V100 | **$15.20** | Multi-GPU training (7 B–30 B params; e.g., Falcon-40B, Mixtral 8×7B) | Up to 30 B params |
| 100 GB | `p4d.24xlarge` (A100 GPU) | 96 | 1,152 | 8 × A100 (40 GB each) | **$32.77** | High-end DL or fine-tuning for very large models (30 B–70 B+ params; e.g., LLaMA-3-70B, GPT-3) | Up to 70 B+ params |
| 1 TB+ | `p3.16xlarge` (V100 GPU) | 64 | 488 | 8 × V100 | **$30.40** | Extreme-scale transformer training (30 B–60 B params) with distributed data parallelism | Up to 65 B params |
| 1 TB+ | `p4d.24xlarge` (A100 GPU) | 96 | 1,152 | 8 × A100 | **$32.77** | Batch or distributed training for foundation-scale models (60 B–70 B+) | Up to 70 B+ params |


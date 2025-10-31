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
|------------------:|------------------------------|----------|------------------|---------|--------------------------|--------------------|------------------------------|
| < 1GB | `ml.t3.medium` | 2 | 4 | None | $0.04 | Preprocessing, lightweight model training | Up to 100M params |
| < 1GB | `ml.m5.large` | 2 | 8 | None | $0.10 | Preprocessing, regression, feature engineering, small model training | Up to 500M params |
| < 1GB | `g4dn.xlarge` (T4 GPU) | 4 | 16 | 1× NVIDIA T4 (16 GB VRAM) | $0.75 | GPU processing for small-scale deep learning, cost-effective GPU option | Up to **3B** params |
| < 1GB | `p3.2xlarge` (V100 GPU) | 8 | 61 | 1× NVIDIA V100 (16 GB VRAM) | **$3.83** | High-performance GPU processing, faster deep learning | Up to **7B** params |
| 10GB | `ml.c5.2xlarge` | 8 | 16 | None | $0.34 | CPU-heavy processing, model training | Up to 500M params |
| 10GB | `ml.m5.2xlarge` | 8 | 32 | None | $0.38 | Preprocessing, feature engineering, model training | Up to 1B params |
| 10GB | `g4dn.2xlarge` (T4 GPU) | 8 | 32 | 1× NVIDIA T4 | $0.94 | Moderate-scale deep learning, cost-effective GPU inference | Up to **3B–4B** params |
| 10GB | `p3.2xlarge` (V100 GPU) | 8 | 61 | 1× NVIDIA V100 | **$3.83** | Faster GPU processing for deep learning | Up to **7B** params |
| 50GB | `ml.c5.4xlarge` | 16 | 64 | None | $0.77 | CPU-heavy processing, large model training | Up to 1B params |
| 50GB | `ml.m5.4xlarge` | 16 | 64 | None | $0.77 | Preprocessing, feature engineering, large model training | Up to 1B params |
| 50GB | `g4dn.4xlarge` (T4 GPU) | 16 | 64 | 1× NVIDIA T4 | $1.48 | Moderate-scale deep learning | Up to **4B** params |
| 100GB | `g4dn.8xlarge` (T4 GPU) | 32 | 128 | 1× NVIDIA T4 | **$2.76** | Large-scale model training with cost-effective GPU | Up to **5B** params |
| 100GB | `p3.8xlarge` (V100 GPU) | 32 | 244 | 4× NVIDIA V100 | **$15.20** | High-performance GPU processing for large DL models | Up to **30B** params |
| 100GB | `p4d.24xlarge` (A100 GPU) | 96 | 1,152 | 8× NVIDIA A100 (40 GB each) | **$32.77** | High-performance DL for large datasets with batch streaming | Up to **70B+** params |
| 1TB+ | `p3.16xlarge` (V100 GPU) | 64 | 488 | 8× NVIDIA V100 | **$30.40** | Extreme-scale deep learning, large transformer training | Up to **65B** params |
| 1TB+ | `p4d.24xlarge` (A100 GPU) | 96 | 1,152 | 8× NVIDIA A100 | **$32.77** | Deep learning with batch processing for massive datasets | Up to **70B+** params |



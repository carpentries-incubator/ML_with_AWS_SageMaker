---
title: 'Instances for ML'
---

This table provides recommendations for selecting AWS instances based on dataset size, computational needs, and cost considerations.

| **Dataset Size** | **Recommended Instance Type** | **vCPU** | **Memory (GiB)** | **GPU** | **Price per Hour (USD)** | **Suitable Tasks** |
|-----------------|------------------------------|----------|------------------|---------|--------------------------|--------------------|
| < 1GB          | `ml.t3.medium`                | 2        | 4                | None    | $0.04                    | Preprocessing, lightweight model training |
| < 1GB          | `ml.m5.large`                 | 2        | 8                | None    | $0.10                    | Preprocessing, regression, feature engineering, small model training |
| < 1GB          | `g4dn.xlarge` (T4 GPU)        | 4        | 16               | 1x NVIDIA T4 | $0.75 | GPU processing for small-scale deep learning, cost-effective GPU option |
| < 1GB          | `p3.2xlarge` (V100 GPU)       | 8        | 61               | 1x NVIDIA V100 | $3.83 | High-performance GPU processing, faster training for deep learning models, higher cost but faster than `g4dn` |
| 10GB          | `ml.c5.xlarge`                | 4        | 8                | None    | $0.17                    | CPU-heavy processing, model training |
| 10GB          | `ml.m5.xlarge`                | 4        | 16               | None    | $0.19                    | Preprocessing, feature engineering, model training |
| 10GB          | `g4dn.xlarge` (T4 GPU)        | 4        | 16               | 1x NVIDIA T4 | $0.75 | Moderate-scale deep learning, cost-effective GPU training |
| 10GB          | `p3.2xlarge` (V100 GPU)       | 8        | 61               | 1x NVIDIA V100 | $3.83 | Faster GPU processing for deep learning, better suited for larger models if budget allows |
| 50GB          | `ml.c5.2xlarge`               | 8        | 16               | None    | $0.34                    | CPU-heavy processing, large model training |
| 50GB          | `ml.m5.2xlarge`               | 8        | 32               | None    | $0.38                    | Preprocessing, feature engineering, large model training |
| 50GB          | `g4dn.xlarge` (T4 GPU)        | 4        | 16               | 1x NVIDIA T4 | $0.75 | Moderate-scale deep learning, balanced performance and cost |
| 100GB         | `g4dn.2xlarge` (T4 GPU)       | 8        | 32               | 1x NVIDIA T4 | $0.94 | Large-scale model training with cost-effective GPU acceleration |
| 100GB         | `ml.m5.4xlarge`               | 16       | 64               | None    | $0.77                    | Preprocessing, feature engineering, batch processing, large model training |
| 100GB         | `p3.2xlarge` (V100 GPU)       | 8        | 61               | 1x NVIDIA V100 | $3.83 | High-performance GPU processing for large deep learning models (e.g., transformers, CNNs) |

### Notes:
- The **g4dn** instances are **cost-effective GPU options**, suitable for moderate-scale deep learning tasks.
- The **p3** instances offer **high-performance GPU processing**, best suited for large deep learning models requiring fast training times.
- The **m5** and **c5** instances are optimized for **CPU-heavy tasks**, such as preprocessing, feature engineering, and model training without GPUs.

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


| **Dataset Size** | **Max Model Size (Approx.)** | **Recommended Instance** | **vCPU** | **Memory (GiB)** | **GPU** | **Price per Hour (USD)** | **Suitable Tasks** |
|------------------:|------------------------------|------------------------------|----------|------------------|---------|--------------------------|------------------------------------------------------|
| < 1 GB | Up to 100 M params | ml.t3.medium | 2 | 4 | None | $0.04 | Lightweight preprocessing or small models |
| < 1 GB | Up to 500 M params | ml.m5.large | 2 | 8 | None | $0.10 | Regression, feature engineering, small CNNs or tree ensembles |
| < 1 GB | Up to 3 B params | g4dn.xlarge (T4 GPU) | 4 | 16 | 1 × T4 (16 GB) | $0.75 | Cost-effective GPU training for compact DL models |
| < 1 GB | Up to 8–10 B params | g5.xlarge (A10G GPU) | 4 | 16 | 1 × A10G (24 GB) | $1.21 | Stronger inference & training; good for 7B LLMs |
| < 1 GB | Up to 7 B params | p3.2xlarge (V100 GPU) | 8 | 61 | 1 × V100 (16 GB) | $3.83 | High-performance GPU jobs; fine-tuning or inference |

| 10 GB | Up to 500 M params | ml.c5.2xlarge | 8 | 16 | None | $0.34 | CPU-heavy processing, tabular/ensemble models |
| 10 GB | Up to 1 B params | ml.m5.2xlarge | 8 | 32 | None | $0.38 | Feature engineering, boosting or linear models |
| 10 GB | Up to 3–4 B params | g4dn.2xlarge (T4 GPU) | 8 | 32 | 1 × T4 | $0.94 | Moderate-scale DL; training/inference |
| 10 GB | Up to 10–12 B params | g5.2xlarge (A10G GPU) | 8 | 32 | 1 × A10G | $1.69 | Larger 7–10B models; faster than g4dn |
| 10 GB | Up to 7 B params | p3.2xlarge (V100 GPU) | 8 | 61 | 1 × V100 | $3.83 | Deep learning workloads |

| 50 GB | Up to 1 B params | ml.c5.4xlarge | 16 | 64 | None | $0.77 | Large CPU modeling / preprocessing |
| 50 GB | Up to 1 B params | ml.m5.4xlarge | 16 | 64 | None | $0.77 | Classic ML on wide data |
| 50 GB | Up to 4 B params | g4dn.4xlarge (T4 GPU) | 16 | 64 | 1 × T4 | $1.48 | Moderate DL models; batch inference |
| 50 GB | Up to 14 B params | g5.4xlarge (A10G GPU) | 16 | 64 | 1 × A10G | $2.54 | Solid for 7–14B models (fp16 or 4/8-bit) |

| 100 GB | Up to 5 B params | g4dn.8xlarge (T4 GPU) | 32 | 128 | 1 × T4 | $2.76 | Large-scale DL training |
| 100 GB | Up to 40–50 B params | g5.12xlarge (A10G GPU) | 48 | 192 | 4 × A10G | $10.18 | Multi-GPU training; distributed finetuning |
| 100 GB | Up to 30 B params | p3.8xlarge (V100 GPU) | 32 | 244 | 4 × V100 | $15.20 | Multi-GPU training |
| 100 GB | Up to 70 B+ params | p4d.24xlarge (A100 GPU) | 96 | 1,152 | 8 × A100 | $32.77 | High-end DL / LLM finetuning |

| 1 TB+ | Up to 65 B params | p3.16xlarge (V100 GPU) | 64 | 488 | 8 × V100 | $30.40 | Extreme-scale transformer training |
| 1 TB+ | Up to 70 B+ params | p4d.24xlarge (A100 GPU) | 96 | 1,152 | 8 × A100 | $32.77 | Distributed training for foundation-scale models |


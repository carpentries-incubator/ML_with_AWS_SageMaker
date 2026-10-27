---
title: Instances for ML
---

The below table provides general recommendations for selecting AWS SageMaker instances based on dataset size, computational needs, and cost considerations. Prices are **us-east-1 on-demand, from the AWS price list of 2026-10-16**; other regions (including the workshop's US East (Ohio), `us-east-2`) are the same or within a few percent. Rows marked *approx.* were not re-verified against that list.

#### General Notes:
- **Minimum RAM** should be at least 1.5X dataset size unless using batch processing (common in deep learning).
- The **m5** and **c5** instances are optimized for CPU-heavy tasks, such as preprocessing, feature engineering, and model training without GPUs. Tabular and tree-ensemble models (XGBoost, random forests) belong here; a GPU does not make them faster.
- **GPU choices** depend on the task. The GPUs you will meet, cheapest first:
    - **T4** (16 GB, `g4dn`): the cheapest GPU; fine for small CNNs, inference of models up to a few billion parameters in 4- or 8-bit, and learning the workflow. No bf16 support, so modern LLM fine-tuning is slow or fails on it.
    - **A10G** (24 GB, `g5`): the lesson's workhorse for fine-tuning: bf16, enough memory for a 3B vision-language model with LoRA, or a 7B LLM in 4-bit.
    - **L4** (24 GB, `g6`): the same memory as an A10G at a lower price, somewhat slower for training, good for inference and embeddings.
    - **L40S** (48 GB, `g6e`): twice the memory of an A10G; what the optional GPU-notebook episode uses to serve a 27B model in FP8 (27 GB of weights). The `ml.g6e.xlarge` is the GPU notebook instance that costs about `$60` a day if left running.
    - **A100 / H100** (`p4d` / `p5`): multi-GPU, 8 per instance, for training or fine-tuning models of tens of billions of parameters. Nothing in this lesson needs one; they are listed so the price is not a surprise.
- The **g4dn** instances are cost-effective GPU options, suitable for moderate-scale deep learning tasks.
- **Free Tier Eligibility**: Some smaller instance types, such as `ml.t3.medium`, may be eligible for the AWS Free Tier, which provides limited hours of usage per month. Free Tier eligibility can vary, so check [AWS Free Tier details](https://aws.amazon.com/free/) before launching instances to avoid unexpected costs.

#### How to read an instance name, and what the price covers

An instance is a fixed box: `ml.` + family + generation + size. The family letter says what the box is built around (`t` and `m` general CPU, `c` compute-optimised CPU, `g` one or a few GPUs, `p` the multi-GPU data-centre boxes); the number is the generation (`g5` is newer than `g4dn`); a trailing letter is a variant (`g4dn` adds NVMe disks, `g6e` is the L40S version of `g6`); and the size (`xlarge`, `2xlarge`, ... `48xlarge`) scales vCPUs, RAM, and on GPU families the number of GPUs together. You cannot rent half a box or add a GPU to one.

**The hourly price is for the whole box, not per GPU.** An `ml.p5.48xlarge` is eight H100s for `$63.30` an hour, about `$7.90` per H100-hour; an `ml.p4d.24xlarge` is eight A100s for `$25.25`, about `$3.15` per A100-hour; an `ml.g5.xlarge` is one A10G for `$1.41`. To turn an estimate into a bill:

```text
instance-hours = GPU-hours ÷ GPUs per box
bill           = instance-hours × price per instance-hour
```

So "32 H100s for 25 hours" is 800 GPU-hours, which is four `ml.p5.48xlarge` for 25 hours, 100 instance-hours, about `$6,330`. The same 800 GPU-hours on one box would take 100 hours of wall-clock time and cost the same.

| **Dataset Size** | **Max Model Size (Approx.)** | **Recommended Instance** | **vCPU** | **Memory (GiB)** | **GPU** | **Price per Hour (USD)** | **Suitable Tasks** |
|------------------:|------------------------------|------------------------------|----------|------------------|---------|--------------------------|------------------------------------------------------|
| < 1 GB | Up to 100 M params | ml.t3.medium | 2 | 4 | None | `$0.05` | Controller notebook; lightweight preprocessing or small models |
| < 1 GB | Up to 500 M params | ml.m5.large | 2 | 8 | None | `$0.115` | Regression, feature engineering, small CNNs or tree ensembles |
| < 1 GB | Up to 1 B params | ml.m5.xlarge | 4 | 16 | None | `$0.23` | Classic ML with room for a wider feature frame |
| < 1 GB | Up to 3 B params | ml.g4dn.xlarge (T4) | 4 | 16 | 1 × T4 (16 GB) | `$0.74` | Cost-effective GPU training for compact DL models |
| < 1 GB | Up to 3 B params | ml.g6.xlarge (L4) | 4 | 16 | 1 × L4 (24 GB) | `$1.13` | Inference and embeddings; cheaper than g5 for serving |
| < 1 GB | Up to 8–10 B params | ml.g5.xlarge (A10G) | 4 | 16 | 1 × A10G (24 GB) | `$1.41` | LoRA fine-tuning of 3B models (this lesson); 7B LLM inference in 4-bit |
| < 1 GB | Up to 13 B params | ml.g6e.xlarge (L40S) | 4 | 32 | 1 × L40S (48 GB) | `$2.61` | 27B model served in FP8 (optional GPU-notebook episode) |
| 10 GB | Up to 500 M params | ml.c5.2xlarge | 8 | 16 | None | `$0.34` (approx.) | CPU-heavy processing, tabular/ensemble models |
| 10 GB | Up to 1 B params | ml.m5.2xlarge | 8 | 32 | None | `$0.46` (approx.) | Feature engineering, boosting or linear models |
| 10 GB | Up to 3–4 B params | ml.g4dn.2xlarge (T4) | 8 | 32 | 1 × T4 | `$0.94` | Moderate-scale DL; training/inference |
| 10 GB | Up to 3–4 B params | ml.g6.2xlarge (L4) | 8 | 32 | 1 × L4 | `$1.22` | Moderate-scale DL with more RAM; batch inference |
| 10 GB | Up to 10–12 B params | ml.g5.2xlarge (A10G) | 8 | 32 | 1 × A10G | `$1.52` | Larger 7–10B models; faster than g4dn |
| 10 GB | Up to 13 B params | ml.g6e.2xlarge (L40S) | 8 | 64 | 1 × L40S | `$2.80` | 27B in FP8 with more RAM and disk; RAG serving over larger corpora |
| 50 GB | Up to 1 B params | ml.c5.4xlarge | 16 | 32 | None | `$0.68` (approx.) | Large CPU modeling / preprocessing |
| 50 GB | Up to 1 B params | ml.m5.4xlarge | 16 | 64 | None | `$0.92` (approx.) | Classic ML on wide data |
| 50 GB | Up to 14 B params | ml.g5.4xlarge (A10G) | 16 | 64 | 1 × A10G | `$2.03` | Solid for 7–14B models (fp16 or 4/8-bit) |
| 50 GB | Up to 14 B params | ml.g6e.4xlarge (L40S) | 16 | 128 | 1 × L40S | `$3.76` | 27B in FP8 with headroom, or 13B in bf16 |
| 100 GB | Up to 40–50 B params | ml.g5.12xlarge (A10G) | 48 | 192 | 4 × A10G | `$7.09` | Multi-GPU training; distributed fine-tuning |
| 100 GB | Up to 70 B params | ml.g6e.12xlarge (L40S) | 48 | 384 | 4 × L40S | `$13.12` | Multi-GPU fine-tuning of 30–70B models with LoRA |
| 100 GB | Up to 70 B+ params | ml.p4d.24xlarge (A100) | 96 | 1,152 | 8 × A100 (40 GB) | `$25.25` | High-end DL / LLM fine-tuning |
| 1 TB+ | Up to 70 B+ params | ml.p4d.24xlarge (A100) | 96 | 1,152 | 8 × A100 (40 GB) | `$25.25` | Distributed training for foundation-scale models |
| 1 TB+ | 70 B+ params, full fine-tunes | ml.p5.48xlarge (H100) | 192 | 2,048 | 8 × H100 (80 GB) | `$63.30` | Full (non-LoRA) fine-tuning and pretraining; where five-figure bills come from |

A quick way to read the table: find the row whose memory is at least 1.5× your dataset, then move down to the first GPU row only if the task is deep learning. For the workshop, the controller is always `ml.t3.medium`, the Titanic training jobs use `ml.m5.large` or `ml.g4dn.xlarge`, the fine-tuning job uses `ml.g5.xlarge`, and the RAG Processing jobs use `ml.g5.xlarge` (embedding) and `ml.g6e.xlarge` (the 27B generator).

---
title: 'Account preparation checklist (quotas, Bedrock, tags)'
---

Everything that has to be requested or configured on the class AWS account before the workshop, sized for **50 students**. GPU quota increases often take days and sometimes a support exchange, so file them three to four weeks ahead. Instance usage per episode is taken from the code in the episodes; if you change an instance type in an episode, change the quota here.

## SageMaker service quotas

Console: **Service Quotas → AWS services → Amazon SageMaker**, in the region the class uses. Names below are the quota names as the console lists them.

| Quota | Request | Used by |
|---|---|---|
| `ml.t3.medium for notebook instance usage` | 60 | controller notebook, every episode |
| `ml.g6e.xlarge for notebook instance usage` | 60 | GPU RAG episode (Qwen3.8-27B via vLLM). **Default is usually 0** |
| `Total number of notebook instances` | 120 | two per student |
| `ml.m5.large for training job usage` | 120 | XGBoost episode; PyTorch part 2; tuning (2 parallel jobs each) |
| `ml.m5.xlarge for training job usage` | 120 | PyTorch part 2 runs `instance_count=2` |
| `ml.g4dn.xlarge for training job usage` | 60 | PyTorch part 2 GPU comparison |
| `ml.g5.xlarge for training job usage` | 60 | fine-tuning episode (one job per student at once) |
| `ml.g6e.xlarge for training job usage` | 20 | optional: fine-tuning Exercise 2 |
| `Maximum number of instances across all training jobs` | 250 | must exceed the largest concurrent total above |
| `ml.g5.xlarge for processing job usage` | 60 | Processing Jobs episode (two jobs per student, sequential); fine-tuning homework |
| `ml.g6e.xlarge for processing job usage` | 60 | only if the Processing Jobs episode is moved to the 27B model |
| `Maximum number of instances across all processing jobs` | 60 | |

Also check `ml.g5.xlarge for training warm pool usage` is left at its default of 0; the episodes do not use warm pools and a nonzero quota lets a student leave an idle GPU billing.

## Bedrock

Console: **Service Quotas → Amazon Bedrock**, and **Bedrock → Model access**.

- Enable model access for the two models the Bedrock episode uses: `deepseek.v3-v1:0` (generation) and `amazon.titan-embed-text-v2:0` (embeddings).
- Request increases to the **tokens per minute** and **requests per minute** quotas for both. Fifty notebooks each embedding ~3,000 chunks in a loop will exceed the default embedding RPM; ask for 5–10x whatever the console shows.
- The student execution role needs, in addition to `bedrock:InvokeModel`: `bedrock:CreateInferenceProfile`, `bedrock:GetInferenceProfile`, `bedrock:ListInferenceProfiles`, `bedrock:GetFoundationModel`, `bedrock:TagResource`. `InvokeModel` must be allowed on **both** the application inference profile ARN and the underlying model ARN. Without these, profile creation fails and students fall back to untagged calls, which is the usual reason Bedrock spend shows up untagged.

## Cost allocation tags

- Confirm with the Public Cloud team that `Name`, `Project`, `Purpose` are active cost allocation keys on the account ([KB 72454](https://kb.wisc.edu/72454)). The lesson uses exactly those three everywhere.
- Set a Budgets alert on the account at the planned class total, and a Cost Explorer saved report filtered on `Project`.

## Data and files

- Shared read-only bucket with the Badger Scribe data and your full fine-tune outputs: see the fine-tuning cost page.
- WattBot data: students upload `data.zip` contents to their own bucket in the Bedrock episode; nothing to stage.

## Capacity fallbacks to have on a slide

- GPU notebook episode: if `ml.g6e.xlarge` capacity is short, `ml.g6e.2xlarge` ($2.80/h, same GPU) is the only same-memory alternative; a 24 GB instance cannot hold the 27B model.
- Fine-tuning episode: `ml.g6.xlarge` ($1.13/h, L4, 24 GB) runs the mini job at about the same speed as `ml.g5.xlarge`.

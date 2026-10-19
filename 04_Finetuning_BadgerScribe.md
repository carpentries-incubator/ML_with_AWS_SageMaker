---
title: "Fine-tuning a Vision-Language Model with Training Jobs"
teaching: 60
exercises: 45
---

:::::::::::::::::::::::::::::::::::::: questions

- When is fine-tuning worth it, and how is it different from RAG or prompting?
- Why does fine-tuning cost anywhere from a dollar to tens of thousands, and how do I estimate where my run falls *before* I launch it?
- How do I run a GPU fine-tune as a SageMaker training job so the meter stops the moment it finishes?
- How do I measure whether the fine-tune actually helped?

::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: objectives

- Explain what drives the cost of a fine-tuning run (model size × tokens × epochs × number of attempts) and why the same three habits apply at $1 and at $10,000: a time cap, a small first run, and an estimate written down before launch.
- Launch a LoRA fine-tune of a 3B vision-language model (CHURRO-3B) as a `HuggingFace` training job on a single 24 GB GPU, with tags, a unique job name and a `max_run` cap.
- Read a job's billable time and compare it to the estimate.
- Compare the base model and the fine-tuned model on held-out pages using the competition's own character error rate (CER).
- Know how to scale the same job up to the full dataset and, optionally, to a Kaggle submission (homework).

::::::::::::::::::::::::::::::::::::::::::::::::

## Why a fine-tuning episode

Everything so far in the RAG episodes left the model's weights alone. Fine-tuning changes them, and that puts it in a different class of workload:

- It needs a GPU for the whole run, not just for inference calls.
- It is iterative. Nobody gets the learning rate, the image resolution or the data split right the first time, so the first run is never the last.
- The cost of a run depends on things that are easy to get wrong by 10x: model size, epochs, image resolution, instance type, and whether the instance is still running after you stopped paying attention.

That combination is why fine-tuning is where research teams see the surprise bills. It is also, done carefully, the single biggest lever you have: the Badger Scribe organizers report that LoRA on ~100 pages takes a small open model from 0.23 to 0.06 CER on survey notebooks in about a GPU-hour. This episode is about getting that kind of gain for a few dollars, and about knowing in advance when a run is going to cost a few thousand instead.

### The task: Badger Scribe

[Badger Scribe](https://www.kaggle.com/competitions/badger-scribe) is a Kaggle challenge built from UW Digital Collections material: 19th-century surveyors' field notebooks, craftsmen's account books, and German immigrant letters in Kurrent script. Given a page image, produce a verbatim transcription. The score is character error rate (CER): edits per reference character, lower is better.

This lesson uses the German letters, the Anna Seifert correspondence held by the Max Kade Institute: 191 letters and postcards written in Kurrent between 1932 and 1942 by one woman to her son, transcribed by the Institute's Kurrent Transcription Group. It is the part of the competition data where every label was produced by a person, and it is most of the hidden test set.

| split | pages | documents | label source |
|---|---|---|---|
| train | 295 | 4 letter bundles | all human-transcribed, ~620 characters per page |
| test | 141 | 5 other letter bundles | labels held by Kaggle |

Small is the point. 295 labelled pages is a realistic amount of data for a library or a lab to have, and it is enough to move a model that was pre-trained on the right kind of material. Kurrent is also a genuinely different script from modern German cursive, so this is a case where the model has to learn something, not just adjust its style.

We will fine-tune [CHURRO-3B](https://github.com/stanford-oval/Churro), a 3-billion-parameter open-weight vision-language model from Stanford that was itself fine-tuned from Qwen2.5-VL-3B on roughly 100,000 pages of historical documents in 46 language clusters. It already reads old handwriting better than its base model does; we are teaching it *these* hands and *these* transcription conventions.

::::::::::::::::::::::::::::::::::::: callout

### Fine-tune, RAG, or prompt?

Ask what the model is missing. If it lacks **facts** (your documents, your data), retrieval gives it those at inference time and no training is needed. If it lacks a **skill or a style** (reading Kurrent, following your lab's labelling conventions, producing output in a strict format), you need to change the weights, and that is fine-tuning. Badger Scribe is squarely the second case: no amount of context will teach a model what a Kurrent "e" looks like.

Full fine-tuning updates all 3 billion weights and needs several times the model's size in GPU memory for optimizer state. **LoRA** (low-rank adaptation) freezes the model and trains a few million extra parameters in small matrices bolted onto the attention and MLP layers. The adapter is tens of megabytes, trains on a single 24 GB GPU, and recovers most of the gain of a full fine-tune for a task like this. We use LoRA.

::::::::::::::::::::::::::::::::::::::::::::::::

## Step 0: Estimate the cost before you launch anything

This is the habit that matters. Before launching a fine-tune, write down a number for what it should cost. If the real bill is 3x your number, something is wrong and you want to know on the first run, not the fiftieth.

### What fine-tuning costs, from one dollar to ten thousand

The cost of a run is GPU-hours × the hourly rate, and GPU-hours scale with **model size × tokens seen × epochs**, multiplied by **how many runs you do before one works**. Our class run sits at the very bottom of that curve. The rows below use current SageMaker on-demand prices in `us-east-1` (AWS price list, October 2026):

| Run | Hardware | GPU-hours | Cost |
|---|---|---|---|
| **This episode:** 3B, LoRA, 120 Kurrent pages × 2 epochs | 1 × A10G (`ml.g5.xlarge`, $1.41/h), ~25 min | 0.4 | **~$0.60** |
| Homework: 3B, LoRA, all 295 pages × 2 epochs | 1 × A10G, ~45 min | 0.75 | ~$1.05 |
| 7B full fine-tune, ~50k documents, a few epochs | 8 × A100 (`ml.p4d.24xlarge`, $25.25/h), ~10 h | 80 | ~$250 |
| 70B LoRA, one day | 8 × A100, 24 h | 192 | ~$600 |
| CHURRO-3B itself: full fine-tune, 100k pages, 5 epochs | 32 × H100 for ~25 h (4 × `ml.p5.48xlarge`, $63.30/h) | 800 | ~$6,300 |
| 70B full fine-tune, two days | 4 × `ml.p5.48xlarge`, 48 h | 1,536 | ~$12,000 |
| Any row above × a 10-run hyperparameter sweep | | | × 10 |

Two rows deserve a second look. The CHURRO row is the model you are about to fine-tune: Stanford spent on the order of 800 H100-hours making it, and you get the benefit for sixty cents because you are *adapting* it, not retraining it. Starting from a model that already knows most of the task is the single largest cost saving available, and it is a choice you make before any GPU is involved. The sweep row is the other honest one. Nobody's first run is their last, so a "$250 fine-tune" is usually a $1,000 to $2,500 project, and that is the number that surprises research groups. (The 7B and 70B rows are rule-of-thumb magnitudes; the CHURRO figures come from a published review of the paper.)

The habits in this episode are cheap practice at sixty cents a run. At $250 a run they are the difference between a project and a budget incident.

### Back of the envelope: will it fit, and how long will it take?

Two rules of thumb answer most of the question before you touch a console. They are good to within a factor of two, which is what you need to tell a $5 run from a $500 one.

**1. Memory: will it fit on the GPU?** Count bytes per parameter for what you are doing, multiply by the parameter count, then add a few GB for activations (more for long sequences and big batches).

| What you are doing | Bytes per parameter | Why | 3B | 7B | 27B | 70B |
|---|---|---|---|---|---|---|
| Inference, 4-bit weights | 0.5 | weights only | 1.5 GB | 3.5 GB | 14 GB | 35 GB |
| Inference, FP8 | 1 | weights only | 3 GB | 7 GB | 27 GB | 70 GB |
| Inference, bf16 | 2 | weights only | 6 GB | 14 GB | 54 GB | 140 GB |
| LoRA fine-tune | ~2.5 | frozen bf16 weights + a small adapter with its gradients and optimizer | 8 GB | 18 GB | 68 GB | 175 GB |
| Full fine-tune, mixed precision | 16 | bf16 weights 2 + bf16 gradients 2 + fp32 master weights 4 + Adam's two moments 8 | 48 GB | 112 GB | 430 GB | 1.1 TB |

The jump from the inference rows to the training rows is the point. Training cannot run on quantized weights the way inference can: gradient updates are tiny numbers that vanish in 8-bit or 4-bit precision, so every trainable parameter keeps a full-precision master copy plus two Adam moments, 16 bytes against 2. That is why a 7B model you can *run* on a 24 GB GPU needs 112 GB plus activations to *fully fine-tune*, which means several GPUs with the model sharded across them. LoRA sidesteps it by freezing the model and training a few million adapter parameters, which is why this episode fits on one 24 GB card.

**2. Time: how many GPU-hours?** Training compute is roughly proportional to parameters × tokens seen:

```
tokens      = examples × tokens per example × epochs
FLOPs       ≈ 6 × parameters × tokens      (full fine-tune)
            ≈ 4 × parameters × tokens      (LoRA: no weight gradients for the frozen base)
GPU-seconds = FLOPs / (GPU peak bf16 FLOPS × utilization)
```

Utilization is the number people forget. A GPU running one page at a time with gradient checkpointing manages 20–30% of its peak; a well-tuned multi-GPU job reaches 40–50%. Use 25% unless you have measured otherwise. Then cost = GPU-hours × hourly rate + the fixed startup overhead per job (~10 minutes here).

Worked for the class run: 120 pages × ~1,600 tokens (1,300 visual + 300 text) × 2 epochs = 384k tokens. LoRA on a 3.75B model (3B language model plus the vision tower): 4 × 3.75e9 × 3.84e5 ≈ 5.8e15 FLOPs. An A10G peaks at about 70 TFLOPS in bf16; at 25% utilization that is 17 TFLOPS, so about 340 seconds, call it 6 minutes. The "~10 min" in the table below adds the image preprocessing on four vCPUs and the checkpointing overhead. Same arithmetic, same answer within the factor of two.

**More GPUs do not make it cheaper.** GPU-hours are the cost; spreading them across eight GPUs changes the wall-clock time, not the bill, and only helps if the job is large enough to keep all eight busy. You add GPUs when the memory row above says one GPU cannot hold the job, not to save money.

::::::::::::::::::::::::::::::::::::: challenge

### Estimate a run you have not launched

A lab wants to **fully fine-tune a 7B model** on 50,000 documents of ~2,000 tokens for 3 epochs. They are choosing between `ml.p4d.24xlarge` (8 × A100 40 GB, $25.25/h, ~312 TFLOPS bf16 per GPU) and renting a single `ml.g5.xlarge`. Using the two rules: does it fit on the g5? How many GPU-hours on the p4d, and what does one run cost? What does a realistic project cost?

:::::::::::::::::::::::: solution

It does not fit on the g5: full fine-tuning a 7B model needs ~112 GB plus activations, and the g5 has 24 GB. On the p4d the 8 × 40 GB = 320 GB holds it with the model sharded.

Tokens: 50,000 × 2,000 × 3 = 3 × 10^8. FLOPs: 6 × 7 × 10^9 × 3 × 10^8 ≈ 1.3 × 10^19. Per GPU at 312 TFLOPS × 40% utilization ≈ 125 TFLOPS: 1.0 × 10^5 GPU-seconds ≈ 28 GPU-hours, which is 3.5 hours of wall-clock on 8 GPUs, about $90 for one run.

A realistic project is not one run. The first run finds a data bug, the second a bad learning rate, then two or three are compared, then the one that worked is rerun on the final data: five to ten runs, so $450 to $900, before anyone touches a hyperparameter sweep. That is the number to put in the proposal, and it is why the three habits matter more as the model gets bigger.

:::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::::::::::::::

### The instance

The model we fine-tune fits on one 24 GB GPU. Single-GPU and multi-GPU SageMaker training instances, same price list:

| Instance | GPU | GPU memory | $/hour |
|---|---|---|---|
| `ml.g4dn.xlarge` | 1 × T4 | 16 GB | $0.74 |
| `ml.g6.xlarge` | 1 × L4 | 24 GB | $1.13 |
| `ml.g5.xlarge` | 1 × A10G | 24 GB | $1.41 |
| `ml.g6e.xlarge` | 1 × L40S | 48 GB | $2.61 |
| `ml.g5.12xlarge` | 4 × A10G | 96 GB | $7.09 |
| `ml.p4d.24xlarge` | 8 × A100 | 320 GB | $25.25 |
| `ml.p5.48xlarge` | 8 × H100 | 640 GB | $63.30 |

Multi-GPU instances are the right tool when the model, its optimizer state and a batch do not fit on one GPU: a full fine-tune of a 7B model, or anything at 70B. They are the wrong tool when the job *does* fit on one GPU, because the price goes up 5x to 45x and a job too small to split gets no faster. A 3B LoRA run fits with room to spare, so `ml.g5.xlarge` is the workhorse for this episode. A T4 is too small and too slow for a 3B vision model with page-sized images; `ml.g6.xlarge` is a fine substitute if g5 capacity is short; `ml.g6e.xlarge` is roughly 2–3x faster for less than 2x the price, so it is often *cheaper per run* (Exercise 2).

### The run

A training job bills from the moment its instance is ready until the job ends, including the minutes spent pulling the container and downloading data and model weights. In class we train on 120 German-letter pages for 2 epochs and score 15 held-out pages before and after, all inside one job:

| Phase | Time | Cost |
|---|---|---|
| Startup: container pull, 2 GB of images, 7 GB of model weights | ~10 min | $0.24 |
| Score 15 holdout pages with the base model, ~10 s each | ~3 min | $0.07 |
| Training: 120 pages × 2 epochs, ~2.5 s per page with gradient checkpointing | ~10 min | $0.24 |
| Score the same 15 pages with the fine-tuned model | ~3 min | $0.07 |
| **One run** | **~25 min** | **~$0.60** |

Training time scales linearly with pages × epochs, and roughly linearly with image resolution (every 28 × 28 pixel block of the page becomes one token in the model's sequence). Doubling `max_pixels` doubles the time; doubling epochs doubles it again. Those two knobs, left at careless values, are why the estimate must be written down before the first run and compared to the real `BillableTimeInSeconds` after it.

We set `max_run` to about twice the estimate. If a run goes past that, SageMaker kills it and we lose at most a dollar, not a weekend.

### The whole episode

| What | Instance | Time | Cost |
|---|---|---|---|
| Controller notebook (no GPU needed) | `ml.t3.medium` | 3 h | $0.15 |
| Mini fine-tune with before/after scoring | `ml.g5.xlarge` | ~25 min | $0.60 |
| Exercise 1: same job at 2 megapixels | `ml.g5.xlarge` | ~35 min | $0.80 |
| **Total** | | | **~$1.55** |

Compare the classic mistake: doing all of this interactively on a `ml.g5.xlarge` *notebook* instance. The class itself costs about the same, but a notebook instance keeps billing after you close the tab. Forgotten over a weekend: $90. Forgotten for a month: $1,029. A training job cannot be forgotten, because it ends.

::::::::::::::::::::::::::::::::::::: callout

### The three habits

1. **Cap it.** Every job gets `max_run`. Every notebook GPU instance gets stopped when you stand up.
2. **Start small.** The first launch of any training script uses a slice of the data and a short cap. The in-class run below *is* that small first run; the homework scales it up only after it has been seen to work.
3. **Estimate it, then check.** Write the number down. After the run, read `BillableTimeInSeconds` from the job description and compare.

::::::::::::::::::::::::::::::::::::::::::::::::

## Setup

::::::::::::::::::::::::::::::::::::: instructor

### Before class (instructor checklist)

Students never download the data themselves; four things have to be in place before the session. Each is written out in the instructor pages.

1. **Stage the data in the shared read-only bucket** from the lesson's data release (`badger-scribe-data-v1`): about ten minutes from CloudShell, commands and the read-only bucket policy in *Shared data bucket: staging before each term* (the same bucket holds the WattBot data for the RAG episodes). The bucket must survive the end-of-workshop cleanup, or be recreated from the release next term.
2. **Run the full fine-tune once** (the estimator in the Homework section, all 251 training pages, `eval_pages` 40) and copy `train_summary.json`, `eval_before.csv` and `eval_after.csv` from its `model.tar.gz` to `badger-scribe/instructor/full-run/` in the shared bucket. Step 3 compares against these. Replace the time estimates in Step 0 with the job's real `BillableTimeInSeconds`.
3. **Quotas**: `ml.g5.xlarge for training job usage` at least the class size (request 100). Default is often 0 to 4 and increases take days; file three to four weeks ahead. Full list in *Account preparation checklist*.
4. **Tell students the bucket name** at the start; it is the `shared_bucket` value in the first code cell.

Fallback instance if g5 capacity is short: `ml.g6.xlarge` (L4, 24 GB, $1.13/h).

::::::::::::::::::::::::::::::::::::::::::::::::

This episode runs from a **CPU** notebook instance (`ml.t3.medium` or `ml.m5.xlarge`). The notebook is only a controller: it stages data, launches jobs, and reads results. The GPU lives in the job.

There is no data upload in this episode. The competition data is published as a [release on the lesson repository](https://github.com/carpentries-incubator/ML_with_AWS_SageMaker/releases) (three zip files: labels, train images, test images), and your instructor has staged it in a shared read-only bucket:

- `s3://<shared-bucket>/badger-scribe/train.csv`, `test.csv`, `metadata.csv`, `metric.py`, `sample_submission.csv`
- `s3://<shared-bucket>/badger-scribe/images/<page_id>.jpg`: train and test pages together, about 2 GB
- `s3://<shared-bucket>/badger-scribe/instructor/full-run/`: the output of a full fine-tune your instructor ran before class, for comparison in Step 3

The training job reads the images straight from that bucket; the only files your notebook downloads are the small CSVs. Outputs go to your own bucket from the S3 episode, under `badger-scribe/`.

::::::::::::::::::::::::::::::::::::: callout

### Working on your own account?

Download the three assets of the data release and sync them into a bucket of yours under the same `badger-scribe/` prefix, then set `shared_bucket` below to that bucket. From a terminal with the AWS CLI (the notebook instance's terminal works):

```bash
%%bash
V=1   # data release version; check the releases page for the latest
R=https://github.com/carpentries-incubator/ML_with_AWS_SageMaker/releases/download/badger-scribe-data-v$V
mkdir -p badger-scribe && cd badger-scribe
for a in labels images-train images-test; do
  curl -L -o $a.zip $R/badger-scribe-v$V-$a.zip && unzip -q -o $a.zip && rm $a.zip
done
cd .. && aws s3 sync badger-scribe/ s3://<your-bucket>/badger-scribe/
```

No Kaggle account is needed for any of this. The Step 3 comparison against an instructor full run will not apply; run the homework's full fine-tune yourself instead.

::::::::::::::::::::::::::::::::::::::::::::::::

```python
import os
import json
import time

import boto3
import pandas as pd
import sagemaker
from sagemaker import get_execution_role

session = sagemaker.Session()
region = session.boto_region_name
role = get_execution_role()
s3_client = boto3.client("s3", region_name=region)

shared_bucket = "ml-marathon-shared"   # <-- EDIT ME: read-only bucket with the competition data
bucket_name = "sinkorswim-doejohn"     # <-- EDIT ME: your own bucket from the S3 episode
username = "doejohn"                   # <-- EDIT ME: lowercase letters/digits only; used in job names

data_prefix = "badger-scribe"
train_s3_uri = f"s3://{shared_bucket}/{data_prefix}/"   # training channel: csvs + images/

local_data_dir = "./data"
os.makedirs(local_data_dir, exist_ok=True)

print("Region:", region)
print("Role:", role)
```

```python
for fname in ["train.csv", "test.csv", "metadata.csv", "metric.py"]:
    s3_client.download_file(shared_bucket, f"{data_prefix}/{fname}", os.path.join(local_data_dir, fname))

train_df = pd.read_csv(os.path.join(local_data_dir, "train.csv"), keep_default_na=False)
test_df = pd.read_csv(os.path.join(local_data_dir, "test.csv"), keep_default_na=False)

print("train pages:", len(train_df), " test pages:", len(test_df))
print(train_df.groupby(["category", "label_source"]).size())
```

Look at one page and its transcription. Note what "verbatim" means here: the writer's spelling and punctuation, umlauts and `ß` kept as written, Kurrent letterforms mapped to their Latin equivalents, and `#` for a character the transcriber could not read.

```python
row = train_df[train_df.category == "kade_letters"].iloc[3]
print(row.page_id, "|", row.category, "|", row.label_source)
print(row.text[:600])
```

```python
from IPython.display import Image, display

page_path = os.path.join(local_data_dir, f"{row.page_id}.jpg")
s3_client.download_file(shared_bucket, f"{data_prefix}/images/{row.page_id}.jpg", page_path)
display(Image(filename=page_path, width=500))
```

### The scripts

Jobs run in containers, so the code they execute has to be shipped with the job. Clone the lesson repository and copy the scripts into a `code/` directory:

```python
!git clone https://github.com/carpentries-incubator/ML_with_AWS_SageMaker.git
```

```python
import shutil

os.makedirs("code", exist_ok=True)
for f in ["badger_common.py", "finetune_churro_lora.py", "churro_transcribe_batch.py"]:
    shutil.copy(f"ML_with_AWS_SageMaker/scripts/{f}", f"code/{f}")

with open("code/requirements.txt", "w") as f:
    f.write("\n".join(["peft>=0.14", "accelerate", "pillow", "pandas"]))

print(os.listdir("code"))
```

- `badger_common.py`: the prompt, image loading, and a copy of the competition's CER. Shared by the other two so the model is trained on exactly the prompt it is later asked.
- `finetune_churro_lora.py`: the training job entry point. Holds out 15% of pages per category, filters to the categories you ask for, scores a sample of holdout pages with the base model, trains a LoRA adapter, scores the same pages again, and saves everything.
- `churro_transcribe_batch.py`: used in the homework. Transcribes any list of pages with the base model or with the adapter merged in, and writes `predictions.csv` in the submission format.

Open `code/finetune_churro_lora.py` and read it before launching it. The pieces worth understanding:

```python
# Only the language model's projection layers get LoRA weights.
# The vision tower ("visual") stays frozen.
lora = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
    target_modules=r"^(?!.*visual).*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$",
)

# The loss is computed on the transcription only. Prompt and image tokens
# are masked to -100 so the model is not graded on repeating the question.
labels[i, :n_prompt] = -100
```

And the one setting that controls cost more than any other:

```python
# Every 28x28 block of the page becomes one token. 1280 blocks ~ 1 megapixel.
DEFAULT_MAX_PIXELS = 1280 * 28 * 28
```

### Tags

As in every episode: tags do not propagate from the notebook to the jobs it launches. Pass them to every estimator.

```python
name = "John Doe"        # replace with your name
project = "sinkorswim"   # replace with your team name

def make_job_tags(purpose: str):
    return [
        {"Key": "Name", "Value": name},
        {"Key": "Project", "Value": project},
        {"Key": "Purpose", "Value": purpose},
    ]
```

## Step 1: Launch the fine-tune

One job does everything: scores 15 held-out German-letter pages with the base model, trains on 120 German-letter pages for 2 epochs, scores the same 15 pages again, and saves the adapter. The hyperparameters below are what the Step 0 estimate was built from.

```python
from sagemaker.huggingface import HuggingFace

instance_type = "ml.g5.xlarge"
model_output_path = f"s3://{bucket_name}/{data_prefix}/models/"

estimator = HuggingFace(
    entry_point="finetune_churro_lora.py",
    source_dir="code/",
    role=role,
    tags=make_job_tags("finetune-churro-mini"),
    instance_type=instance_type,
    instance_count=1,
    volume_size=50,                       # GB of disk for the container, weights and images
    max_run=45 * 60,                      # 45 minutes: about twice the estimate, never "whatever it takes"
    transformers_version="4.56",
    pytorch_version="2.8",
    py_version="py312",
    output_path=model_output_path,
    sagemaker_session=session,
    hyperparameters={
        "model_id": "stanford-oval/churro-3B",
        "max_train_pages": 120,           # a slice in class; the homework uses all 250 training pages
        "epochs": 2,
        "lr": 2e-4,
        "lora_r": 16,
        "batch_size": 1,
        "grad_accum": 4,                  # effective batch of 4 pages -> ~60 optimizer steps
        "max_pixels": 1280 * 28 * 28,     # ~1 megapixel per page
        "eval_pages": 15,
        "eval_before": 1,
    },
)
```

Before launching, check that you do not already have a job running. An empty list means go.

```python
session.sagemaker_client.list_training_jobs(StatusEquals="InProgress", NameContains="badger")["TrainingJobSummaries"]
```

Now launch. Two details in this one line matter:

- `job_name` is set explicitly. SageMaker requires training job names to be unique in the account, so running this cell a second time fails immediately with `ResourceInUse` and nothing is launched. To run again on purpose, change `v1` to `v2`.
- `wait=True` (the default) blocks the notebook until the job ends and streams its log here. For a 25-minute job that is the right choice: you cannot accidentally queue a second launch while you wait, and everything worth doing in the meantime happens in the console, not in this notebook.

```python
training_job_name = f"badger-{username}-mini-v1"

estimator.fit({"train": train_s3_uri}, job_name=training_job_name, wait=True)
```

::::::::::::::::::::::::::::::::::::: callout

### Interrupting the cell does not stop the job

The square "interrupt kernel" button stops the *notebook* from watching the job. The GPU instance keeps running and keeps billing until the script finishes or `max_run` expires. To actually stop a job, use the console (**SageMaker AI → Training → Training jobs → select → Stop**) or:

```python
# session.sagemaker_client.stop_training_job(TrainingJobName=training_job_name)
```

Billing stops within a minute or two. If you launched with the wrong instance, the wrong data or the wrong number of epochs, stop it immediately; the cents already spent are not a reason to let it run.

::::::::::::::::::::::::::::::::::::::::::::::::

### While it runs

Open a second browser tab on the SageMaker console: **SageMaker AI → Training → Training jobs**. Find your job and watch its status move through `Starting`, `Downloading`, `Training`, `Uploading`, `Completed`. The job page shows the instance type, the `max_run` you set, the tags, and a link to the CloudWatch log, which is where every `print()` from the script goes. The log is the same text the notebook cell is streaming, but it stays available after the kernel is gone, which is what you will use when a job you launched yesterday fails overnight.

Things to find in the log while you wait:

- `pages: 295 total -> 251 train / 44 holdout`: the split, before the 120-page slice.
- `trainable params: ...`: what fraction of the 3B parameters LoRA actually trains.
- `[eval before] 15 holdout pages in ...s (... s/page) -> macro CER ...`: the base model's score, and the per-page time that the homework inference jobs will cost.
- The Trainer's loss lines every 5 steps.

## Step 2: Read the results and the bill

When the cell returns, ask SageMaker what it charged for. This is the number to compare against the estimate every time.

```python
def job_cost_report(job_name: str, price_per_hour: float):
    desc = session.sagemaker_client.describe_training_job(TrainingJobName=job_name)
    billable = desc["BillableTimeInSeconds"]
    training = desc.get("TrainingTimeInSeconds", 0)
    print(f"job:            {job_name}")
    print(f"status:         {desc['TrainingJobStatus']}")
    print(f"training time:  {training / 60:.1f} min  (script running)")
    print(f"billable time:  {billable / 60:.1f} min  (what you pay for, incl. download)")
    print(f"cost:           ${billable / 3600 * price_per_hour:.2f} at ${price_per_hour}/h")
    return desc

g5_price = 1.41
desc = job_cost_report(training_job_name, g5_price)
adapter_s3_uri = desc["ModelArtifacts"]["S3ModelArtifacts"]
print("adapter:", adapter_s3_uri)
```

The gap between *training time* and *billable time* is the fixed overhead of every job: container pull, data download, model download. It is the same whether you train for one minute or one hour, which is why a small first run is cheap and why batching work into fewer, longer jobs is cheaper than many tiny ones.

The whole output of the job is a `model.tar.gz` of a few tens of megabytes: the LoRA adapter, the processor configuration, the holdout list, the two evaluation files and a summary. The 3B base model is never copied anywhere.

```python
import tarfile

run_dir = os.path.join(local_data_dir, training_job_name)
os.makedirs(run_dir, exist_ok=True)
bucket, key = adapter_s3_uri.replace("s3://", "").split("/", 1)
s3_client.download_file(bucket, key, os.path.join(run_dir, "model.tar.gz"))
with tarfile.open(os.path.join(run_dir, "model.tar.gz")) as tar:
    tar.extractall(run_dir)
print(sorted(os.listdir(run_dir)))

summary = json.load(open(os.path.join(run_dir, "train_summary.json")))
print(json.dumps(summary, indent=2))
```

### Before vs. after

Both evaluations ran on the same 15 pages, so the difference is the fine-tune and nothing else.

```python
before = pd.read_csv(os.path.join(run_dir, "eval_before.csv"), keep_default_na=False)
after = pd.read_csv(os.path.join(run_dir, "eval_after.csv"), keep_default_na=False)

comparison = before[["page_id", "cer"]].merge(after[["page_id", "cer"]], on="page_id", suffixes=("_base", "_tuned"))
comparison["gain"] = comparison.cer_base - comparison.cer_tuned
print(comparison.sort_values("gain", ascending=False).round(3).to_string(index=False))
print(f"\nmean CER  base: {comparison.cer_base.mean():.3f}   tuned: {comparison.cer_tuned.mean():.3f}")
```

Look at the page that improved most. The fine-tuned model should have picked up the conventions (umlauts and `ß` as written, no modernised spelling, Kurrent `s` and `e` forms read correctly) as well as this particular writer's hand.

```python
best = comparison.sort_values("gain", ascending=False).iloc[0].page_id
ref = before.set_index("page_id").loc[best, "text_ref"]
print("REFERENCE\n", ref[:500])
print("\nBASE\n", before.set_index("page_id").loc[best, "text_pred"][:500])
print("\nTUNED\n", after.set_index("page_id").loc[best, "text_pred"][:500])
```

::::::::::::::::::::::::::::::::::::: challenge

### Was the estimate right?

1. Compare the billable time from `job_cost_report` with the ~25 minutes estimated in Step 0. Which phase was off, and by how much? (`train_minutes` and `seconds_per_page` in `train_summary.json` tell you.)
2. Using your measured seconds per page, estimate the homework run: 251 pages × 2 epochs, plus 40 evaluation pages before and after, plus the same startup overhead. Write the number down; you will compare it to the real bill after the homework.

:::::::::::::::::::::::: solution

1. Expect startup to be close to 10 minutes and the two evaluations to be 2–4 minutes each. Training is the line most likely to differ: if you measured 2.5 s/page, 240 page-passes is 10 minutes; at 4 s/page it is 16.
2. At 2.5 s/page: 502 passes is 21 minutes, plus ~13 minutes of evaluation at 10 s/page, plus 10 minutes startup, so about 45 minutes and $1.05. Scale every line by the ratio you measured.

:::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::::::::::::::

## Step 3: What a full run buys

Your instructor ran the same script before class on all 251 training pages (2 epochs, 40 evaluation pages). Its output is in the shared bucket. The holdout split is deterministic, so its evaluation pages include yours.

```python
full_dir = os.path.join(local_data_dir, "instructor-full-run")
os.makedirs(full_dir, exist_ok=True)
for fname in ["train_summary.json", "eval_before.csv", "eval_after.csv"]:
    s3_client.download_file(shared_bucket, f"{data_prefix}/instructor/full-run/{fname}", os.path.join(full_dir, fname))

full_summary = json.load(open(os.path.join(full_dir, "train_summary.json")))
print("full run:", full_summary["train_pages"], "pages,", full_summary["train_minutes"], "min of training")
print("CER before :", full_summary["holdout_eval_before"]["per_category"])
print("CER after  :", full_summary["holdout_eval_after"]["per_category"])
```

Put the three models side by side on the German-letter pages they have in common:

```python
full_after = pd.read_csv(os.path.join(full_dir, "eval_after.csv"), keep_default_na=False)
three = (comparison
         .merge(full_after[["page_id", "cer"]].rename(columns={"cer": "cer_full"}), on="page_id", how="inner")
         [["page_id", "cer_base", "cer_tuned", "cer_full"]])
print(three.round(3).to_string(index=False))
print("\nmean on", len(three), "shared pages:", three[["cer_base", "cer_tuned", "cer_full"]].mean().round(3).to_dict())
```

Three numbers: the base model, 10 minutes of training on 120 pages, and 20 minutes on 251. The second gap is what more data and more time buy, and it cost about fifty cents. Whether that dollar is worth it depends on the gap; whether the next $250 is worth it is the same question at a different scale, and the only way to answer it is to measure, as you just did.

::::::::::::::::::::::::::::::::::::: callout

### Why the holdout number flatters you

The script holds out 15% of *pages*, so pages from the same letter bundle appear in both the training and holdout halves. The competition's hidden test set is split by *document*: the five test bundles are letters the model has never seen. All the letters are from one writer, which softens the gap here, but her hand, pen and paper change over a decade, so expect the leaderboard CER to be somewhat worse than the holdout CER. A document-level holdout is the fix; it is left as an exercise because with only four bundles in train, holding one out means giving up a quarter of the data.

::::::::::::::::::::::::::::::::::::::::::::::::

## Step 4: What did this episode cost?

Add it up from the job descriptions, not from memory.

```python
total = 0.0
for job in session.sagemaker_client.list_training_jobs(NameContains=f"badger-{username}", MaxResults=50)["TrainingJobSummaries"]:
    d = session.sagemaker_client.describe_training_job(TrainingJobName=job["TrainingJobName"])
    cost = d.get("BillableTimeInSeconds", 0) / 3600 * g5_price
    total += cost
    print(f"{job['TrainingJobName']:<40} {d['TrainingJobStatus']:<10} {d.get('BillableTimeInSeconds', 0) / 60:5.1f} min  ${cost:.2f}")
print(f"\nGPU spend so far: ${total:.2f}  (plus the CPU notebook at $0.05/h)")
```

Then compare that number to the estimate in Step 0. If you are within 50%, your estimating is good enough to be trusted with a bigger run. If you are not, find out which line was wrong before launching anything else.

::::::::::::::::::::::::::::::::::::: challenge

### Exercise 1: The resolution knob

Relaunch the same job with `"max_pixels": 2560 * 28 * 28` (~2 megapixels) and the job name `badger-{username}-mini-v2`. Before it starts, write down your prediction for its billable time. Then compare `train_minutes` and `seconds_per_page` in the two summaries, and the before/after CER. Did the extra pixels help?

:::::::::::::::::::::::: solution

Time per page roughly doubles: the sequence is twice as long and attention cost grows faster than linearly. On these scans the strokes are legible well below 1 megapixel, so the extra pixels mostly buy tokens that carry no information, and CER usually moves very little. Resolution is the first knob to turn *down* when a fine-tune is over budget, and the first to turn *up* if the model is misreading small characters such as the `½` fractions.

:::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: challenge

### Exercise 2: Cheaper per run, or just faster?

`ml.g6e.xlarge` (one L40S, 48 GB) costs $2.61/h, 1.85x the price of `ml.g5.xlarge`, and runs this workload roughly 2.5–3x faster. It also has twice the GPU memory, so `batch_size` can be 2 or 4 instead of 1, which uses the GPU more efficiently.

Using the Step 0 "The run" table, estimate the cost of the class job on `ml.g6e.xlarge`. Which phase does *not* get faster?

:::::::::::::::::::::::: solution

Startup (~10 min, $0.44) does not speed up: the container pull and the downloads are the same. Training drops from ~10 to ~4 minutes and each evaluation from ~3 to ~1 minute, so the job is about 16 minutes and ~$0.70: a little *more* than the g5 run, because on a job this short the fixed overhead dominates. On the homework run (50 minutes of training) the same arithmetic gives ~30 minutes and ~$1.30, cheaper than g5. The lesson is that the hourly rate is the wrong thing to compare; the cost per run is what matters, and the fixed overhead is a larger share of it on faster hardware. The catch is quotas: the account needs a service quota for `ml.g6e.xlarge` training jobs, and those often default to zero.

:::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: challenge

### Exercise 3: Managed spot training

SageMaker can run a training job on spare ("spot") capacity at a discount of up to 70% in exchange for the risk of being interrupted. Add `use_spot_instances=True` and `max_wait=3 * 60 * 60` (the total time allowed including waiting for capacity; it must be at least `max_run`) to the estimator and relaunch under a new job name.

What else would the script need before this is safe to rely on for a 10-hour run?

:::::::::::::::::::::::: solution

Checkpointing. If the instance is reclaimed, the job restarts and, as written, starts training from scratch. For a 25-minute job that costs 25 minutes; for a ten-hour job it can cost the whole run. The fix is to save checkpoints to `/opt/ml/checkpoints` (which SageMaker syncs to S3 via `checkpoint_s3_uri`) with `save_strategy="steps"` and resume from the latest one when the script starts. Spot is a large saving on long runs once that is in place; for a short class exercise with 50 people launching at once, the interruptions and capacity waits are not worth it.

:::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::::::::::::::

## Homework: the full run and (optionally) a Kaggle submission

Everything below is the same pattern at a larger scale: about $1.05 for the fine-tune and $0.80 for inference, roughly an hour and a half of wall-clock time that you do not need to sit through. Launch, go do something else, come back.

### The full fine-tune

Same estimator with the page slice removed, all 251 training pages, and a 90-minute cap. Write your estimate down first (Step 2's challenge). Launch with `wait=False` here only because you will not be watching: it returns immediately and you poll later. The explicit job name still protects you from a double launch.

```python
full_estimator = HuggingFace(
    entry_point="finetune_churro_lora.py",
    source_dir="code/",
    role=role,
    tags=make_job_tags("finetune-churro-full"),
    instance_type=instance_type,
    instance_count=1,
    volume_size=50,
    max_run=90 * 60,                      # 90 minutes: twice the estimate
    transformers_version="4.56",
    pytorch_version="2.8",
    py_version="py312",
    output_path=model_output_path,
    sagemaker_session=session,
    hyperparameters={
        "model_id": "stanford-oval/churro-3B",
        "epochs": 2,
        "lr": 2e-4,
        "lora_r": 16,
        "batch_size": 1,
        "grad_accum": 8,
        "max_pixels": 1280 * 28 * 28,
        "holdout_frac": 0.15,
        "eval_pages": 40,
        "eval_before": 1,
    },
)

full_job_name = f"badger-{username}-full-v1"
full_estimator.fit({"train": train_s3_uri}, job_name=full_job_name, wait=False)
print("launched:", full_job_name)
```

```python
# Come back later and run this until it says Completed.
desc = session.sagemaker_client.describe_training_job(TrainingJobName=full_job_name)
print(desc["TrainingJobStatus"], "-", desc.get("SecondaryStatus"))
if desc["TrainingJobStatus"] == "Completed":
    desc = job_cost_report(full_job_name, g5_price)
    full_adapter_s3_uri = desc["ModelArtifacts"]["S3ModelArtifacts"]
```

### Base vs. fine-tuned over the holdout and the test set

The evaluation inside the training job covers a sample. To score the whole holdout *and* produce predictions for the 141 test pages, run the transcription script as a Processing job, once without the adapter and once with it. Each run transcribes about 185 pages in roughly 15 minutes.

First, build the list of pages and upload it together with the metadata.

```python
holdout_df = pd.read_csv(os.path.join(local_data_dir, training_job_name, "holdout_page_ids.csv"))

pages_df = pd.concat([
    train_df[train_df.page_id.isin(holdout_df.page_id)][["page_id", "doc_id", "category", "text"]],
    test_df[["page_id", "doc_id", "category"]].assign(text=""),
], ignore_index=True)
print(len(pages_df), "pages to transcribe")

infer_prefix = f"{data_prefix}/inference"
pages_df.to_csv(os.path.join(local_data_dir, "pages.csv"), index=False)
for fname in ["pages.csv", "metadata.csv"]:
    s3_client.upload_file(os.path.join(local_data_dir, fname), bucket_name, f"{infer_prefix}/{fname}")

images_s3_uri = f"s3://{shared_bucket}/{data_prefix}/images/"
infer_data_s3_uri = f"s3://{bucket_name}/{infer_prefix}/"
```

Then one function that launches a transcription job. Three inputs are mounted into the container: the images (from the shared bucket), your CSVs, and, optionally, the adapter.

```python
from sagemaker.huggingface import HuggingFaceProcessor
from sagemaker.processing import ProcessingInput, ProcessingOutput

def run_transcription_job(label: str, adapter_uri: str = None):
    output_uri = f"s3://{bucket_name}/{data_prefix}/predictions/{label}/"
    inputs = [
        ProcessingInput(source=images_s3_uri, destination="/opt/ml/processing/input/images"),
        ProcessingInput(source=infer_data_s3_uri, destination="/opt/ml/processing/input/data"),
    ]
    arguments = ["--model_id", "stanford-oval/churro-3B", "--pages_csv", "pages.csv",
                 "--batch_size", "4", "--max_pixels", str(1280 * 28 * 28)]
    if adapter_uri:
        inputs.append(ProcessingInput(source=adapter_uri, destination="/opt/ml/processing/input/adapter"))
        arguments += ["--adapter_tar", "/opt/ml/processing/input/adapter/model.tar.gz"]

    processor = HuggingFaceProcessor(
        role=role,
        tags=make_job_tags(f"transcribe-{label}"),
        instance_type=instance_type,
        instance_count=1,
        volume_size_in_gb=50,
        transformers_version="4.56",
        pytorch_version="2.8",
        py_version="py312",
        sagemaker_session=session,
        max_runtime_in_seconds=45 * 60,   # 45 minute cap; the estimate is 15 minutes
    )
    processor.run(
        code="churro_transcribe_batch.py",
        source_dir="code/",
        inputs=inputs,
        outputs=[ProcessingOutput(output_name="predictions", source="/opt/ml/processing/output", destination=output_uri)],
        arguments=arguments,
        job_name=f"badger-{username}-transcribe-{label}-v1",   # unique, like the training jobs
        wait=False, logs=False,
    )
    return output_uri

base_output_uri = run_transcription_job("base")
tuned_output_uri = run_transcription_job("tuned", adapter_uri=full_adapter_s3_uri)
```

Both jobs run in parallel on their own GPU, which doubles the instantaneous spend but halves the wall-clock time; the total cost is the same. Watch them under **SageMaker AI → Data preparation → Processing jobs**, and when both are `Completed`, score the holdout rows with `metric.py`, the exact file that scores the leaderboard.

```python
import subprocess

def fetch_predictions(output_uri: str, label: str) -> pd.DataFrame:
    bucket, key = output_uri.replace("s3://", "").split("/", 1)
    local = os.path.join(local_data_dir, f"predictions_{label}.csv")
    s3_client.download_file(bucket, key + "predictions.csv", local)
    return pd.read_csv(local, keep_default_na=False)

preds = {label: fetch_predictions(uri, label) for label, uri in [("base", base_output_uri), ("tuned", tuned_output_uri)]}

holdout_solution = train_df[train_df.page_id.isin(holdout_df.page_id)][["page_id", "text", "category"]]
holdout_solution.to_csv(os.path.join(local_data_dir, "holdout_solution.csv"), index=False)

for label, df in preds.items():
    df[df.page_id.isin(holdout_solution.page_id)].to_csv(os.path.join(local_data_dir, f"holdout_pred_{label}.csv"), index=False)
    print(f"\n=== {label} ===")
    print(subprocess.run(
        ["python", os.path.join(local_data_dir, "metric.py"),
         "--solution", os.path.join(local_data_dir, "holdout_solution.csv"),
         "--submission", os.path.join(local_data_dir, f"holdout_pred_{label}.csv")],
        capture_output=True, text=True).stdout)
```

### Submit

The `tuned` prediction file already contains every test page. Keep only those rows, in the order `test.csv` lists them. If you want a leaderboard score, this file is what you upload on the competition's **Submit Prediction** page (two submissions per day; this is the only step that needs a Kaggle account, and it is optional).

```python
submission = test_df[["page_id"]].merge(preds["tuned"], on="page_id", how="left").fillna({"text": ""})
assert len(submission) == len(test_df) and submission.text.str.len().gt(0).all(), "every test page needs a prediction"

submission_path = os.path.join(local_data_dir, "submission_churro_lora.csv")
submission.to_csv(submission_path, index=False)
s3_client.upload_file(submission_path, bucket_name, f"{data_prefix}/submissions/submission_churro_lora.csv")
print("wrote", submission_path, len(submission), "rows")
```

Finally, rerun Step 4 and compare the total with the estimate you wrote down before the full run.

## Cleanup

- The training and processing jobs have already ended; there is nothing to stop. Check **Training jobs** and **Processing jobs** in the console anyway: nothing should say `InProgress`.
- Delete what you no longer need from `s3://<your-bucket>/badger-scribe/` (the `models/` and `predictions/` prefixes). The adapter is small, but prediction outputs accumulate across runs.
- **Stop your notebook instance.** It is a CPU instance at five cents an hour, which is exactly the kind of charge that goes unnoticed for a semester.

::::::::::::::::::::::::::::::::::::: keypoints

- Fine-tuning cost is GPU-hours × hourly rate, and GPU-hours scale with model size × tokens × epochs × attempts. The same recipe runs from under a dollar (3B, LoRA, a few hundred pages) to tens of thousands (70B, full fine-tune, days on multi-GPU nodes). Starting from a model that already knows most of the task is the biggest saving available.
- Training jobs, not GPU notebooks, are the safe way to run it: they end, `max_run` guarantees it, and a unique `job_name` prevents a second copy launching by accident. Interrupting a notebook cell does not stop a job.
- Estimate before launching (startup overhead + pages × epochs × seconds per page, at the instance's rate), then compare to `BillableTimeInSeconds`.
- Image resolution (`max_pixels`) and epochs change cost by multiples; the instance size changes it by 5–45x, which is right when the model does not fit on one GPU and waste when it does.
- Measure the gain on the same held-out pages before and after, with the task's own metric, and remember that a page-level holdout is optimistic when the real test is split by document.

::::::::::::::::::::::::::::::::::::::::::::::::

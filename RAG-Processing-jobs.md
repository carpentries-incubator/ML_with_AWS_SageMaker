---
title: "RAG with Processing Jobs"
teaching: 30
exercises: 20
---

:::::::::::::::::::::::::::::::::::::: questions

- When does a RAG pipeline need a self-hosted, open-weight model instead of Bedrock, and what is the safest way to rent the GPU for it?
- How do we run the embedding and generation steps of the WattBot pipeline as SageMaker Processing jobs that end on their own?
- How do we pass data in and out of a job through S3, and tag the job so its cost is traceable?

::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: objectives

- Explain why Bedrock's catalog is not enough when you need a specific open-weight model, and why a Processing job is the default way to self-host one for batch work.
- Launch a GPU Processing job that embeds the cached WattBot chunks with a self-hosted embedding model and writes the vectors to S3.
- Launch a second Processing job that serves Qwen3.8-27B (an open-weight model Bedrock does not carry) with vLLM and runs the full RAG loop over the training questions.
- Read a job's inputs, outputs, arguments and runtime cap, and relate its duration to what it cost.
- Score the batch predictions with the WattBot metric and compare against the Bedrock run.

::::::::::::::::::::::::::::::::::::::::::::::::


## When Bedrock does not have your model

The Bedrock episode is the default for RAG: per-token billing, nothing left running, two API calls. Bedrock's catalog is curated, though. It carries Amazon's models, Anthropic, Meta, Mistral, DeepSeek, and a few others, and within a family only some sizes, usually a generation behind: for Qwen, the serverless lineup is still the Qwen3 series. Hugging Face hosts well over a million open-weight checkpoints. The one you want (the newest release, a specific size, a fine-tune for scientific text, a model a paper just published) is usually not on Bedrock, and often never will be.

When that happens you host the model yourself on a GPU you rent, and the question becomes *how*. There are three ways to rent a GPU on SageMaker:

| | Bills | Stops when | Best for |
|---|---|---|---|
| GPU notebook instance | by the hour from start to stop | **you** stop it | interactive debugging, prototyping at the keyboard |
| Processing or training job | by the second while the job runs | the script ends, or `max_runtime_in_seconds` | batch work: embed a corpus, answer a list of questions, train |
| Inference endpoint | by the hour while deployed | you delete it | serving a model to an application with steady traffic |

For research RAG, almost all the work is batch: embed everything once, answer a list of questions, score them. That is a job, and a job is the one option that cannot be forgotten. This episode runs the WattBot pipeline that way. The notebook itself is a small CPU instance and never touches the model; it stages data, launches two GPU jobs, and reads the results back.

The generator is **Qwen3.8-27B**, a 27-billion-parameter open-weight model (Apache 2.0, released August 2026) that is not on Bedrock. Its weights are served in FP8 through vLLM: inference needs no gradients, so the weights are the memory cost, and 27B parameters at one byte each is about 27 GB, which fits one 48 GB GPU (`ml.g6e.xlarge`) with room for the KV cache. In bf16 the same model is 54 GB and would need a multi-GPU instance for no gain in output quality. (The fine-tuning episode has the full memory arithmetic; the short version is that inference runs on reduced-precision weights and training cannot.)

::::::::::::::::::::::::::::::::::::: callout

### What this episode costs

| Job | Instance | $/h | Expected duration | Cost |
|---|---|---|---|---|
| Embed ~3,000 chunks | `ml.g5.xlarge` (A10G, 24 GB) | $1.41 | ~8 min incl. startup | ~$0.20 |
| RAG over the training questions with Qwen3.8-27B | `ml.g6e.xlarge` (L40S, 48 GB) | $2.61 | ~30 min: 5 startup, ~5 installing vLLM, ~5 downloading 27 GB of weights, the rest generating | ~$1.30 |
| Controller notebook | `ml.t3.medium` | $0.05 | 2 h | $0.10 |

About $1.60 for the episode, and the GPUs are off the moment each script ends. Both jobs carry `max_runtime_in_seconds`, so a hung job costs at most the cap. Compare the same pipeline on a GPU notebook instance (the optional GPU-notebook episode at the end of the lesson): the hardware is the same, but the meter runs until someone stops it, and a forgotten `ml.g6e.xlarge` is $63 a day.

The trade-off is latency. A job takes minutes to start, so this pattern is for batch questions, not a chatbot. For interactive use, Bedrock answers in seconds; for interactive use of a model Bedrock does not have, an inference endpoint does, at an hourly rate you have to remember to turn off.

::::::::::::::::::::::::::::::::::::::::::::::::

In this episode we will:

- Reuse the chunks cached in your bucket by the Bedrock episode and the inputs in the shared bucket.
- Run two short-lived GPU Processing jobs: one computes embeddings for all chunks; the second serves Qwen3.8-27B with vLLM and runs the full RAG loop (retrieval + generation) over all training questions.
- Score the result and compare it with the Bedrock run.


## Setup

### The scripts

Jobs run in containers, so the code they execute has to be shipped with the job: a Processing job cannot import functions defined in this notebook. The lesson repository has the three files we need as standalone scripts. Clone it:

```python
!test -d ML_with_AWS_SageMaker || git clone https://github.com/carpentries-incubator/ML_with_AWS_SageMaker.git
```

- `scripts/embedding_inference.py`: generic embedding script (JSONL with a `text` field in, `embeddings.npy` out).
- `scripts/wattbot_rag_batch.py`: the WattBot RAG loop as a batch job: loads the inputs, serves Qwen with vLLM, writes `wattbot_solutions.csv`.
- `scripts/wattbot_rag.py`: the shared pipeline logic (retrieval, prompts, answer parsing, `run_single_qa`, the scorer) that the Bedrock episode imported too. The batch script imports it, so it ships with the job; the notebook imports it for scoring.

Each job gets its own source directory, because the two jobs need different Python packages and SageMaker installs whatever `requirements.txt` it finds next to the script. `code_embed/` holds the embedding job, `code_rag/` the RAG job, and a copy of `wattbot_rag.py` sits next to the notebook for the scoring step at the end.

```python
!mkdir -p code_embed code_rag
!cp ML_with_AWS_SageMaker/scripts/embedding_inference.py code_embed/
!cp ML_with_AWS_SageMaker/scripts/wattbot_rag_batch.py ML_with_AWS_SageMaker/scripts/wattbot_rag.py code_rag/
!cp ML_with_AWS_SageMaker/scripts/wattbot_rag.py .
!ls code_embed code_rag
```

Next, set up the AWS SDK, SageMaker session, and bucket names.

```python
import os
import json

import boto3
import numpy as np
import pandas as pd
import sagemaker
from sagemaker import get_execution_role
from sagemaker.huggingface import HuggingFaceProcessor
from sagemaker.processing import ProcessingInput, ProcessingOutput

session = sagemaker.Session()
region = session.boto_region_name
role = get_execution_role()

shared_bucket = "ml-marathon-shared"   # <-- EDIT ME: read-only inputs, from your instructor
bucket_name = "sinkorswim-doejohn"     # <-- EDIT ME: your own bucket (chunks in, outputs out)
s3_client = boto3.client("s3", region_name=region)

local_data_dir = "./data"
os.makedirs(local_data_dir, exist_ok=True)

print("Region:", region)
print("Role:", role)
print("Bucket:", bucket_name)
```


## Step 1 – Check the inputs

The jobs read three files from S3: `train_QA.csv` and `metadata.csv` from the shared bucket, and `chunks.jsonl` from your bucket, where the Bedrock episode cached it. The PDFs themselves are not needed again: chunking was done once, and the jobs work from the chunks. We download the three files to the notebook only to confirm they exist and to look at them; the jobs fetch their own copies from S3.

```python
train_qa_key = "wattbot/train_QA.csv"
metadata_key = "wattbot/metadata.csv"
chunks_s3_key = "wattbot/chunks.jsonl"

train_qa_s3 = f"s3://{shared_bucket}/{train_qa_key}"
metadata_s3 = f"s3://{shared_bucket}/{metadata_key}"
chunks_s3_uri = f"s3://{bucket_name}/{chunks_s3_key}"

def smart_read_csv(path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin-1")

train_qa_path = os.path.join(local_data_dir, "train_QA.csv")
metadata_path = os.path.join(local_data_dir, "metadata.csv")
local_chunks_path = os.path.join(local_data_dir, "chunks.jsonl")

if not os.path.exists(train_qa_path):
    s3_client.download_file(shared_bucket, train_qa_key, train_qa_path)
if not os.path.exists(metadata_path):
    s3_client.download_file(shared_bucket, metadata_key, metadata_path)
if not os.path.exists(local_chunks_path):
    s3_client.download_file(bucket_name, chunks_s3_key, local_chunks_path)   # written by the Bedrock episode

train_df = smart_read_csv(train_qa_path)
metadata_df = smart_read_csv(metadata_path)
with open(local_chunks_path, "r", encoding="utf-8") as f:
    chunked_docs = [json.loads(line) for line in f]

print("Train QAs:", len(train_df))
print("Metadata rows:", len(metadata_df))
print("Chunks:", len(chunked_docs), "from", chunks_s3_uri)
train_df.head(3)
```

If the last download fails with `NoSuchKey`, go back to the Bedrock episode and run its chunking step; this episode starts from its output.


## Step 2 – Processing Job 1: embed all chunks on a GPU

Now we launch a short-lived Hugging Face **Processing job** that:

1. Downloads `chunks.jsonl` from S3.
2. Loads `thenlper/gte-large` from Hugging Face.
3. Encodes each chunk into an embedding vector.
4. Saves the full matrix as `embeddings.npy` back to S3.

We use the same `embedding_inference.py` script across projects; here it expects a JSONL file with a `text` field.

### Requirements

The Hugging Face container has `transformers` and PyTorch but not `sentence-transformers`. A `requirements.txt` in the job's `source_dir` is installed before the script runs. The RAG job's requirements are different (vLLM, plus `sentence-transformers` for the query embedder), which is why it has its own directory.

```python
with open("code_embed/requirements.txt", "w") as f:
    f.write("sentence-transformers\n")

with open("code_rag/requirements.txt", "w") as f:
    f.write("\n".join(["vllm>=0.17", "sentence-transformers"]) + "\n")   # 0.17+ is needed for Qwen3.8

print("code_embed:", os.listdir("code_embed"))
print("code_rag:  ", os.listdir("code_rag"))
```

::::::::::::::::::::::::::::::::::::: callout

### The RAG job's pip install is part of its bill

vLLM ships its own CUDA kernels and pins its own PyTorch, so `pip install vllm` in the RAG job pulls several gigabytes and replaces the container's `torch` and `transformers` with the versions it wants. That is normal (the container is thrown away when the job ends) but it takes about 5 of the job's ~30 minutes, on a GPU instance that is billing the whole time: roughly 20 cents per run. For a job you launch often, bake the install into a custom container image instead; for a job you launch twice in a workshop, paying for the install is cheaper than building the image.

::::::::::::::::::::::::::::::::::::::::::::::::

### Cost tracking: tag every job

Tags do **not** propagate in AWS. The tags you put on your notebook instance cover the notebook's hourly cost and nothing else. Each Processing job below launches its own GPU instance, which is a separate billable resource with no tags unless we pass them at launch, exactly as we did for training and tuning jobs earlier in the workshop. An untagged job shows up in Cost Explorer as anonymous SageMaker spend that nobody can attribute to a team.

Define the tags once and pass `tags=job_tags` to **every** `HuggingFaceProcessor` you create. Use a distinct `Purpose` per job so you can tell the embedding job from the batch RAG job on the bill.

```python
name = "DoeJohn"         # replace with your name
project = "SinkOrSwim"   # replace with your challenge name

def make_job_tags(purpose: str):
    return [
        {"Key": "Name", "Value": name},
        {"Key": "Project", "Value": project},
        {"Key": "Purpose", "Value": purpose},
    ]
```

### Launch the embedding job

```python
embedding_model_id = "thenlper/gte-large"

emb_output_prefix = "wattbot/embeddings"
emb_output_path = f"s3://{bucket_name}/{emb_output_prefix}/"

embedding_processor = HuggingFaceProcessor(
    base_job_name="WattBot-embed-gte-large",
    role=role,
    tags=make_job_tags("Inference"),   # tags are per job; nothing is inherited from the notebook
    instance_type="ml.g5.xlarge",
    instance_count=1,
    transformers_version="4.56",
    pytorch_version="2.8",
    py_version="py312",
    sagemaker_session=session,
    max_runtime_in_seconds=2 * 60 * 60,   # backstop: the estimate is 8 minutes
)

embedding_processor.run(
    code="embedding_inference.py",
    source_dir="code_embed/",
    inputs=[
        ProcessingInput(
            source=chunks_s3_uri,
            destination="/opt/ml/processing/input",
        )
    ],
    outputs=[
        ProcessingOutput(
            output_name="embeddings",
            source="/opt/ml/processing/output",
            destination=emb_output_path,
        )
    ],
    arguments=[
        "--model_id", embedding_model_id,
        "--input_filename", "chunks.jsonl",
        "--text_key", "text",
        "--input_dir", "/opt/ml/processing/input",
        "--output_dir", "/opt/ml/processing/output",
    ],
)

print("Embedding job complete.")
```

### Check on the running job in the AWS Console

To view the job running from the AWS Console, you can visit SageMaker AI, and then find the "Data Preparation" dropdown menu on the left side panel. Click that to find "Processing jobs". If you're in us-east-1, the following link should bring you there: [https://us-east-1.console.aws.amazon.com/sagemaker/home?region=us-east-1#/processing-jobs](https://us-east-1.console.aws.amazon.com/sagemaker/home?region=us-east-1#/processing-jobs)

It takes ~5-8 minutes in total for the job to complete, most of it provisioning the instance and installing the requirements. This is the downside of launching jobs, but the good news is that we only need to launch one embedding job for our RAG pipeline. This strategy also ensures we're only paying for the GPU while the job runs.


### Sanity-check the embeddings locally

We can download `embeddings.npy` back into the notebook and inspect its shape to confirm the job ran successfully.

```python
local_embeddings_path = os.path.join(local_data_dir, "embeddings.npy")
embeddings_key = f"{emb_output_prefix}/embeddings.npy"

s3_client.download_file(bucket_name, embeddings_key, local_embeddings_path)
chunk_embeddings = np.load(local_embeddings_path)

print("Embeddings shape:", chunk_embeddings.shape, "for", len(chunked_docs), "chunks")
```


## Step 3 – Processing Job 2: full WattBot RAG over all questions

For the second job, we pass four inputs:

- `chunks.jsonl` – serialized chunks (your bucket)
- `embeddings.npy` – precomputed chunk embeddings (your bucket, written by Job 1)
- `train_QA.csv` – training questions (shared bucket)
- `metadata.csv` – to resolve `ref_id` → URL (shared bucket)

The script `wattbot_rag_batch.py` is short, because the pipeline lives in `scripts/wattbot_rag.py`, the same module the Bedrock episode imported:

- `retrieve_top_k`: cosine similarity over the embedding matrix
- `run_single_qa`: retrieval, the answer call, the hybrid unanswerable rule (retrieval threshold OR the model's own `is_blank`), the explanation call
- `compute_wattbot_score`: the metric we use at the end

What the batch script adds is the two model callables: `embed_query` wraps the same `gte-large` sentence-transformer the embedding job used (the question has to be embedded with the same model as the chunks), and `generate` wraps a vLLM engine serving Qwen3.8-27B with thinking mode off. The job writes `wattbot_solutions.csv` in the WattBot submission format.

```python
emb_output_s3 = f"s3://{bucket_name}/{emb_output_prefix}/embeddings.npy"

print("chunks:    ", chunks_s3_uri)
print("embeddings:", emb_output_s3)
print("train_QA:  ", train_qa_s3)
print("metadata:  ", metadata_s3)
```

Two settings on this processor matter more than on the embedding job. `volume_size_in_gb=60` attaches a 60 GB volume to the instance, mounted under `/opt/ml/processing`; the container's own root disk is small, so the 27 GB of model weights (and vLLM's install) must not land there. `env={"HF_HOME": ...}` points Hugging Face's download cache at that volume, which is where the weights go.

```python
rag_output_prefix = "wattbot/solutions"
rag_output_path = f"s3://{bucket_name}/{rag_output_prefix}/"

generation_model_id = "Qwen/Qwen3.8-27B-FP8"   # open-weight, not on Bedrock; ~27 GB in FP8

rag_processor = HuggingFaceProcessor(
    base_job_name="WattBot-rag-batch",
    role=role,
    tags=make_job_tags("Inference"),   # separate job, separate tags
    instance_type="ml.g6e.xlarge",     # 1 x L40S, 48 GB: holds the FP8 27B model plus KV cache
    instance_count=1,
    volume_size_in_gb=60,              # 27 GB of weights + vLLM install; mounted at /opt/ml/processing
    env={"HF_HOME": "/opt/ml/processing/hf_cache"},   # put the weight download on that volume
    transformers_version="4.56",
    pytorch_version="2.8",
    py_version="py312",
    sagemaker_session=session,
    max_runtime_in_seconds=2 * 60 * 60,   # cap: the estimate is 30 minutes
)

rag_processor.run(
    code="wattbot_rag_batch.py",
    source_dir="code_rag/",            # wattbot_rag_batch.py + wattbot_rag.py + requirements.txt
    inputs=[
        ProcessingInput(
            source=chunks_s3_uri,
            destination="/opt/ml/processing/input/chunks",
        ),
        ProcessingInput(
            source=emb_output_s3,
            destination="/opt/ml/processing/input/embeddings",
        ),
        ProcessingInput(
            source=train_qa_s3,
            destination="/opt/ml/processing/input/train",
        ),
        ProcessingInput(
            source=metadata_s3,
            destination="/opt/ml/processing/input/metadata",
        ),
    ],
    outputs=[
        ProcessingOutput(
            output_name="solutions",
            source="/opt/ml/processing/output",
            destination=rag_output_path,
        )
    ],
    arguments=[
        "--input_dir", "/opt/ml/processing/input",
        "--output_dir", "/opt/ml/processing/output",
        "--embedding_model_id", embedding_model_id,
        "--qwen_model_id", generation_model_id,
        "--top_k", "8",
        "--retrieval_threshold", "0.25",
    ],
)

print("RAG batch job complete.")
```

While it runs, open the job's CloudWatch log from the console page above. You will see, in order: pip installing vLLM, the weight download (27 GB in shards), vLLM loading the model and reporting how much GPU memory it reserved, then one line per question. The whole job is the 30 minutes in the cost callout; the generation is the last third of it.


## Step 4 – Download predictions and evaluate

Finally, we download `wattbot_solutions.csv` from S3, inspect a few rows, and compute the WattBot score against `train_QA.csv` with the same `compute_wattbot_score` the Bedrock episode used.

```python
solutions_key = f"{rag_output_prefix}/wattbot_solutions.csv"
local_solutions_path = os.path.join(local_data_dir, "wattbot_solutions.csv")

s3_client.download_file(bucket_name, solutions_key, local_solutions_path)
solutions_df = pd.read_csv(local_solutions_path)
solutions_df.head()
```

```python
from wattbot_rag import compute_wattbot_score

results_df = compute_wattbot_score(
    train_qa_path=train_qa_path,
    preds_path=local_solutions_path,
    gt_is_na_col="is_blank",
    pred_is_na_col="is_blank",
    n_examples=10,
)
```

Compare the overall score with the Bedrock run. The retrieval logic, prompts and scoring are identical, so any difference is the two models: `gte-large` versus Titan for retrieval, Qwen3.8-27B versus DeepSeek V3 for generation.

::::::::::::::::::::::::::::::::::::: challenge

### What did the two jobs cost?

Ask SageMaker what it recorded for each job: `session.sagemaker_client.describe_processing_job(ProcessingJobName=...)` returns `CreationTime`, `ProcessingStartTime` and `ProcessingEndTime`. The job names are `embedding_processor.latest_job.job_name` and `rag_processor.latest_job.job_name`. For each job, compute the minutes between creation and start (provisioning, before the instance is billed) and between start and end (the billed window), and price the billed window at $1.41/h for the `ml.g5.xlarge` and $2.61/h for the `ml.g6e.xlarge`. Then, from the RAG job's CloudWatch log timestamps, estimate how much of its billed window was setup (pip install and weight download) and how much was generation. How does the total compare with the cost callout, and with the Bedrock run?

:::::::::::::::::::::::: solution

```python
def job_cost(job_name: str, price_per_hour: float):
    d = session.sagemaker_client.describe_processing_job(ProcessingJobName=job_name)
    provisioning = (d["ProcessingStartTime"] - d["CreationTime"]).total_seconds() / 60
    billed = (d["ProcessingEndTime"] - d["ProcessingStartTime"]).total_seconds() / 60
    cost = billed / 60 * price_per_hour
    print(f"{job_name}: {d['ProcessingJobStatus']}")
    print(f"  provisioning (not billed): {provisioning:.1f} min")
    print(f"  billed window:             {billed:.1f} min at ${price_per_hour}/h = ${cost:.2f}")
    return cost

total = job_cost(embedding_processor.latest_job.job_name, 1.41)
total += job_cost(rag_processor.latest_job.job_name, 2.61)
print(f"GPU total for the episode: ${total:.2f}")
```

Typical numbers: the embedding job bills 5-6 minutes (about $0.13), the RAG job 22-28 minutes (about $1.00-$1.20), so roughly $1.20-$1.40 of GPU time against the callout's $1.50. In the RAG job's log, the first 10-12 minutes of the billed window are pip and the download; the model loads for another minute or two; the remaining ~10 minutes are the questions, about 15 seconds each for two generation calls. So well over half of what you paid for was setup, which is the argument for a custom container (skip the pip) and for keeping an `HF_HOME` cache on S3 (skip the download) once a job is run routinely. The Bedrock run of the same questions cost cents, and the self-hosted route is only justified here because the model is not in the catalog.

:::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: challenge

### The runtime cap

Both processors set `max_runtime_in_seconds=2 * 60 * 60`. Predict what would happen if the RAG job were launched with `max_runtime_in_seconds=600` instead: what does the job status become, what is in the output prefix, and what did it cost? Then compute the most a hung job can bill under each job's actual cap, and compare it with the cost of an `ml.g6e.xlarge` notebook instance left running over a weekend (64 hours).

:::::::::::::::::::::::: solution

With a 600-second cap the RAG job is killed by SageMaker ten minutes after `ProcessingStartTime`, while it is still installing vLLM or downloading the weights. The status becomes `Stopped` (the failure reason reads `MaxRuntimeExceeded`), the output prefix is empty because the script never reached the point of writing `wattbot_solutions.csv`, and the bill is the full 600 seconds: 600 / 3600 × $2.61 ≈ $0.44 for nothing. A cap has to leave room for the setup that precedes the work, which is why ours is four times the 30-minute estimate rather than ten percent over it.

Under the caps we set, the worst case for a job that hangs (a stuck download, a model that never finishes loading) is 2 h × $1.41 = **$2.82** for the embedding job and 2 h × $2.61 = **$5.22** for the RAG job. A `ml.g6e.xlarge` notebook instance forgotten over a weekend is 64 h × $2.61 = **$167**, and nothing caps it. That is the difference between a job and an instance: the job's worst case is a number you chose before you launched it.

:::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: keypoints

- Bedrock's catalog is curated and lags releases; when you need a specific open-weight model you rent a GPU and serve it yourself.
- Of the ways to rent a GPU, a Processing job is the one that cannot be forgotten: it bills while the script runs and ends on its own, with `max_runtime_in_seconds` as the backstop. Use it for batch work; use Bedrock or an endpoint when you need answers in seconds.
- Inference runs on reduced-precision weights: Qwen3.8-27B in FP8 is ~27 GB and fits one 48 GB GPU; bf16 would need two.
- A job's inputs and outputs are S3 prefixes mounted into the container; the notebook only stages files and reads results, so it stays a cheap CPU instance.
- The pipeline logic (`wattbot_rag.py`) ships with the job unchanged from the Bedrock episode; only the two model callables differ.
- Tags do not propagate from the notebook: every job gets its own `tags=`.

::::::::::::::::::::::::::::::::::::::::::::::::

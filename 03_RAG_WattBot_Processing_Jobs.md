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


# RAG with Processing Jobs

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

About $1.60 for the episode, and the GPUs are off the moment each script ends. Both jobs carry `max_runtime_in_seconds`, so a hung job costs at most the cap. Compare the same pipeline on a GPU notebook instance (the optional episode after this one): the hardware is the same, but the meter runs until someone stops it, and a forgotten `ml.g6e.xlarge` is $63 a day.

The trade-off is latency. A job takes minutes to start, so this pattern is for batch questions, not a chatbot. For interactive use, Bedrock answers in seconds; for interactive use of a model Bedrock does not have, an inference endpoint does, at an hourly rate you have to remember to turn off.

::::::::::::::::::::::::::::::::::::::::::::::::

In this episode we will:

- Reuse the chunks cached in your bucket by the Bedrock episode and the inputs in the shared bucket.
- Run two short-lived GPU Processing jobs: one computes embeddings for all chunks; the second serves Qwen3.8-27B with vLLM and runs the full RAG loop (retrieval + generation) over all training questions.
- Score the result and compare it with the Bedrock run.


## Setup

We'll first need to clone in some .py files that contain helper functions for embedding and RAG processing jobs. Since we're using containerized Processing jobs, we can't just import local Python functions from the notebook. Instead, we create standalone scripts that the jobs can run.

```python
!git clone https://github.com/carpentries-incubator/ML_with_AWS_SageMaker.git
```

Create a /code directory and copy over the relevant scripts from ML_with_AWS_SageMaker/scripts into /code:

```python

- `embedding_inference.py` – generic embedding script
- `wattbot_rag_batch.py` – WattBot-specific RAG logic for batch processing job
```

Next, setup your AWS SDK, SageMaker session, and S3 bucket information.
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
# bucket_region = "us-east-1"
s3_client = boto3.client("s3", region_name=region)

local_data_dir = "./data"
os.makedirs(local_data_dir, exist_ok=True)
corpus_dir = local_data_dir + "/pdfs/"
os.makedirs(corpus_dir, exist_ok=True)

print("Region:", region)
print("Role:", role)
print("Bucket:", bucket_name)

```


## Step 1 – Load WattBot metadata and training questions

We reuse the same `metadata.csv`, `train_QA.csv` and `corpus.zip` from the earlier RAG episodes, read from the shared bucket. If they are not already on the notebook file system, we download them.



```python

```


```python

def smart_read_csv(path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin-1")

metadata_path = os.path.join(local_data_dir, "metadata.csv")
train_qa_path = os.path.join(local_data_dir, "train_QA.csv")
corpus_path = os.path.join(corpus_dir, "corpus.zip")

if not os.path.exists(metadata_path):
    s3_client.download_file(shared_bucket, "wattbot/metadata.csv", metadata_path)
if not os.path.exists(train_qa_path):
    s3_client.download_file(shared_bucket, "wattbot/train_QA.csv", train_qa_path)
if not os.path.exists(corpus_path):
    s3_client.download_file(shared_bucket, "wattbot/corpus.zip", corpus_path)

metadata_df = smart_read_csv(metadata_path)
train_df = smart_read_csv(train_qa_path)

print("Metadata rows:", len(metadata_df))
print("Train QAs:", len(train_df))
train_df.head(3)

```


```python
import zipfile
with zipfile.ZipFile(corpus_path, 'r') as zip_ref:
    zip_ref.extractall(corpus_dir)

```


```python
corpus_dir
```


## Step 2 – Verify chunks exist on S3 (from the Bedrock episode)

For our processing job, we'll reuse the same chunks generated in the Bedrock episode. The code below just verifies you have this file available in S3 (for calling from the processing job).

```python
# load chunks from YOUR bucket (written by the Bedrock episode)
chunks_s3_key = 'wattbot/chunks.jsonl'
chunks_s3_uri = f"s3://{bucket_name}/{chunks_s3_key}"
local_chunks_path = os.path.join(local_data_dir, "chunks.jsonl")
if not os.path.exists(local_chunks_path):
    s3_client.download_file(bucket_name, chunks_s3_key, local_chunks_path)
with open(local_chunks_path, "r", encoding="utf-8") as f:
    chunked_docs = [json.loads(line) for line in f]
print(f"Loaded {len(chunked_docs)} chunks from {local_chunks_path}")

```


## Step 3 – Processing Job 1: embed all chunks on a GPU

Now we launch a short-lived Hugging Face **Processing job** that:

1. Downloads `chunks.jsonl` from S3.
2. Loads `thenlper/gte-large` from Hugging Face.
3. Encodes each chunk into an embedding vector.
4. Saves the full matrix as `embeddings.npy` back to S3.

We use the same `embedding_inference.py` script across projects; here it expects a JSONL file with a `text` field.


### But first...
we have to create a requirements.txt file that will add additional libraries to the HuggingFaceProcessor we use below, which builds the environment we'll run our embedding_inference.py script in. For the processing job to recognize this dependence, we'll add it to the source_dir (code/) referenced when we call embedding_processor.run() below.


```python
requirements = [
    "sentence-transformers",   # embedding job
    "vllm>=0.17",              # RAG job: serves Qwen3.8-27B (FP8); needs 0.17+ for this model
]

req_path = "code/requirements.txt"
with open(req_path, "w") as f:
    f.write("\n".join(requirements))

print(f"Created requirements.txt at {req_path}")

```


### Cost tracking: tag every job

Tags do **not** propagate in AWS. The tags you put on your notebook instance cover the notebook's hourly cost and nothing else. Each Processing job below launches its own GPU instance, which is a separate billable resource with no tags unless we pass them at launch, exactly as we did for training and tuning jobs earlier in the workshop. An untagged job shows up in Cost Explorer as anonymous SageMaker spend that nobody can attribute to a team.

Define the tags once and pass `tags=job_tags` to **every** `HuggingFaceProcessor` you create. Use a distinct `Purpose` per job so you can tell the embedding job from the batch RAG job on the bill.

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

```python

embedding_model_id = "thenlper/gte-large"
script_path = "embedding_inference.py"

emb_output_prefix = "wattbot/embeddings"
emb_output_path = f"s3://{bucket_name}/{emb_output_prefix}/"

embedding_processor = HuggingFaceProcessor(
    base_job_name="WattBot-embed-gte-large",
    role=role,
    tags=make_job_tags("RAG-embed"),   # tags are per job; nothing is inherited from the notebook
    instance_type="ml.g5.xlarge",
    instance_count=1,
    transformers_version="4.56",
    pytorch_version="2.8",
    py_version="py312",
    sagemaker_session=session,
    max_runtime_in_seconds=2 * 60 * 60,
)

embedding_processor.run(
    code=script_path,
    source_dir="code/",
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

### Check on running job in AWS Console
To view the job running from the AWS Console, you can visit SageMaker AI, and then find the "Data Preparation" dropdown menu on the left side panel. Click that to find "Processing jobs". If you're in us-east-1, the following link should bring you there: [https://us-east-1.console.aws.amazon.com/sagemaker/home?region=us-east-1#/processing-jobs](https://us-east-1.console.aws.amazon.com/sagemaker/home?region=us-east-1#/processing-jobs)

It may take ~5 minutes in total for the job to complete. This is the downside of launching jobs, but the good news is that we only need to launch one embedding job for our RAG pipeline. This strategy also ensures we're only paying for GPUs when we need them during the processing job.


### Sanity-check the embeddings locally

We can download `embeddings.npy` back into the notebook and inspect its shape to confirm the job ran successfully.



```python

local_embeddings_path = os.path.join(local_data_dir, "embeddings.npy")
embeddings_key = f"{emb_output_prefix}/embeddings.npy"

s3_client.download_file(bucket_name, embeddings_key, local_embeddings_path)
chunk_embeddings = np.load(local_embeddings_path)

print("Embeddings shape:", chunk_embeddings.shape)

```


## Step 5 – Processing Job 2: full WattBot RAG over all questions

For the second job, we pass four inputs:

- `chunks.jsonl` – serialized chunks
- `embeddings.npy` – precomputed chunk embeddings
- `train_QA.csv` – training questions (to compare with ground truth)
- `metadata.csv` – to resolve `ref_id` → URL

The script `wattbot_rag_batch.py` reuses the RAG helpers from the notebook GPU episode:

- cosine similarity + `retrieve_top_k`
- `retrieve_context_for_question`
- `answer_phase_for_question` (Qwen answer, answer_value, ref_ids, is_blank)
- `explanation_phase_for_question`
- `run_single_qa` (hybrid unanswerable logic: retrieval threshold + LLM is_blank)

The job writes out `wattbot_solutions.csv` in the WattBot submission format.



```python

# Inputs: the CSVs from the shared bucket, the embeddings from yours
train_qa_key = "wattbot/train_QA.csv"
metadata_key = "wattbot/metadata.csv"

train_qa_s3 = f"s3://{shared_bucket}/{train_qa_key}"
metadata_s3 = f"s3://{shared_bucket}/{metadata_key}"
emb_output_s3 = f"s3://{bucket_name}/{emb_output_prefix}/embeddings.npy"

print("train_QA:", train_qa_s3)
print("metadata:", metadata_s3)
print("embeddings:", emb_output_s3)

```


```python

rag_script = "wattbot_rag_batch.py"

rag_output_prefix = "wattbot/solutions"
rag_output_path = f"s3://{bucket_name}/{rag_output_prefix}/"

generation_model_id = "Qwen/Qwen3.8-27B-FP8"   # open-weight, not on Bedrock; ~27 GB in FP8

rag_processor = HuggingFaceProcessor(
    base_job_name="WattBot-rag-batch",
    role=role,
    tags=make_job_tags("RAG-batch"),   # separate job, separate tags
    instance_type="ml.g6e.xlarge",     # 1 x L40S, 48 GB: holds the FP8 27B model plus KV cache
    instance_count=1,
    volume_size_in_gb=60,              # vLLM install + 27 GB of weights
    transformers_version="4.56",
    pytorch_version="2.8",
    py_version="py312",
    sagemaker_session=session,
    max_runtime_in_seconds=2 * 60 * 60,   # cap: the estimate is 30 minutes
)

rag_processor.run(
    code=rag_script,
    source_dir="code/",
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
    ],
)


print("RAG batch job complete.")

```


## Step 6 – Download predictions and evaluate

Finally, we download `wattbot_solutions.csv` from S3, inspect a few rows, and (optionally) compute the WattBot score against `train_QA.csv` using the `Score.py` logic.



```python

solutions_key = f"{rag_output_prefix}/wattbot_solutions.csv"
local_solutions_path = os.path.join(local_data_dir, "wattbot_solutions.csv")

s3_client.download_file(bucket_name, solutions_key, local_solutions_path)
solutions_df = pd.read_csv(local_solutions_path)
solutions_df.head()

```


```python
import pandas as pd
import numpy as np

def _to_bool_flag(x):
    """Convert typical truthy/falsey strings to bool."""
    if isinstance(x, str):
        s = x.strip().lower()
        if s in {"1", "True", "true", "yes"}:
            return True
        if s in {"0", "False", "false", "no"}:
            return False
    return bool(x)

def _parse_float_or_none(x):
    try:
        return float(str(x).strip())
    except Exception:
        return None

def _answer_value_correct(gt_val, pred_val, rel_tol=1e-3):
    """
    gt_val, pred_val: values from answer_value columns.
    rel_tol = 0.001 => 0.1% relative tolerance.
    """
    gt_str = str(gt_val).strip()
    pred_str = str(pred_val).strip()
    
    # If either is 'is_blank', treat as categorical
    if gt_str.lower() == "is_blank" or pred_str.lower() == "is_blank":
        return gt_str.lower() == pred_str.lower()
    
    gt_num = _parse_float_or_none(gt_val)
    pred_num = _parse_float_or_none(pred_val)
    
    # If both numeric, use relative tolerance
    if gt_num is not None and pred_num is not None:
        if gt_num == 0:
            return abs(pred_num - gt_num) <= rel_tol  # small absolute tolerance around 0
        rel_err = abs(pred_num - gt_num) / max(abs(gt_num), 1e-12)
        return rel_err <= rel_tol
    
    # Otherwise, fall back to normalized string match
    return gt_str.lower() == pred_str.lower()

def _ref_id_jaccard(gt_ref, pred_ref):
    """
    Jaccard overlap between sets of ref_ids.
    Strings may contain semicolon-separated IDs, or 'is_blank'.
    Case-insensitive.
    """
    def to_set(s):
        if s is None:
            return set()
        s = str(s).strip()
        if not s or s.lower() == "is_blank":
            return set()
        parts = [p.strip().lower() for p in s.split(";") if p.strip()]
        return set(parts)
    
    gt_set = to_set(gt_ref)
    pred_set = to_set(pred_ref)
    
    if not gt_set and not pred_set:
        return 1.0
    union = gt_set | pred_set
    if not union:
        return 0.0
    inter = gt_set & pred_set
    return len(inter) / len(union)

def compute_wattbot_score(
    train_qa_path="train_QA.csv",
    preds_path="train_solutions_qwen.csv",
    id_col="id",
    gt_answer_col="answer_value",
    gt_ref_col="ref_id",
    gt_is_na_col="is_NA",   # can also pass "is_blank" or None
    pred_answer_col="answer_value",
    pred_ref_col="ref_id",
    pred_is_na_col=None,    # can pass "is_blank", or leave None to auto
    n_examples=10,          # how many incorrect examples to print
):
    """
    Compare your solutions to train_QA.csv using a WattBot-style score.

    NA logic:
    - If an explicit NA column is found/used (e.g. is_NA), we use it via _to_bool_flag.
    - If you pass gt_is_na_col="is_blank" or pred_is_na_col="is_blank",
      we *derive* NA from answer_value == "is_blank" instead of expecting a real column.
    - If no NA column is available at all, we derive from answer_value == "is_blank".

    Also prints up to `n_examples` rows where the model is not perfect
    (answer_score < 1, ref_id_score < 1, or is_NA_score < 1).
    """
    gt = pd.read_csv(train_qa_path)
    preds = pd.read_csv(preds_path)
    
    # Inner join on id to be strict
    merged = gt.merge(preds, on=id_col, suffixes=("_gt", "_pred"))
    if merged.empty:
        raise ValueError("No overlapping ids between ground truth and predictions.")

    # ----- ground truth NA flags -----
    if gt_is_na_col is not None and gt_is_na_col in merged.columns:
        # Use explicit column (e.g. "is_NA")
        gt_is_na_series = merged[gt_is_na_col].map(_to_bool_flag)
    elif gt_is_na_col is not None and gt_is_na_col.lower() == "is_blank":
        # Special meaning: derive NA from answer_value_gt == "is_blank"
        gt_is_na_series = merged[f"{gt_answer_col}_gt"].astype(str).str.lower().eq("is_blank")
        merged["gt_is_blank_flag"] = gt_is_na_series
    else:
        # Fallback: if we have is_NA or is_blank col, use it; else derive
        if "is_NA" in merged.columns:
            gt_is_na_series = merged["is_NA"].map(_to_bool_flag)
        elif "is_blank" in merged.columns:
            gt_is_na_series = merged["is_blank"].map(_to_bool_flag)
        else:
            gt_is_na_series = merged[f"{gt_answer_col}_gt"].astype(str).str.lower().eq("is_blank")
            merged["gt_is_blank_flag"] = gt_is_na_series

    # ----- prediction NA flags -----
    if pred_is_na_col is not None and pred_is_na_col in merged.columns:
        pred_is_na_series = merged[pred_is_na_col].map(_to_bool_flag)
    elif pred_is_na_col is not None and pred_is_na_col.lower() == "is_blank":
        # Same convention: derive from answer_value_pred
        pred_is_na_series = merged[f"{pred_answer_col}_pred"].astype(str).str.lower().eq("is_blank")
        merged["pred_is_blank_flag"] = pred_is_na_series
    else:
        # Auto-detect or derive if no NA column in preds
        if "is_NA" in merged.columns:
            pred_is_na_series = merged["is_NA"].map(_to_bool_flag)
        elif "is_blank" in merged.columns:
            pred_is_na_series = merged["is_blank"].map(_to_bool_flag)
        else:
            pred_is_na_series = merged[f"{pred_answer_col}_pred"].astype(str).str.lower().eq("is_blank")
            merged["pred_is_blank_flag"] = pred_is_na_series

    ans_scores = []
    ref_scores = []
    na_scores = []
    
    for idx, row in merged.iterrows():
        gt_ans = row[f"{gt_answer_col}_gt"]
        pred_ans = row[f"{pred_answer_col}_pred"]
        gt_ref = row[f"{gt_ref_col}_gt"]
        pred_ref = row[f"{pred_ref_col}_pred"]
        
        gt_is_na = bool(gt_is_na_series.iloc[idx])
        pred_is_na = bool(pred_is_na_series.iloc[idx])
        
        # 1. answer_value component
        ans_correct = _answer_value_correct(gt_ans, pred_ans)
        ans_scores.append(1.0 * ans_correct)
        
        # 2. ref_id Jaccard
        ref_j = _ref_id_jaccard(gt_ref, pred_ref)
        ref_scores.append(ref_j)
        
        # 3. is_NA component (simple: must match ground truth flag)
        na_scores.append(1.0 if gt_is_na == pred_is_na else 0.0)
    
    merged["answer_score"] = ans_scores
    merged["ref_id_score"] = ref_scores
    merged["is_NA_score"] = na_scores
    
    merged["wattbot_score"] = (
        0.75 * merged["answer_score"]
        + 0.15 * merged["ref_id_score"]
        + 0.10 * merged["is_NA_score"]
    )
    
    print(f"Rows compared: {len(merged)}")
    print(f"Mean answer_value score: {merged['answer_score'].mean():.4f}")
    print(f"Mean ref_id score:       {merged['ref_id_score'].mean():.4f}")
    print(f"Mean is_NA score:        {merged['is_NA_score'].mean():.4f}")
    print(f"Overall WattBot score:   {merged['wattbot_score'].mean():.4f}")
    
    # ----- Show some incorrect examples -----
    incorrect = merged[
        (merged["answer_score"] < 1.0)
        | (merged["ref_id_score"] < 1.0)
        | (merged["is_NA_score"] < 1.0)
    ]
    
    if not incorrect.empty and n_examples > 0:
        print("\nExamples of incorrect / partially correct responses "
              f"(up to {n_examples} rows):\n")
        # Grab up to n_examples "worst" rows by wattbot_score
        for _, row in incorrect.sort_values("wattbot_score").head(n_examples).iterrows():
            q = row["question_gt"] if "question_gt" in row.index else None
            print("-" * 80)
            print(f"id: {row[id_col]}")
            if q is not None:
                print(f"Question: {q}")
            print(f"GT answer_value:   {row[f'{gt_answer_col}_gt']}")
            print(f"Pred answer_value: {row[f'{pred_answer_col}_pred']}")
            print(f"GT ref_id:         {row[f'{gt_ref_col}_gt']}")
            print(f"Pred ref_id:       {row[f'{pred_ref_col}_pred']}")
            print(f"answer_score: {row['answer_score']:.3f}, "
                  f"ref_id_score: {row['ref_id_score']:.3f}, "
                  f"is_NA_score: {row['is_NA_score']:.3f}, "
                  f"wattbot_score: {row['wattbot_score']:.3f}")
        print("-" * 80)
    
    return merged

```


```python
results_df = compute_wattbot_score(
    train_qa_path="./data/train_QA.csv",
    preds_path="./data/wattbot_solutions.csv",
    gt_is_na_col="is_blank",
    pred_is_na_col="is_blank",
)

```

::::::::::::::::::::::::::::::::::::: keypoints

- Bedrock's catalog is curated and lags releases; when you need a specific open-weight model you rent a GPU and serve it yourself.
- Of the ways to rent a GPU, a Processing job is the one that cannot be forgotten: it bills while the script runs and ends on its own, with `max_runtime_in_seconds` as the backstop. Use it for batch work; use Bedrock or an endpoint when you need answers in seconds.
- Inference runs on reduced-precision weights: Qwen3.8-27B in FP8 is ~27 GB and fits one 48 GB GPU; bf16 would need two.
- A job's inputs and outputs are S3 prefixes mounted into the container; the notebook only stages files and reads results, so it stays a cheap CPU instance.
- Tags do not propagate from the notebook: every job gets its own `tags=`.

::::::::::::::::::::::::::::::::::::::::::::::::

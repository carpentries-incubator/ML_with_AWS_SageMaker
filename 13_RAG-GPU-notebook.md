---
title: "(Optional) RAG with a Notebook GPU"
teaching: 30
exercises: 15
---

:::::::::::::::::::::::::::::::::::::: questions

- How can we run a basic Retrieval-Augmented Generation (RAG) pipeline entirely from a single GPU-backed SageMaker notebook?
- How do we go from cached text chunks to a searchable embedding space using a self-hosted model?
- How can we generate WattBot-style answers (including citations and evidence) that follow the competition’s scoring conventions?

::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: objectives

- Verify that our SageMaker notebook instance has a working GPU and compatible Python environment.
- Reuse the WattBot metadata and questions from the shared bucket, and the `chunks.jsonl` the Bedrock episode cached in your own bucket.
- Build the embedding matrix and the retrieval step with a self-hosted embedding model, in plain NumPy.
- Build a small retrieval helper that finds the most relevant chunks for a question using cosine similarity in embedding space.
- Serve a 27B open-weight generator (Qwen3.8-27B in FP8, via vLLM) on the notebook's GPU and wire it to the retriever to produce WattBot-format answers (including `answer`, `ref_id`, `ref_url`, and `supporting_materials`).
- Explain why inference uses quantized weights rather than bf16, and what a self-hosted GPU costs per hour, per forgotten weekend, and per forgotten month.
- Add a second LLM pass that generates short explanations and marks whether the evidence comes from text, figures, tables, or a combination.

::::::::::::::::::::::::::::::::::::::::::::::::



::::::::::::::::::::::::::::::::::::: callout

### Optional, self-study episode

This episode is not taught in the workshop. **The Bedrock and Processing Jobs episodes are the default ways to run RAG; this one is for the case where you need to work on a self-hosted model by hand, and only if you will remember to stop the instance.** **Do not run this notebook on the controller instance from the other episodes. Create a new GPU notebook instance for it (Setup below). On a CPU instance the vLLM install and the 27 GB model load will exhaust memory and freeze your JupyterLab.** It runs the same pipeline as the Processing Jobs episode on a GPU notebook instance, which is the right tool for prototyping alone at a keyboard and the wrong one for a room of a hundred people with meters running. Work through it on your own when you want to see model loading, GPU memory and generation as plain Python. Read the cost callout first.

::::::::::::::::::::::::::::::::::::::::::::::::

## Why run RAG on a GPU notebook?

The Bedrock and Processing Jobs episodes are the better patterns for most research work. Bedrock serves the models for you, and a Processing job rents a GPU only long enough to finish a batch. A notebook GPU gives you something different, an interactive place to load a model, watch GPU memory, try retrieval strategies and change prompts without repackaging a job after every experiment.

That matters when you need a model Bedrock does not offer and want to work on it by hand. This episode runs the WattBot pipeline with **Qwen3.8-27B** as the generator, a 27-billion-parameter open-weight (Apache 2.0) model released in August 2026 that Bedrock does not carry, loaded in FP8 through vLLM beside a local embedding model. Everything a managed model API hides, loading an embedding model, loading a generator, fitting both in GPU memory, is a cell you can run and change.

A 27B model is large enough that the choice of GPU and weight precision matters. At about one byte per weight, FP8 puts the weights near 27 GB, and a 48 GB L40S leaves room for the embedding model and the attention cache that generation needs. The 24 GB GPU the earlier episodes used cannot hold this setup. That makes the notebook a good place to learn about model serving, and an expensive place to leave idle.

**Treat this as an interactive experiment, not a deployment pattern.** Create the separate GPU notebook described below. Do not run it on the CPU controller from the other episodes.

::::::::::::::::::::::::::::::::::::: callout

### This instance bills `$2.61` an hour until you stop it, and nothing stops it for you

A GPU notebook instance is a rental with the meter running. It bills by the hour from the moment it starts until the moment *you* stop it, whether it is generating answers, sitting at an idle prompt, or showing a closed browser tab on a laptop that went home for the weekend. Nothing in this episode turns it off for you. The instance we use, `ml.g6e.xlarge` (one L40S GPU, 48 GB), costs **`$2.61` per hour** on-demand in `us-east-1` (AWS price list, October 2026). That is:

| Left running for | Cost |
|---|---|
| This episode (~2 hours) | ~`$5` |
| Overnight (12 h) | ~`$31` |
| A weekend (64 h) | ~`$167` |
| A week | ~`$438` |
| A month | **~`$1,906`** |

The same evaluation on Bedrock cost well under a dollar. One forgotten weekend on this instance costs a thousand times more than the work it was rented for. A forgotten GPU notebook is the most common way research groups overspend cloud credits.

Two further charges people do not expect:

- **A stopped instance still bills for its storage.** We give this notebook a 100 GB volume for model weights. SageMaker charges `$0.14` per GB-month for notebook storage, so a stopped-but-not-deleted instance costs about **`$14` a month** forever. When you are done with the episode, delete the instance, not just stop it, unless you will use it again within days.
- **Idle is billed the same as busy.** There is no discount for a GPU that is loaded and waiting. Loading a 27B model takes a few minutes, which tempts people to leave the instance up "so I don't have to reload". That is the `$1,906`.

What keeps this safe:

1. **Stop the instance when you stand up**, every time, from the [Notebook instances page](https://us-east-2.console.aws.amazon.com/sagemaker/home?region=us-east-2#/notebook-instances) (select it, **Stop**). Then look at the status column until it says `Stopped`.
2. **Set a reminder** on your phone or calendar before you start the instance, for two hours from now.
3. **Prefer jobs for anything beyond exploration**: a Processing job (the Processing Jobs episode) or a training job ends on its own, and `max_runtime_in_seconds` (`max_run` on a training job) guarantees it.

::::::::::::::::::::::::::::::::::::::::::::::::

## What we'll build

We reuse the [WattBot 2026](https://www.kaggle.com/competitions/WattBot2026) document chunks and evaluation questions prepared in the Bedrock episode. The pipeline keeps its purpose. What changes is that its two managed model calls become two models running on this GPU, `thenlper/gte-large` for embeddings and Qwen3.8-27B for answers. Qwen3.8-27B is the newer and larger relative of the Qwen3-8B the Processing Jobs episode ran, and sits within a few points of a frontier commercial model on agent tasks and science questions (see that episode's chart), which is why it is worth the trouble of hosting.

First we load the cached chunks and embed them into a searchable matrix. Then we implement cosine-similarity retrieval so a question brings back the most relevant passages. Finally we load Qwen through vLLM, pass those passages into the same WattBot prompts, and generate answers, citations and short explanations that label the evidence type, `[text]`, `[table]`, `[figure]` or a combination such as `[text+table]`. Scoring those answers against the competition's references tells us how the self-hosted pipeline performs, not only whether it runs.

The workflow is kept close to the earlier episodes on purpose. With the documents, prompts and scoring the same, the differences in model serving are the only thing left to compare.


## Notebook + dataset setup

For this episode you need a notebook instance with a GPU that can hold the FP8 27B model plus the embedding model plus working memory for generation. Create a new notebook instance (same steps as the controller-notebook episode, same three tags) with:

- **Instance type:** `ml.g6e.xlarge` (1 × L40S, 48 GB GPU memory, `$2.61`/h on the [Instances for ML](https://carpentries-incubator.github.io/ML_with_AWS_SageMaker/instances-for-ML.html) page). The FP8 weights are ~27 GB; the rest is for the embedding model and the KV cache that generation needs for long prompts. The L40S runs FP8 natively. A 24 GB instance such as `ml.g5.xlarge` cannot hold this model; a 16 GB `ml.g4dn.xlarge` cannot hold anything in this size class.
- **Volume size:** `100` GB (the default of 5 GB will not fit the weights or the vLLM install). This volume bills at `$0.14` per GB-month even while the instance is stopped.
- **Tags:** the same `Project`, `Name`, `Purpose` keys as every other resource; `Purpose` is `Notebook`.

Once it is `InService`, open JupyterLab, clone the lesson repository as in the controller-notebook episode (this is a new instance, so it has no copy yet), and open `/ML_with_AWS_SageMaker/notebooks/13_RAG-GPU-notebook.ipynb`. Select the **`conda_pytorch`** kernel.

See [Instances for ML](https://carpentries-incubator.github.io/ML_with_AWS_SageMaker/instances-for-ML.html) for further guidance. This instance bills by the hour from the moment it starts, whether or not it is doing anything. **Stop it when you leave**, and **delete it** when the episode is over.

We reuse the same two buckets as the Bedrock episode, the shared bucket for `metadata.csv` and `train_QA.csv`, and your own bucket for the `wattbot/chunks.jsonl` you cached there. If you skipped that episode, go back and run its chunking step first. Your notebook instance carries the tags you gave it when you created it and your bucket carries its own tags, so in this episode those two resources are the only things being billed.


###  Verify GPU and basic environment


```python
# nvidia-smi lists the GPU, its memory and the driver; it fails when the instance has no GPU
!nvidia-smi || echo "No GPU detected; the next cell stops the notebook"
```


```python
# also verify you have selected the conda_pytorch kernel
import os
os.environ["HF_HOME"] = "/home/ec2-user/SageMaker/hf_cache"   # before any Hugging Face import: every model download goes to the volume, not the small root disk
import torch

# Stop here, and so stop a Run All, unless this instance has a GPU big enough for the 27B model:
# the install below is several gigabytes and the weights are 27 GB, neither worth downloading onto the wrong machine
if not torch.cuda.is_available():
    raise RuntimeError("No GPU: this notebook needs an ml.g6e.xlarge notebook instance. Stop this instance and create one as described above.")
gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
if gpu_memory_gb < 40:
    raise RuntimeError(f"GPU has {gpu_memory_gb:.0f} GB; the FP8 27B model needs a 48 GB GPU (ml.g6e.xlarge). Stop this instance and create one as described above.")
print(f"GPU: {torch.cuda.get_device_name(0)}, {gpu_memory_gb:.0f} GB")
```

### Install vLLM first

The install may replace PyTorch and then needs a kernel restart, so it comes before anything is loaded.

::::::::::::::::::::::::::::::::::::: callout

### Installing vLLM takes a few minutes, and the meter is running

vLLM is the open-source inference engine that loads an open-weight model onto the GPU and answers chat requests from Python, the self-hosted counterpart of Bedrock's runtime (the Processing Jobs episode introduces it). `pip install vllm` downloads several gigabytes (it brings its own CUDA kernels and may replace the kernel's PyTorch build) and takes 5–10 minutes. Qwen3.8 needs vLLM 0.17 or newer. The 27 GB of model weights take another 4–6 minutes to download the first time. At `$2.61` an hour that is about 60 cents of waiting, which is fine; what is not fine is doing it twice because the instance was deleted and recreated. Keep the Hugging Face cache on the 100 GB volume (the `HF_HOME` line in the serving cell below does that) so the weights survive a stop/start.

If the install changes PyTorch, restart the kernel (**Kernel → Restart**) before continuing.

::::::::::::::::::::::::::::::::::::::::::::::::

```python
# Several GB; it may replace torch, so restart the kernel (Kernel -> Restart) if the log says it did
%pip install -q "vllm>=0.17"   # Qwen3.8 needs vLLM 0.17+, which brings its own PyTorch
```


## Import data from bucket into notebook

The same setup as the Bedrock episode, session, region, role, an S3 client, the two bucket names and a local data folder, plus a small download helper.

```python
import os
import json
import time

import boto3
import pandas as pd
import numpy as np

import sagemaker
from sagemaker import get_execution_role


# Initialize SageMaker + AWS basics
session = sagemaker.Session()
region = session.boto_region_name
role = get_execution_role()
s3_client = boto3.client("s3", region_name=region)

print("Region:", region)
print("Role:", role)

```


```python
def download_s3_object(bucket: str, key: str, local_path: str) -> None:
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    print(f"Downloading s3://{bucket}/{key} -> {local_path}")
    s3_client.download_file(bucket, key, local_path)
```


```python
!mkdir -p /home/ec2-user/SageMaker/wattbot   # a project folder beside the lesson clone (this is a new instance, so it starts empty)
%cd /home/ec2-user/SageMaker/wattbot

shared_bucket = "ml-marathon-shared"   # the instructor-staged, read-only bucket with the data and model weights; change only on your own account
bucket_name = "sinkorswim-doejohn"     # <-- EDIT ME: your own bucket (chunks live here)

# Local working directory in the notebook instance
local_data_dir = "./data"

session_start = time.time()   # the meter on this GPU instance started when it booted; this marks where the episode's work began

print("Local data dir:", local_data_dir)
```

### Get the shared WattBot helpers

The retrieval, prompts, answer parsing, per-question orchestration and the scorer are the same code the Bedrock and Processing Jobs episodes used, `scripts/wattbot_rag.py` in the lesson repository. Copy the module next to this notebook. What this episode adds is the two model callables, `embed_query` and `generate`, running on the notebook's own GPU.

```python
!cp /home/ec2-user/SageMaker/ML_with_AWS_SageMaker/scripts/wattbot_rag.py /home/ec2-user/SageMaker/ML_with_AWS_SageMaker/scripts/hf_embedder.py /home/ec2-user/SageMaker/ML_with_AWS_SageMaker/scripts/Score.py .
```

Import the pieces of the shared RAG module we reuse, the prompts, the retrieval and scoring functions, and the per-question runner.

```python
from wattbot_rag import (
    ANSWER_SYSTEM_PROMPT,
    build_docid_to_url,
    compute_wattbot_score,
    embed_query_with,
    explanation_system_prompt,
    retrieve_top_k,
    run_single_qa,
)
```

Download the two WattBot input files from the shared bucket.

```python
# Download metadata.csv and train_QA.csv from the shared bucket
metadata_key = "wattbot/metadata.csv"
train_qa_key = "wattbot/train_QA.csv"

metadata_path = os.path.join(local_data_dir, "metadata.csv")
train_qa_path = os.path.join(local_data_dir, "train_QA.csv")

download_s3_object(shared_bucket, metadata_key, metadata_path)
download_s3_object(shared_bucket, train_qa_key, train_qa_path)



```

## Step 1 – Model libraries and safe CSV loading


```python
import torch
from hf_embedder import HFEmbedder   # thirty lines on plain transformers: encode(list of strings) -> unit-length matrix; the Processing jobs use the same file
```

A CSV reader that tries a few encodings, because some metadata files are not valid UTF-8.

```python
def smart_read_csv(path: str) -> pd.DataFrame:
    """Try several encodings when reading a CSV file.

    Some CSVs (especially those with special characters in author names or titles)
    may not be valid UTF-8. This helper rotates through common encodings and raises
    the last error only if all fail.
    """
    encodings = ["utf-8", "latin1", "ISO-8859-1", "cp1252"]
    last_error = None
    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception as e:
            last_error = e
    if last_error is not None:
        raise last_error
    raise RuntimeError(f"Unable to read CSV at {path}")


train_df = smart_read_csv(train_qa_path)
metadata_df = smart_read_csv(metadata_path)

print("train_QA.csv columns:", train_df.columns.tolist())
print("metadata.csv columns:", metadata_df.columns.tolist())
print("\nNumber of training QAs:", len(train_df))
print("Number of metadata rows:", len(metadata_df))

train_df.head(15)
```

## Step 2 – Load the cached chunks from S3

In the Bedrock episode we turned every PDF into page-level records, split those pages into overlapping chunks, and uploaded the result to S3 as `chunks.jsonl`. Chunking is CPU work, so there is no reason to repeat it while a GPU is billing. We download the cached file and load it.


```python
chunks_s3_key = "wattbot/chunks.jsonl"          # in YOUR bucket, written by the Bedrock episode
chunks_jsonl_path = os.path.join(local_data_dir, "chunks.jsonl")

def load_chunked_docs_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

if not os.path.exists(chunks_jsonl_path):
    download_s3_object(bucket_name, chunks_s3_key, chunks_jsonl_path)

chunked_docs = load_chunked_docs_jsonl(chunks_jsonl_path)
print("Chunked docs:", len(chunked_docs))
chunked_docs[0] if chunked_docs else None
```

Each chunk keeps a pointer back to its document and page metadata (`doc_id`, `title`, `url`, `page_num`, `page_label`, `chunk_idx_in_page`), which we use later to cite sources.


## Step 3 – Build an embedding matrix

Now we embed each chunk into a vector with an open **embedding model**:

### `thenlper/gte-large`

The embedder the Processing Jobs episode used has 335 million parameters, 0.7 GB in bf16, 1,024-dimensional vectors, and about 1,100 chunks a minute on an A10G, so the 9,000 chunks here take a few minutes on this GPU.

We will:

1. Load the embedding model on GPU if available.
2. Encode all chunks in batches.
3. Store the resulting matrix as a `torch.Tensor` or `numpy.ndarray` along with the original `chunked_docs` list.

Later, we’ll implement a small retrieval helper that does cosine-similarity search over this matrix, with no indexing library.



```python
# We'll use a stronger embedding model now that we have a GPU.
# This model has ~335M parameters and benefits from GPU acceleration,
# and fits beside the 27B generator on this 48 GB GPU.
embedding_model_id = "thenlper/gte-large"

print("CUDA available for embeddings:", torch.cuda.is_available())

# Single shared embedder object that we can pass around; it uses the GPU when one is available
embedder = HFEmbedder(embedding_model_id)


```

Embed every chunk in batches on the GPU and stack the results into one matrix, one row per chunk.

```python
def embed_texts(embedder, docs, batch_size: int = 32) -> np.ndarray:
    """Embed all chunk texts into a dense matrix of shape (N, D)."""
    texts = [d["text"] for d in docs]
    all_embeddings = []
    start = time.time()
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        emb = embedder.encode(
            batch,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=True,   # unit-length vectors, so a dot product is the cosine similarity
        )
        all_embeddings.append(emb)
    embeddings = np.vstack(all_embeddings) if all_embeddings else np.zeros((0, 1024))
    print(f"Computed embeddings for {len(texts)} chunks in {time.time() - start:.1f}s")
    return embeddings

```


```python

chunk_embeddings = embed_texts(embedder, chunked_docs)
chunk_embeddings.shape

```


### 4. Build a simple retrieval step (cosine similarity)

Retrieval here is cosine similarity between the question's vector and every chunk's vector, with the top *k* chunks as context, and nothing more. For nine thousand chunks that is one matrix multiply. A vector database (FAISS, Chroma, Weaviate) is the same step built for millions of chunks or many concurrent users, with approximate search, metadata filters and a shared index; swapping it in changes nothing else in the pipeline.

`retrieve_top_k` from `wattbot_rag.py` is that search. Normalize the query and the matrix, one matrix multiply for the cosine similarities, `argsort` for the top *k*. Open the file to see it; it is a dozen lines. `embed_query_with` wraps the embedder as the `embed_query(question)` callable the rest of the pipeline expects.

```python
embed_query = embed_query_with(embedder)   # embed_query(question) -> vector, with the same model that embedded the chunks

# Quick sanity check for `retrieve_top_k` on the first training question
first_row = train_df.iloc[0]
test_question = first_row["question"]
print("Sample question:", test_question)

test_retrieved = retrieve_top_k(
    query_embedding=embed_query(test_question),
    chunk_embeddings=chunk_embeddings,
    chunked_docs=chunked_docs,
    k=3,
)

print(f"Top {len(test_retrieved)} retrieved chunks:")
for r in test_retrieved:
    snippet = r["text"].replace("\n", " ")
    if len(snippet) > 160:
        snippet = snippet[:160] + "..."
    print(f"- score={r['score']:.3f} | doc_id={r['doc_id']} | page={r['page_label']} | snippet={snippet}")
```


### 5. Serve the generator, Qwen3.8-27B in FP8, with vLLM

For this episode we use **Qwen3.8-27B**, an open-weight model that Bedrock does not carry, in its **FP8** release (`Qwen/Qwen3.8-27B-FP8`), served with [vLLM](https://docs.vllm.ai/). It is a dense 27B model with a hybrid attention design (most layers use a linear-attention variant with a fixed-size state, so the KV cache stays small even for long prompts), a native 262K context, and a "thinking" mode that is on by default and that we turn off, because WattBot wants a JSON answer, not a reasoning trace.

**Why FP8 weights and not bf16?** Inference is not training. No gradients, no optimizer state, nothing is updated, so the only memory that matters is the weights themselves plus the KV cache for the prompt being processed. A 27B model in bf16 is about 54 GB of weights and does not fit a 48 GB GPU; the same weights in 8-bit floating point are about 27 GB, which fits with room for the embedding model and generation. FP8 is near-lossless for a model this size, and it is faster, because decoding is limited by how fast the GPU can read weights from memory. For inference on rented GPUs, reduced-precision weights are the norm (FP8 where the GPU supports it, 4-bit integer where it does not or where memory is tighter); bf16 is what you use when you have memory to spare and a model small enough not to need it.

**Why vLLM and not `transformers.pipeline`?** vLLM is an inference server. It has the kernels for FP8 and for this model's hybrid attention, batches requests, and manages the KV cache so long prompts do not run out of memory. It is what you would deploy behind an API, and its offline `LLM` class is a two-line replacement for a `transformers.pipeline`.

We will:

1. Call the model once to propose an answer and supporting evidence.
2. Call it a **second time** with a smaller prompt to generate a short explanation (<= 100 characters).

```python
import os
# Keep model weights on the notebook's 100 GB volume rather than the small root disk.

from vllm import LLM, SamplingParams

qwen_model_id = "Qwen/Qwen3.8-27B-FP8"   # pre-quantized FP8 checkpoint; vLLM reads that from its config, so no quantization= argument is needed

load_start = time.time()
llm = LLM(
    model=qwen_model_id,                         # Hugging Face id; the weights download to HF_HOME on first use
    max_model_len=8192,                          # longest prompt + answer we need; the model allows 262K, we do not
    gpu_memory_utilization=0.85,                 # leave ~7 GB for the embedding model and PyTorch overhead
    limit_mm_per_prompt={"image": 0, "video": 0},  # text only: skip reserving memory for the vision encoder
)
print(f"Model loaded in {(time.time() - load_start) / 60:.1f} min (download on first use, then weights into GPU memory)")
print("vLLM engine ready:", qwen_model_id)
```

```python
!nvidia-smi --query-gpu=memory.used,memory.total --format=csv
```

The number above is what you are paying for, a 48 GB GPU, most of it now holding model weights and a reserved KV cache. It stays allocated, and billed, until you stop the instance.

```python
def call_qwen_chat(system_prompt: str, user_prompt: str, max_new_tokens: int = 384) -> str:
    """Send a chat prompt to the vLLM engine and return only the generated text."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    params = SamplingParams(temperature=0.0, max_tokens=max_new_tokens)
    outputs = llm.chat(
        messages,
        sampling_params=params,
        chat_template_kwargs={"enable_thinking": False},   # answer directly; no reasoning trace
        use_tqdm=False,
    )
    return outputs[0].outputs[0].text.strip()

print("Generator model and helper loaded.")

```

One throwaway question, to confirm the generator answers before spending time on the real ones.

```python

# Quick sanity check for `call_qwen_chat`
test_system_prompt = "You are a concise assistant who answers simple questions clearly."
test_user_prompt = "What is 2 + 2? Answer in one short sentence."

test_response = call_qwen_chat(
    system_prompt=test_system_prompt,
    user_prompt=test_user_prompt,
    max_new_tokens=32,
)
print(f"Generator ({qwen_model_id}) test response: {test_response}")

```


### 6. The prompts

`call_qwen_chat` has the signature the shared pipeline expects, `generate(system_prompt, user_prompt, max_new_tokens) -> str`, and `embed_query` has the other one. `run_single_qa` in `wattbot_rag.py` makes two calls per question with them:

- The first asks for a JSON object with `answer`, `answer_value` (normalized scalar, range or category), `answer_unit`, `ref_id` (list of document ids), `is_blank` and `supporting_materials`. The retrieved chunks go in the user turn, labelled `[doc_id, page N]` so the model can cite them, along with the expected `answer_unit` from `train_QA.csv` as a hint, so the number comes back in the unit the ground truth uses.
- The second, made only when the first produced an answer, asks for a short explanation. With `with_evidence_type=True` it also asks the model to name the **main type of evidence** (`text`, `table`, `figure`, or a combination) in the strict format `evidence_type: explanation`, and the code turns that into `[evidence_type] explanation` for the CSV, so the tag is parsed rather than trusted.

```python
print(ANSWER_SYSTEM_PROMPT)
print("-" * 80)
print(explanation_system_prompt(with_evidence_type=True))
```

### 7. Run over the full WattBot training set

For every row of `train_QA.csv`, `run_single_qa`:

1. Embeds the question and retrieves the top-*k* chunks.
2. Asks Qwen for the answer JSON and parses it (a malformed reply counts as unanswerable).
3. Decides answerability with a **hybrid rule**. The question is `is_blank` if the model said so **or** the top retrieval cosine score is below `retrieval_threshold`. Weak retrieval means the model is about to answer from memory instead of the papers, which is what RAG exists to prevent. A lower threshold marks fewer questions unanswerable, a higher one more; change it, rerun, and rescore to see the trade-off.
4. Copies `answer_unit` from the row (never guessed), maps each `ref_id` to its URL via `metadata.csv`, and makes the explanation call.
5. Returns one submission-format row; we collect them into `train_solutions_qwen.csv`.

```python
docid_to_url = build_docid_to_url(metadata_df)
from wattbot_rag import drop_questions_without_papers
train_df = drop_questions_without_papers(train_df, {d["doc_id"] for d in chunked_docs})   # as in the other RAG episodes: questions citing papers the corpus lacks are not scored

all_results = []
max_N = np.inf   # set to e.g. 5 for a smoke test
run_start = time.time()

for idx, row in train_df.iterrows():
    if idx >= max_N:
        break
    print("#" * 96)
    print(f"QUESTION: {row['question']}")

    res = run_single_qa(
        row=row,
        embed_query=embed_query,             # the embedding model on this GPU
        generate=call_qwen_chat,             # the vLLM engine on this GPU; the Bedrock episode passed its converse wrapper here
        chunk_embeddings=chunk_embeddings,
        chunked_docs=chunked_docs,
        docid_to_url=docid_to_url,
        top_k=8,
        retrieval_threshold=0.25,
        with_evidence_type=True,             # ask the explanation call to tag [text], [table], [figure]
        verbose=True,
    )
    print(f"ANSWER: {res['answer']}")
    print(f"ref_ids: {res['ref_id']}")
    print(f"EXPLANATION: {res['explanation']}")
    all_results.append(res)

solutions_df = pd.DataFrame(all_results)
print(f"Answered {len(solutions_df)} questions in {(time.time() - run_start) / 60:.1f} min on the notebook GPU")
solutions_path = os.path.join(local_data_dir, "train_solutions_qwen.csv")
solutions_df.to_csv(solutions_path, index=False)

print(f"Saved solutions for {len(solutions_df)} questions to: {solutions_path}")
solutions_df.head(20)
```

### Compare answers to ground truth

WattBot scores each prediction with the competition's own `Score.py`, 0.75 for `answer_value` (numeric within ±0.1% relative tolerance, categorical match after normalisation, zero if you answer a question the corpus cannot answer), 0.20 for citation F1 against the ground-truth `ref_id` set, and 0.05 for a clean refusal when the answer is `is_blank`. `compute_wattbot_score` from `wattbot_rag.py` calls it for the headline number, adds a per-row score, and prints the worst rows with the model's explanation.

```python
results_df = compute_wattbot_score(
    train_qa_path=train_qa_path,
    preds_path=solutions_path,
    n_examples=10,
)
```

Change `retrieval_threshold` or `top_k` in the loop above, rerun it, and rescore. With the model loaded, a pass over all 245 questions takes tens of minutes, and a `max_N` of 50 a few. Then stop the instance.

## Recap and next steps

In this episode, we:

- Pulled the WattBot corpus chunks that the Bedrock episode cached on S3 into our notebook environment, and reused the pipeline code (`wattbot_rag.py`) from the Bedrock and Processing Jobs episodes unchanged.
- Created vector embeddings for each chunk and used similarity search to retrieve relevant context for a given question.
- Used an LLM to generate answers from retrieved context and wrote results out to a CSV for later scoring and analysis.
- Handled unanswerable questions with an `is_blank` flag so the system can explicitly say “I don’t know” when the evidence isn’t there.

This is a first pass at a RAG pipeline. Next steps, in rough order of what they usually gain:

- **Stronger models**: a larger or domain-tuned embedding model, or a more capable generator, if the GPU budget allows.

- **Add a reranking step**: Instead of sending the top-k raw nearest neighbors directly to the LLM, use a cross-encoder or reranker model to re-score those candidates and send only the best ones.

- **Handle figures and tables more carefully**: Many key numbers live in tables, figure captions, or plots. Consider:
  - OCR / table-parsing tools (e.g., `pytesseract`, table extractors, PDF parsers).
  - Multimodal models that can embed or interpret figures and diagrams, not just text.
  - Separate chunking strategies for captions, tables, and main text.

- **Enrich chunks with metadata**: Attach metadata like section headings (e.g., *Methods*, *Results*), paper ID, year, or paragraph type. You can:
  - Filter or boost chunks by metadata at retrieval time.
  - Use metadata in the prompt so the LLM knows where evidence is coming from.

- **Models tuned on scientific text** (arXiv, PubMed), which read equations and technical language more reliably and invent less when the prose is dense.

Models, retrieval and preprocessing can each be swapped without changing the rest of the pipeline.

### What did this episode cost?

There is no job record to ask here; the bill is simply the instance's hours. This counts from the setup cell, so the boot, the install and every pause since are in it, and it keeps counting until you stop the instance.

```python
g6e_price = 2.61   # $/hour, ml.g6e.xlarge on-demand, from the Instances for ML page
hours = (time.time() - session_start) / 3600
print(f"this notebook has been running {hours * 60:.0f} min since the setup cell: ${hours * g6e_price:.2f} so far, and counting")
print(f"the same pipeline on the 53-question sample billed about $0.50 as Processing jobs; on Bedrock, about fifteen cents")
```

The Processing Jobs episode does the same work, with the same `wattbot_rag.py`, as short-lived SageMaker jobs, so the GPU only bills while it is embedding or generating. Stop this instance now, and delete it if you are done with the episode.



::::::::::::::::::::::::::::::::::::: keypoints

- **Why self-host at all**: Bedrock's catalog is curated and lags releases; most open-weight models (including Qwen3.8-27B) are not on it. When the model you need is not offered, you rent a GPU and serve it yourself.
- **Inference uses reduced-precision weights**: no gradients means the weights are the memory cost. FP8 puts a 27B model on one 48 GB GPU (`ml.g6e.xlarge`); bf16 would need 54 GB, an 80 GB GPU or a multi-GPU instance.
- **The meter never stops on its own**: `ml.g6e.xlarge` is `$2.61`/h, `$167` per forgotten weekend, `$1,906` per forgotten month, plus `$14`/month for the storage volume even when stopped. Stop it when you stand up; delete it when the episode is over.
- **Retrieval is a matrix multiply**: cosine similarity over the in-memory embedding matrix is enough for a corpus this size; a vector database is the same step for a corpus of millions.
- **Units come from the data**: `answer_unit` is copied from `train_QA.csv`, never guessed by the model.
- **Two calls per question**: one answers and cites; a second, shorter one writes the explanation and names the evidence type.
- **The competition's format**: `is_blank` marks unanswerable questions and missing fields, as the scorer expects.
::::::::::::::::::::::::::::::::::::::::::::::::

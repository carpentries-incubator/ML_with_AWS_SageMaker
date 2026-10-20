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
- Implement a simple, explicit “from scratch” embedding pipeline with a self-hosted sentence-transformer, without relying on FAISS or production vector DBs.
- Build a small retrieval helper that finds the most relevant chunks for a question using cosine similarity in embedding space.
- Serve a 27B open-weight generator (Qwen3.8-27B in FP8, via vLLM) on the notebook's GPU and wire it to the retriever to produce WattBot-format answers (including `answer`, `ref_id`, `ref_url`, and `supporting_materials`).
- Explain why inference uses quantized weights rather than bf16, and what a self-hosted GPU costs per hour, per forgotten weekend, and per forgotten month.
- Add a second LLM pass that generates short explanations and marks whether the evidence comes from text, figures, tables, or a combination.

::::::::::::::::::::::::::::::::::::::::::::::::



::::::::::::::::::::::::::::::::::::: callout

### Optional, self-study episode

This episode is not taught in the workshop. It runs the same self-hosted pipeline as the Processing Jobs episode, but interactively on a GPU notebook instance, which is the right tool for prototyping alone at a keyboard and the wrong one for a room of a hundred people with meters running. Work through it on your own when you want to see model loading, GPU memory and generation as plain Python you can poke at. Read the cost callout first.

::::::::::::::::::::::::::::::::::::::::::::::::

## When Bedrock does not have your model

In the Bedrock episode we built the full WattBot pipeline on **Amazon Bedrock**, the recommended default for RAG inference because it bills per token and leaves nothing running between calls. Bedrock's catalog is curated, though. It has Amazon's own models, Anthropic, Meta, Mistral, DeepSeek, and a handful of others, and even within a family it carries a few sizes and lags the releases: for Qwen, the serverless lineup is still the Qwen3 generation (Qwen3-32B, Qwen3-235B and the Qwen3-Coder models). Hugging Face hosts well over a million open-weight checkpoints. The one you want (the newest release, a specific size, a specific fine-tune, a model tuned on scientific text) is usually not on Bedrock, and often never will be.

When that happens you host the model yourself on a GPU you rent. This episode does that in its simplest form: a single GPU notebook instance, with **Qwen3.8-27B** as the generator. It is a 27-billion-parameter open-weight (Apache 2.0) model released in August 2026 that is not offered on Bedrock, large enough to follow the WattBot answer format reliably, and small enough to fit on one 48 GB GPU once its weights are stored in 8-bit floating point. Everything Bedrock abstracts away (loading an embedding model, loading a generator, fitting both in GPU memory) becomes visible as plain Python. Treat it as a learning exercise and a prototyping workbench, not a deployment pattern.

::::::::::::::::::::::::::::::::::::: callout

### This instance bills $2.61 an hour until you stop it, and nothing stops it for you

A GPU notebook instance is a rental with the meter running. It bills by the hour from the moment it starts until the moment *you* stop it, whether it is generating answers, sitting at an idle prompt, or showing a closed browser tab on a laptop that went home for the weekend. Nothing in this episode turns it off for you. The instance we use, `ml.g6e.xlarge` (one L40S GPU, 48 GB), costs **$2.61 per hour** on-demand in `us-east-1` (AWS price list, October 2026). That is:

| Left running for | Cost |
|---|---|
| This episode (~2 hours) | ~$5 |
| Overnight (12 h) | ~$31 |
| A weekend (64 h) | ~$167 |
| A week | ~$438 |
| A month | **~$1,906** |

The same evaluation on Bedrock cost well under a dollar. One forgotten weekend on this instance costs a thousand times more than the work it was rented for. A forgotten GPU notebook is the most common way research groups overspend cloud credits.

Two further charges people do not expect:

- **A stopped instance still bills for its storage.** We give this notebook a 100 GB volume for model weights. SageMaker charges $0.14 per GB-month for notebook storage, so a stopped-but-not-deleted instance costs about **$14 a month** forever. When you are done with the episode, delete the instance, not just stop it, unless you will use it again within days.
- **Idle is billed the same as busy.** There is no discount for a GPU that is loaded and waiting. Loading a 27B model takes a few minutes, which tempts people to leave the instance up "so I don't have to reload". That is the $1,906.

The habits that keep this safe:

1. **Stop the instance when you stand up**, every time, from the console (**Notebook instances → select → Stop**). Then look at the status column until it says `Stopped`.
2. **Set a reminder** on your phone or calendar before you start the instance, for two hours from now.
3. **Prefer jobs for anything beyond exploration**: a Processing job (the Processing Jobs episode) or a training job ends on its own, and `max_runtime_in_seconds` (`max_run` on a training job) guarantees it.

::::::::::::::::::::::::::::::::::::::::::::::::

Self-hosting on a notebook GPU is nonetheless the right starting point for understanding the alternatives, because it is:

- Easy to understand on the AWS side (pick a GPU instance, pick a model, go)
- Ideal for learning what retrieval and generation do, step by step
- The fastest loop for experimenting with prompts and models that Bedrock does not carry

The Processing Jobs episode runs this same pipeline as short-lived jobs, so the GPU only bills while it is embedding or generating; that is the pattern to use for real work. Later patterns (inference endpoints, batch transform) build on the same idea.


## Overview: WattBot RAG on a single notebook GPU

In this episode we rerun the [WattBot 2025](https://www.kaggle.com/competitions/WattBot2025/overview) RAG pipeline from the Bedrock episode, this time entirely from a single GPU-backed SageMaker notebook with self-hosted models.

In this episode we will:

1. **Reuse the WattBot data you prepared in the Bedrock episode.**
   - Pull `train_QA.csv` and `metadata.csv` from the shared bucket and the cached `chunks.jsonl` from your own.
2. **Implement the model steps explicitly in code.**
   - Embed chunks with a sentence-transformer (`thenlper/gte-large`) running on the notebook's GPU.
   - Implement cosine-similarity search over the embedding matrix.
3. **Serve an open-weight generator that Bedrock does not offer.**
   - Load Qwen3.8-27B with FP8 weights through vLLM on a single `ml.g6e.xlarge` GPU.
   - Construct WattBot-style answers that we can compare against `train_QA.csv`.
4. **Add an explanation pass.**
   - Use an LLM to look at the retrieved evidence, the answer, and citations.
   - Generate a short explanation and label the **evidence type**: `[text]`, `[table]`, `[figure]` or a combination such as `[text+table]`.


## Notebook + dataset setup

For this episode you need a notebook instance with a GPU that can hold the FP8 27B model plus the embedding model plus working memory for generation. Create a new notebook instance (same steps as the controller-notebook episode, same three tags) with:

- **Instance type:** `ml.g6e.xlarge` (1 × L40S, 48 GB GPU memory, $2.61/h). The FP8 weights are ~27 GB; the rest is for the embedding model and the KV cache that generation needs for long prompts. The L40S runs FP8 natively. A 24 GB instance such as `ml.g5.xlarge` cannot hold this model; a 16 GB `ml.g4dn.xlarge` cannot hold anything in this size class.
- **Volume size:** `100` GB (the default of 5 GB will not fit the weights or the vLLM install). This volume bills at $0.14 per GB-month even while the instance is stopped.
- **Tags:** the same `Project`, `Name`, `Purpose` keys as every other resource; `Purpose` is `Notebook`.

See [Instances for ML](https://carpentries-incubator.github.io/ML_with_AWS_SageMaker/instances-for-ML.html) for further guidance. This instance bills by the hour from the moment it starts, whether or not it is doing anything: **stop it when you leave**, and **delete it** when the episode is over.

We reuse the same two buckets as the Bedrock episode: the shared bucket for `metadata.csv` and `train_QA.csv`, and your own bucket for the `wattbot/chunks.jsonl` you cached there. If you skipped that episode, go back and run its chunking step first. Your notebook instance carries the tags you gave it when you created it and your bucket carries its own tags, so in this episode those two resources are the only things being billed.


###  Verify GPU and basic environment


```python
# nvidia-smi lists the GPU, its memory and the driver; it fails when the instance has no GPU
!nvidia-smi || echo "No GPU detected – please switch to a GPU-backed instance (ml.g6e.xlarge) before running this notebook."
```


```python
# also verify you have selected a PyTorch (conda_pytorch_*) kernel
import torch
print("torch cuda available:", torch.cuda.is_available())   # False means PyTorch cannot use the GPU even if nvidia-smi sees it
print("num gpus:", torch.cuda.device_count())
```

### Install vLLM first

We install the inference server before loading anything else, because the install may replace PyTorch and then needs a kernel restart; doing it now means nothing is lost by restarting.

::::::::::::::::::::::::::::::::::::: callout

### Installing vLLM takes a few minutes, and the meter is running

`pip install vllm` downloads several gigabytes (it brings its own CUDA kernels and may replace the kernel's PyTorch build) and takes 5–10 minutes. Qwen3.8 needs vLLM 0.17 or newer. The 27 GB of model weights take another 4–6 minutes to download the first time. At $2.61 an hour that is about 60 cents of waiting, which is fine; what is not fine is doing it twice because the instance was deleted and recreated. Keep the Hugging Face cache on the 100 GB volume (the `HF_HOME` line in the serving cell below does that) so the weights survive a stop/start.

If the install changes PyTorch, restart the kernel (**Kernel → Restart**) before continuing.

::::::::::::::::::::::::::::::::::::::::::::::::

```python
# Several GB; it may replace torch, so restart the kernel (Kernel -> Restart) if the log says it did
%pip install -q "vllm>=0.17"
```


## Import data from bucket into notebook


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
shared_bucket = "ml-marathon-shared"   # <-- EDIT ME: read-only inputs, from your instructor
bucket_name = "sinkorswim-doejohn"     # <-- EDIT ME: your own bucket (chunks live here)

# Local working directory in the notebook instance
local_data_dir = "./data"

print("Local data dir:", local_data_dir)
```

### Get the shared WattBot helpers

The retrieval, prompts, answer parsing, per-question orchestration and the scorer are the same code the Bedrock and Processing Jobs episodes used: `scripts/wattbot_rag.py` in the lesson repository. Copy the module next to this notebook. What this episode adds is the two model callables, `embed_query` and `generate`, running on the notebook's own GPU.

```python
!cp /home/ec2-user/SageMaker/ML_with_AWS_SageMaker/scripts/wattbot_rag.py .
```

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
from sentence_transformers import SentenceTransformer   # embedding-model wrapper: encode(list of strings) -> matrix
```


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

Now we embed each chunk into a vector using a **sentence-transformer** model. For WattBot, a strong and relatively efficient choice is:

### `thenlper/gte-large` (Recommended baseline embedder)

- Size / parameters:  ~335M parameters, roughly 1.3–1.4 GB in BF16/FP16 when loaded on GPU. Fits cleanly on T4 (16 GB), L4, A10G, A10, A100, and all g5.* instances.  Offers noticeably better retrieval quality than smaller 100M–150M models without requiring high-end GPU memory. Runs comfortably on g4dn.xlarge, g5.xlarge, or g5.2xlarge during workshops. Lets participants see meaningful improvements from chunking and retrieval methods without excessive compute cost.

- Intended use:  General-purpose retrieval and semantic search across academic PDFs, sustainability reports, and mixed-domain long-form documents. Stronger semantic coherence than gte-base or MiniLM, but still lightweight enough for workshop hardware.

- Throughput expectations:
  - CPU only: workable for small corpora (<2k chunks) but slow for anything larger.  
  - GPU (T4, L4, A10G, A100) with batch sizes around 64–128:  
    - 20k–40k chunks/min on L4 or A10G  
    - 10k–15k chunks/min on T4  
    - 50k+ chunks/min on A100  
      
We will:

1. Load the embedding model on GPU if available.
2. Encode all chunks in batches.
3. Store the resulting matrix as a `torch.Tensor` or `numpy.ndarray` along with the original `chunked_docs` list.

Later, we’ll implement a small retrieval helper that does cosine-similarity search over this matrix—no additional indexing library required.



```python
# We'll use a stronger embedding model now that we have a GPU.
# This model has ~335M parameters and benefits from GPU acceleration,
# but is still reasonable to run on a single 24 GB GPU.
embedding_model_id = "thenlper/gte-large"

use_cuda_for_embeddings = torch.cuda.is_available()
print("CUDA available for embeddings:", use_cuda_for_embeddings)

# Single shared embedder object that we can pass around.
embedder = SentenceTransformer(
    embedding_model_id,
    device="cuda" if use_cuda_for_embeddings else "cpu"   # on the GPU; CPU works too, but much more slowly
)


```


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
    embeddings = np.vstack(all_embeddings) if all_embeddings else np.zeros((0, 768))
    print(f"Computed embeddings for {len(texts)} chunks in {time.time() - start:.1f}s")
    return embeddings

```


```python

chunk_embeddings = embed_texts(embedder, chunked_docs)
chunk_embeddings.shape

```


### 4. Build a simple retrieval step (cosine similarity)

We are **not** using a heavy vector database in this episode.

Instead, we:

1. Embed each chunk with `thenlper/gte-large` (done above).
2. Embed each question.
3. Compute cosine similarity between the question embedding and all chunk embeddings.
4. Take the top–k most similar chunks as our retrieved context.

This keeps the retrieval logic completely transparent for teaching, while still matching the *spirit* of
production systems that use FAISS, Chroma, Weaviate, etc.

#### When might FAISS or a vector database be worth exploring?

For small–to–medium experiments (a few thousand to maybe tens of thousands of chunks), this "plain NumPy + cosine
similarity" approach is usually enough. You might consider FAISS or a full vector DB when:

- **Your corpus gets big**  
  Once you’re in the hundreds of thousands to millions of chunks, brute-force similarity search can become slow
  and memory-hungry. FAISS and friends provide *approximate nearest neighbor* search that scales much better.

- **You need low-latency, repeated queries**  
  If many users (or a web app) will hit your RAG system concurrently, you’ll want:
  - fast indexing,
  - efficient caching, and
  - sub-second query latency.  
  Vector DBs are designed for this use case.

- **You need rich filtering or metadata search**  
  Vector DBs often support:
  - filtering by metadata (e.g., `paper = "chung2025"`, `year > 2021`),
  - combining keyword + vector search (“hybrid search”),
  - role-based access control and multi-tenant setups.

- **You want to share an index across services**  
  If multiple notebooks, microservices, or teams need to reuse the **same embedding index**, a shared FAISS index or
  hosted vector DB is much easier to manage than passing around `.npy` files.

- **You need GPU-accelerated or distributed search**  
  FAISS can use GPUs and sharding to speed up search on very large embedding collections. This is overkill for our
  teaching demo (and the Wattbot project in general), but very relevant for production-scale systems.

In this episode we deliberately stick with a simple in-memory index so the retrieval step is easy to inspect and
debug. In later episodes (or your own projects), you can **swap out the retrieval layer** for FAISS or a vector DB
without changing the overall RAG architecture: the model still sees “top–k retrieved chunks” as context.


`retrieve_top_k` from `wattbot_rag.py` is that search: normalize the query and the matrix, one matrix multiply for the cosine similarities, `argsort` for the top *k*. Open the file to see it; it is a dozen lines. `embed_query_with` wraps the sentence-transformer as the `embed_query(question)` callable the rest of the pipeline expects.

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


### 5. Serve the generator: Qwen3.8-27B in FP8, via vLLM

For this episode we use **Qwen3.8-27B**, an open-weight model that Bedrock does not carry, in its **FP8** release (`Qwen/Qwen3.8-27B-FP8`), served with [vLLM](https://docs.vllm.ai/). It is a dense 27B model with a hybrid attention design (most layers use a linear-attention variant with a fixed-size state, so the KV cache stays small even for long prompts), a native 262K context, and a "thinking" mode that is on by default and that we turn off, because WattBot wants a JSON answer, not a reasoning trace.

**Why FP8 weights and not bf16?** Inference is not training. No gradients, no optimizer state, nothing is updated, so the only memory that matters is the weights themselves plus the KV cache for the prompt being processed. A 27B model in bf16 is about 54 GB of weights and does not fit a 48 GB GPU; the same weights in 8-bit floating point are about 27 GB, which fits with room for the embedding model and generation. FP8 is near-lossless for a model this size, and it is faster, because decoding is limited by how fast the GPU can read weights from memory. For inference on rented GPUs, reduced-precision weights are the norm (FP8 where the GPU supports it, 4-bit integer where it does not or where memory is tighter); bf16 is what you use when you have memory to spare and a model small enough not to need it.

**Why vLLM and not `transformers.pipeline`?** vLLM is an inference server: it has the kernels for FP8 and for this model's hybrid attention, batches requests, and manages the KV cache so long prompts do not run out of memory. It is what you would deploy behind an API, and its offline `LLM` class is a two-line replacement for a `transformers.pipeline`.

We will:

1. Call the model once to propose an answer and supporting evidence.
2. Call it a **second time** with a smaller prompt to generate a short explanation (<= 100 characters).

```python
import os
# Keep model weights on the notebook's 100 GB volume rather than the small root disk.
os.environ["HF_HOME"] = "/home/ec2-user/SageMaker/hf_cache"

from vllm import LLM, SamplingParams

qwen_model_id = "Qwen/Qwen3.8-27B-FP8"   # pre-quantized FP8 checkpoint; vLLM reads that from its config, so no quantization= argument is needed

llm = LLM(
    model=qwen_model_id,                         # Hugging Face id; the weights download to HF_HOME on first use
    max_model_len=8192,                          # longest prompt + answer we need; the model allows 262K, we do not
    gpu_memory_utilization=0.85,                 # leave ~7 GB for the embedding model and PyTorch overhead
    limit_mm_per_prompt={"image": 0, "video": 0},  # text only: skip reserving memory for the vision encoder
)
print("vLLM engine ready:", qwen_model_id)
```

```python
!nvidia-smi --query-gpu=memory.used,memory.total --format=csv
```

The number above is what you are paying for: a 48 GB GPU, most of it now holding model weights and a reserved KV cache. It stays allocated, and billed, until you stop the instance.

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
3. Decides answerability with a **hybrid rule**: the question is `is_blank` if the model said so **or** the top retrieval cosine score is below `retrieval_threshold`. Weak retrieval means the model is about to answer from memory instead of the papers, which is what RAG exists to prevent. A lower threshold marks fewer questions unanswerable, a higher one more; change it, rerun, and rescore to see the trade-off.
4. Copies `answer_unit` from the row (never guessed), maps each `ref_id` to its URL via `metadata.csv`, and makes the explanation call.
5. Returns one submission-format row; we collect them into `train_solutions_qwen.csv`.

```python
docid_to_url = build_docid_to_url(metadata_df)

all_results = []
max_N = np.inf   # set to e.g. 5 for a smoke test

for idx, row in train_df.iterrows():
    if idx >= max_N:
        break
    print("#" * 96)
    print(f"QUESTION: {row['question']}")

    res = run_single_qa(
        row=row,
        embed_query=embed_query,             # the sentence-transformer on this GPU
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
solutions_path = os.path.join(local_data_dir, "train_solutions_qwen.csv")
solutions_df.to_csv(solutions_path, index=False)

print(f"Saved solutions for {len(solutions_df)} questions to: {solutions_path}")
solutions_df.head(20)
```

### Compare answers to ground truth

WattBot evaluates each prediction using a weighted score that combines three components. Most of the credit (0.75) comes from the `answer_value`, which must match the ground truth after normalization (numeric answers must be within ±0.1% relative tolerance; categorical values must match exactly). An additional 0.15 comes from `ref_id`, where partial credit is given based on the Jaccard overlap between your cited document IDs and the ground-truth set. The final 0.10 comes from correctly marking unanswerable questions: if a question is truly unanswerable, you must set `answer_value`, `ref_id`, and `supporting_materials` to `is_blank`. Any other combination scores zero for this component.

| Component      | Weight | What counts as correct |
|----------------|--------|------------------------|
| answer_value   | 0.75   | Numeric within ±0.1% relative tolerance; categorical exact match; `is_blank` if unanswerable |
| ref_id         | 0.15   | Jaccard overlap with the ground-truth reference set (case-insensitive) |
| is_NA          | 0.10   | All required fields set to `is_blank` when the question is truly unanswerable |



`compute_wattbot_score` from `wattbot_rag.py` implements that metric, joins predictions to `train_QA.csv` on `id`, and prints the worst-scoring rows. `train_QA.csv` has no separate "unanswerable" column, so `gt_is_na_col="is_blank"` derives that flag from `answer_value == "is_blank"` on both sides. The scorer accepts `ref_id` either as `;`-joined ids (what the pipeline writes) or as the Python-list strings in `train_QA.csv`.

```python
results_df = compute_wattbot_score(
    train_qa_path=train_qa_path,
    preds_path=solutions_path,
    gt_is_na_col="is_blank",
    pred_is_na_col="is_blank",
    n_examples=10,
)
```

Change `retrieval_threshold` or `top_k` in the loop above, rerun it, and rescore: with the model loaded, a full pass over the training set takes a few minutes, which is what a GPU notebook is for. Then stop the instance.

## Recap and next steps

In this episode, we:

- Pulled the WattBot corpus chunks that the Bedrock episode cached on S3 into our notebook environment, and reused the pipeline code (`wattbot_rag.py`) from the Bedrock and Processing Jobs episodes unchanged.
- Created vector embeddings for each chunk and used similarity search to retrieve relevant context for a given question.
- Used an LLM to generate answers from retrieved context and wrote results out to a CSV for later scoring and analysis.
- Handled unanswerable questions with an `is_blank` flag so the system can explicitly say “I don’t know” when the evidence isn’t there.

This is just a first pass at a RAG pipeline: it works, but there’s a lot of headroom to improve both accuracy and robustness. Some natural next steps:

- **Increase the size/quality of models used for embedding and generation**: Try stronger embedding models (e.g., larger sentence-transformers or domain-tuned embeddings) and more capable LLMs for answer generation, especially if you have GPU budget.

- **Add a reranking step**: Instead of sending the top-k raw nearest neighbors directly to the LLM, use a cross-encoder or reranker model to re-score those candidates and send only the best ones.

- **Handle figures and tables more carefully**: Many key numbers live in tables, figure captions, or plots. Consider:
  - OCR / table-parsing tools (e.g., `pytesseract`, table extractors, PDF parsers).
  - Multimodal models that can embed or interpret figures and diagrams, not just text.
  - Separate chunking strategies for captions, tables, and main text.

- **Enrich chunks with metadata**: Attach metadata like section headings (e.g., *Methods*, *Results*), paper ID, year, or paragraph type. You can:
  - Filter or boost chunks by metadata at retrieval time.
  - Use metadata in the prompt so the LLM knows where evidence is coming from.

- **Look for LLMs tuned for scientific literature**: Experiment with models that are explicitly trained or finetuned on scientific text (e.g., arXiv / PubMed) so they:
  - Parse equations and technical language more reliably.
  - Are less likely to hallucinate when reading dense scientific prose.

As you iterate, the goal is to treat this notebook as a baseline RAG “workbench”: you can swap in better models, smarter retrieval strategies, and richer document preprocessing without changing the overall pipeline structure.

The Processing Jobs episode does the same work, with the same `wattbot_rag.py`, as short-lived SageMaker jobs, so the GPU only bills while it is embedding or generating. Stop this instance now, and delete it if you are done with the episode.



::::::::::::::::::::::::::::::::::::: keypoints

- **Why self-host at all**: Bedrock's catalog is curated and lags releases; most open-weight models (including Qwen3.8-27B) are not on it. When the model you need is not offered, you rent a GPU and serve it yourself.
- **Inference uses reduced-precision weights**: no gradients means the weights are the memory cost. FP8 puts a 27B model on one 48 GB GPU (`ml.g6e.xlarge`); bf16 would need 54 GB and a multi-GPU instance.
- **The meter never stops on its own**: `ml.g6e.xlarge` is $2.61/h, $167 per forgotten weekend, $1,906 per forgotten month, plus $14/month for the storage volume even when stopped. Stop it when you stand up; delete it when the episode is over.
- **Local-first RAG**: For teaching (and small corpora), we avoid an external vector database
  and instead perform cosine similarity search over in-memory embeddings.
- **Ground-truth units**: The `answer_unit` column is always copied directly
  from `train_QA.csv`, never guessed by the LLM.
- **Two-stage LLM use**: One call focuses on *answering and citing*; a second,
  lighter call produces a short explanation tagged with an evidence type.
- **WattBot conventions**: We respect the Kaggle competition format,
  using `is_blank` for unanswerable questions and for missing fields.
- **Scalability path**: The same logic can later be swapped to FAISS/Chroma
  and larger models, while preserving the interface used here.
::::::::::::::::::::::::::::::::::::::::::::::::

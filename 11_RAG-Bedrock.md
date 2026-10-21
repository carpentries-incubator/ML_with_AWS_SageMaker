---
title: "RAG with Bedrock"
teaching: 30
exercises: 20
---

:::::::::::::::::::::::::::::::::::::: questions

- How do we build a complete RAG pipeline on AWS without provisioning a GPU?
- How do we turn a collection of PDFs into retrievable, citable chunks stored in S3?
- How do we call Bedrock-hosted embedding and generation models so that the spend is tagged to our project?

::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: objectives

- Read the WattBot metadata, questions and PDF corpus from the shared bucket, and cache what you derive from them (`chunks.jsonl`) in your own bucket for the later episodes.
- Convert PDFs into page-level records and overlapping text chunks.
- Create tagged application inference profiles so that every Bedrock call is attributed to your project.
- Embed chunks with Amazon Titan Text Embeddings and retrieve relevant context with cosine similarity.
- Generate WattBot-format answers and explanations with a Bedrock-hosted model (DeepSeek V3) and score them against `train_QA.csv`.
- Build a second index from page images with a multimodal embedding model and measure what it changes for the table and figure questions.

::::::::::::::::::::::::::::::::::::::::::::::::

Open the notebook from: `/ML_with_AWS_SageMaker/notebooks/11_RAG-Bedrock.ipynb`. Select the **`conda_pytorch`** kernel.
This is the first hands-on RAG episode, and it starts where the overview told you to start: **Amazon Bedrock**. We build a complete RAG pipeline for the [WattBot 2026](https://www.kaggle.com/competitions/WattBot2026) challenge from a small, CPU-only SageMaker notebook. Nothing in this episode needs a GPU: the core model work — **both text embeddings and answer generation** — runs on Bedrock-hosted models that bill per token and leave nothing running between calls. We'll use:

- an **Amazon Titan Text Embeddings V2** model to turn WattBot chunks into vectors, and  
- a **DeepSeek V3** model hosted on Bedrock to generate answers and explanations.

We also do the corpus preparation here: unpack the WattBot PDF corpus, split it into pages and overlapping
chunks, and cache the chunks on S3. The Processing Jobs episode (and the optional GPU-notebook episode) reuse those chunks and run the same
retrieval, evaluation, and scoring logic on self-hosted models, so you can compare cost, latency, and
complexity directly.

## Why Bedrock for WattBot?

When you self-host models (as the Processing Jobs episode does next), you are responsible for picking a model,
managing versions, and making sure your instance has enough VRAM. That’s fine for experiments,
but it can get painful once multiple teams or challenges want to reuse the same pipeline.

Running your **embedding + generation** steps on Amazon Bedrock gives you a few nice properties:

- **Managed, up-to-date models.** You can use high‑quality models from Anthropic, Amazon, DeepSeek, and
  others without worrying about container images or CUDA versions.
- **Pay for what you use (in tokens).** Instead of paying for a GPU instance that might sit
  idle, you pay per token (input + output) when you call the model. For some workloads this
  is cheaper; for large offline batches with smaller models, a dedicated GPU can still win.
- **Easier sharing and governance.** It’s easier to standardize on a small set of Bedrock
  models across courses, hackathons, or labs than to manage many separate GPU instances.

In this notebook, both the **embedding** and **answer/explanation** steps run on
Bedrock-hosted models. The embeddings are saved as `embeddings_bedrock.npy` so you can compare them
against the self-hosted run in the Processing Jobs episode.


### Models used in this episode

We’ll work with **Amazon Bedrock–hosted foundation models** for both embedding and generation:

- **Amazon Titan Text Embeddings V2** (`amazon.titan-embed-text-v2:0`)

  - General‑purpose text‑embedding model for semantic search, retrieval, clustering, and classification.
  - Supports configurable embedding dimensions (256, 512, or 1,024) and has presets tuned for retrieval or binary indexing.
  - AWS does not publish the exact number of parameters for Titan models; you can treat it as a modern transformer specialized for embeddings rather than free‑form text generation. 

- **DeepSeek V3** (`deepseek.v3-v1:0` via Bedrock)

  - An open‑weight mixture‑of‑experts model (671B parameters in total, about 37B active per token) that Bedrock serves on‑demand, so there is no GPU to manage.
  - Particularly useful when you want many calls (e.g., one per question) and care about low latency and lower per‑token pricing compared to flagship models (check the [Bedrock pricing page](https://aws.amazon.com/bedrock/pricing/) for current rates).
  - The same code works for any chat model in the catalog. Bedrock has one chat call, `converse`, that takes the same input for every provider (a system prompt and a list of user and assistant messages) and returns the reply in the same shape, so switching `base_generation_model_id` to a Claude, Llama, or Mistral model ID changes nothing else.

- **(Optional) Multimodal models for tables and figures**

  - Bedrock also exposes **multimodal models** that can reason over images, charts, and document layouts (for example, Claude models with vision, or Amazon Titan Multimodal Embeddings). These are a good fit if much of your evidence lives in **figures, tables, or scanned PDFs**.
  - To use them from Bedrock you send **both text and image content** in a single request:
      - Pre‑process PDFs by rendering pages (or cropping individual tables/figures) to images using a tool like `pdf2image` or a headless browser.
      - Base64‑encode those images and include them as image parts alongside text in the model request.
      - For multimodal embeddings, you call a Titan multimodal embedding model with an `inputImage` (and optionally `inputText`) payload to obtain a single vector that mixes visual and textual information.
  - This notebook stays with **text‑only** embeddings + generation to keep the workflow simple, but the same RAG pattern extends naturally to multimodal models once you add an image‑extraction step to your preprocessing pipeline.

For the full list of available models (Claude variants, Amazon models, and partner models), open the [Model catalog](https://us-east-2.console.aws.amazon.com/bedrock/home?region=us-east-2#/model-catalog) in the Amazon Bedrock console. Each entry has a model card with capabilities, typical use cases, and pricing, for exploring alternatives for your own RAG system.

::::::::::::::::::::::::::::::::::::: callout

### Choosing the two models

The catalog tells you what exists and what it costs; it does not tell you what is good. Three places do, and all three are free to read:

- **Embedding models**: the [MTEB leaderboard](https://huggingface.co/spaces/mteb/leaderboard) scores embedding models on retrieval benchmarks. Filter to the retrieval task and your language; the top of the table changes every few months, and Titan V2 is a sound middle-of-the-pack choice that Bedrock serves without setup.
- **Generation models**: [LMArena](https://lmarena.ai/) ranks chat models by blind human preference, and [Artificial Analysis](https://artificialanalysis.ai/) plots quality against price per million tokens, which is the chart that matters for RAG: the answer is written from retrieved text, so a mid-priced model often reads as well as a flagship at a tenth of the cost.
- **Your own questions**: the benchmarks rank models on other people's data. `train_QA.csv` has labelled questions for this corpus, and the evaluation at the end of this episode scores a model on them and prices the run from its token counts. Changing `base_generation_model_id` and rerunning is the whole experiment. A model that wins on a leaderboard and loses on your questions has told you something the leaderboard cannot.

The habit is the same as for instances: pick the cheapest candidate that scores well enough, measure it on your data, and move up only when the measurement says to.

::::::::::::::::::::::::::::::::::::::::::::::::


## Notebook + dataset setup

For this episode you need a SageMaker notebook instance, but **not a GPU one**. A small CPU instance such as `ml.t3.medium` is enough, because the models run on Bedrock rather than in the notebook. The notebook's execution role must allow Bedrock access, which the default SageMaker role does not. On the class account that is the `ml-sagemaker-bedrock-use` role you picked when creating the notebook instance. If your instance was created with the older `ml-sagemaker-use` role, the first Bedrock call fails with an access-denied error: stop the instance, choose **Edit**, change the IAM role under *Permissions and encryption*, click **Update notebook instance**, and wait for the status to go from *Updating* back to *Stopped* (one to three minutes) before starting it. Reopening Edit during that wait shows the old role; the details page shows the new one once the update has landed. On your own account, attach the Bedrock policy from the *Account preparation checklist* instructor page to your role before starting.

See [Instances for ML](https://carpentries-incubator.github.io/ML_with_AWS_SageMaker/instances-for-ML.html) for further guidance on instance types.


### Two buckets, no uploads

This episode reads from two buckets and you create neither of them here:

- **The shared bucket** (your instructor gives you its name) holds the WattBot inputs under `wattbot/`: `metadata.csv` (the index of papers), `train_QA.csv` (labelled questions and answers), `corpus.zip` (the PDFs) and `pages.jsonl` (the text of every PDF page, extracted once). It is read-only and there is one copy for the whole class.
- **Your bucket** from the S3 episode (`challenge-lastnamefirstname`) is where this episode writes what it derives, under a `wattbot/` prefix: the chunked text (`wattbot/chunks.jsonl`), the embeddings (`wattbot/embeddings/`) and the predictions (`wattbot/solutions/`). Its tags cover its storage cost; your notebook instance carries the tags you gave it when you created it, and in this episode those two resources plus the per-token Bedrock calls (tagged through an inference profile, see "Cost tracking" below) are the only things being billed.

::::::::::::::::::::::::::::::::::::: callout

### Working through the lesson on your own?

If there is no instructor-staged bucket, stage the three input files in your own bucket under the same keys and point both variables at it: `metadata.csv` and `train_QA.csv` are in `data/data.zip` in the lesson repository, `corpus.zip` is built once by `scripts/build_wattbot_corpus.py`, and `instructors/shared-data-bucket.md` has the upload commands. Then set `shared_bucket = bucket_name` in the first cell below and everything else works unchanged.

::::::::::::::::::::::::::::::::::::::::::::::::

### Step 1 – Configure the notebook

Three things are new here compared with the training episodes, so the setup is split into four cells.

The session, region and S3 client are as before.

```python
import os
import json
import time
from typing import Dict, Any, List
from pathlib import Path

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError
import pandas as pd
import numpy as np
import sagemaker

session = sagemaker.Session()
region = session.boto_region_name
s3 = boto3.client("s3", region_name=region)
```

**The models.** RAG needs two: an embedding model, which turns a piece of text into a vector so that similar passages land near each other, and a generation model, the chat model that writes the answer from the passages it is given. Both are catalog models on Bedrock, named by a model ID; the [model catalog](https://us-east-2.console.aws.amazon.com/bedrock/home?region=us-east-2#/model-catalog) in the console lists everything available in the region with its ID and price, and the [supported models page](https://docs.aws.amazon.com/bedrock/latest/userguide/models-supported.html) in the documentation is the same list as a table. Prices are per million tokens on the [Bedrock pricing page](https://aws.amazon.com/bedrock/pricing/): in October 2026, DeepSeek V3 on demand is `$0.62` in and `$1.85` out, and Titan Text Embeddings V2 is `$0.02` in, which is why the whole WattBot run below costs cents. We write the IDs down now but will not call them by these IDs directly: the cost-tracking step below wraps each one in a tagged inference profile first, because that is the only way Bedrock spend gets attributed to a project.

```python
# Base model IDs. Any chat model in the Bedrock catalog works as the generator (a Claude or Llama ID, say),
# because converse, Bedrock's chat call, takes the same messages and returns the same shape for every provider.
base_generation_model_id = "deepseek.v3-v1:0"             # writes the answers; $0.62 in / $1.85 out per million tokens (Oct 2026)
base_embedding_model_id = "amazon.titan-embed-text-v2:0"  # turns text into vectors for retrieval; $0.02 per million tokens
```

**The files.** The inputs live in the shared bucket: the paper list, the questions with their answers, and the zipped corpus of PDFs. Everything this episode derives from them goes to your own bucket, under the same `wattbot/` prefix: the chunked text, the chunk embeddings, and the answers the pipeline produces.

```python
!mkdir -p /home/ec2-user/SageMaker/wattbot   # a project folder beside the lesson clone, so data and outputs stay out of the repository
%cd /home/ec2-user/SageMaker/wattbot

shared_bucket = "ml-marathon-shared"   # the instructor-staged, read-only bucket with the data and model weights; change only on your own account
bucket_name = "sinkorswim-doejohn"     # <-- EDIT ME: your own bucket from the S3 episode

# inputs, in the shared bucket
metadata_key = "wattbot/metadata.csv"      # one row per paper: id, title, url
train_key = "wattbot/train_QA.csv"         # the questions, with reference answers for scoring
corpus_key = "wattbot/corpus.zip"          # the PDFs, one per paper
shared_embeddings_key = "wattbot/embeddings_bedrock.npy"   # the instructor's embedding matrix, used when use_instructor_embeddings is True
use_instructor_embeddings = True   # True in class: download the instructor's matrix instead of embedding 9,000 chunks yourself (ten minutes at the quota). False to do it yourself

# derived, written to YOUR bucket
chunks_key = "wattbot/chunks.jsonl"                                # the corpus cut into passages (end of the chunking step)
embeddings_key = "wattbot/embeddings/embeddings_bedrock.npy"       # one vector per chunk
solutions_key = "wattbot/solutions/wattbot_solutions_bedrock.csv"  # the pipeline's answers, in the competition's format

local_data_dir = "./data"              # working copies on this notebook instance
os.makedirs(local_data_dir, exist_ok=True)
```

**Two Bedrock clients.** Bedrock has two APIs. `bedrock-runtime` is the one that calls models (`converse`, the chat call, for answers; `invoke_model`, the raw call with a provider-specific body, for embeddings); it is what every answer and every embedding in this episode goes through. `bedrock` is the control plane: listing models, creating the inference profiles that carry our tags. The runtime client gets a retry setting because on-demand Bedrock throughput is rate-limited per model: a loop of a few thousand embedding calls will occasionally be throttled, and the adaptive mode backs off and retries instead of raising.

```python
bedrock_runtime = boto3.client(
    "bedrock-runtime",
    region_name=region,
    config=Config(retries={"max_attempts": 10, "mode": "adaptive"}),   # back off and retry when throttled
)
bedrock = boto3.client("bedrock", region_name=region)                  # control plane: list models, create inference profiles
```

### Step 2 – Get the shared WattBot helpers

The retrieval, prompting, answer parsing and scoring logic of the WattBot pipeline is the same whether the models run on Bedrock, in a Processing job or on a notebook GPU. It lives once in the lesson repository, in `scripts/wattbot_rag.py`, and every RAG episode imports it; the only code that differs per episode is the two model calls. Clone the repository and copy the module next to this notebook (the Processing Jobs episode ships the same file inside its jobs).

```python
!cp /home/ec2-user/SageMaker/ML_with_AWS_SageMaker/scripts/wattbot_rag.py /home/ec2-user/SageMaker/ML_with_AWS_SageMaker/scripts/Score.py .
```

```python
from wattbot_rag import (
    ANSWER_SYSTEM_PROMPT,
    build_docid_to_url,
    compute_wattbot_score,
    retrieve_top_k,
    run_single_qa,
)
```

Open `wattbot_rag.py` in the Jupyter file browser as you go: each helper is short, and the docstring at the top says which episode uses what.

## Step 3 – Load the metadata and training questions

Pull the two CSV files from the shared bucket into the notebook and inspect them. Two helpers first: `download_from_s3` copies one object from the shared bucket (or any bucket you name) into the local data folder, and `smart_read_csv` reads a CSV by trying a few encodings in turn, because author names and titles in `metadata.csv` are not always valid UTF-8.

```python
import zipfile


def download_from_s3(key: str, local_name: str, bucket: str = shared_bucket) -> str:
    """Download a file from S3 (the shared bucket by default) to local_data_dir."""
    local_path = os.path.join(local_data_dir, local_name)
    print(f"Downloading s3://{bucket}/{key} -> {local_path}")
    s3.download_file(bucket, key, local_path)
    return local_path


def smart_read_csv(path: str) -> pd.DataFrame:
    """Read a CSV, trying utf-8 and then the common Windows and Latin encodings."""
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
```

Now the two files: `train_QA.csv` is the labelled questions and answers we score against, `metadata.csv` is one row per paper with its id, title and URL.

```python
train_qa_path = download_from_s3(train_key, "train_QA.csv")      # labelled questions + answers: what we score against
metadata_path = download_from_s3(metadata_key, "metadata.csv")   # one row per paper (id, title, url): the corpus index

train_df = smart_read_csv(train_qa_path)
metadata_df = smart_read_csv(metadata_path)

print("train_QA.csv columns:", train_df.columns.tolist())
print("metadata.csv columns:", metadata_df.columns.tolist())
print("\nNumber of training QAs:", len(train_df))
print("Number of metadata rows:", len(metadata_df))

train_df.head(15)
```

## Step 4 – Get the PDF corpus

The corpus is the set of PDFs listed in `metadata.csv`, one per paper, named `<id>.pdf`. Downloading them from the publishers' sites is slow and unreliable when fifty notebooks do it at once, so the instructors did it once with `scripts/build_wattbot_corpus.py` and staged the result as `corpus.zip` in the shared bucket. We download that one file, unpack it, and delete the zip: the instance's disk is 20 GB at most and the PDFs alone are 400 MB.

```python
PDF_DIR = os.path.join(local_data_dir, "pdfs")
os.makedirs(PDF_DIR, exist_ok=True)

corpus_path = download_from_s3(corpus_key, "corpus.zip")
with zipfile.ZipFile(corpus_path, "r") as zf:
    zf.extractall(PDF_DIR)
os.remove(corpus_path)   # the zip is 400 MB and the PDFs are now unpacked; the notebook disk is small

pdf_files = sorted(f for f in os.listdir(PDF_DIR) if f.lower().endswith(".pdf"))
print(f"{len(pdf_files)} PDFs in {PDF_DIR}; metadata lists {len(metadata_df)} papers")
```

::::::::::::::::::::::::::::::::::::: callout

### Where the corpus came from

`metadata.csv` has a `url` column pointing at each paper. `scripts/build_wattbot_corpus.py` in the lesson repository walks that column, downloads every PDF, zips them, and then extracts the text of every page into `pages.jsonl`; it reports any URL that failed so the instructor can decide what to do about it. If you are working on your own, run it once and upload both files to your bucket under `wattbot/` (with `shared_bucket = bucket_name`, as in the callout above). That is the whole difference between "the data" and "a copy of the data": the corpus is built once and read many times.

::::::::::::::::::::::::::::::::::::::::::::::::

## Step 5 – Turn PDFs into page-level “documents”

The next unit of work is a list of **page-level records**, one per PDF page. The instructor's staging script already built it for this corpus (`pages.jsonl` in the shared bucket), so below we define the function that does the extraction, then load the staged result rather than rerun it. Each record stores:

- `text`: page text (as extracted by `pypdf`).
- `doc_id`: short ID from `metadata.csv` (e.g., `strubell2019`).
- `title`: title of the document.
- `url`: original PDF URL.
- `page_num`: zero-based page index.
- `page_label`: label used inside the PDF (often 1-based).

Later, we will **chunk these pages** into smaller overlapping segments for embedding.

### Why we page-chunk first

We split the PDF into **pages before chunking** because pages give us a stable, easy-to-interpret unit.  
This helps with:

- **Keeping metadata** (doc ID, URL, page labels) tied to the text.  
- **Debugging retrieval** — it’s much easier to understand what the model saw if we know which page(s) were used.  
- **Cleaning text** before making smaller overlapping chunks.  
- **Flexibility later** — once pages are structured, we can try different chunk sizes or strategies without re-extracting the PDF.

In short: **pages first → then chunks** keeps the workflow cleaner and easier to reason about.

```python
%pip install -q pypdf
```

Open each PDF listed in the metadata with `pypdf` and turn every page into one record: its text, the document id, title and URL, and the page number. The function is the reference for what a page record is; on a CPU notebook it takes several minutes over this corpus, so the cell after it first looks for `pages.jsonl`, the same records written once when the corpus was staged.

```python
from pypdf import PdfReader

def pdfs_to_page_docs(metadata: pd.DataFrame, pdf_dir: str) -> List[Dict[str, Any]]:
    """Load each PDF into a list of page-level dictionaries.

    Each dict has keys: text, doc_id, title, url, page_num, page_label, total_pages.
    """
    page_docs: List[Dict[str, Any]] = []

    for _, row in metadata.iterrows():
        doc_id = str(row["id"]).strip()
        title = str(row.get("title", "")).strip()
        url = str(row.get("url", "")).strip()

        pdf_path = os.path.join(pdf_dir, f"{doc_id}.pdf")
        if not os.path.exists(pdf_path):
            print(f"Missing PDF for {doc_id}, skipping.")
            continue

        try:
            reader = PdfReader(pdf_path)
        except Exception as e:
            print(f"Failed to read {pdf_path}: {e}")
            continue

        total_pages = len(reader.pages)
        for i, page in enumerate(reader.pages):
            try:
                text = page.extract_text() or ""
            except Exception as e:
                print(f"Failed to extract text from {doc_id} page {i}: {e}")
                text = ""

            text = text.strip()
            if not text:
                # Still keep the page so we know it exists, but mark it as empty
                text = "[[EMPTY PAGE TEXT – see original PDF for tables/figures]]"

            page_docs.append(
                {
                    "text": text,
                    "doc_id": doc_id,
                    "title": title,
                    "url": url,
                    "page_num": i,
                    "page_label": str(i + 1),
                    "total_pages": total_pages,
                }
            )

    return page_docs


start = time.time()
try:
    pages_path = download_from_s3("wattbot/pages.jsonl", "pages.jsonl")   # staged next to corpus.zip by the corpus build script
    page_docs = [json.loads(line) for line in open(pages_path)]
    print(f"Loaded {len(page_docs)} pre-extracted page records in {time.time() - start:.1f} s")
except ClientError:
    page_docs = pdfs_to_page_docs(metadata_df, PDF_DIR)
    print(f"Extracted {len(page_docs)} page-level records from {len(metadata_df)} PDFs in {(time.time() - start) / 60:.1f} min")
page_docs[0] if page_docs else None
```

A paper whose PDF the corpus build could not download has no pages here, and a question that cites it can only score zero, which would measure the download rather than the pipeline. `drop_questions_without_papers` removes those questions from the evaluation set and says which papers were missing; the unanswerable questions, which cite nothing, stay.

```python
from wattbot_rag import drop_questions_without_papers

train_df = drop_questions_without_papers(train_df, {p["doc_id"] for p in page_docs})
print(f"{len(train_df)} questions can be answered from this corpus")
```

## Step 6 – Text chunking

RAG systems typically break documents into **chunks** so that:

- Each chunk is long enough to carry meaningful context.
- No chunk is so long that it blows up the embedding/LLM context window.

For this workshop we will implement a **simple sliding-window chunker** that operates on characters:

- `chunk_size_chars`: maximum characters per chunk (e.g., 1,000–1,500).
- `chunk_overlap_chars`: overlap between consecutive chunks (e.g., 200).

In your own work, you may wish to plug in more sophisticated *semantic chunking* methods (e.g., splitting on headings, section titles, or sentence boundaries). For now, we'll keep the implementation explicit and easy to debug.

```python
def split_text_into_chunks(
    text: str,
    chunk_size_chars: int = 1200,     # max characters per chunk (~300 tokens)
    chunk_overlap_chars: int = 200,   # consecutive chunks share this much, so a sentence cut at the edge survives
) -> List[str]:
    """Split `text` into overlapping character-based chunks.

    This is a simple baseline; more advanced versions might:
    - split on sentence boundaries, or
    - merge short paragraphs and respect section headings.
    """
    text = text.strip()
    if not text:
        return []

    chunks: List[str] = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = min(start + chunk_size_chars, text_len)
        chunk = text[start:end]
        chunks.append(chunk)
        if end == text_len:
            break
        # Move the window forward, keeping some overlap
        start = end - chunk_overlap_chars

    return chunks


def make_chunked_docs(
    page_docs: List[Dict[str, Any]],
    chunk_size_chars: int = 1200,
    chunk_overlap_chars: int = 200,
) -> List[Dict[str, Any]]:
    """Turn page-level records into smaller overlapping text chunks.

    Each chunk keeps a pointer back to its document and page metadata.
    """
    chunked: List[Dict[str, Any]] = []
    for page in page_docs:
        page_text = page["text"]
        chunks = split_text_into_chunks(
            page_text,
            chunk_size_chars=chunk_size_chars,
            chunk_overlap_chars=chunk_overlap_chars,
        )
        for idx, chunk_text in enumerate(chunks):
            chunked.append(
                {
                    "text": chunk_text,
                    "doc_id": page["doc_id"],
                    "title": page["title"],
                    "url": page["url"],
                    "page_num": page["page_num"],
                    "page_label": page["page_label"],
                    "total_pages": page["total_pages"],
                    "chunk_idx_in_page": idx,   # position within the page; the rest is copied from the page record
                }
            )
    return chunked
```

Chunking is cheap CPU work, but there is no reason to repeat it on every run. We save the chunks locally as `data/chunks.jsonl` (one JSON record per line) and reload the file if it already exists.

```python
chunks_jsonl_path = os.path.join(local_data_dir, "chunks.jsonl")

def save_chunked_docs_jsonl(path, chunks):
    with open(path, "w", encoding="utf-8") as f:
        for rec in chunks:
            json.dump(rec, f, ensure_ascii=False)
            f.write("\n")


def load_chunked_docs_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

# -------------------------------------------------------------------
# Cached chunking logic
# -------------------------------------------------------------------
if os.path.exists(chunks_jsonl_path):
    print(f"Found existing chunk file: {chunks_jsonl_path}")
    chunked_docs = load_chunked_docs_jsonl(chunks_jsonl_path)
    print("Loaded chunked docs:", len(chunked_docs))
else:
    print("No chunk file found. Running chunking step...")
    chunked_docs = make_chunked_docs(page_docs)
    save_chunked_docs_jsonl(chunks_jsonl_path, chunked_docs)
    print(f"Saved chunked docs to {chunks_jsonl_path}")

# Show first chunk
print("Raw pages:", len(page_docs))
print("Chunked docs:", len(chunked_docs))
chunked_docs[0] if chunked_docs else None
```

```python
# Upload to YOUR bucket so the Processing Jobs episode (and future runs) can reuse the
# chunks without repeating the unpacking and chunking steps.
print(f"Uploading chunked docs to s3://{bucket_name}/{chunks_key} ...")
s3.upload_file(chunks_jsonl_path, bucket_name, chunks_key)
print("Upload complete.")
```


## Step 7 – Cost tracking: Bedrock calls carry no tags unless you route them through a profile

Every SageMaker job in this lesson is tagged at launch, because tags never propagate from the notebook. Bedrock is stricter still: **an on-demand `invoke_model` or `converse` call has no tags parameter at all.** If you call `deepseek.v3-v1:0` or `amazon.titan-embed-text-v2:0` by its model ID, the usage lands in the account's Bedrock bill as anonymous per-token charges. The notebook's tags, the bucket's tags, and the tags you set on anything else are irrelevant to it. On a shared account that means nobody can tell which team spent what.

The fix is not a tag on the call; it is a different thing to call. An **application inference profile** is a small resource you create in your own account: an alias that points at one base model and carries your three tags. It has its own ARN, and that ARN goes wherever a model ID goes. Nothing else changes: same model, same price per token, same answers. The only difference is in the bill, where usage made through the alias shows up with the alias's tags.

Before and after, in the generation call:

| | `modelId` you pass | Who it is billed to |
|---|---|---|
| Raw model ID | `deepseek.v3-v1:0` | nobody: anonymous DeepSeek spend on the account |
| Your profile | `arn:aws:bedrock:us-east-2:<account-id>:application-inference-profile/adewnu8wqxpy` | `Name=DoeJohn, Project=SinkOrSwim, Purpose=Inference` |

So "cost tracking on Bedrock" is: create the alias once per model (one for DeepSeek, one for Titan, because an alias points at exactly one model), keep its ARN in a variable, and use that variable as `modelId` from then on. The profile is free; you pay only for tokens, as before.

One call per model does all of it. `tagged_profile_arn` in `wattbot_rag.py` looks for a profile named after your project and user name, creates it with the three tags if it is not there, and returns its ARN. Open the file if you want to see the forty lines; nothing in them is worth memorising. What is worth remembering is the sentence above: one tagged profile per model, its ARN in place of the model ID.

```python
from wattbot_rag import tagged_profile_arn

name = "DoeJohn"         # replace with your name
project = "SinkOrSwim"   # replace with your challenge name

# From here on, these ARNs are what we pass as modelId. The base IDs are never sent to the runtime again.
bedrock_model_id = tagged_profile_arn(bedrock, base_generation_model_id, "wattbot-gen", name, project)
embedding_model_id_bedrock = tagged_profile_arn(bedrock, base_embedding_model_id, "wattbot-embed", name, project)
```

::::::::::::::::::::::::::::::::::::: callout

### The same thing from a terminal

Outside a notebook it is one CLI command per model, which is all the helper wraps:

```bash
aws bedrock create-inference-profile --inference-profile-name SinkOrSwim-DoeJohn-wattbot-gen \
  --model-source copyFrom=arn:aws:bedrock:us-east-2::foundation-model/deepseek.v3-v1:0 \
  --tags key=Name,value=DoeJohn key=Project,value=SinkOrSwim key=Purpose,value=Inference
```

It prints the profile ARN; pass that as `modelId`. The helper adds only the reuse check, so rerunning does not create duplicates.

:::::::::::::::::::::::::::::::::::::::::::::::::


::::::::::::::::::::::::::::::::::::: callout

### Checking the profiles

- You can confirm the routing in the Bedrock console under **Inference and Assessment → Cross-region inference → Application inference profiles**, or with `bedrock.list_inference_profiles(typeEquals="APPLICATION")`.
- The keys must be the three approved ones, `Name`, `Project` and `Purpose`, spelled exactly; any other key is ignored by the bill (the S3 episode says why).
- The profile itself has no hourly or storage charge. You only pay per token, exactly as before.

:::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: challenge

### Who is paying for these tokens?

List the application inference profiles in the account and find your two. For each, print its name, the model it points at, and its tags (`bedrock.list_tags_for_resource` takes the profile ARN). Then answer: if a teammate skipped this step and called `deepseek.v3-v1:0` by its raw model ID for the rest of the episode, what would Cost Explorer show when the organizers filter the month's Bedrock spend by `Project`?

:::::::::::::::::::::::: solution

```python
profiles = bedrock.list_inference_profiles(typeEquals="APPLICATION")["inferenceProfileSummaries"]
print(f"{len(profiles)} application inference profiles in this account/region")
for p in profiles:
    if p["inferenceProfileArn"] in {bedrock_model_id, embedding_model_id_bedrock}:
        tags = bedrock.list_tags_for_resource(resourceARN=p["inferenceProfileArn"])["tags"]
        print(p["inferenceProfileName"], "->", [m["modelArn"].split("/")[-1] for m in p["models"]])
        print("   tags:", {t["key"]: t["value"] for t in tags})
```

Filtering by `Project` shows **only** the usage that went through a tagged profile. The teammate's calls are still on the bill, under the Bedrock service line for `deepseek.v3-v1:0`, but with no `Project`, `Name` or `Purpose` value: in the grouped view they fall into the "No tag key: Project" bucket, together with every other untagged call in the account. There is no way to attribute them after the fact; tags are applied when the usage is metered, not later. That is why the base model ID is never passed to the runtime in this episode.

:::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::::::::::::::

## Step 8 – Embed the chunks with Titan

Everything the RAG loop needs is now in memory: the chunks, the questions, and a way to resolve a document id to its URL for the `ref_url` column.

```python
print(f"Chunks: {len(chunked_docs)}")
print(f"Train QAs: {len(train_df)}")
print(f"Metadata rows: {len(metadata_df)}")

docid_to_url = build_docid_to_url(metadata_df)   # doc_id -> url, for the ref_url column of the submission
print(f"docid_to_url has {len(docid_to_url)} entries.")
```

Titan Text Embeddings V2 takes one string per `invoke_model` call and returns a 1,024-dimensional normalized vector. `embedding_model_id_bedrock` is the ARN of our tagged profile for it, so every call is attributed to our `Project`/`Name`/`Purpose` tags. Three cells: where the result is saved, the call that embeds one string, and the loop that embeds every chunk once and caches the matrix. First the paths.

```python
data_dir = Path(local_data_dir)
emb_save_path = data_dir / "embeddings_bedrock.npy"             # the finished matrix, one row per chunk
emb_partial_path = data_dir / "embeddings_bedrock.partial.npy"  # progress file, so an interrupted run resumes
```

The call itself. `invoke_model` takes a provider-specific JSON body; Titan's is `{"inputText": ...}`, and its response carries the vector under `embedding`. `embed_query` is the same call wrapped for retrieval time, when we embed one question at a time.

```python
def bedrock_embed_text(text: str, model_id: str = embedding_model_id_bedrock) -> List[float]:
    """Call the Titan embedding model (through our tagged profile) for one string."""
    response = bedrock_runtime.invoke_model(
        modelId=model_id,
        contentType="application/json",
        accept="application/json",
        body=json.dumps({"inputText": text}),   # Titan's own request shape; add "dimensions": 256 or 512 to shrink the 1,024 default
    )
    response_body = json.loads(response["body"].read())
    embedding = response_body.get("embedding")   # a list of floats, one per dimension
    if embedding is None:
        raise ValueError(f"No 'embedding' found in response: {response_body}")
    return embedding


def embed_query(question: str) -> np.ndarray:
    """The retrieval-time half of the pipeline: the same model, one question at a time."""
    return np.asarray(bedrock_embed_text(question), dtype="float32")
```

Embedding the corpus, about 9,000 chunks, is ten minutes and a few cents, so do not do it twice. The cell after next looks for the finished matrix locally, then in the shared bucket if `use_instructor_embeddings` is set, then in your bucket, and only then computes it, saving progress every 500 chunks so an interrupted run picks up where it stopped.

::::::::::::::::::::::::::::::::::::: callout

### In a class, use the instructor's copy

Bedrock allows this account 300,000 Titan embedding tokens per minute, and the quota is not adjustable. This corpus is about 3 million tokens, so one notebook alone needs ten minutes at the full quota; a room of them shares it, so everyone's loop slows to a crawl of retries. That is what `use_instructor_embeddings = True` in Step 1 is for: the cell below then downloads the matrix your instructor computed with the same chunks and the same model from the shared bucket and skips the calls. Working on your own, set it to `False` and let the loop run; that is the step to practise.

::::::::::::::::::::::::::::::::::::::::::::::::

One small helper asks S3 whether an object exists, and one checks that a cached matrix belongs to these chunks: a matrix from an earlier corpus has a different number of rows, and using it would pair every question with the wrong passages.

```python
def _s3_object_exists(bucket: str, key: str) -> bool:
    try:
        s3.head_object(Bucket=bucket, Key=key)
        return True
    except ClientError:
        return False


def load_embeddings_if_matching(path) -> Any:
    """The matrix at `path` if it has one row per chunk, else None (a matrix from a different corpus)."""
    matrix = np.load(path)
    if matrix.shape[0] == len(chunked_docs):
        return matrix
    print(f"{path} has {matrix.shape[0]} rows but there are {len(chunked_docs)} chunks: a matrix from a different corpus, ignoring it")
    return None


# Cache order: local file, the instructor's copy (in class), your bucket, then recompute (and upload for next time)
chunk_embeddings = None
if emb_save_path.exists():
    chunk_embeddings = load_embeddings_if_matching(emb_save_path)
if chunk_embeddings is None and use_instructor_embeddings and _s3_object_exists(shared_bucket, shared_embeddings_key):
    s3.download_file(shared_bucket, shared_embeddings_key, str(emb_save_path))
    print(f"Downloaded the instructor's embeddings from s3://{shared_bucket}/{shared_embeddings_key}")
    chunk_embeddings = load_embeddings_if_matching(emb_save_path)
if chunk_embeddings is None and _s3_object_exists(bucket_name, embeddings_key):
    s3.download_file(bucket_name, embeddings_key, str(emb_save_path))
    print(f"Downloaded your cached embeddings from s3://{bucket_name}/{embeddings_key}")
    chunk_embeddings = load_embeddings_if_matching(emb_save_path)
if chunk_embeddings is None:
    # Resume from a partial save if a previous run was interrupted
    all_embeddings = list(np.load(emb_partial_path)) if emb_partial_path.exists() else []
    start = len(all_embeddings)
    print(f"Computing embeddings via Bedrock from chunk {start} of {len(chunked_docs)}...")
    t0 = time.time()

    for idx in range(start, len(chunked_docs)):
        all_embeddings.append(bedrock_embed_text(chunked_docs[idx]["text"]))
        if (idx + 1) % 500 == 0 or idx + 1 == len(chunked_docs):
            np.save(emb_partial_path, np.array(all_embeddings, dtype="float32"))
            print(f"Embedded {idx + 1} / {len(chunked_docs)} chunks (progress saved)")

    print(f"Embedded {len(chunked_docs) - start} chunks in {time.time() - t0:.1f} seconds")
    chunk_embeddings = np.array(all_embeddings, dtype="float32")   # the embedding matrix: one row per chunk, one column per dimension
    np.save(emb_save_path, chunk_embeddings)
    emb_partial_path.unlink(missing_ok=True)
    s3.upload_file(str(emb_save_path), bucket_name, embeddings_key)
    print(f"Saved embeddings to {emb_save_path} and s3://{bucket_name}/{embeddings_key}")

assert chunk_embeddings.shape[0] == len(chunked_docs), "embeddings and chunks are out of step; delete the cached files and rerun"
print("Embeddings shape:", chunk_embeddings.shape)
```

## Step 9 – Retrieve: cosine similarity over the embedding matrix

Retrieval is pure NumPy: embed the question with the same model, take the cosine similarity against every chunk, and keep the top *k*. `retrieve_top_k` from `wattbot_rag.py` does that (open the file to see it: it is a normalize-and-matrix-multiply). No vector database is needed for a few thousand chunks; the optional GPU-notebook episode discusses when one becomes worth it.

```python
first_row = train_df.iloc[0]
print("Sample question:", first_row["question"])

# Cosine similarity of the question vector against every row of the matrix; keep the k best
test_retrieved = retrieve_top_k(embed_query(first_row["question"]), chunk_embeddings, chunked_docs, k=3)
for r in test_retrieved:   # score is the cosine similarity; 1.0 would be an identical vector
    snippet = r["text"].replace("\n", " ")[:160]
    print(f"- score={r['score']:.3f} | doc_id={r['doc_id']} | page={r['page_label']} | {snippet}...")
```

## Step 10 – Generate: DeepSeek V3 through the `converse` API

Bedrock has two runtime APIs. `invoke_model` takes a provider-specific request body (which is why the Titan call above builds a `{"inputText": ...}` body by hand). `converse` is provider-neutral: a system prompt, a list of messages, and an `inferenceConfig`, in the same shape for DeepSeek, Claude, Llama and the rest. We use `converse` for generation so the model is a one-line swap. The response carries the generated text and a `usage` block with the input and output token counts, which is what Bedrock bills on; we keep every `usage` block so we can price the run at the end.

```python
bedrock_usage = []   # one {"inputTokens": ..., "outputTokens": ...} per call

def call_bedrock_chat(
    system_prompt: str,
    user_prompt: str,
    max_new_tokens: int = 512,
    model_id: str = bedrock_model_id,   # the tagged profile ARN, never the raw model ID
) -> str:
    """Send one system + user turn to a Bedrock chat model and return the generated text."""
    try:
        response = bedrock_runtime.converse(
            modelId=model_id,
            # system: instructions that apply to the whole exchange
            system=[{"text": system_prompt}],
            # messages: the conversation so far, here a single user turn
            messages=[{"role": "user", "content": [{"text": user_prompt}]}],
            # inferenceConfig: generation settings; temperature 0 makes the output repeatable
            inferenceConfig={"maxTokens": max_new_tokens, "temperature": 0},
        )
    except ClientError as e:
        print(f"ERROR calling Bedrock model {model_id}: {e}")
        raise

    bedrock_usage.append(response["usage"])   # token counts for this call, kept so we can price the run later
    # Some models return extra content blocks (e.g. reasoning); keep only the text ones.
    text_blocks = [block["text"] for block in response["output"]["message"]["content"] if "text" in block]
    return "\n".join(text_blocks).strip()


print(call_bedrock_chat("You are a concise assistant.", "What is 2 + 2? Answer in one short sentence.", max_new_tokens=32))
print("usage of that call:", bedrock_usage[-1])
```

That function has the signature the shared pipeline expects, `generate(system_prompt, user_prompt, max_new_tokens) -> str`, and `embed_query` has the other one, `embed_query(question) -> vector`. Those two callables are the only difference between this episode and the self-hosted ones.

### What the pipeline asks the model

`run_single_qa` in `wattbot_rag.py` makes two calls per question. The first asks for a JSON object in the WattBot submission format, from the retrieved context only; the second, made only when the first produced an answer, asks for a one-to-three-sentence explanation of how the cited material supports it. The system prompt of the first call is the contract that makes the output machine-readable:

```python
print(ANSWER_SYSTEM_PROMPT)
```

A question is marked unanswerable (`is_blank` everywhere, as the competition requires) when the model says so **or** when the best retrieval score is below `retrieval_threshold`.

## Step 11 – Run the WattBot evaluation with Bedrock

Now we can loop over the questions, run retrieval + Bedrock generation, and write a `wattbot_solutions_bedrock.csv` file (locally and to your bucket).

`train_QA.csv` has 245 questions, and at two calls each that is about twenty minutes. For class we score a stratified sample instead: `sample_questions_by_type` draws ten questions from each evidence type (Quote, Table, Figure, Math, unanswerable, cross-paper, reconcile), 55 questions after overlaps, so every row of the breakdown in the next step has something in it. The first 55 rows of the file would be nearly all Quote questions. The seed fixes the draw, and the Processing Jobs episode uses the same one, so the two runs are compared on the same questions. Set `per_type=0` to run the whole file.

The Processing Jobs episode reuses this same loop with self-hosted models; the only difference there is that the embedding, answer
and explanation phases call local models instead of Bedrock-hosted ones.

```python
from wattbot_rag import sample_questions_by_type

eval_df = sample_questions_by_type(train_df, per_type=10, seed=0)   # per_type=0 for all 245
print(f"Scoring {len(eval_df)} of {len(train_df)} questions")

results = []
run_start = time.time()

for _, row in eval_df.iterrows():
    print("#" * 96)
    print(f"QUESTION: {row['question']}")

    out = run_single_qa(
        row=row,                             # one train_QA.csv row: id, question, expected answer_unit
        embed_query=embed_query,             # the two model callables: the only episode-specific inputs
        generate=call_bedrock_chat,
        chunk_embeddings=chunk_embeddings,   # the index: the matrix plus the chunks it was built from
        chunked_docs=chunked_docs,
        docid_to_url=docid_to_url,           # for the ref_url column
        top_k=8,                             # chunks handed to the model as context
        retrieval_threshold=0.25,            # best cosine score below this => unanswerable (is_blank)
        verbose=True,
    )

    print(f"ANSWER: {out['answer']}")
    print(f"ref_ids: {out['ref_id']}")
    print(f"EXPLANATION: {out['explanation']}")
    results.append(out)

results_df = pd.DataFrame(results)
print(f"Answered {len(results_df)} questions in {(time.time() - run_start) / 60:.1f} min")

output_dir = "outputs"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "wattbot_solutions_bedrock.csv")
results_df.to_csv(output_path, index=False)
s3.upload_file(output_path, bucket_name, solutions_key)
print(f"Wrote predictions to {output_path} and s3://{bucket_name}/{solutions_key}")

results_df.head()
```

## Step 12 – Score against the training answers

WattBot scores each prediction with three components: 0.75 for `answer_value` (numeric within ±0.1% relative tolerance, categorical match after normalisation, and zero if you answer a question the corpus cannot answer), 0.20 for the F1 between your `ref_id` set and the ground-truth citations, and 0.05 for refusing cleanly, with every evidence field blank, when the answer is `is_blank`. The number comes from `Score.py`, the competition's own metric file, which `compute_wattbot_score` in `wattbot_rag.py` calls for the headline score so that what you see here is what the leaderboard would report. It also prints the error rate by evidence type: whether each answer rests on a quoted sentence, a table, a figure, or a calculation. The competition's solution file carries those flags; the public `train_QA.csv` only notes them in its `explanation` column, so the wrapper derives them from there. A question can carry several. Read that table before changing anything: figure and table questions failing while quote questions pass points at text extraction, which loses figures and flattens tables, not at the model. The wrapper adds a per-row score, joins on `id`, and prints the worst-scoring rows so you can see *why* the pipeline lost points. `train_QA.csv` has no separate "unanswerable" column, so `gt_is_na_col="is_blank"` tells the scorer to derive that flag from `answer_value == "is_blank"` on both sides.

```python
scored_df = compute_wattbot_score(
    train_qa_path=train_qa_path,    # ground truth
    preds_path=output_path,         # our predictions, joined to it on id
    n_examples=20,
)
```

### Why were the wrong ones wrong?

The score tells you *which* questions failed, not why. Two different failures look the same in that table. Either retrieval never surfaced the right paper, so the model had nothing to work with, or the right passage was in front of it and it still declined or misread it. The cell below separates them for the lowest-scoring rows: it joins the model's own explanation and the passages it cited back onto the score, then reruns retrieval for the question (one embedding call) and reports whether the ground-truth paper was among the top-k chunks and what the best cosine score was.

```python
from wattbot_rag import retrieve_top_k, split_ref_ids

worst = scored_df.sort_values("wattbot_score").head(8)
by_id = results_df.set_index("id")   # the model's answer, explanation and cited passages, keyed by question id

for _, row in worst.iterrows():
    qid = row["id"]
    gt_refs = set(split_ref_ids(row["ref_id_gt"]))   # train_QA.csv stores the list as a string
    retrieved = retrieve_top_k(embed_query(row["question_gt"]), chunk_embeddings, chunked_docs, k=8)   # the merge suffixed both sides' columns
    retrieved_docs = [r["doc_id"] for r in retrieved]
    print("=" * 96)
    print(f"{qid}: {row['question_gt']}")
    print(f"  expected: {row['answer_value_gt']}  from {sorted(gt_refs)}")
    print(f"  predicted: {row['answer_value_pred']}  citing {row['ref_id_pred']}")
    print(f"  retrieval: top cosine {retrieved[0]['score']:.3f}; top-8 papers {retrieved_docs}")
    print(f"  ground-truth paper retrieved: {'yes' if gt_refs & set(retrieved_docs) else 'NO'}")
    if by_id.loc[qid, "explanation"] != "is_blank":
        print(f"  model's explanation: {by_id.loc[qid, 'explanation']}")
        print(f"  passages it cited: {str(by_id.loc[qid, 'supporting_materials'])[:400]}")
```

Read the two lines at the bottom of each block together. "Ground-truth paper retrieved: NO" is a retrieval problem: a different chunk size, a reranker, or a better embedding model is the fix, and no generator will help. "Yes" with `is_blank` is the model declining on evidence it had, which the system prompt and the retrieval threshold control. "Yes" with a wrong value is a reading error, and the cited passage usually shows what it latched onto: a nearby number, a different unit, a different model in the same table. And sometimes the explanation shows the model read correctly and the label is the problem: a question asking what inference "can account for" labelled 90 where three papers in the corpus give 60, 70 and 80 to 90 percent, so a range is arguably the better answer, or a fact that appears in several papers with the label naming only one of them. Labels are written by people. When the explanation convinces you, the fix is to the label, and the competition organisers want to hear about it. Each of those is a different lever, and the score alone would have you pulling the wrong one.

::::::::::::::::::::::::::::::::::::: challenge

### What did that run cost?

Every `converse` response carried a `usage` block, and `call_bedrock_chat` kept them all in `bedrock_usage`. Sum the input and output tokens over the evaluation run and price them at the current on-demand rates for DeepSeek V3 on the [Bedrock pricing page](https://aws.amazon.com/bedrock/pricing/) (fill the two prices in below). Add the embedding step: ~9,000 chunks of 1,200 characters at Titan Text Embeddings V2's `$0.02` per million input tokens. How does the total compare with the overview's estimate for "the whole WattBot pipeline on Bedrock", and with one hour of the `ml.g6e.xlarge` the Processing Jobs episode rents?

:::::::::::::::::::::::: solution

The two prices are from the [Bedrock pricing page](https://aws.amazon.com/bedrock/pricing/) for DeepSeek on-demand in October 2026; update them for the model and region you used. The rest sums the usage blocks and prints the cost.

```python
price_in_per_m = 0.62    # $ per million input tokens: DeepSeek v3.2 on-demand in us-east-2, October 2026; check the pricing page for your model and region
price_out_per_m = 1.85   # $ per million output tokens, same source

in_tokens = sum(u["inputTokens"] for u in bedrock_usage)
out_tokens = sum(u["outputTokens"] for u in bedrock_usage)
gen_cost = in_tokens / 1e6 * price_in_per_m + out_tokens / 1e6 * price_out_per_m

# Titan bills per input token; ~4 characters per token is a fair estimate for English prose
embed_tokens = sum(len(ch["text"]) for ch in chunked_docs) / 4
embed_cost = embed_tokens / 1e6 * 0.02

print(f"{len(bedrock_usage)} generation calls: {in_tokens:,} input + {out_tokens:,} output tokens -> ${gen_cost:.3f}")
print(f"~{embed_tokens:,.0f} embedding tokens -> ${embed_cost:.3f}")
print(f"Total ~${gen_cost + embed_cost:.2f}; one hour of ml.g6e.xlarge is $2.61")
```

With 55 questions, two calls each and ~3,000-token prompts, the run is on the order of 350K input and 30K output tokens: about a quarter of a dollar at DeepSeek V3's prices, plus about six cents of embeddings for the 3 million tokens of corpus. All 245 questions is about a dollar, which is the overview's estimate for the whole pipeline, and it is a small fraction of a single GPU-hour.

:::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::::::::::::::

## Step 13 – Multimodal embeddings for the table and figure questions

Look back at the evidence-type table in Step 12. The Table and Figure rows are usually the worst, and the reason is upstream of the model. `pypdf` gives us a page's *text*: a table comes out as a run of numbers with its column headers somewhere else on the page, and a figure comes out as its caption and nothing more. A question whose answer is a bar in a plot has nothing in the chunks to find, and a question whose answer is a cell in a table is being read from a flattened copy.

A **multimodal embedding model** is one model that maps text *and* images into the same vector space, so a text question can be compared with an image directly: the picture of a table lands near a sentence about its contents. Bedrock has two: [Amazon Nova Multimodal Embeddings](https://aws.amazon.com/blogs/aws/amazon-nova-multimodal-embeddings-now-available-in-amazon-bedrock/) (text, images, documents, video and audio, released 2025) and the older Titan Multimodal Embeddings G1 (text and images). Both are per-call, like Titan Text, and billed per image plus per text token; the [Bedrock pricing page](https://aws.amazon.com/bedrock/pricing/) has the rates, on the order of a hundredth of a cent per image. One catch: in October 2026 both are offered in `us-east-1` only (check the [model catalog](https://us-east-1.console.aws.amazon.com/bedrock/home?region=us-east-1#/model-catalog) for that region). A Bedrock call is an HTTPS request and the region is a parameter, so from Ohio that is a second client pointed at Virginia; it also means the page images leave the region, which is the thing to check against your institution's data policy before doing this with documents that are not public papers.

This step builds a second index next to the text one, measures whether it finds the right paper for table and figure questions more often, and then hands the retrieved page images to a model that can look at them. Each cell is short because the plumbing is in `wattbot_rag.py`.

### Which pages, and why whole pages

We do not try to cut the tables and figures out of the PDFs. Finding their boundaries needs a layout model, which is slow on a CPU notebook, and a page image already contains the table or figure *with* its caption and the sentences around it, which is what the embedding model needs to match a question. What we do instead is cheap: `pypdf` keeps caption lines even where it loses everything else, so a line starting with "Table 3" or "Figure 2" marks a page worth rendering. `find_caption_pages` scans the page records from Step 5 for them.

```python
from wattbot_rag import find_caption_pages, render_page_png, embed_multimodal, image_blocks, hit_at_k

caption_pages = find_caption_pages(page_docs)
n_tables = sum(any(c.startswith("Table") for c in p["captions"]) for p in caption_pages)
n_figures = sum(any(c.startswith("Figure") for c in p["captions"]) for p in caption_pages)
print(f"{len(caption_pages)} of {len(page_docs)} pages carry a caption: {n_tables} with tables, {n_figures} with figures")
caption_pages[:3]
```

Rendering is `pypdfium2`, the Python binding of the PDF engine Chrome uses; `render_page_png` renders one page at 100 dpi, enough to read axis labels, and returns the PNG bytes. Expect a few minutes for the corpus.

```python
%pip install -q pypdfium2
```

```python
page_image_dir = data_dir / "page_images"
page_image_dir.mkdir(exist_ok=True)

t0 = time.time()
for i, p in enumerate(caption_pages):
    png_path = page_image_dir / f"{p['doc_id']}_p{p['page_num']:03d}.png"
    p["png_path"] = str(png_path)
    if not png_path.exists():
        png_path.write_bytes(render_page_png(os.path.join(PDF_DIR, f"{p['doc_id']}.pdf"), p["page_num"]))
    if (i + 1) % 200 == 0:
        print(f"Rendered {i + 1} / {len(caption_pages)} pages")
print(f"{len(caption_pages)} page images in {page_image_dir} ({time.time() - t0:.0f} s)")
```

### Embed the page images

The embedding model lives in `us-east-1`, so this is a second runtime client and a second control-plane client there, and a tagged profile created through the latter so these calls are attributed like the rest. `embed_multimodal` builds the model's request body (Nova's says whether a vector is for the index or for a query, and that the image is a document page rather than a photo) and returns a unit vector. Set `multimodal_model_id` to `amazon.titan-embed-image-v1` to try the older model; the function handles both.

```python
multimodal_region = "us-east-1"
base_multimodal_model_id = "amazon.nova-2-multimodal-embeddings-v1:0"

bedrock_runtime_mm = boto3.client("bedrock-runtime", region_name=multimodal_region,
                                  config=Config(retries={"max_attempts": 10, "mode": "adaptive"}))
bedrock_mm = boto3.client("bedrock", region_name=multimodal_region)
multimodal_model_id = tagged_profile_arn(bedrock_mm, base_multimodal_model_id, "wattbot-embed-mm", name, project)

v = embed_multimodal(bedrock_runtime_mm, multimodal_model_id, image_png=Path(caption_pages[0]["png_path"]).read_bytes())
print(f"{caption_pages[0]['doc_id']} page {caption_pages[0]['page_label']} -> vector of {v.shape[0]} dimensions")
```

One call per page, cached like the text matrix: locally, then in your bucket, then computed. The question side uses the same model with `purpose="query"`.

```python
page_emb_path = data_dir / "embeddings_pages_multimodal.npy"
page_emb_key = "wattbot/embeddings/embeddings_pages_multimodal.npy"

if page_emb_path.exists():
    page_embeddings = np.load(page_emb_path)
elif _s3_object_exists(bucket_name, page_emb_key):
    s3.download_file(bucket_name, page_emb_key, str(page_emb_path))
    page_embeddings = np.load(page_emb_path)
else:
    t0 = time.time()
    rows = []
    for i, p in enumerate(caption_pages):
        rows.append(embed_multimodal(bedrock_runtime_mm, multimodal_model_id, image_png=Path(p["png_path"]).read_bytes()))
        if (i + 1) % 200 == 0:
            print(f"Embedded {i + 1} / {len(caption_pages)} page images ({time.time() - t0:.0f} s)")
    page_embeddings = np.vstack(rows)
    np.save(page_emb_path, page_embeddings)
    s3.upload_file(str(page_emb_path), bucket_name, page_emb_key)
    print(f"Embedded {len(caption_pages)} page images in {(time.time() - t0) / 60:.1f} min")

assert page_embeddings.shape[0] == len(caption_pages), "page embeddings and caption pages are out of step; delete the cached file and rerun"
print("Page embeddings shape:", page_embeddings.shape)

def embed_query_mm(question: str) -> np.ndarray:
    return embed_multimodal(bedrock_runtime_mm, multimodal_model_id, text=question, purpose="query")
```

### Does the image index find the right paper more often?

Retrieval is the same `retrieve_top_k` as Step 9, over the page matrix instead of the chunk matrix. `train_QA.csv` names the paper each answer comes from, so the cleanest measure is the **hit rate**: the fraction of questions for which a ground-truth paper is among the top-k results. We compute it for the text index and the image index on the Table questions, the Figure questions and, as a control, the Quote questions, where the image index should have no advantage.

```python
k = 8
groups = {"Table": train_df[train_df["Table"] == 1], "Figure": train_df[train_df["Figure"] == 1],
          "Quote only": train_df[(train_df["Quote"] == 1) & (train_df["Table"] == 0) & (train_df["Figure"] == 0)].head(40)}

print(f"{'questions':<12}{'n':>4}{'text index':>14}{'image index':>14}")
for label, qs in groups.items():
    gt = [split_ref_ids(r) for r in qs["ref_id"]]
    text_hits = [[c["doc_id"] for c in retrieve_top_k(embed_query(q), chunk_embeddings, chunked_docs, k=k)] for q in qs["question"]]
    image_hits = [[c["doc_id"] for c in retrieve_top_k(embed_query_mm(q), page_embeddings, caption_pages, k=k)] for q in qs["question"]]
    print(f"{label:<12}{len(qs):>4}{hit_at_k(text_hits, gt):>14.2f}{hit_at_k(image_hits, gt):>14.2f}")
```

Two numbers per row, and three readings. If the image index wins on Table and Figure and ties on Quote, the embedding model is doing what it was built for. If it loses everywhere, check what it was given: a page at 100 dpi with a dense table is readable, a page of nine small panels may not be. If both indexes already find the paper for nearly every question, retrieval was never the problem for this corpus and the loss is in the reading, which the next cell tests.

### Let the model see the page

Finding the page is half of it; DeepSeek V3 reads text only, so a figure retrieved as an image is no use to it. `converse` accepts image blocks in a user message for models that can see, and the swap is the model ID: Amazon Nova 2 Lite reads images and documents and is in the catalog for Ohio through the cross-region profile below (check the [model catalog](https://us-east-2.console.aws.amazon.com/bedrock/home?region=us-east-2#/model-catalog); `us.amazon.nova-lite-v1:0` is the cheaper, weaker alternative). `call_vision_chat` is `call_bedrock_chat` with an `images` argument: for each attached page it adds a text label naming the paper, then the image, so the model can cite what it reads.

```python
base_vision_model_id = "us.amazon.nova-2-lite-v1:0"
vision_model_id = tagged_profile_arn(bedrock, base_vision_model_id, "wattbot-vision", name, project)
vision_usage = []

def call_vision_chat(system_prompt: str, user_prompt: str, max_new_tokens: int = 512, images=()) -> str:
    """One system + user turn, with zero or more (label, png_bytes) page images attached to the user turn."""
    content = [{"text": user_prompt}]
    if images:
        content.append({"text": "The page images below are from the same papers; read tables and figures from them where the text above is incomplete."})
        content.extend(image_blocks(images))
    response = bedrock_runtime.converse(
        modelId=vision_model_id,
        system=[{"text": system_prompt}],
        messages=[{"role": "user", "content": content}],
        inferenceConfig={"maxTokens": max_new_tokens, "temperature": 0},
    )
    vision_usage.append(response["usage"])
    return "\n".join(b["text"] for b in response["output"]["message"]["content"] if "text" in b).strip()

print(call_vision_chat("You are a concise assistant.", "What is in the attached page? One sentence.", 64,
                       images=[(f"{caption_pages[0]['doc_id']}, page {caption_pages[0]['page_label']}", Path(caption_pages[0]["png_path"]).read_bytes())]))
```

Two runs over the Table and Figure questions, both through the vision model: one with the text chunks only, one with the top three retrieved page images attached as well. The first is the control. Without it, a better score with images could just be a different generation model, and the comparison would say nothing about the images.

```python
tf_df = train_df[(train_df["Table"] == 1) | (train_df["Figure"] == 1)]
print(f"{len(tf_df)} table and figure questions, two runs, two calls each")

def run_tf(with_images: bool) -> pd.DataFrame:
    rows = []
    for _, row in tf_df.iterrows():
        images = []
        if with_images:
            for pg in retrieve_top_k(embed_query_mm(row["question"]), page_embeddings, caption_pages, k=3):
                images.append((f"{pg['doc_id']}, page {pg['page_label']}", Path(pg["png_path"]).read_bytes()))
        generate = lambda s, u, max_new_tokens=512, images=images: call_vision_chat(s, u, max_new_tokens, images=images)
        rows.append(run_single_qa(row=row, embed_query=embed_query, generate=generate, chunk_embeddings=chunk_embeddings,
                                  chunked_docs=chunked_docs, docid_to_url=docid_to_url, top_k=8, retrieval_threshold=0.25))
    return pd.DataFrame(rows)

t0 = time.time()
tf_text_only = run_tf(with_images=False)
tf_text_only.to_csv(os.path.join(output_dir, "wattbot_solutions_vision_text_only.csv"), index=False)
tf_with_images = run_tf(with_images=True)
tf_with_images.to_csv(os.path.join(output_dir, "wattbot_solutions_vision_with_images.csv"), index=False)
print(f"Both runs in {(time.time() - t0) / 60:.1f} min")
```

Score both the same way as Step 12. The scorer compares the rows it finds, so the breakdown is over these questions only; the `Table` and `Figure` lines are the ones to read, against each other and against the DeepSeek run.

```python
for label, path in [("text chunks only", "wattbot_solutions_vision_text_only.csv"), ("text chunks + page images", "wattbot_solutions_vision_with_images.csv")]:
    print("=" * 96)
    print(f"Nova 2 Lite, {label}")
    compute_wattbot_score(train_qa_path=train_qa_path, preds_path=os.path.join(output_dir, path), n_examples=3)
```

::::::::::::::::::::::::::::::::::::: callout

### What to expect, and what it cost

Figure questions should move the most: the text index had only a caption to match and the text model had nothing to read, and the image gives it both. Table questions move less, because a flattened table is still mostly readable, and some will get worse: three page images is a lot of tokens for a model to search, and a wrong page retrieved with confidence is a new way to misread. The hit-rate table tells you which of those happened. For the bill, every page image is roughly a thousand input tokens at Nova 2 Lite's rate on the [pricing page](https://aws.amazon.com/bedrock/pricing/), so the run with images is about three times the control; `vision_usage` holds both runs' usage blocks, and the embedding of the page images is the per-image rate times `len(caption_pages)`.

::::::::::::::::::::::::::::::::::::::::::::::::

## Wrap‑up: Bedrock versus GPU‑based runs

You now have a complete WattBot evaluation running on Bedrock, with per‑token billing and nothing left running. The Processing Jobs episode reruns the same evaluation on a self‑hosted model, and the optional GPU-notebook episode does it once more interactively, so by the end you will have three versions to compare:

1. **Bedrock** (this episode) using hosted embedding and generation models with per‑token billing.
2. **SageMaker Processing job** running an open‑source model in batch with on-demand compute.
3. **Notebook GPU instance** (optional) using the same locally loaded open‑source model.

When deciding between these options in practice:

- Use **Bedrock or other hosted APIs** when:
  - You want to try the latest frontier models quickly.  
  - You only need to run a modest number of questions, or you are still prototyping.  
  - You prefer a simple, token‑based cost model and don’t want to manage GPU capacity.

- Use **self‑hosted models on GPU instances** when:
  - You need a specific open-weight model that the catalog does not carry.  
  - You expect to run large batches repeatedly (e.g., many thousands of questions).  
  - You want tight control over which architectures/checkpoints you run or fine‑tune.  
  - You already have institutional access to cost‑effective on‑prem or cloud GPUs.

The core **RAG evaluation logic stays identical** across all three episodes, which is the main takeaway:
once you have a clean retrieval + normalization pipeline (like WattBot’s, in `wattbot_rag.py`), swapping out the generator
is mostly a matter of re‑implementing `embed_query` and `generate`
for each compute option you care about.


## Concluding remarks: Bedrock models are one piece of the RAG puzzle

In this episode we used Bedrock-hosted models for **both** embedding and
generation. Larger, higher-quality models can definitely help a ton — especially on
messy real-world questions — but it's important to remember that they are still just
**one component** in your RAG system.

- **Bigger or newer models do not magically fix weak retrieval.** If your chunks
  are poorly aligned with the questions, a very strong LLM will still struggle.
- **Most of the long‑term accuracy gains in RAG systems come from the plumbing
  around the LLMs**, including:
  - smarter / semantic chunking strategies
  - good metadata and filtering
  - reranking or multi‑stage retrieval
  - domain‑specific heuristics and post‑processing
- **Cost and latency live in tension with quality.** Larger models (or higher
  token budgets) often improve answers, but at the cost of more inference time
  and higher per‑request spend. Bedrock makes it easier to experiment with that
  tradeoff by switching models without rewriting your pipeline.

As you adapt this notebook to your own projects, treat the LLM choice as **one
tunable component** in a larger system. Iterating on chunking, indexing, and
retrieval policies will almost always give you more headroom than swapping
between already-good models.


::::::::::::::::::::::::::::::::::::: keypoints

- A complete RAG pipeline on AWS needs no GPU: a CPU notebook plus Bedrock-hosted embedding and generation models is enough.
- Corpus preparation (unpacking the staged PDFs, page extraction, chunking) is cheap CPU work; cache `chunks.jsonl` on S3 so later episodes and runs can reuse it.
- Bedrock calls carry no tags; route them through a tagged application inference profile or the spend is anonymous on the bill.
- `converse` is the provider-neutral way to call a chat model on Bedrock; its `usage` block is what you are billed on. DeepSeek V3 is the generator here, and any catalog model is a one-line swap.
- Bedrock bills per token and leaves nothing running between calls, which is why it is the default route for RAG inference.
- The model is one tunable component; most accuracy gains come from chunking, retrieval, and post-processing.

::::::::::::::::::::::::::::::::::::::::::::::::

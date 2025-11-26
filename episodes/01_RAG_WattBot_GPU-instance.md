---
title: "RAG with a Notebook GPU"
teaching: 30
exercises: 15
---

:::::::::::::::::::::::::::::::::::::: questions

- How can we run a basic Retrieval-Augmented Generation (RAG) pipeline entirely from a single GPU-backed SageMaker notebook?
- How do we go from raw PDFs and CSV files to a searchable embedding space for WattBot documents?
- How can we generate WattBot-style answers (including citations and evidence) that follow the competition’s scoring conventions?

::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: objectives

- Verify that our SageMaker notebook instance has a working GPU and compatible Python environment.
- Load the WattBot metadata and question–answer files from local storage and inspect their structure.
- Download all referenced PDFs from `metadata.csv` and turn them into a collection of text pages with useful metadata attached.
- Implement a simple, explicit “from scratch” text-chunking and embedding pipeline without relying on FAISS or production vector DBs.
- Build a small retrieval helper that finds the most relevant chunks for a question using cosine similarity in embedding space.
- Wire the retriever to a local Qwen 7B-style generator to produce WattBot-format answers (including `answer`, `ref_id`, `ref_url`, and `supporting_materials`).
- Add a second LLM pass that generates short explanations and marks whether the evidence comes from text, figures, tables, or a combination.

::::::::::::::::::::::::::::::::::::::::::::::::



## Working with AWS for RAG Experiments 

In the previous episode, we briefly introduced several approaches for implementing RAG in AWS. Here, we are simply selecting a good GPU instance that can handle whatever RAG system we want to build. This approach is:

- Very easy to understand core on the AWS side of things (just select GPU instance and you're good to move on)
- Ideal for learning retrieval and generation steps  
- Great for experimentation and debugging  

However, it is **not the most cost‑efficient method**. In upcoming episodes we will introduce more efficient and production‑aligned GPU strategies, including:

- On-demand GPU tasks  
- Fully managed asynchronous jobs  
- Serverless or streaming LLM inference  
- SageMaker batch transform & RAG pipelines  
- Embedding jobs that run only when needed  

Those techniques bring you closer to best practice for scalable and budget‑friendly research computing.

**Remember to Shut Down Your AWS Instance**: GPU notebook instances continue billing **even when idle**.  Always:

- Save your work  
- Shut down or stop the instance when not in use
- Verify the status in the AWS console  

This habit prevents accidental ongoing GPU charges.


## Overview: WattBot RAG on a single notebook GPU

In this episode we build a **minimal but realistic RAG pipeline** from the [WattBot 2025](https://www.kaggle.com/competitions/WattBot2025/overview) challenge that runs entirely from a single GPU-backed SageMaker notebook.

In this episode we will:

1. **Work directly with the WattBot data.**
   - Use `train_QA.csv` and `metadata.csv` from the competition dataset.
   - Download all referenced PDFs (our RAG corpus) using the URLs in `metadata.csv`.
2. **Implement the core RAG steps explicitly in code.**
   - Read PDFs, extract per-page text, and attach document metadata.
   - Chunk text into overlapping segments suitable for embedding.
   - Embed chunks with a sentence-transformer (`thenlper/gte-base`)
   - Implement cosine-similarity search over the embedding matrix.
3. **Connect to a local Qwen-style generator.**
   - Use a quantized 7B model on a GPU-backed instance (e.g., `ml.g5.xlarge`).
   - Construct WattBot-style answers that we can compare against `train_QA.csv`.
4. **Add an explanation pass.**
   - Use an LLM to look at the retrieved evidence, the answer, and citations.
   - Generate a short explanation and label the **evidence type**: `[Quote]`, `[Table]`, `[Figure]`, or `[Mixed]`.


## Notebook + dataset setup

For this episode, we assume you are running on an AWS SageMaker notebook instance with a GPU, such as:

- `ml.g5.xlarge` (recommended) or
- `ml.g4dn.xlarge` (may work with smaller models / more aggressive quantization).

See [Instances for ML](https://carpentries-incubator.github.io/ML_with_AWS_SageMaker/instances-for-ML.html) for further guidance.


### Step 1 – Download `data.zip` locally

We’ll use the **WattBot 2025** dataset. Download the workshop data archive to your laptop or desktop:

- Open this link in your browser: https://github.com/carpentries-incubator/ML_with_AWS_SageMaker/blob/main/data/data.zip
- Save `data.zip` somewhere you can find it easily and unzip the folder contents

This archive should include a `data/wattbot/` folder containing:

- `metadata.csv` – index of all WattBot papers.
- `train_QA.csv` – labeled questions + ground truth answers.

### Step 2 – Create a WattBot S3 bucket

In the AWS console:

1. Go to **S3**.
2. Create a new bucket named something like:  
   `teamname-yourname-wattbot`
3. Keep **Block all public access** enabled.
4. (Optional, but recommended) Add tags so we can track costs:  
   - `Project = your-team-name`  
   - `Name = your-name`  
   - `Purpose = RAG-demo`

### Step 3 – Upload the WattBot files to S3

1. In your new bucket, click **Upload**.
2. Drag the `data/wattbot/` folder contents from `data.zip` into the upload dialog.
3. Upload it so that your bucket contains paths like:

   - `metadata.csv`
   - `train_QA.csv`

We’ll pull these files from S3 into the notebook in the next steps.


###  Verify GPU and basic environment


```python
!nvidia-smi || echo "No GPU detected – please switch to a GPU-backed instance (e.g., ml.g5.xlarge) before running this notebook."
```


```python
# also verify you've selected teh conda_pytorch_p310 kernel
import torch
print("torch cuda available:", torch.cuda.is_available())
print("num gpus:", torch.cuda.device_count())

```

## Import data from bucket into notebook


```python
import os
import json
import time
import math
from typing import List, Dict, Any

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
# TODO: update this to your bucket name
bucket_name = "chris-rag"  # <-- EDIT ME

# Local working directory in the notebook instance
local_data_dir = "./data"

print("Local data dir:", local_data_dir)

```


```python
# Download metadata.csv and train_QA.csv
metadata_key = "metadata.csv"
train_qa_key = "train_QA.csv"

metadata_path = os.path.join(local_data_dir, metadata_key)
train_qa_path = os.path.join(local_data_dir, train_qa_key)

download_s3_object(bucket_name, metadata_key, metadata_path)
download_s3_object(bucket_name, train_qa_key, train_qa_path)



```

## Step 1 – Imports, paths, and safe CSV loading


```python
import os
import time
import json
import math
import zipfile
from typing import List, Dict, Any, Tuple

import requests
import numpy as np
import pandas as pd

import torch
from torch import nn

from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer


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

## Step 2 – Download all PDFs from `metadata.csv`

Next we will...

1. Read the `url` column from `metadata.csv`.
2. Download each PDF via HTTP and save it locally as `<id>.pdf` under `pdfs/`.
3. Report any failures (e.g., missing or malformed URLs) at the end.
4. Upload zipped version of corpus to S3




```python
PDF_DIR = os.path.join(local_data_dir, "pdfs")
os.makedirs(PDF_DIR, exist_ok=True)

def download_all_pdfs_from_urls(
    metadata: pd.DataFrame,
    local_pdf_dir: str,
    url_col: str = "url",
    id_col: str = "id",
    timeout: int = 20,
) -> None:
    """Download all PDFs referenced in `metadata` using their URLs.

    - Saves each file as `<id>.pdf` in `local_pdf_dir`.
    - Strips whitespace from the URL (to avoid trailing spaces becoming `%20`).
    - Skips rows with missing or non-HTTP URLs.
    - Prints a short summary of any failures.
    """
    os.makedirs(local_pdf_dir, exist_ok=True)
    errors: List[Tuple[str, str]] = []

    print(f"Saving PDFs to: {local_pdf_dir}\n")

    for _, row in metadata.iterrows():
        doc_id = str(row[id_col]).strip()

        raw_url = row.get(url_col, None)
        if not isinstance(raw_url, str):
            errors.append((doc_id, "URL is not a string"))
            continue

        pdf_url = raw_url.strip()  # important: strip trailing whitespace
        if not pdf_url.startswith("http"):
            errors.append((doc_id, f"Invalid URL: {pdf_url!r}"))
            continue

        local_path = os.path.join(local_pdf_dir, f"{doc_id}.pdf")

        try:
            print(f"Downloading {doc_id} from {pdf_url} ...")
            resp = requests.get(pdf_url, timeout=timeout, allow_redirects=True)
            resp.raise_for_status()

            content_type = resp.headers.get("Content-Type", "")

            if "pdf" not in content_type.lower() and not pdf_url.lower().endswith(".pdf"):
                print(f"  Warning: Content-Type for {doc_id} does not look like PDF ({content_type})")

            with open(local_path, "wb") as f:
                f.write(resp.content)

        except Exception as e:
            print(f"  -> FAILED for {doc_id}: {e}")
            errors.append((doc_id, str(e)))

    if errors:
        print("\nSome PDFs could not be downloaded:")
        for doc_id, err in errors:
            print(f"  {doc_id}: {err}")
    else:
        print("\nAll PDFs downloaded successfully!")


download_all_pdfs_from_urls(
    metadata_df,
    PDF_DIR,
    url_col="url",
    id_col="id",
    timeout=20,
)

len(os.listdir(PDF_DIR))
```

### Zip all PDFs and upload to S3

Once we have all PDFs locally, it can be convenient and efficient to:

1. Zip them into a single file (e.g., `wattbot_pdfs.zip`).  
2. Upload that ZIP archive to an S3 bucket, such as `s3://<your-wattbot-bucket>/data/wattbot/wattbot_pdfs.zip`.

We’ll include a short code example here, but feel free to skip this during the workshop if time is tight.



```python
import os
import zipfile
import boto3

def zip_and_upload_pdfs(
    local_pdf_dir: str,
    bucket: str,
    zip_name: str = "corpus.zip"
) -> str:
    """
    Zips all PDFs in local_pdf_dir and uploads the ZIP file to:
        s3://<bucket>/<prefix>/<zip_name>

    Returns the full S3 URI of the uploaded zip file.
    """

    # Ensure directory exists
    if not os.path.exists(local_pdf_dir):
        raise ValueError(f"Directory not found: {local_pdf_dir}")

    # Path for the ZIP file
    zip_path = os.path.join(local_pdf_dir, zip_name)

    # Create ZIP archive
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for fname in os.listdir(local_pdf_dir):
            if fname.lower().endswith(".pdf"):
                fpath = os.path.join(local_pdf_dir, fname)
                zipf.write(fpath, arcname=fname)
                print(f"Added to ZIP: {fname}")

    print(f"\nZIP created: {zip_path}")

    # Upload to S3
    s3_client = boto3.client("s3")
    s3_key = f"{zip_name}"

    print(f"Uploading to s3://{bucket}/{s3_key} ...")
    s3_client.upload_file(zip_path, bucket, s3_key)
    print("Upload complete.")

    return f"s3://{bucket}/{s3_key}"


zip_s3_uri = zip_and_upload_pdfs(
    local_pdf_dir=PDF_DIR,
    bucket=bucket_name
)

```

## Step 3 – Turn PDFs into page-level “documents”

Next, we convert each PDF into a list of **page-level records**. Each record stores:

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
!pip install pypdf
```


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


page_docs = pdfs_to_page_docs(metadata_df, PDF_DIR)
print(f"Loaded {len(page_docs)} page-level records from {len(metadata_df)} PDFs.")
page_docs[0] if page_docs else None
```

## Step 4 – Simple, explicit text chunking

RAG systems typically break documents into **chunks** so that:

- Each chunk is long enough to carry meaningful context.
- No chunk is so long that it blows up the embedding/LLM context window.

For this workshop we will implement a **simple sliding-window chunker** that operates on characters:

- `chunk_size_chars`: maximum characters per chunk (e.g., 1,000–1,500).
- `chunk_overlap_chars`: overlap between consecutive chunks (e.g., 200).

In our own work, you may wish to plug in more sophisticated *semantic chunking*  methods(e.g., splitting on headings, section titles, or sentence boundaries). For now, we'll keep the implementation explicit and easy to debug.



```python
def split_text_into_chunks(
    text: str,
    chunk_size_chars: int = 1200,
    chunk_overlap_chars: int = 200,
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
                    "chunk_idx_in_page": idx,
                }
            )
    return chunked



```


```python
import os, json

chunks_s3_key = 'chunks.jsonl'
chunks_jsonl_path = os.path.join(local_data_dir, chunks_s3_key)

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
# Upload to S3 so future runs (or other instances) can reuse
print(f"Uploading chunked docs to s3 ...")
s3_client.upload_file(chunks_jsonl_path, bucket_name, chunks_s3_key)
print("Upload complete.")
```


## Step 5 – Build an embedding matrix

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

import numpy as np
import time
from sentence_transformers import SentenceTransformer

# We'll use a stronger embedding model now that we have a GPU.
# This model has ~335M parameters and benefits from GPU acceleration,
# but is still reasonable to run on a single 24 GB GPU.
embedding_model_id = "thenlper/gte-large"

use_cuda_for_embeddings = torch.cuda.is_available()
print("CUDA available for embeddings:", use_cuda_for_embeddings)

# Single shared embedder object that we can pass around.
embedder = SentenceTransformer(
    embedding_model_id,
    device="cuda" if use_cuda_for_embeddings else "cpu"
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
            normalize_embeddings=True,
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


### 6. Build a simple retrieval step (cosine similarity)

We are **not** using a heavy vector database in this first episode.

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


```python

from typing import List, Dict, Any

def cosine_similarity_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between rows of a and rows of b."""
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
    return np.dot(a_norm, b_norm.T)

def retrieve_top_k(
    query_embedding: np.ndarray,
    chunk_embeddings: np.ndarray,
    chunked_docs: List[Dict[str, Any]],
    k: int = 5,
) -> List[Dict[str, Any]]:
    """Return top-k most similar chunks for a query embedding."""
    if chunk_embeddings.shape[0] == 0:
        return []

    # query_embedding is 1D (D,)
    sims = cosine_similarity_matrix(query_embedding.reshape(1, -1), chunk_embeddings)[0]
    top_idx = np.argsort(-sims)[:k]

    results: List[Dict[str, Any]] = []
    for idx in top_idx:
        doc = chunked_docs[idx]
        results.append(
            {
                "score": float(sims[idx]),
                "text": doc["text"],
                "doc_id": doc["doc_id"],
                "page_num": doc["page_num"],
                "title": doc["title"],
                "url": doc["url"],
            }
        )
    return results

```


```python

# Quick sanity check for `retrieve_top_k` on the first training question
first_row = train_df.iloc[0]
test_question = first_row["question"]
print("Sample question:", test_question)

test_q_emb = embedder.encode(
    [test_question],
    convert_to_numpy=True,
    normalize_embeddings=True,
)[0]

test_retrieved = retrieve_top_k(
    query_embedding=test_q_emb,
    chunk_embeddings=chunk_embeddings,
    chunked_docs=chunked_docs,
    k=3,
)

print(f"Top {len(test_retrieved)} retrieved chunks:")
for r in test_retrieved:
    snippet = r["text"].replace("\n", " ")
    if len(snippet) > 160:
        snippet = snippet[:160] + "..."
    print(f"- score={r['score']:.3f} | doc_id={r['doc_id']} | page={r['page_num']} | snippet={snippet}")

```


### 7. Load the Qwen model for answer generation

For this episode we use **Qwen2.5-7B-Instruct** via the Hugging Face `transformers` library.

- Parameter count: ~7 billion.
- VRAM needs: ~14–16 GB in bfloat16 / 4-bit; fine for `ml.g5.xlarge` or a similar single-GPU instance.
- Intended use here: short, grounded answers plus a normalized `answer_value`.

We will:

1. Call Qwen once to propose an answer and supporting evidence.
2. Call Qwen a **second time** with a smaller prompt to generate a short explanation (<= 100 characters).



```python

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

qwen_model_id = "Qwen/Qwen2.5-7B-Instruct"

use_cuda_for_llm = torch.cuda.is_available()
print("CUDA available for LLM:", use_cuda_for_llm)

tokenizer_qwen = AutoTokenizer.from_pretrained(qwen_model_id)

if use_cuda_for_llm:
    llm_dtype = torch.bfloat16
    model_qwen = AutoModelForCausalLM.from_pretrained(
        qwen_model_id,
        dtype=llm_dtype,
        device_map=None,  # load on a single GPU
    ).to("cuda")
    generation_device = 0
else:
    llm_dtype = torch.float32
    model_qwen = AutoModelForCausalLM.from_pretrained(
        qwen_model_id,
        dtype=llm_dtype,
        device_map=None,
    )
    generation_device = -1  # CPU

qwen_pipe = pipeline(
    "text-generation",
    model=model_qwen,
    tokenizer=tokenizer_qwen,
    device=generation_device,
    max_new_tokens=384,
)

def call_qwen_chat(system_prompt: str, user_prompt: str, max_new_tokens: int = 384) -> str:
    """Use Qwen chat template and return only the newly generated text."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    prompt_text = tokenizer_qwen.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    outputs = qwen_pipe(
        prompt_text,
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )
    full = outputs[0]["generated_text"]
    generated = full[len(prompt_text):]
    return generated.strip()

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


### 8. Build prompts for answers and explanations

We keep the prompts **very explicit**:

- The first call asks Qwen to return JSON with:
  - `answer` (short text),
  - `answer_value` (normalized scalar or category),
  - `ref_id` (comma‑separated doc ids, e.g. `"jegham2025"`),
  - `supporting_material` (short quote or paraphrase).

- The second call asks Qwen to generate a **single sentence explanation** (<= 100 characters).
  We will prepend an evidence type tag (e.g. `[text]` or `[text+table]`) in code rather than
  asking the model to output it.



```python

def format_context_for_prompt(retrieved_chunks):
    """Format retrieved chunks so the LLM can see where text came from."""
    blocks = []
    for r in retrieved_chunks:
        header = f"[DOC {r['doc_id']} | page {r['page_num']} | score {r['score']:.3f}]"
        blocks.append(header + "\n" + r["text"])
    return "\n\n".join(blocks)

explanation_system_prompt = (
    "You are helping annotate how an answer is supported by a research paper. "
    "You will see a question, an answer, and the supporting text used. "
    "Your job is to (1) choose the MAIN type of evidence and "
    "(2) give a VERY short explanation (<= 100 characters). "
    "Valid evidence types are: text, figure, table, text+figure, table+figure, etc. "
    "Respond in the strict format: evidence_type: explanation"
)

def build_explanation_prompt(question, answer, supporting_materials, ref_id_list):
    ref_str = ", ".join(ref_id_list) if ref_id_list else "unknown"
    return f"""Question: {question}

Answer: {answer}

Supporting materials:
{supporting_materials}

Cited document ids: {ref_str}

Remember:
- evidence_type in [text, figure, table, text+figure, table+figure, etc.]
- explanation <= 100 characters
- Format: evidence_type: explanation
"""

```


### 9. Run over the full WattBot training set

Now we:

1. Iterate over **all** questions in `train_QA.csv`.
2. Retrieve the top-\(k\) chunks for each question.
3. Ask Qwen for an answer proposal (JSON).
4. Derive:
   - `answer` and `answer_value` from the JSON,
   - `answer_unit` **copied directly from the ground truth** (never guessed),
   - `ref_id` from the JSON,
   - `ref_url` by mapping `ref_id` to `metadata.csv`,
   - `supporting_material` from the JSON,
   - `evidence_type` from the supporting text,
   - `explanation` via a second Qwen call, prefixed with `[evidence_type]`.
5. Save `wattbot_solutions.csv` in the project folder.



```python
import re
from decimal import Decimal

def normalize_answer_value(raw_answer_value, answer_text, answer_unit, is_blank):
    """
    Normalize answer_value into the conventions used by train_QA:
      - 'is_blank' for unanswerable questions
      - plain numeric strings without units, commas, or scientific notation
      - booleans as 1/0
      - categorical strings (e.g., 'ML.ENERGY Benchmark') unchanged
      - ranges like '[0.02,0.1]' preserved as-is
    """
    s = str(raw_answer_value).strip()
    if is_blank:
        return "is_blank"
    if not s or s.lower() == "is_blank":
        return "is_blank"

    # Preserve ranges like [0.02,0.1]
    if s.startswith("[") and s.endswith("]"):
        return s

    lower = s.lower()

    # Booleans -> 1/0
    if lower in {"true", "false"}:
        return "1" if lower == "true" else "0"

    # Pure categorical (no digits) -> leave as-is
    if not any(ch.isdigit() for ch in s):
        return s

    # Try to extract the first numeric token from either the raw string or the answer text
    txt_candidates = [s, str(answer_text)]
    match = None
    for txt in txt_candidates:
        if not txt:
            continue
        match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", str(txt).replace(",", ""))
        if match:
            break

    if not match:
        # Fallback: strip obvious formatting characters
        cleaned = s.replace(",", "").replace("%", "").strip()
        return cleaned or "is_blank"

    num_str = match.group(0)

    # Format without scientific notation, trim trailing zeros
    try:
        d = Decimal(num_str)
        normalized = format(d.normalize(), "f")
    except Exception:
        try:
            f = float(num_str)
            normalized = ("%.15f" % f).rstrip("0").rstrip(".")
        except Exception:
            normalized = num_str

    return normalized

```

### Running the full RAG pipeline for one question

At this point we have all the building blocks we need:

- an **embedder** to turn questions into vectors  
- a **retriever** (`retrieve_top_k`) to grab the most relevant text chunks  
- a **chat helper** (`call_qwen_chat`) to talk to Qwen and get JSON back  
- a small post-processing helper (`normalize_answer_value`) to clean numbers

In the next few cells we tie these pieces together. We keep the code split into
small helper functions so learners can follow each step:

1. Retrieve context for a question.  
2. Ask the LLM for an answer, references, and a quote.  
3. Clean and normalize the structured fields (answer_value, ref_id, is_blank).  
4. Ask a second LLM call for a short explanation and evidence type.


### 🔍 Retrieving Relevant Context
This function embeds the question and fetches the top‐K most relevant text chunks. It’s the first step of the RAG pipeline and determines what evidence the LLM can see.



```python
# Build a lookup from document id -> URL using metadata
docid_to_url = {
    str(row["id"]).strip(): row["url"]
    for _, row in metadata_df.iterrows()
    if isinstance(row.get("url", None), str)
}

def retrieve_context_for_question(question, embedder, chunk_embeddings, chunked_docs, top_k: int = 8):
    """Embed the question and retrieve the top-k most similar chunks."""
    q_emb = embedder.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )[0]
    retrieved = retrieve_top_k(q_emb, chunk_embeddings, chunked_docs, k=top_k)
    context = format_context_for_prompt(retrieved)
    return retrieved, context


```

### First LLM Step: Producing an Answer
Here we prompt the model to:
- Decide if the question is answerable
- Extract a numeric/categorical answer
- Identify supporting evidence
- Select relevant document IDs



```python
def answer_phase_for_question(qid, question, answer_unit, context):
    """
    First LLM call: ask Qwen for an answer, answer_value, is_blank, ref_ids,
    and a short supporting quote. Then normalize these fields.
    """
    # System prompt: what role Qwen should play
    system_prompt_answer = (
        "You answer questions about AI energy, carbon, and water from academic papers.\n"
        "You are given:\n"
        "- a question\n"
        "- retrieved text chunks from the relevant paper(s)\n\n"
        "You must:\n"
        "1. Decide if the question can be answered from the provided context.\n"
        "2. If answerable, extract a concise numeric or short-text answer_value.\n"
        "3. Use the provided answer_unit EXACTLY as given (do NOT invent units).\n"
        "4. Select one or more document ids as ref_id from the supplied chunks.\n"
        "5. Copy a short supporting quote (<= 300 chars) into supporting_materials.\n"
        "6. If the context is insufficient, mark is_blank = true and set all\n"
        "   other fields to 'is_blank' except answer_unit (keep it as provided).\n"
        "Return a JSON object with fields:\n"
        "  answer (string)\n"
        "  answer_value (string)\n"
        "  is_blank (true or false)\n"
        "  ref_id (list of doc_id strings)\n"
        "  supporting_materials (string)\n"
    )

    context_block = context if context.strip() else "[NO CONTEXT FOUND]"

    # User prompt: question + unit hint + retrieved context
    user_prompt_answer = f"""Question: {question}
Expected answer unit: {answer_unit}

Retrieved context:
{context_block}

Return JSON ONLY with keys:
  answer (string)
  answer_value (string)
  is_blank (true or false)
  ref_id (list of doc_id strings)
  supporting_materials (string)
"""

    raw_answer = call_qwen_chat(system_prompt_answer, user_prompt_answer, max_new_tokens=384)

    # Try to parse JSON from the model output
    parsed = {
        "answer": "",
        "answer_value": "is_blank",
        "is_blank": True,
        "ref_id": [],
        "supporting_materials": "is_blank",
    }
    try:
        first_brace = raw_answer.find("{")
        last_brace = raw_answer.rfind("}")
        if first_brace != -1 and last_brace != -1:
            json_str = raw_answer[first_brace : last_brace + 1]
        else:
            json_str = raw_answer
        candidate = json.loads(json_str)
        if isinstance(candidate, dict):
            parsed.update(candidate)
    except Exception as e:
        print(f"JSON parse error for question {qid}: {e}")
        # fall back to defaults in `parsed`

    # Normalize parsed fields
    is_blank = bool(parsed.get("is_blank", False))
    ref_ids = parsed.get("ref_id") or []
    if isinstance(ref_ids, str):
        ref_ids = [ref_ids]
    ref_ids = [str(r).strip() for r in ref_ids if str(r).strip()]

    answer = str(parsed.get("answer", "")).strip()
    answer_value = str(parsed.get("answer_value", "")).strip() or "is_blank"
    answer_value = normalize_answer_value(
        raw_answer_value=answer_value,
        answer_text=answer,
        answer_unit=answer_unit,
        is_blank=is_blank,
    )
    supporting_materials = str(parsed.get("supporting_materials", "")).strip()

    # If context is empty or model says blank, force is_blank behaviour
    if not context.strip() or is_blank:
        is_blank = True
        answer = ""
        answer_value = "is_blank"
        ref_ids = []
        supporting_materials = "is_blank"

    # String formatting for ref_id / ref_url to match training style
    if not ref_ids:
        ref_id_str = "is_blank"
        ref_url_str = "is_blank"
    else:
        ref_id_str = str(ref_ids)

        # Resolve ref_url via metadata
        ref_url = "is_blank"
        for rid in ref_ids:
            if rid in docid_to_url:
                ref_url = docid_to_url[rid]
                break
        if not ref_url:
            ref_url = "is_blank"
        ref_url_str = str([ref_url])

    return answer, answer_value, is_blank, ref_ids, supporting_materials, ref_id_str, ref_url_str



```

### Second LLM Step: Explanation and Evidence Type
Now that we have an answer, we produce a short explanation and classify the evidence type. This step matches WattBot’s expected metadata.



```python
def explanation_phase_for_question(question, answer, supporting_materials, ref_ids, is_blank):
    """
    Second LLM call: ask Qwen to label an evidence_type and provide a short
    explanation tying the answer back to the supporting materials.
    """
    if is_blank:
        # For unanswerable questions we just propagate a sentinel.
        evidence_type = "other"
        explanation = "is_blank"
        return evidence_type, explanation

    expl_user_prompt = build_explanation_prompt(
        question=question,
        answer=answer,
        supporting_materials=supporting_materials,
        ref_id_list=ref_ids,
    )
    raw_expl = call_qwen_chat(
        explanation_system_prompt,
        expl_user_prompt,
        max_new_tokens=64,
    )

    # Take the first non-empty line (we expect something like "text: short reason")
    first_line = ""
    for line in raw_expl.splitlines():
        if line.strip():
            first_line = line.strip()
            break

    if ":" in first_line:
        etype, expl = first_line.split(":", 1)
        evidence_type = etype.strip().lower() or "other"
        explanation = expl.strip()
    else:
        evidence_type = "other"
        explanation = first_line.strip() or "short justification"

    # Keep explanations short for the CSV
    if len(explanation) > 100:
        explanation = explanation[:100]

    return evidence_type, explanation

```

###  Orchestration: `run_single_qa`
This high‐level function ties together retrieval, answering, normalization, and explanation into one full pass over a single question.



### Handling unanswerable questions

Some WattBot questions truly **cannot** be answered from the retrieved papers.  
We use a simple hybrid rule to detect these cases:

- We look at the **top retrieval score** (cosine similarity).  
- We also use the LLM's own `is_blank` flag from the first JSON response.  

If **either** of these says the evidence is too weak, we mark the question as unanswerable
and set all relevant fields (`answer_value`, `ref_id`, `supporting_materials`) to `is_blank`.

The `THRESHOLD` inside `run_single_qa` controls how strict this behaviour is:

- lower values → fewer questions marked unanswerable  
- higher values → more questions marked unanswerable  

You can change `THRESHOLD` and then re-run the notebook and `Score.py` to see
how this trade-off affects your final WattBot score.



```python

def run_single_qa(
    row,
    embedder,
    chunk_embeddings,
    chunked_docs,
    top_k: int = 8,
):
    """Run retrieval + Qwen for a single training QA row.

    This is the high-level orchestration function that calls three helpers:

    1. retrieve_context_for_question  -> get relevant text chunks
    2. answer_phase_for_question      -> generate answer from context, returning citations and supporting materials
    3. explanation_phase_for_question -> evidence type + short explanation
    """

    # Confidence threshold for retrieval.
    # If the top similarity score is below this value, we treat the question
    # as unanswerable, even if the LLM tried to produce an answer.
    THRESHOLD = 0.25

    qid = row["id"]
    question = row["question"]
    answer_unit = row.get("answer_unit", "")

    # 1. Retrieval step
    retrieved, context = retrieve_context_for_question(
        question=question,
        embedder=embedder,
        chunk_embeddings=chunk_embeddings,
        chunked_docs=chunked_docs,
        top_k=top_k,
    )

    top_score = retrieved[0]["score"] if retrieved else 0.0

    # 2. Answer + refs + supporting materials (LLM's view)
    (
        answer,
        answer_value,
        is_blank_llm,
        ref_ids,
        supporting_materials,
        ref_id_str,
        ref_url_str,
    ) = answer_phase_for_question(
        qid=qid,
        question=question,
        answer_unit=answer_unit,
        context=context,
    )

    # Hybrid is_blank decision:
    # - if retrieval is weak (top_score < THRESHOLD)
    # - OR the LLM marks is_blank = true
    # then we treat the question as unanswerable.
    is_blank = bool(is_blank_llm) or (top_score < THRESHOLD)

    if is_blank:
        answer = ""
        answer_value = "is_blank"
        ref_ids = []
        ref_id_str = "is_blank"
        ref_url_str = "is_blank"
        supporting_materials = "is_blank"

    # Always copy answer_unit from train_QA.csv (do NOT let the LLM invent it)
    answer_unit = row.get("answer_unit", "")

    # 3. Explanation + evidence_type
    evidence_type, explanation = explanation_phase_for_question(
        question=question,
        answer=answer,
        supporting_materials=supporting_materials,
        ref_ids=ref_ids,
        is_blank=is_blank,
    )

    return {
        "id": qid,
        "question": question,
        "answer": answer,
        "answer_value": answer_value,
        "answer_unit": answer_unit,
        "is_blank": "true" if is_blank else "false",
        "ref_id": ref_id_str,
        "ref_url": ref_url_str,
        "supporting_materials": supporting_materials,
        "evidence_type": evidence_type,
        "explanation": explanation,
    }

```


```python
# -------------------------------------------------------------------
# Run over max_N training questions (this can take a while!)
# -------------------------------------------------------------------
all_results = []
error_log = []
max_N = np.inf

for idx, row in train_df.iterrows():
    if idx >= max_N:
        break
    question = row["question"]
    print(f"########################################################################################################\nQUESTION: {question}")

    res = run_single_qa(
        row=row,
        embedder=embedder,
        chunk_embeddings=chunk_embeddings,
        chunked_docs=chunked_docs,
        top_k=8,
    )
    answer = res["answer"]
    ref_ids = res["ref_id"]

    explanation = res["explanation"]
    print(f"ANSWER: {answer}")
    print(f"ref_ids: {ref_ids}")
    print(f"EXPLANATION: {explanation}")
    
    all_results.append(res)

solutions_df = pd.DataFrame(all_results)
solutions_path = os.path.join(local_data_dir, "train_solutions_qwen.csv")
solutions_df.to_csv(solutions_path, index=False)

print(f"Saved solutions for {len(solutions_df)} questions to: {solutions_path}")
print(f"Number of questions with errors (filled as blank): {len(error_log)}")

solutions_df.head(20)

```

### Compare answers to ground truth

WattBot evaluates each prediction using a weighted score that combines three components. Most of the credit (0.75) comes from the `answer_value`, which must match the ground truth after normalization (numeric answers must be within ±0.1% relative tolerance; categorical values must match exactly). An additional 0.15 comes from `ref_id`, where partial credit is given based on the Jaccard overlap between your cited document IDs and the ground-truth set. The final 0.10 comes from correctly marking unanswerable questions: if a question is truly unanswerable, you must set `answer_value`, `ref_id`, and `supporting_materials` to `is_blank`. Any other combination scores zero for this component.

| Component      | Weight | What counts as correct |
|----------------|--------|------------------------|
| answer_value   | 0.75   | Numeric within ±0.1% relative tolerance; categorical exact match; `is_blank` if unanswerable |
| ref_id         | 0.15   | Jaccard overlap with the ground-truth reference set (case-insensitive) |
| is_NA          | 0.10   | All required fields set to `is_blank` when the question is truly unanswerable |



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
    preds_path="./data/train_solutions_qwen.csv",
    gt_is_na_col="is_blank",   # or "is_blank" / None depending on how you mark NAs
    n_examples=10,
)
```

## Recap and next steps

In this episode, we:

- Loaded a small corpus of AI / ML energy papers into our notebook environment.
- Split long documents into manageable chunks and cached those chunks to disk so we don’t have to re-run the chunking step every time.
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

In the next episodes, we will repeat largely the same exact RAG pipeline using slightly different approaches on AWS (processing jobs and Bedrock).



::::::::::::::::::::::::::::::::::::: keypoints

- **Notebook setup**: Start by provisioning a GPU-backed notebook instance
  (e.g., `ml.g5.xlarge`) so that both the embedding model and Qwen2.5-7B
  can run comfortably.
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



```python

```


```python

```


```python

```

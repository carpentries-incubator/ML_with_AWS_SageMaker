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

- Set up an S3 bucket with the WattBot metadata, questions, and PDF corpus, and tag it for cost tracking.
- Convert PDFs into page-level records and overlapping text chunks, and cache them as `chunks.jsonl` on S3 for reuse in later episodes.
- Create tagged application inference profiles so that every Bedrock call is attributed to your project.
- Embed chunks with Amazon Titan Text Embeddings and retrieve relevant context with cosine similarity.
- Generate WattBot-format answers and explanations with a Bedrock-hosted model and score them against `train_QA.csv`.

::::::::::::::::::::::::::::::::::::::::::::::::

This is the first hands-on RAG episode, and it starts where the overview told you to start: **Amazon Bedrock**. We build a complete RAG pipeline for the [WattBot 2025](https://www.kaggle.com/competitions/WattBot2025/overview) challenge from a small, CPU-only SageMaker notebook. Nothing in this episode needs a GPU: the core model work — **both text embeddings and answer generation** — runs on Bedrock-hosted models that bill per token and leave nothing running between calls. We'll use:

- an **Amazon Titan Text Embeddings V2** model to turn WattBot chunks into vectors, and  
- an **Anthropic Claude** model hosted on Bedrock to generate answers and explanations.

We also do the corpus preparation here: download the WattBot PDFs, split them into pages and overlapping
chunks, and cache the chunks on S3. The two episodes that follow reuse those chunks and run the same
retrieval, evaluation, and scoring logic on self-hosted models, so you can compare cost, latency, and
complexity directly.

## Why Bedrock for WattBot?

When you self-host models (as the notebook GPU and Processing Job episodes do later), you are responsible for picking a model,
managing versions, and making sure your instance has enough VRAM. That’s fine for experiments,
but it can get painful once multiple teams or challenges want to reuse the same pipeline.

Running your **embedding + generation** steps on Amazon Bedrock gives you a few nice properties:

- **Managed, up‑to‑date models.** You can use high‑quality models from Anthropic, Amazon, and
  others without worrying about container images or CUDA versions.
- **Pay for what you use (in tokens).** Instead of paying for a GPU instance that might sit
  idle, you pay per token (input + output) when you call the model. For some workloads this
  is cheaper; for large offline batches with smaller models, a dedicated GPU can still win.
- **Easier sharing and governance.** It’s easier to standardize on a small set of Bedrock
  models across courses, hackathons, or labs than to manage many separate GPU instances.

In this notebook, both the **embedding** and **answer/explanation** steps run on
Bedrock-hosted models. The embeddings are saved as `embeddings_bedrock.npy` so you can compare them
against the self-hosted runs in the next two episodes.


### Models used in this episode

We’ll work with **Amazon Bedrock–hosted foundation models** for both embedding and generation:

- **Amazon Titan Text Embeddings V2** (`amazon.titan-embed-text-v2:0`)

  - General‑purpose text‑embedding model for semantic search, retrieval, clustering, and classification.
  - Supports configurable embedding dimensions (for example 256–8,192) and has presets tuned for retrieval or binary indexing.
  - AWS does not publish the exact number of parameters for Titan models; you can treat it as a modern transformer specialized for embeddings rather than free‑form text generation. 

- **Anthropic Claude 3 Haiku** (`anthropic.claude-3-haiku-20240307-v1:0` via Bedrock)

  - A fast, mid‑sized Claude model that balances cost and quality for workloads like RAG, chat, and lightweight analysis.
  - Particularly useful when you want many calls (e.g., one per question) and care about low latency and lower per‑token pricing compared to flagship models such as Claude Opus or Claude 3.5 Sonnet. 
  - Anthropic does not publish exact parameter counts for Claude models; Haiku sits in the “smallest / fastest” tier within the Claude 3 family.

- **(Optional) Multimodal models for tables and figures**

  - Bedrock also exposes **multimodal models** that can reason over images, charts, and document layouts (for example, Claude 3.5 Sonnet with vision, or Amazon Titan Multimodal Embeddings). These are a good fit if much of your evidence lives in **figures, tables, or scanned PDFs**.
  - To use them from Bedrock you send **both text and image content** in a single request:
      - Pre‑process PDFs by rendering pages (or cropping individual tables/figures) to images using a tool like `pdf2image` or a headless browser.
      - Base64‑encode those images and include them as image parts alongside text in the model request.
      - For multimodal embeddings, you call a Titan multimodal embedding model with an `inputImage` (and optionally `inputText`) payload to obtain a single vector that mixes visual and textual information.
  - This notebook stays with **text‑only** embeddings + generation to keep the workflow simple, but the same RAG pattern extends naturally to multimodal models once you add an image‑extraction step to your preprocessing pipeline.

For a full catalog of available models (including other Claude variants, Amazon models, and partner models), open the **Model catalog** in the Amazon Bedrock console. Each entry provides a model card with capabilities, typical use cases, and pricing details so learners can explore alternatives for their own RAG systems. 


## Notebook + dataset setup

For this episode you need a SageMaker notebook instance, but **not a GPU one**. A small CPU instance such as `ml.t3.medium` is enough, because the models run on Bedrock rather than in the notebook. The notebook's execution role must allow Bedrock access; the exact permissions are listed in the "Permissions and verification" callout below.

See [Instances for ML](https://carpentries-incubator.github.io/ML_with_AWS_SageMaker/instances-for-ML.html) for further guidance on instance types.


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
4. Add tags so we can track costs:  
   - `Project = your-team-name`  
   - `Name = your-name`  
   - `Purpose = RAG-demo`

   These tags cover the bucket's storage costs only. Your notebook instance carries the tags you gave it when you created it, and in this episode those two resources are the only things being billed. Bedrock calls are tagged separately, through an inference profile (see "Cost tracking" below), and the SageMaker jobs in later episodes each need their own tags as well.
5. Once the bucket is created, you'll be brought to a page that shows all of your current buckets (and those on our shared account). We'll have to edit our bucket's policy to allow ourselves proper access to any files stored there (e.g., read from bucket, write to bucket). To set these permissions...

a. Click on the name of your bucket to bring up additional options and settings.
   b. Click the Permissions tab
   c. Scroll down to Bucket policy and click Edit. Paste the following policy, editing the bucket name "sinkorswim-doejohn-wattbot" to reflect your bucket's nameAs we did in the "setting up S3 episode, edit your bucket's policy to include the following:

```json
{
	"Version": "2012-10-17",
	"Statement": [
		{
			"Effect": "Allow",
			"Principal": {
			    "AWS": [
			        "arn:aws:iam::183295408236:role/ml-sagemaker-use",
			        "arn:aws:iam::183295408236:role/ml-sagemaker-bedrock-use"
		        ]
			},
			"Action": [
				"s3:GetObject",
				"s3:PutObject",
				"s3:DeleteObject",
				"s3:ListMultipartUploadParts"
			],
			"Resource": [
				"arn:aws:s3:::sinkorswim-chrisendemann-titanic",
				"arn:aws:s3:::sinkorswim-chrisendemann-titanic/*"
			]
		}
	]
}
```

### Step 3 – Upload the WattBot files to S3

1. In your new bucket, click **Upload**.
2. Drag the `data/wattbot/` folder contents from `data.zip` into the upload dialog.
3. Upload it so that your bucket contains paths like:

   - `metadata.csv`
   - `train_QA.csv`

We’ll pull these files from S3 into the notebook in the next steps.



```python
import os
import json
from typing import Dict, Any, List
from pathlib import Path

import boto3
import pandas as pd
import numpy as np

# from sentence_transformers import SentenceTransformer
from botocore.exceptions import ClientError

# ---- AWS configuration ----
import sagemaker

session = sagemaker.Session()
region = session.boto_region_name

# Base model IDs. These are the models we actually want to call, but we will NOT pass
# them to invoke_model directly -- see "Cost tracking" below for why.
# Swap for Sonnet/Opus if you have access and want higher quality.
base_generation_model_id = "deepseek.v3-v1:0"
base_embedding_model_id = "amazon.titan-embed-text-v2:0"

# S3 bucket you created above, and the keys we will read and write there.
bucket_name = "chris-rag"  # <-- change to your bucket
chunks_key = "chunks.jsonl"
train_key = "train_QA.csv"
metadata_key = "metadata.csv"

# Local working directory in the notebook instance
local_data_dir = "./data"
os.makedirs(local_data_dir, exist_ok=True)

# AWS clients
s3 = boto3.client("s3", region_name=region)
bedrock_runtime = boto3.client("bedrock-runtime", region_name=region)  # invoke models
bedrock = boto3.client("bedrock", region_name=region)                  # manage profiles, list models

```

## Step 4 – Load the metadata and training questions

Pull the two CSV files you uploaded to S3 into the notebook and inspect them.


```python
import requests
import zipfile
from typing import Tuple


def download_from_s3(key: str, local_name: str) -> str:
    """Download a file from S3 to local_data_dir and return the local path."""
    local_path = os.path.join(local_data_dir, local_name)
    print(f"Downloading s3://{bucket_name}/{key} -> {local_path}")
    s3.download_file(bucket_name, key, local_path)
    return local_path


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


train_qa_path = download_from_s3(train_key, "train_QA.csv")
metadata_path = download_from_s3(metadata_key, "metadata.csv")

train_df = smart_read_csv(train_qa_path)
metadata_df = smart_read_csv(metadata_path)

print("train_QA.csv columns:", train_df.columns.tolist())
print("metadata.csv columns:", metadata_df.columns.tolist())
print("\nNumber of training QAs:", len(train_df))
print("Number of metadata rows:", len(metadata_df))

train_df.head(15)
```

## Step 5 – Download all PDFs from `metadata.csv`

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

## Step 6 – Turn PDFs into page-level “documents”

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

## Step 7 – Simple, explicit text chunking

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

chunks_jsonl_path = os.path.join(local_data_dir, chunks_key)

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
# Upload to S3 so the next two episodes (and future runs) can reuse the chunks
# without repeating the PDF download and chunking steps.
print(f"Uploading chunked docs to s3 ...")
s3.upload_file(chunks_jsonl_path, bucket_name, chunks_key)
print("Upload complete.")
```


## Cost tracking: Bedrock calls carry no tags unless you route them through a profile

In the SageMaker episodes that follow, we tag every job at launch because tags never propagate from the notebook. Bedrock is stricter still: **an on-demand `invoke_model` or `converse` call has no tags parameter at all.** If you call `anthropic.claude-3-haiku-...` or `amazon.titan-embed-text-v2:0` by its model ID, the usage lands in the account's Bedrock bill as anonymous per-token charges. The notebook's tags, the bucket's tags, and the tags you set on anything else are irrelevant to it. On a shared account that means nobody can tell which team spent what.

The mechanism Bedrock provides for this is an **application inference profile**: a small, free resource that points at a base model and carries cost allocation tags. You create the profile once, then pass the **profile ARN** instead of the model ID as `modelId` in every call. Usage billed through the profile carries the profile's tags into Cost Explorer. Two details to get right:

- You need **one profile per base model** (the profile is model‑specific), so we create one for the generation model and one for the embedding model.
- Bedrock's tag format uses lowercase `key` / `value`, unlike SageMaker's `Key` / `Value`. The tag *names* (`Name`, `Project`, `Purpose`) stay the same so the Cost Explorer filters line up across services.

The helper below reuses a profile if one with the same name already exists, so re‑running the notebook does not create duplicates.

```python
import re

name = "John Doe"        # replace with your name
project = "sinkorswim"   # replace with your team name
purpose = "RAG-bedrock"

# Bedrock tags are lowercase key/value (SageMaker jobs use Key/Value)
profile_tags = [
    {"key": "Name", "value": name},
    {"key": "Project", "value": project},
    {"key": "Purpose", "value": purpose},
]

def _slug(text: str) -> str:
    # Profile names allow letters, digits, and single spaces/hyphens/underscores, max 64 chars
    return re.sub(r"[^0-9a-zA-Z]+", "-", text).strip("-")[:30]

def _source_arn(model_id: str) -> str:
    """ARN that the profile should point at.

    Plain model IDs (e.g. 'amazon.titan-embed-text-v2:0') map to a foundation-model ARN.
    Cross-region IDs (e.g. 'us.anthropic.claude-3-5-sonnet-...') are system-defined
    inference profiles, so we copy from that profile's ARN instead.
    """
    if model_id.split(".")[0] in {"us", "eu", "apac", "global", "jp", "au"}:
        return bedrock.get_inference_profile(inferenceProfileIdentifier=model_id)["inferenceProfileArn"]
    return bedrock.get_foundation_model(modelIdentifier=model_id)["modelDetails"]["modelArn"]

def get_or_create_inference_profile(base_model_id: str, label: str) -> str:
    """Return the ARN of a tagged application inference profile for base_model_id."""
    profile_name = f"{_slug(project)}-{_slug(name)}-{label}"

    # Reuse an existing profile with this name, if any
    paginator = bedrock.get_paginator("list_inference_profiles")
    for page in paginator.paginate(typeEquals="APPLICATION"):
        for summary in page["inferenceProfileSummaries"]:
            if summary["inferenceProfileName"] == profile_name:
                print(f"Reusing inference profile {profile_name}: {summary['inferenceProfileArn']}")
                return summary["inferenceProfileArn"]

    resp = bedrock.create_inference_profile(
        inferenceProfileName=profile_name,
        description=f"{project} / {name} WattBot RAG ({label}) -> {base_model_id}",
        modelSource={"copyFrom": _source_arn(base_model_id)},
        tags=profile_tags,
    )
    print(f"Created inference profile {profile_name}: {resp['inferenceProfileArn']}")
    return resp["inferenceProfileArn"]

# From here on, these ARNs are what we pass as modelId. The base IDs are only used
# inside request bodies that require a model name.
bedrock_model_id = get_or_create_inference_profile(base_generation_model_id, "wattbot-gen")
embedding_model_id_bedrock = get_or_create_inference_profile(base_embedding_model_id, "wattbot-embed")
```

::::::::::::::::::::::::::::::::::::: callout

### Permissions and verification

- The notebook's execution role needs `bedrock:CreateInferenceProfile`, `bedrock:GetInferenceProfile`, `bedrock:ListInferenceProfiles`, `bedrock:GetFoundationModel`, and `bedrock:TagResource` in addition to `bedrock:InvokeModel`. The `InvokeModel` permission must cover **both** the profile ARN and the underlying model ARN, or calls through the profile are denied. If `create_inference_profile` raises `AccessDeniedException`, ask the account administrator for these permissions rather than falling back to the raw model ID.
- You can confirm the routing in the Bedrock console under **Inference and Assessment → Cross-region inference → Application inference profiles**, or with `bedrock.list_inference_profiles(typeEquals="APPLICATION")`.
- Tags only become filters in Cost Explorer once the tag keys have been **activated as cost allocation tags** in the Billing console (an account‑admin, once‑per‑key action; on a shared workshop account the organizers do this). Until then the tags exist on the resource but do not show up on reports.
- The profile itself has no hourly or storage charge. You only pay per token, exactly as before.

:::::::::::::::::::::::::::::::::::::::::::::::::


```python
# Everything the RAG loop needs is now in memory.
print(f"Chunks: {len(chunked_docs)}")
print(f"Train QAs: {len(train_df)}")
print(f"Metadata rows: {len(metadata_df)}")
```


```python
def retrieve_context_for_question_bedrock(
    question: str,
    chunk_embeddings: np.ndarray,
    chunked_docs,
    top_k: int = 8,
):
    """
    Retrieve top-k chunks for a question using Bedrock embeddings.

    We call the Bedrock embedding model (via `bedrock_embed_text`) to
    embed the question, then compute cosine similarity against the
    pre-computed `chunk_embeddings` array.
    """
    # Embed the question with the same Bedrock model used for chunks
    q_emb = bedrock_embed_text(question)

    # Use the same cosine similarity + top-k helper as before
    retrieved = retrieve_top_k(q_emb, chunk_embeddings, chunked_docs, k=top_k)
    return retrieved, q_emb

```


```python
from sentence_transformers import SentenceTransformer

```


```python
# Build a mapping from doc_id -> URL so we can surface links in our outputs
docid_to_url = {}
for _, row in metadata_df.iterrows():
    doc_id = str(row.get("id", "")).strip()
    url = row.get("url", "")
    if doc_id and isinstance(url, str) and url.strip():
        docid_to_url[doc_id] = url.strip()

print(f"docid_to_url has {len(docid_to_url)} entries.")
```


```python
# ----------------------------------------------------------------------------------
# Bedrock embeddings for WattBot chunks
# ----------------------------------------------------------------------------------
# embedding_model_id_bedrock was set above to the ARN of our tagged application
# inference profile for amazon.titan-embed-text-v2:0, so every embedding call is
# attributed to our Project/Name/Purpose tags.

data_dir = Path("data")
data_dir.mkdir(exist_ok=True)
emb_save_path = data_dir / "embeddings_bedrock.npy"

def bedrock_embed_text(text: str, model_id: str = embedding_model_id_bedrock):
    """Call a Bedrock embedding model for a single input string."""
    body = json.dumps({"inputText": text})
    response = bedrock_runtime.invoke_model(
        modelId=model_id,
        body=body,
    )
    response_body = json.loads(response["body"].read())
    embedding = response_body.get("embedding")
    if embedding is None:
        raise ValueError(f"No 'embedding' found in response: {response_body}")
    return embedding


# -------------------------------------------------------------------------
# If an embedding file already exists, skip recomputing and load it instead
# -------------------------------------------------------------------------
if emb_save_path.exists():
    print(f"Found existing embeddings at {emb_save_path}. Skipping re-computation.")
    chunk_embeddings = np.load(emb_save_path)
else:
    print("No existing embeddings found. Computing via Bedrock...")

    all_embeddings = []
    for idx, ch in enumerate(chunked_docs):
        if (idx + 1) % 250 == 0:
            print(f"Embedding chunk {idx+1} / {len(chunked_docs)}")
        text = ch.get("text", "")
        emb = bedrock_embed_text(text)
        all_embeddings.append(emb)

    chunk_embeddings = np.array(all_embeddings, dtype="float32")

    # Save embeddings for reuse
    np.save(emb_save_path, chunk_embeddings)
    print(f"Saved embeddings to {emb_save_path}")

```


```python
# Save embeddings so we can reuse them later without re-calling Bedrock
np.save(emb_save_path, chunk_embeddings)
print("Saved Bedrock chunk embeddings to", emb_save_path)
print("Embeddings shape:", chunk_embeddings.shape)
```


```python
# ---------------------- similarity + retrieval ----------------------

# ---------------------- similarity + retrieval ----------------------

def cosine_similarity_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cosine similarity between two sets of vectors.

    This helper is intentionally defensive: it will accept Python lists,
    list-of-lists, or NumPy arrays and cast everything to float32 arrays
    before computing similarities.
    """
    a = np.asarray(a, dtype="float32")
    b = np.asarray(b, dtype="float32")

    # Ensure 2D
    if a.ndim == 1:
        a = a.reshape(1, -1)
    if b.ndim == 1:
        b = b.reshape(1, -1)

    a_norm = a / np.linalg.norm(a, axis=1, keepdims=True)
    b_norm = b / np.linalg.norm(b, axis=1, keepdims=True)
    return np.matmul(a_norm, b_norm.T)



def retrieve_top_k(
    query_embedding: np.ndarray,
    chunk_embeddings: np.ndarray,
    chunked_docs: List[Dict[str, Any]],
    k: int = 8,
) -> List[Dict[str, Any]]:
    """Return the top–k chunks for a single query embedding.

    Accepts query/collection embeddings as either NumPy arrays or lists.
    """
    # Defensive casting in case we accidentally pass in lists
    query = np.asarray(query_embedding, dtype="float32").reshape(1, -1)
    chunks = np.asarray(chunk_embeddings, dtype="float32")

    sims = cosine_similarity_matrix(query, chunks)[0]

    top_idx = np.argsort(-sims)[:k]

    results = []
    for idx in top_idx:
        ch = chunked_docs[idx]
        results.append(
            {
                "score": float(sims[idx]),
                "text": ch["text"],
                "doc_id": ch.get("doc_id", ""),
                "title": ch.get("title", ""),
                "url": ch.get("url", ""),
                "page_num": ch.get("page_num", None),
                "page_label": ch.get("page_label", None),
            }
        )
    return results



def format_context_for_prompt(retrieved_chunks: List[Dict[str, Any]]) -> str:
    """Turn retrieved chunk dicts into a compact context string for the LLM."""
    lines = []
    for i, ch in enumerate(retrieved_chunks, start=1):
        label = ch.get("doc_id", f"chunk_{i}")
        page = ch.get("page_label", ch.get("page_num", ""))
        header = f"[{label}, page {page}]".strip()
        txt = ch["text"].replace("\n", " ")
        lines.append(f"{header}: {txt}")
    return "\n".join(lines)


def retrieve_context_for_question(
    question: str,
    chunk_embeddings: np.ndarray,
    chunked_docs,
    top_k: int = 8,
):
    """Use Bedrock embeddings to retrieve the top-k chunks for a question."""
    # Embed question with Bedrock and make sure we end up with a 1D float32 vector
    q_vec = bedrock_embed_text(question)
    q_emb = np.asarray(q_vec, dtype="float32")

    retrieved = retrieve_top_k(q_emb, chunk_embeddings, chunked_docs, k=top_k)
    return retrieved, q_emb


```


```python
# ---------------------- answer normalization ----------------------

def normalize_answer_value(raw_value: str) -> str:
    """Normalize answer_value according to WattBot conventions."""
    if raw_value is None:
        return "is_blank"

    s = str(raw_value).strip()

    if not s or s.lower() == "none":
        return "is_blank"

    if s.startswith("[") and s.endswith("]"):
        return s

    if s.lower() == "is_blank":
        return "is_blank"

    # If there is whitespace, keep only the first token
    if " " in s:
        first, *_ = s.split()
        s = first

    # Remove commas
    s = s.replace(",", "")

    try:
        val = float(s)
        if val.is_integer():
            return str(int(val))
        return f"{val:.10g}"  # avoid scientific notation
    except ValueError:
        return s

```


```python
def call_bedrock_claude(
    system_prompt: str,
    user_prompt: str,
    model_id: str = bedrock_model_id,
    max_tokens: int = 512,
    temperature: float = 0.3,
) -> str:
    """
    Call a Bedrock chat model (Anthropic 4.x / Claude 3.5 / Llama 3.x, etc.)
    that uses the OpenAI-style chat completions schema.
    """
    # OpenAI-style chat body – this is what your error message is asking for
    body = {
        "model": base_generation_model_id,  # the body wants the model *name*; modelId below is our tagged profile ARN
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    request = json.dumps(body)
    try:
        response = bedrock_runtime.invoke_model(modelId=model_id, body=request)
    except ClientError as e:
        print(f"ERROR calling Bedrock model {model_id}: {e}")
        raise

    model_response = json.loads(response["body"].read())

    # OpenAI-style response: choices[0].message.content
    try:
        text = model_response["choices"][0]["message"]["content"]
    except Exception:
        # Fallback / debug
        print("Unexpected model response:", model_response)
        raise

    return text.strip()

```


```python
# ---------------------- explanation helpers ----------------------

def explanation_system_prompt() -> str:
    return (
        "You are an AI assistant that explains how evidence supports answers about "
        "energy, water, and carbon footprint of AI models.\n\n"
        "Instructions:\n"
        "- Write 1–3 sentences.\n"
        "- Directly explain how the cited supporting materials justify the answer.\n"
        "- Do NOT include any planning text, meta-reasoning, or tags like <reasoning>.\n"
        "- Do NOT start with phrases like 'We need to answer'—just give the explanation."
    )



def explanation_system_prompt() -> str:
    return (
        "You are an AI assistant that explains how evidence supports answers about "
        "energy, water, and carbon footprint. Focus on clear, factual reasoning, "
        "and refer directly to the cited documents when appropriate."
    )


def bedrock_explanation_phase_for_question(
    qid: str,
    question: str,
    answer: str,
    supporting_materials: str,
    model_id: str = bedrock_model_id,
) -> str:
    sys_prompt = explanation_system_prompt()
    prompt = build_explanation_prompt(question, answer, supporting_materials)
    raw_explanation = call_bedrock_claude(
        system_prompt=sys_prompt,
        user_prompt=prompt,
        model_id=model_id,
        max_tokens=256,
    )
    return raw_explanation.strip()


# ---------------------- answer phase (JSON contract) ----------------------

def bedrock_answer_phase_for_question(
    qid: str,
    question: str,
    retrieved_chunks: List[Dict[str, Any]],
    model_id: str = bedrock_model_id,
):
    """Use Bedrock to answer a single WattBot question given retrieved chunks."""
    context = format_context_for_prompt(retrieved_chunks)

    system_prompt = (
        "You are WattBot, a question-answering assistant for energy, water, and carbon footprint.\n"
        "You must answer questions using ONLY the provided context from scientific papers.\n"
        "If the context does not contain enough information to answer or infer,\n"
        "you must mark the question as unanswerable.\n\n"
        "You must respond with a single JSON object with the following keys:\n"
        "- answer: natural language answer, including numeric value and units if applicable.\n"
        "- answer_value: normalized numeric (0 for false, 1 for true), or categorical value with NO units or symbols;\n"
        "  use 'is_blank' if the question is unanswerable.\n"
        "- answer_unit: unit string (e.g., kWh, gCO2, %, is_blank).\n"
        "- ref_id: list of document IDs that support the answer, e.g., ['ID1', 'ID2'].\n"
        "- is_blank: true if unanswerable, false otherwise.\n"
        "- supporting_materials: short quote or table/figure pointer from the context.\n"
    )

    user_prompt = (
        "Use the context below to answer the question. "
        "Return ONLY a JSON object, no extra commentary.\n\n"
        f"Question: {question}\n\n"
        f"Context:\n{context}\n"
    )

    raw_answer = call_bedrock_claude(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        model_id=model_id,
        max_tokens=512,
    )

    parsed = {
        "answer": "",
        "answer_value": "is_blank",
        "answer_unit": "is_blank",
        "ref_id": [],
        "is_blank": True,
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

        parsed["answer"] = candidate.get("answer", "").strip()
        parsed["answer_value"] = normalize_answer_value(candidate.get("answer_value", "is_blank"))
        parsed["answer_unit"] = str(candidate.get("answer_unit", "is_blank")).strip() or "is_blank"

        ref_id = candidate.get("ref_id", [])
        if isinstance(ref_id, str):
            ref_ids = [ref_id]
        elif isinstance(ref_id, list):
            ref_ids = [str(x).strip() for x in ref_id if x]
        else:
            ref_ids = []
        parsed["ref_id"] = ref_ids

        is_blank_flag = candidate.get("is_blank", False)
        parsed["is_blank"] = bool(is_blank_flag)

        supp = candidate.get("supporting_materials", "is_blank")
        parsed["supporting_materials"] = str(supp).strip() or "is_blank"

    except Exception as e:
        print(f"JSON parse error for question {qid}; defaulting to is_blank. Error: {e}")

    return (
        parsed["answer"],
        parsed["answer_value"],
        parsed["is_blank"],
        parsed["ref_id"],
        parsed["supporting_materials"],
    )

```


```python
def run_single_qa_bedrock(
    row,
    chunk_embeddings: np.ndarray,
    chunked_docs,
    docid_to_url: dict,
    top_k: int = 8,
    retrieval_threshold: float = 0.25,
    model_id: str = "anthropic.claude-3-haiku-20240307-v1:0",
):
    """
    Full pipeline for a single question using Bedrock for both retrieval-time
    embeddings and generation.
    """
    qid = row["id"]
    question = row["question"]

    # 1. Retrieve supporting chunks using Bedrock embeddings for the query
    retrieved, q_emb = retrieve_context_for_question_bedrock(
        question=question,
        chunk_embeddings=chunk_embeddings,
        chunked_docs=chunked_docs,
        top_k=top_k,
    )

    top_score = retrieved[0]["score"] if retrieved else 0.0

    # 2. Call Bedrock Claude to produce answer JSON
    (
        answer,
        answer_value,
        is_blank_llm,
        ref_ids,
        supporting_materials,
    ) = bedrock_answer_phase_for_question(
        qid=qid,
        question=question,
        retrieved_chunks=retrieved,
        model_id=model_id,
    )

    # --------------------------------------------------------
    # 3. DECISION: retrieval_threshold OR model blank?
    # --------------------------------------------------------
    # NOTE: we only tell the user when it *actually* gets blanked.
    if is_blank_llm:
        print(f"[diag][{qid}] → Model returned is_blank (LLM could not answer).")
    elif top_score < retrieval_threshold:
        print(
            f"[diag][{qid}] → Retrieval blocked: top cosine={top_score:.3f} "
            f"< threshold={retrieval_threshold:.3f}"
        )
    is_blank = bool(is_blank_llm) or (top_score < retrieval_threshold)

    if is_blank:
        answer = "Unable to answer with confidence based on the provided documents."
        answer_value = "is_blank"
        answer_unit = "is_blank"
        ref_ids = []
        ref_id_str = "is_blank"
        ref_url_str = "is_blank"
        supporting_materials = "is_blank"
        explanation = ""
    else:
        answer_value = normalize_answer_value(answer_value)
        answer_unit = "is_blank"

        if isinstance(ref_ids, list) and ref_ids:
            ref_id_str = ";".join(ref_ids)
            urls = []
            for rid in ref_ids:
                url = docid_to_url.get(str(rid), "")
                if url:
                    urls.append(url)
            ref_url_str = ";".join(urls) if urls else "is_blank"
        else:
            ref_id_str = "is_blank"
            ref_url_str = "is_blank"

        explanation = bedrock_explanation_phase_for_question(
            qid=qid,
            question=question,
            answer=answer,
            supporting_materials=supporting_materials,
            model_id=model_id,
        )

    return {
        "id": qid,
        "question": question,
        "answer": answer,
        "answer_value": answer_value,
        "answer_unit": answer_unit,
        "ref_id": ref_id_str,
        "ref_url": ref_url_str,
        "supporting_materials": supporting_materials,
        "explanation": explanation,
    }

```

## Run the WattBot evaluation with Bedrock

Now we can loop over all questions in `train_QA.csv`, run retrieval + Bedrock
generation, and write a `wattbot_solutions_bedrock.csv` file.

The next two episodes reuse this same loop with self-hosted models; the only difference there is that the answer
and explanation phases call a local Qwen model instead of a Bedrock-hosted one.



```python
results = []

# For quick smoke tests, you can slice train_df (e.g., train_df.head(5))
for _, row in train_df.iterrows():
    question = row["question"]
    print("#" * 96)
    print(f"QUESTION: {question}")

    out = run_single_qa_bedrock(
        row=row,
        chunk_embeddings=chunk_embeddings,
        chunked_docs=chunked_docs,
        docid_to_url=docid_to_url,
        top_k=20,
        retrieval_threshold=0.1,
        model_id=bedrock_model_id,
    )

    answer = out["answer"]
    ref_ids = out["ref_id"]
    explanation = out["explanation"]

    print(f"ANSWER: {answer}")
    print(f"ref_ids: {ref_ids}")
    print(f"EXPLANATION: {explanation}")

    results.append(out)

results_df = pd.DataFrame(results)

output_dir = "outputs"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "wattbot_solutions_bedrock.csv")

results_df.to_csv(output_path, index=False)
print(f"Wrote predictions to {output_path}")

results_df.head()

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
# ------------------------------------------------------------------
# Normalize reference IDs + answer ranges after results are created
# ------------------------------------------------------------------

from typing import Any
import re
import numpy as np

def normalize_ref_ids(refs: Any) -> str:
    """
    Normalize reference IDs to a Python-list-style string.

    Output format examples:
      Input                       → Output
      ---------------------------------------------------------
      "chen2024"                 → "['chen2024']"
      ['chen2024']               → "['chen2024']"
      "[chen2024]"               → "['chen2024']"
      "['chen2024']"             → "['chen2024']"

      "chen2024;smith2023"       → "['chen2024', 'smith2023']"
      "chen2024, smith2023"      → "['chen2024', 'smith2023']"
      "[wu2021b;wu2021a]"        → "['wu2021b', 'wu2021a']"
      ['wu2021b','wu2021a']      → "['wu2021b', 'wu2021a']"

      None                       → "is_blank"
      "is_blank"                 → "is_blank"

    Rules:
      - "is_blank" stays exactly "is_blank".
      - Semicolons are treated as separators (→ commas).
      - Strips stray brackets, quotes, spaces.
      - Produces Python-list-style: ['id'] or ['id1', 'id2'].
    """
    import numpy as np

    # ----- 1. Handle blanks -----
    if refs is None or str(refs).strip() == "is_blank":
        return "is_blank"

    # ----- 2. True iterable input -----
    if isinstance(refs, (list, tuple, np.ndarray)):
        cleaned = [str(x).strip().strip("[]'\" ") for x in refs if str(x).strip()]
        return "[" + ", ".join(f"'{c}'" for c in cleaned) + "]"

    # ----- 3. Treat as string -----
    s = str(refs).strip()

    # Strip outer brackets if present (e.g., "[chen2024]" or "['chen2024']")
    if s.startswith("[") and s.endswith("]"):
        s = s[1:-1].strip()

    # Replace semicolons with commas
    s = s.replace(";", ",")

    # Split, strip quotes/spaces
    parts = [p.strip().strip("'\"") for p in s.split(",") if p.strip()]

    if len(parts) == 0:
        return "is_blank"

    if len(parts) == 1:
        return f"['{parts[0]}']"

    return "[" + ", ".join(f"'{p}'" for p in parts) + "]"



def normalize_answer_value(val: Any) -> str:
    """
    Normalize answer_value so that:
      - single numbers stay as-is (300 -> "300")
      - ranges get bracketed ("300-1000" -> "[300,1000]")
      - lists/tuples become bracketed ranges
    """
    import re
    import numpy as np

    # list / tuple / array → always a range
    if isinstance(val, (list, tuple, np.ndarray)):
        vals = []
        for v in val:
            # convert ints cleanly
            if isinstance(v, (int, float)) and float(v).is_integer():
                vals.append(str(int(v)))
            else:
                vals.append(str(v))
        return "[" + ",".join(vals) + "]"

    # numeric scalar → leave alone
    if isinstance(val, (int, float)):
        if float(val).is_integer():
            return str(int(val))
        return str(val)

    # string cases
    if isinstance(val, str):
        s = val.strip()

        # already bracketed
        if s.startswith("[") and s.endswith("]"):
            return s

        # detect range: 300-1000 or 300 – 1000
        m = re.match(r"^\s*([0-9]+(?:\.[0-9]+)?)\s*[-–—]\s*([0-9]+(?:\.[0-9]+)?)\s*$", s)
        if m:
            a, b = m.groups()
            # strip trailing .0
            a = a.rstrip(".0")
            b = b.rstrip(".0")
            return f"[{a},{b}]"

        # otherwise single value → leave alone
        return s

    # fallback: return string without brackets
    return str(val)


```


```python
import pandas as pd

solutions_df = pd.read_csv(output_dir + "/wattbot_solutions_bedrock.csv")
solutions_df.head()

```


```python
solutions_df["ref_id"] = solutions_df["ref_id"].apply(normalize_ref_ids)
solutions_df["answer_value"] = solutions_df["answer_value"].apply(normalize_answer_value)
solutions_df.head()
```


```python
solutions_df.to_csv(output_dir + "/solutions_normalized.csv", index=False)

```


```python
results_df = compute_wattbot_score(
    train_qa_path="./data/train_QA.csv",
    preds_path=output_dir + "/solutions_normalized.csv",
    gt_is_na_col="is_NA",   # or "is_blank" / None depending on how you mark NAs
    n_examples=20,
)
```

## Wrap‑up: Bedrock versus GPU‑based runs

You now have a complete WattBot evaluation running on Bedrock, with per‑token billing and nothing left running. The next two episodes rerun the same evaluation on self‑hosted models, so by the end you will have three versions to compare:

1. **Bedrock** (this episode) using hosted embedding and generation models with per‑token billing.
2. **Notebook GPU instance** using a locally loaded open‑source model.
3. **SageMaker Processing job** running the same model in batch with on-demand compute.

When deciding between these options in practice:

- Use **Bedrock or other hosted APIs** when:
  - You want to try the latest frontier models quickly.  
  - You only need to run a modest number of questions, or you are still prototyping.  
  - You prefer a simple, token‑based cost model and don’t want to manage GPU capacity.

- Use **self‑hosted models on GPU instances** when:
  - You expect to run large batches repeatedly (e.g., many thousands of questions).  
  - You want tight control over which architectures/checkpoints you run or fine‑tune.  
  - You already have institutional access to cost‑effective on‑prem or cloud GPUs.

The core **RAG evaluation logic stays identical** across all three episodes, which is the main takeaway:
once you have a clean retrieval + normalization pipeline (like WattBot’s), swapping out the generator
is mostly a matter of re‑implementing `answer_phase_for_question` and `explanation_phase_for_question`
for each compute option you care about.


## Concluding remarks: Bedrock models are one piece of the RAG puzzle

In this episode we swapped in Bedrock-hosted models for **both** embedding and
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
- Corpus preparation (PDF download, page extraction, chunking) is cheap CPU work; cache `chunks.jsonl` on S3 so later episodes and runs can reuse it.
- Bedrock calls carry no tags; route them through a tagged application inference profile or the spend is anonymous on the bill.
- Bedrock bills per token and leaves nothing running between calls, which is why it is the default route for RAG inference.
- The model is one tunable component; most accuracy gains come from chunking, retrieval, and post-processing.

::::::::::::::::::::::::::::::::::::::::::::::::
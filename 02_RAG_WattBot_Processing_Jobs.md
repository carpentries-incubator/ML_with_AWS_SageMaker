---
title: "RAG with Processing Jobs"
teaching: 30
exercises: 20
---

:::::::::::::::::::::::::::::::::::::: questions

- TODO

::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: objectives

- TODO

::::::::::::::::::::::::::::::::::::::::::::::::


# RAG with Processing Jobs

In the previous episode, we ran the entire WattBot RAG pipeline on a single GPU-backed SageMaker notebook. That was simple to teach, but the GPU sat idle while we downloaded PDFs, chunked text, and evaluated results.

In this Episode 2 notebook, we will keep the same WattBot corpus and RAG logic, but restructure how we use AWS:

- The notebook itself can run on a small CPU-only instance.
- We regenerate pages and chunks locally, as before.
- We save the chunks to S3.
- We run two short-lived SageMaker Processing jobs on a GPU:
  1. One job computes embeddings for all chunks.
  2. A second job runs the full RAG loop (retrieval + Qwen) over all training questions.

With this approach, we can more effectively use GPU resources only when needed, and we can scale out to larger corpora, models, and hardware more easily. The downside here is that you have to wait for processing jobs to spin up and run in batch mode on your queries. For many research applications of RAG, this is fine. However, if you want a near-real time chatbot you can have back and forth discussion with, this approach will not work. In the following episodes, we will discuss how we can use *Bedrock* or our own *model inference endpoints* to query models more rapidly.


## Setup



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

bucket_name = "chris-rag-2"          # reuse your bucket from Episode 1
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

We reuse the same `metadata.csv` and `train_QA.csv` files from Episode 1. If they are not already on the notebook file system, we download them from S3.



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
    s3_client.download_file(bucket_name, "metadata.csv", metadata_path)
if not os.path.exists(train_qa_path):
    s3_client.download_file(bucket_name, "train_QA.csv", train_qa_path)
if not os.path.exists(corpus_path):
    s3_client.download_file(bucket_name, "corpus.zip", corpus_path)

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


## Step 2 – Regenerate pages and chunks (local)

We reuse the same PDF → pages → overlapping chunks pipeline from Episode 1. For clarity, we keep this logic in the notebook so learners can see exactly how context is constructed.



```python
!pip install pypdf

```


```python
from pypdf import PdfReader
from typing import List, Dict, Any


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


page_docs = pdfs_to_page_docs(metadata_df, corpus_dir)
print(f"Loaded {len(page_docs)} page-level records from {len(metadata_df)} PDFs.")
page_docs[0] if page_docs else None


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


chunked_docs = make_chunked_docs(page_docs)
print("Raw pages:", len(page_docs))
print("Chunked docs:", len(chunked_docs))
chunked_docs[0] if chunked_docs else None
```


## Step 3 – Serialize chunks to JSONL and upload to S3

The Processing jobs will not have access to your Python variables. Instead, we serialize `chunked_docs` to `wattbot_chunks.jsonl` and upload it to S3 under this episode's prefix.

Each line is one JSON object representing a chunk, including its text and metadata.



```python

chunks_jsonl_path = os.path.join(local_data_dir, "wattbot_chunks.jsonl")

with open(chunks_jsonl_path, "w", encoding="utf-8") as f:
    for ch in chunked_docs:
        f.write(json.dumps(ch, ensure_ascii=False) + "\n")

print(f"Wrote {len(chunked_docs)} chunks to {chunks_jsonl_path}")

chunks_key = "wattbot_chunks.jsonl"
s3_client.upload_file(chunks_jsonl_path, bucket_name, chunks_key)

chunks_s3_uri = f"s3://{bucket_name}/{chunks_key}"
print("Chunks JSONL in S3:", chunks_s3_uri)

```


## Step 4 – Processing Job 1: embed all chunks on a GPU

Now we launch a short-lived Hugging Face **Processing job** that:

1. Downloads `wattbot_chunks.jsonl` from S3.
2. Loads `thenlper/gte-large` from Hugging Face.
3. Encodes each chunk into an embedding vector.
4. Saves the full matrix as `embeddings.npy` back to S3.

We use the same `embedding_inference.py` script across projects; here it expects a JSONL file with a `text` field.


### But first...
we have to create a requirements.txt file that will add additional libraries to the HuggingFaceProcessor we use below, which builds the environment we'll run our embedding_inference.py script in. For the processing job to recognize this dependence, we'll add it to the source_dir (code/) referenced when we call embedding_processor.run() below.


```python
requirements = [
    "sentence-transformers",
    # add more packages here if needed
]

req_path = "code/requirements.txt"
with open(req_path, "w") as f:
    f.write("\n".join(requirements))

print(f"Created requirements.txt at {req_path}")

```


```python

embedding_model_id = "thenlper/gte-large"
script_path = "embedding_inference.py"

emb_output_prefix = "embeddings"
emb_output_path = f"s3://{bucket_name}/{emb_output_prefix}/"

embedding_processor = HuggingFaceProcessor(
    base_job_name="WattBot-embed-gte-large",
    role=role,
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
        "--input_filename", "wattbot_chunks.jsonl",
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

- `wattbot_chunks.jsonl` – serialized chunks
- `embeddings.npy` – precomputed chunk embeddings
- `train_QA.csv` – training questions (to compare with ground truth)
- `metadata.csv` – to resolve `ref_id` → URL

The script `wattbot_rag_batch.py` reuses the RAG helpers from Episode 1:

- cosine similarity + `retrieve_top_k`
- `retrieve_context_for_question`
- `answer_phase_for_question` (Qwen answer, answer_value, ref_ids, is_blank)
- `explanation_phase_for_question`
- `run_single_qa` (hybrid unanswerable logic: retrieval threshold + LLM is_blank)

The job writes out `wattbot_solutions.csv` in the WattBot submission format.



```python

# Upload CSVs so the job can read them
train_qa_key = "train_QA.csv"
metadata_key = "metadata.csv"

train_qa_s3 = f"s3://{bucket_name}/{train_qa_key}"
metadata_s3 = f"s3://{bucket_name}/{metadata_key}"
emb_output_s3 = f"s3://{bucket_name}/{emb_output_prefix}/embeddings.npy"

print("train_QA:", train_qa_s3)
print("metadata:", metadata_s3)
print("embeddings:", emb_output_s3)

```


```python

rag_script = "wattbot_rag_batch.py"

rag_output_prefix = "solutions"
rag_output_path = f"s3://{bucket_name}/{rag_output_prefix}/"

rag_processor = HuggingFaceProcessor(
    base_job_name="WattBot-rag-batch",
    role=role,
    instance_type="ml.g5.xlarge",
    instance_count=1,
    transformers_version="4.56",
    pytorch_version="2.8",
    py_version="py312",
    sagemaker_session=session,
    max_runtime_in_seconds=4 * 60 * 60,
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

- TODO

::::::::::::::::::::::::::::::::::::::::::::::::
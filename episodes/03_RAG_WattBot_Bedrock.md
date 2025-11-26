# Episode 03 – Evaluating WattBot RAG with Amazon Bedrock

In the previous episodes you built a basic RAG pipeline for WattBot using a local GPU instance
and then an offline SageMaker Processing job. Both approaches gave you full control over the
models, but you were responsible for provisioning compute and keeping model versions up to date.

In this episode we move the core model work — **both text embeddings and answer generation** —
onto **Amazon Bedrock**. We'll use:

- an **Amazon Titan Text Embeddings V2** model to turn WattBot chunks into vectors, and  
- an **Anthropic Claude** model hosted on Bedrock to generate answers and explanations.

The retrieval, evaluation, and WattBot scoring logic are exactly the same as before; we're just
swapping out the underlying models and where they run. This lets you experiment with hosted,
state‑of‑the‑art models without having to manage GPUs or container images yourself.

## Why Bedrock for WattBot?

For the GPU instance and Processing Job episodes, you were responsible for picking a model,
managing versions, and making sure your instance had enough VRAM. That’s fine for experiments,
but it can get painful once multiple teams or challenges want to reuse the same pipeline.

Running your **embedding + generation** steps on Amazon Bedrock gives you a few nice properties:

- **Managed, up‑to‑date models.** You can use high‑quality models from Anthropic, Amazon, and
  others without worrying about container images or CUDA versions.
- **Pay for what you use (in tokens).** Instead of paying for a GPU instance that might sit
  idle, you pay per token (input + output) when you call the model. For some workloads this
  is cheaper; for large offline batches with smaller models, a dedicated GPU can still win.
- **Easier sharing and governance.** It’s easier to standardize on a small set of Bedrock
  models across courses, hackathons, or labs than to manage many separate GPU instances.

In this notebook, we’ll keep the same WattBot training questions and scoring helper you used
before, and we’ll simply move both the **embedding** and **answer/explanation** steps onto
Bedrock-hosted models.


## Setup: what you should already have

This notebook assumes you have already run the earlier WattBot episodes so that:

- the WattBot corpus has been chunked into `wattbot_chunks.jsonl`
- the WattBot training questions `train_QA.csv` and `metadata.csv` live under a `data/` folder
- (optionally) you have a local embedding file from earlier experiments, e.g. `embeddings.npy`

In this episode we’ll recompute embeddings **using an Amazon Titan Text Embeddings V2 model
on Bedrock**, and we’ll save those vectors out as `embeddings_bedrock.npy`. That keeps this
notebook self‑contained while still letting you compare against the earlier GPU / Processing
Job runs if you want.


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

# Claude 3 Haiku is a good starting point for batch evaluation.
# Swap for Sonnet/Opus if you have access and want higher quality.
bedrock_model_id = "deepseek.v3-v1:0"

# S3 bucket + keys where Episode 02 wrote the artifacts.
# TODO: Update these keys to match your pipeline.
bucket_name = "chris-rag"  # <-- change to your bucket
chunks_key = "wattbot_chunks.jsonl"
# embeddings_key = "embeddings/embeddings.npy"
train_key = "train_QA.csv"
metadata_key = "metadata.csv"

# Local working directory for downloaded artifacts
local_data_dir = "bedrock"
os.makedirs(local_data_dir, exist_ok=True)

# AWS clients
s3 = boto3.client("s3", region_name=region)
bedrock_runtime = boto3.client("bedrock-runtime", region_name=region)

```


```python
def download_from_s3(key: str, local_name: str) -> str:
    """Download a file from S3 to local_data_dir and return the local path."""
    local_path = os.path.join(local_data_dir, local_name)
    print(f"Downloading s3://{bucket_name}/{key} -> {local_path}")
    s3.download_file(bucket_name, key, local_path)
    return local_path


chunks_path = download_from_s3(chunks_key, "wattbot_chunks.jsonl")
# emb_path = download_from_s3(embeddings_key, "embeddings.npy")
train_qa_path = download_from_s3(train_key, "train_QA.csv")
metadata_path = download_from_s3(metadata_key, "metadata.csv")

# Load artifacts
with open(chunks_path, "r", encoding="utf-8") as f:
    chunked_docs = [json.loads(line) for line in f]

# chunk_embeddings = np.load(emb_path)
train_df = pd.read_csv(train_qa_path)

# Robust metadata load: handle possible non-UTF-8 characters
try:
    metadata_df = pd.read_csv(metadata_path)
except UnicodeDecodeError:
    metadata_df = pd.read_csv(metadata_path, encoding="latin1")

print(f"Chunks: {len(chunked_docs)}")
print(f"Train QAs: {len(train_df)}")
# print("Embeddings shape:", chunk_embeddings.shape)

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
embedding_model_id_bedrock = "amazon.titan-embed-text-v2:0"

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
        "model": model_id,  # some models allow omitting this, but it's safe to include
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

This mirrors the logic from Episode 02 – the only difference is that the answer
and explanation phases call a hosted Claude 3 model instead of a local Qwen model.



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

## Wrap‑up: comparing Bedrock to GPU‑based runs

At this point you should have three versions of the WattBot evaluation:

1. **Episode 01 – Notebook GPU instance** using a locally loaded open‑source model.  
2. **Episode 02 – SageMaker Processing job** running the same model in batch with on-demand compute. 
3. **Episode 03 – Bedrock** using a hosted Claude 3 model with per‑token billing.

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


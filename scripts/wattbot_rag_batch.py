import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


# ---------------------- similarity + retrieval ----------------------


def cosine_similarity_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cosine similarity between two sets of vectors."""
    a_norm = a / np.linalg.norm(a, axis=1, keepdims=True)
    b_norm = b / np.linalg.norm(b, axis=1, keepdims=True)
    return np.matmul(a_norm, b_norm.T)


def retrieve_top_k(
    query_embedding: np.ndarray,
    chunk_embeddings: np.ndarray,
    chunked_docs,
    k: int = 8,
):
    """
    Return the top–k chunks for a single query embedding.

    query_embedding: (d,)
    chunk_embeddings: (N, d)
    chunked_docs: list of chunk dicts
    """
    query = query_embedding.reshape(1, -1)
    sims = cosine_similarity_matrix(query, chunk_embeddings)[0]

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


def format_context_for_prompt(retrieved_chunks):
    """Turn retrieved chunk dicts into a compact context string for the LLM."""
    lines = []
    for i, ch in enumerate(retrieved_chunks, start=1):
        label = ch.get("doc_id", f"chunk_{i}")
        page = ch.get("page_label", ch.get("page_num", ""))
        header = f"[{label}, page {page}]".strip()
        txt = ch["text"].replace("\n", " ")
        lines.append(f"{header}: {txt}")
    return "\n".join(lines)


# ---------------------- answer normalization ----------------------


def normalize_answer_value(raw_value: str) -> str:
    """
    Normalize answer_value according to WattBot conventions.

    - No units or symbols in answer_value (e.g., no %, kWh, etc.).
    - Plain numbers (no commas, no scientific notation in the CSV).
    - Ranges should be in the form [low,high].
    - Use 'is_blank' for unanswerable.
    """
    if raw_value is None:
        return "is_blank"

    s = str(raw_value).strip()

    if not s or s.lower() == "none":
        return "is_blank"

    if s.startswith("[") and s.endswith("]"):
        return s

    if s.lower() == "is_blank":
        return "is_blank"

    # If we got something like "1438 kWh", keep just the first token
    if " " in s:
        first, *_ = s.split()
        s = first

    # Strip commas from numbers
    s = s.replace(",", "")

    # Try numeric normalization
    try:
        val = float(s)
        if val.is_integer():
            return str(int(val))
        return f"{val:.10g}"  # avoid scientific notation
    except ValueError:
        # Categorical value (e.g., TRUE, FALSE, Water consumption)
        return s


# ---------------------- explanation helpers ----------------------


def build_explanation_prompt(question: str, answer: str, supporting_materials: str) -> str:
    return (
        "You are explaining answers for an energy, water, and carbon footprint assistant.\n\n"
        f"Question: {question}\n\n"
        f"Answer: {answer}\n\n"
        f"Supporting materials:\n{supporting_materials}\n\n"
        "In 1–3 sentences, explain how the supporting materials justify the answer. "
        "Be precise but concise."
    )


def explanation_system_prompt() -> str:
    return (
        "You are an AI assistant that explains how evidence supports answers about "
        "energy, water, and carbon footprint. Focus on clear, factual reasoning, "
        "and refer directly to the cited documents when appropriate."
    )


# ---------------------- Qwen loading + wrapper ----------------------


def load_qwen_pipeline(model_id: str = "Qwen/Qwen2.5-7B-Instruct"):
    """
    Load Qwen once and return a text-generation pipeline.

    This should be called a single time per processing job and then reused
    for all calls to `call_qwen_chat`.
    """
    print(f"Loading model: {model_id}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )

    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
    )
    return generator


def call_qwen_chat(
    question: str,
    context: str,
    system_prompt: str,
    generator,
    max_new_tokens: int = 512,
):
    """
    Qwen chat wrapper using a pre-loaded `generator` pipeline.
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": (
                "Use the context to answer the question. If the context does not contain "
                "enough information to answer confidently, say explicitly that you are unable "
                "to answer with confidence.\n\n"
                f"Context:\n{context}\n\nQuestion:\n{question}"
            ),
        },
    ]

    prompt_text = ""
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        prompt_text += f"{role.upper()}: {content}\n\n"
    prompt_text += "ASSISTANT:"

    outputs = generator(
        prompt_text,
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )

    return outputs[0]["generated_text"][len(prompt_text) :].strip()


# ---------------------- RAG phases ----------------------


def retrieve_context_for_question(
    question: str,
    embedder,
    chunk_embeddings: np.ndarray,
    chunked_docs,
    top_k: int = 8,
):
    q_emb = embedder.encode([question], convert_to_numpy=True, normalize_embeddings=True)[0]
    retrieved = retrieve_top_k(q_emb, chunk_embeddings, chunked_docs, k=top_k)
    return retrieved, q_emb


def explanation_phase_for_question(
    qid: str,
    question: str,
    answer: str,
    supporting_materials: str,
    generator,
):
    sys_prompt = explanation_system_prompt()
    prompt = build_explanation_prompt(question, answer, supporting_materials)

    raw_explanation = call_qwen_chat(
        question=prompt,
        context="",
        system_prompt=sys_prompt,
        generator=generator,
        max_new_tokens=256,
    )
    return raw_explanation.strip()


def answer_phase_for_question(
    qid: str,
    question: str,
    retrieved_chunks,
    generator,
):
    """
    Use Qwen to answer a single WattBot question given retrieved chunks.
    """
    context = format_context_for_prompt(retrieved_chunks)

    system_prompt = (
        "You are WattBot, a question-answering assistant for energy, water, and carbon footprint.\n"
        "You must answer questions using ONLY the provided context from scientific papers.\n"
        "If the context does not contain enough information to answer with high confidence,\n"
        "you must mark the question as unanswerable.\n\n"
        "You must respond with a single JSON object with the following keys:\n"
        "- answer: natural language answer, including numeric value and units if applicable.\n"
        "- answer_value: normalized numeric or categorical value with NO units or symbols;\n"
        "  use 'is_blank' if the question is unanswerable.\n"
        "- answer_unit: unit string (e.g., kWh, gCO2, %, is_blank).\n"
        "- ref_id: list of document IDs that support the answer.\n"
        "- is_blank: true if unanswerable, false otherwise.\n"
        "- supporting_materials: short quote or table/figure pointer from the context.\n"
    )

    user_prompt = (
        "Use the context below to answer the question. "
        "Return ONLY a JSON object, no extra commentary.\n\n"
        f"Question: {question}\n\n"
        f"Context:\n{context}\n"
    )

    raw_answer = call_qwen_chat(
        question=user_prompt,
        context="",
        system_prompt=system_prompt,
        generator=generator,
        max_new_tokens=512,
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


def run_single_qa(
    row,
    embedder,
    chunk_embeddings: np.ndarray,
    chunked_docs,
    docid_to_url: dict,
    qwen_generator,
    top_k: int = 8,
    retrieval_threshold: float = 0.25,
):
    """
    Full pipeline for a single question.
    """
    qid = row["id"]
    question = row["question"]

    retrieved, q_emb = retrieve_context_for_question(
        question=question,
        embedder=embedder,
        chunk_embeddings=chunk_embeddings,
        chunked_docs=chunked_docs,
        top_k=top_k,
    )

    top_score = retrieved[0]["score"] if retrieved else 0.0

    (
        answer,
        answer_value,
        is_blank_llm,
        ref_ids,
        supporting_materials,
    ) = answer_phase_for_question(
        qid=qid,
        question=question,
        retrieved_chunks=retrieved,
        generator=qwen_generator,
    )

    # Combine retrieval confidence + LLM signal for unanswerable detection
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
        # Final normalization in case the model gave us something messy
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

        explanation = explanation_phase_for_question(
            qid=qid,
            question=question,
            answer=answer,
            supporting_materials=supporting_materials,
            generator=qwen_generator,
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


# ---------------------- main ----------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="/opt/ml/processing/input")
    parser.add_argument("--output_dir", type=str, default="/opt/ml/processing/output")
    parser.add_argument("--embedding_model_id", type=str, default="thenlper/gte-large")
    parser.add_argument("--top_k", type=int, default=8)
    parser.add_argument(
        "--qwen_model_id",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct",
        help="Which Qwen chat model to use for generation.",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Match the folder layout you used in ProcessingInput destinations
    chunks_path = os.path.join(args.input_dir, "chunks", "wattbot_chunks.jsonl")
    emb_path = os.path.join(args.input_dir, "embeddings", "embeddings.npy")
    train_qa_path = os.path.join(args.input_dir, "train", "train_QA.csv")
    metadata_path = os.path.join(args.input_dir, "metadata", "metadata.csv")

    print("Loading inputs...")
    with open(chunks_path, "r", encoding="utf-8") as f:
        chunked_docs = [json.loads(line) for line in f]

    chunk_embeddings = np.load(emb_path)
    train_df = pd.read_csv(train_qa_path)

    # Robust metadata load: handle non-UTF-8 characters
    try:
        metadata_df = pd.read_csv(metadata_path)
    except UnicodeDecodeError:
        metadata_df = pd.read_csv(metadata_path, encoding="latin1")

    print(f"Chunks: {len(chunked_docs)}")
    print(f"Train QAs: {len(train_df)}")
    print("Embeddings shape:", chunk_embeddings.shape)

    # Build docid -> url mapping
    docid_to_url = {}
    for _, row in metadata_df.iterrows():
        doc_id = str(row.get("id", "")).strip()
        url = row.get("url", "")
        if doc_id and isinstance(url, str) and url.strip():
            docid_to_url[doc_id] = url.strip()

    # Load embedding model
    embedder = SentenceTransformer(args.embedding_model_id)

    # Load Qwen ONCE and reuse
    qwen_generator = load_qwen_pipeline(args.qwen_model_id)

    results = []
    for _, row in train_df.iterrows():
        out = run_single_qa(
            row=row,
            embedder=embedder,
            chunk_embeddings=chunk_embeddings,
            chunked_docs=chunked_docs,
            docid_to_url=docid_to_url,
            qwen_generator=qwen_generator,
            top_k=args.top_k,
        )
        results.append(out)

    results_df = pd.DataFrame(results)
    out_path = os.path.join(args.output_dir, "wattbot_solutions.csv")
    results_df.to_csv(out_path, index=False)
    print("Wrote predictions to", out_path)


if __name__ == "__main__":
    main()

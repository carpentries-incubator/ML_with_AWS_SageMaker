"""WattBot RAG helpers shared by the RAG episodes and the batch job script.

The retrieval, prompting, answer parsing, orchestration and scoring logic of
the WattBot pipeline is the same whichever model serves it. This module holds
that logic once, so the episodes differ only in the two model calls they plug
in:

    embed_query(question) -> 1-D numpy vector
    generate(system_prompt, user_prompt, max_new_tokens) -> str

How each place uses it (copy `wattbot_rag.py` next to the notebook, or into
the `code_rag/` directory shipped with a Processing job):

- episodes/11_RAG-Bedrock.md: `embed_query` calls Titan Text Embeddings V2 and
  `generate` calls DeepSeek V3 through `bedrock_runtime.converse`, both via
  tagged application inference profiles. Uses `retrieve_top_k`, `run_single_qa`
  and `compute_wattbot_score`.
- episodes/12_RAG-Processing-jobs.md: the notebook only imports
  `compute_wattbot_score`; the job itself runs `scripts/wattbot_rag_batch.py`.
- scripts/wattbot_rag_batch.py: `embed_query` is a Hugging Face embedder
  (`embed_query_with` over `HFEmbedder`) and `generate` is Qwen3-8B through
  `transformers` (or vLLM with `--engine vllm`).
- episodes/13_RAG-GPU-notebook.md (optional): same as the batch script, but
  interactively, with `with_evidence_type=True` so each explanation is
  prefixed with the kind of evidence it rests on.

Output rows follow the WattBot submission format: id, question, answer,
answer_value, answer_unit, ref_id (";"-joined document ids or "is_blank"),
ref_url, supporting_materials, explanation. `compute_wattbot_score` accepts
ref_id either as ";"-joined ids or as the Python-list strings used in
train_QA.csv ("['wu2021b','patterson2021']").
"""

import json
import re
from decimal import Decimal, InvalidOperation
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

BLANK = "is_blank"
UNANSWERABLE_TEXT = "Unable to answer with confidence based on the provided documents."


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

ANSWER_SYSTEM_PROMPT = (
    "You are WattBot, a question-answering assistant for the energy, water, and carbon "
    "footprint of AI.\n"
    "You must answer questions using ONLY the provided context from scientific papers.\n"
    "If the context does not contain enough information to answer or infer the answer,\n"
    "you must mark the question as unanswerable.\n\n"
    "Respond with a single JSON object and nothing else, with these keys:\n"
    "- answer: natural-language answer, including the numeric value and units if applicable.\n"
    "- answer_value: the bare value with NO units or symbols: a number (1 for true, 0 for false),\n"
    "  a range as [low,high], or a short categorical string; 'is_blank' if unanswerable.\n"
    "- answer_unit: unit string (e.g. kWh, gCO2, %), or 'is_blank' if there is none.\n"
    "- ref_id: list of the document ids (the labels in square brackets in the context)\n"
    "  that support the answer, e.g. ['patterson2021'].\n"
    "- is_blank: true if unanswerable, false otherwise.\n"
    "- supporting_materials: a short quote, or a table/figure pointer, from the context.\n"
)


def build_answer_prompt(question: str, context: str, answer_unit: Optional[str] = None) -> str:
    """User turn for the answer call: the question, an optional unit hint, the context."""
    unit_hint = ""
    if answer_unit and str(answer_unit).strip().lower() not in {"", BLANK, "nan", "none"}:
        unit_hint = f"Express answer_value in this unit: {answer_unit}\n\n"
    return (
        "Use the context below to answer the question. "
        "Return ONLY a JSON object, no extra commentary.\n\n"
        f"Question: {question}\n\n"
        f"{unit_hint}"
        f"Context:\n{context if context.strip() else '[NO CONTEXT FOUND]'}\n"
    )


def explanation_system_prompt(with_evidence_type: bool = False) -> str:
    """System prompt for the second LLM call, which explains how the retrieved
    evidence supports the answer. The with_evidence_type variant also asks the
    model to label the evidence (text, table, figure, ...) in a fixed one-line
    format that the parser downstream relies on."""
    if with_evidence_type:
        return (
            "You are annotating how an answer is supported by a research paper. "
            "You will see a question, an answer, and the supporting material used. "
            "Choose the MAIN type of evidence (one of: text, table, figure, text+table, "
            "text+figure, table+figure) and give a short explanation of at most 100 characters. "
            "Respond on one line in the strict format: evidence_type: explanation"
        )
    return (
        "You are an AI assistant that explains how evidence supports answers about the "
        "energy, water, and carbon footprint of AI models.\n\n"
        "Instructions:\n"
        "- Write 1-3 sentences.\n"
        "- Explain directly how the cited supporting materials justify the answer.\n"
        "- Do NOT include planning text, meta-reasoning, or tags like <reasoning>.\n"
        "- Do NOT start with phrases like 'We need to answer'; just give the explanation."
    )


def build_explanation_prompt(
    question: str,
    answer: str,
    supporting_materials: str,
    ref_ids: Optional[Sequence[str]] = None,
    with_evidence_type: bool = False,
) -> str:
    """User turn for the explanation call."""
    ref_str = ", ".join(ref_ids) if ref_ids else "unknown"
    prompt = (
        f"Question: {question}\n\n"
        f"Answer: {answer}\n\n"
        f"Supporting materials:\n{supporting_materials}\n\n"
        f"Cited document ids: {ref_str}\n\n"
    )
    if with_evidence_type:
        prompt += (
            "Remember:\n"
            "- evidence_type is one of: text, table, figure, text+table, text+figure, table+figure\n"
            "- explanation is at most 100 characters\n"
            "- format: evidence_type: explanation\n"
        )
    else:
        prompt += (
            "In 1-3 sentences, explain how the supporting materials justify the answer. "
            "Be precise but concise."
        )
    return prompt


# ---------------------------------------------------------------------------
# Answer normalization and JSON parsing
# ---------------------------------------------------------------------------

_RANGE_RE = re.compile(r"^\s*([-+]?[0-9]+(?:\.[0-9]+)?)\s*(?:-|–|—|to)\s*([-+]?[0-9]+(?:\.[0-9]+)?)\s*$")


def _try_float(s: str) -> Optional[float]:
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def _format_number(val: float) -> str:
    """Plain decimal string: no commas, no scientific notation, no trailing zeros."""
    if float(val).is_integer():
        return str(int(val))
    try:
        return format(Decimal(repr(float(val))).normalize(), "f")
    except InvalidOperation:
        return f"{val:.10g}"


def normalize_answer_value(raw_value: Any) -> str:
    """Normalize answer_value to the WattBot conventions.

    - 'is_blank' for unanswerable / empty values
    - booleans as 1/0
    - plain numbers without units, commas or scientific notation ("1,438 kWh" -> "1438")
    - ranges as [low,high] ("300-1000" -> "[300,1000]"; "[0.02,0.1]" kept as is)
    - categorical strings (e.g. "ML.ENERGY Benchmark") unchanged
    """
    if raw_value is None:
        return BLANK
    if isinstance(raw_value, (bool, np.bool_)):
        return "1" if raw_value else "0"
    if isinstance(raw_value, (list, tuple, np.ndarray)):
        parts = [normalize_answer_value(v) for v in raw_value]
        return "[" + ",".join(parts) + "]"
    if isinstance(raw_value, (int, float, np.integer, np.floating)):
        if isinstance(raw_value, float) and np.isnan(raw_value):
            return BLANK
        return _format_number(float(raw_value))

    s = str(raw_value).strip()
    if not s or s.lower() in {"none", "null", "nan", BLANK}:
        return BLANK
    if s.startswith("[") and s.endswith("]"):
        return s
    if s.lower() in {"true", "yes"}:
        return "1"
    if s.lower() in {"false", "no"}:
        return "0"

    m = _RANGE_RE.match(s)
    if m:
        a, b = m.groups()
        a = re.sub(r"\.0+$", "", a)  # "300.0" -> "300" (not rstrip(".0"), which eats real zeros)
        b = re.sub(r"\.0+$", "", b)
        return f"[{a},{b}]"

    cleaned = re.sub(r"[,%$€£]", "", s)
    num = _try_float(cleaned)
    if num is None and " " in cleaned:  # "1438 kWh" -> 1438
        num = _try_float(cleaned.split()[0])
    if num is not None:
        return _format_number(num)
    return s


def split_ref_ids(value: Any) -> List[str]:
    """Turn any ref_id representation into a clean, ordered list of ids.

    Accepts a Python list, a ';'- or ','-separated string, or a Python-list
    string such as "['wu2021b','patterson2021']" (the train_QA.csv format).
    """
    if value is None:
        return []
    if isinstance(value, (list, tuple, np.ndarray)):
        items = [str(x) for x in value]
    else:
        s = str(value).strip()
        if not s or s.lower() in {BLANK, "nan", "none"}:
            return []
        items = re.split(r"[;,]", s)
    out: List[str] = []
    for item in items:
        cleaned = item.strip().strip("[]'\" ").strip()
        if cleaned and cleaned.lower() != BLANK and cleaned not in out:
            out.append(cleaned)
    return out


def parse_answer_json(raw_answer: str, qid: str = "") -> Dict[str, Any]:
    """Extract and normalize the JSON object the answer call was asked to return.

    Falls back to an unanswerable record when the model's output is not valid JSON.
    """
    parsed: Dict[str, Any] = {
        "answer": "",
        "answer_value": BLANK,
        "answer_unit": BLANK,
        "ref_id": [],
        "is_blank": True,
        "supporting_materials": BLANK,
    }
    try:
        first_brace = raw_answer.find("{")
        last_brace = raw_answer.rfind("}")
        json_str = raw_answer[first_brace : last_brace + 1] if first_brace != -1 and last_brace != -1 else raw_answer
        candidate = json.loads(json_str)
        if not isinstance(candidate, dict):
            raise ValueError("model output is not a JSON object")
    except Exception as e:  # noqa: BLE001 - any failure means "unparseable"
        print(f"JSON parse error for question {qid}; defaulting to is_blank. Error: {e}")
        return parsed

    parsed["answer"] = str(candidate.get("answer", "") or "").strip()
    parsed["answer_value"] = normalize_answer_value(candidate.get("answer_value", BLANK))
    parsed["answer_unit"] = str(candidate.get("answer_unit", BLANK) or "").strip() or BLANK
    parsed["ref_id"] = split_ref_ids(candidate.get("ref_id", []))
    is_blank = candidate.get("is_blank", False)
    if isinstance(is_blank, str):
        is_blank = is_blank.strip().lower() in {"true", "1", "yes"}
    parsed["is_blank"] = bool(is_blank)
    parsed["supporting_materials"] = str(candidate.get("supporting_materials", BLANK) or "").strip() or BLANK
    return parsed


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------


def cosine_similarity_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cosine similarity between the rows of a (m x d) and the rows of b (n x d)."""
    a = np.asarray(a, dtype="float32")
    b = np.asarray(b, dtype="float32")
    if a.ndim == 1:
        a = a.reshape(1, -1)
    if b.ndim == 1:
        b = b.reshape(1, -1)
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
    return a_norm @ b_norm.T


def retrieve_top_k(
    query_embedding: np.ndarray,
    chunk_embeddings: np.ndarray,
    chunked_docs: List[Dict[str, Any]],
    k: int = 8,
) -> List[Dict[str, Any]]:
    """Return the k chunks whose embeddings are most similar to the query embedding."""
    chunk_embeddings = np.asarray(chunk_embeddings, dtype="float32")
    if chunk_embeddings.shape[0] == 0:
        return []
    sims = cosine_similarity_matrix(np.asarray(query_embedding, dtype="float32").reshape(1, -1), chunk_embeddings)[0]
    top_idx = np.argsort(-sims)[:k]
    results = []
    for idx in top_idx:
        ch = chunked_docs[idx]
        rec = {"text": "", "doc_id": "", "title": "", "url": "", "page_num": None, "page_label": None}
        rec.update(ch)   # a text chunk, or a page-image record (doc_id, page_num, captions, png_path) in the multimodal step
        rec["score"] = float(sims[idx])
        results.append(rec)
    return results


def format_context_for_prompt(retrieved_chunks: List[Dict[str, Any]]) -> str:
    """Compact context string: one '[doc_id, page N]: text' line per chunk."""
    lines = []
    for i, ch in enumerate(retrieved_chunks, start=1):
        label = ch.get("doc_id") or f"chunk_{i}"
        page = ch.get("page_label", ch.get("page_num", ""))
        txt = ch["text"].replace("\n", " ")
        lines.append(f"[{label}, page {page}]: {txt}")
    return "\n".join(lines)


def build_docid_to_url(metadata_df: pd.DataFrame) -> Dict[str, str]:
    """metadata.csv -> {doc_id: url} for filling the ref_url column."""
    docid_to_url: Dict[str, str] = {}
    for _, row in metadata_df.iterrows():
        doc_id = str(row.get("id", "")).strip()
        url = row.get("url", "")
        if doc_id and isinstance(url, str) and url.strip():
            docid_to_url[doc_id] = url.strip()
    return docid_to_url


def embed_query_with(embedder) -> Callable[[str], np.ndarray]:
    """Wrap an embedder with a sentence-transformers-style `encode` as an `embed_query(question)` callable."""

    def embed_query(question: str) -> np.ndarray:
        """One question -> one unit-length vector, with the same model that embedded the chunks."""
        return embedder.encode([question], convert_to_numpy=True, normalize_embeddings=True)[0]

    return embed_query


# ---------------------------------------------------------------------------
# Bedrock cost tracking: a tagged application inference profile per base model
# ---------------------------------------------------------------------------


def tagged_profile_arn(bedrock, base_model_id: str, label: str, name: str, project: str, purpose: str = "Inference") -> str:
    """Return the ARN of a tagged Bedrock application inference profile for `base_model_id`, creating it if needed.

    Bedrock's on-demand calls (`invoke_model`, `converse`) take no tags, so a call made with a raw
    model ID lands on the bill as anonymous spend. An application inference profile is a free,
    model-specific alias that carries cost allocation tags; usage routed through its ARN is
    attributed to them. One profile per base model.

    The profile is named <project>-<name>-<label>, so rerunning finds and reuses it instead of
    creating a duplicate. `bedrock` is a boto3 client for the "bedrock" service (the control
    plane, not "bedrock-runtime"). Raises PermissionError with the role's ARN when the role has no
    Bedrock permissions.
    """
    import boto3
    from botocore.exceptions import ClientError

    def slug(text: str) -> str:   # profile names allow letters, digits and single hyphens, max 64 chars
        return re.sub(r"[^0-9a-zA-Z]+", "-", text).strip("-")[:30]

    def source_arn(model_id: str) -> str:   # what the profile points at
        if model_id.split(".")[0] in {"us", "eu", "apac", "global", "jp", "au"}:   # cross-region IDs are system-defined profiles
            return bedrock.get_inference_profile(inferenceProfileIdentifier=model_id)["inferenceProfileArn"]
        return bedrock.get_foundation_model(modelIdentifier=model_id)["modelDetails"]["modelArn"]

    profile_name = f"{slug(project)}-{slug(name)}-{label}"
    tags = [{"key": "Name", "value": name}, {"key": "Project", "value": project}, {"key": "Purpose", "value": purpose}]   # Bedrock wants lowercase key/value
    try:
        for page in bedrock.get_paginator("list_inference_profiles").paginate(typeEquals="APPLICATION"):
            for summary in page["inferenceProfileSummaries"]:
                if summary["inferenceProfileName"] == profile_name:
                    print(f"Reusing inference profile {profile_name}: {summary['inferenceProfileArn']}")
                    return summary["inferenceProfileArn"]
        resp = bedrock.create_inference_profile(
            inferenceProfileName=profile_name,
            description=f"{project} {name} WattBot RAG {label} on {base_model_id}",   # letters, digits, : and . only, single spaces or hyphens between
            modelSource={"copyFrom": source_arn(base_model_id)},
            tags=tags,
        )
    except ClientError as e:
        if "On Demand" in e.response["Error"].get("Message", "") and not base_model_id.startswith("us."):
            # Some models are listed in the region's catalog but served only through the US cross-region
            # profile (Cohere Embed v4 and Llama 4 in Ohio, for example); the profile ID is the model ID with a us. prefix
            print(f"{base_model_id} is not served on demand in this region; using the cross-region profile us.{base_model_id}")
            return tagged_profile_arn(bedrock, f"us.{base_model_id}", label, name, project, purpose)
        if e.response["Error"]["Code"] == "AccessDeniedException":
            role = boto3.client("sts").get_caller_identity()["Arn"]
            raise PermissionError(
                f"This notebook's role has no Bedrock permissions ({role}). Attach the Bedrock policy from the Account "
                "preparation instructor page to that role (on the class account, tell your instructor), then rerun this cell; "
                "a policy change reaches a running instance within a minute."
            ) from e
        raise
    print(f"Created inference profile {profile_name}: {resp['inferenceProfileArn']}")
    return resp["inferenceProfileArn"]


# ---------------------------------------------------------------------------
# The two LLM phases and the per-question orchestration
# ---------------------------------------------------------------------------

Generate = Callable[[str, str, int], str]


def answer_phase_for_question(
    qid: str,
    question: str,
    retrieved_chunks: List[Dict[str, Any]],
    generate: Generate,
    answer_unit: Optional[str] = None,
    max_new_tokens: int = 512,
) -> Dict[str, Any]:
    """First LLM call: answer JSON (answer, answer_value, ref_id, is_blank, supporting_materials)."""
    context = format_context_for_prompt(retrieved_chunks)
    raw_answer = generate(ANSWER_SYSTEM_PROMPT, build_answer_prompt(question, context, answer_unit), max_new_tokens)
    return parse_answer_json(raw_answer, qid=qid)


def explanation_phase_for_question(
    qid: str,
    question: str,
    answer: str,
    supporting_materials: str,
    generate: Generate,
    ref_ids: Optional[Sequence[str]] = None,
    with_evidence_type: bool = False,
    max_new_tokens: int = 256,
) -> str:
    """Second LLM call: a short explanation, optionally prefixed with '[evidence_type]'."""
    raw = generate(
        explanation_system_prompt(with_evidence_type),
        build_explanation_prompt(question, answer, supporting_materials, ref_ids, with_evidence_type),
        max_new_tokens,
    ).strip()
    if not with_evidence_type:
        return raw

    first_line = next((line.strip() for line in raw.splitlines() if line.strip()), "")
    if ":" in first_line:
        etype, expl = first_line.split(":", 1)
        evidence_type = etype.strip().strip("[]").lower() or "other"
        explanation = expl.strip()
    else:
        evidence_type, explanation = "other", first_line
    return f"[{evidence_type}] {explanation[:100]}"


def run_single_qa(
    row,
    embed_query: Callable[[str], np.ndarray],
    generate: Generate,
    chunk_embeddings: np.ndarray,
    chunked_docs: List[Dict[str, Any]],
    docid_to_url: Dict[str, str],
    top_k: int = 8,
    retrieval_threshold: float = 0.25,
    with_evidence_type: bool = False,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Retrieve, answer, decide answerability, explain: one WattBot row in, one submission row out.

    `row` is a row of train_QA.csv (needs `id` and `question`; `answer_unit` is used
    as a hint if present). A question is marked unanswerable when the model says so
    OR the best retrieval cosine similarity is below `retrieval_threshold`.
    """
    qid = row["id"]
    question = row["question"]
    answer_unit = row.get("answer_unit", BLANK) if hasattr(row, "get") else BLANK
    if not isinstance(answer_unit, str) or not answer_unit.strip():
        answer_unit = BLANK

    retrieved = retrieve_top_k(embed_query(question), chunk_embeddings, chunked_docs, k=top_k)
    top_score = retrieved[0]["score"] if retrieved else 0.0

    parsed = answer_phase_for_question(qid, question, retrieved, generate, answer_unit=answer_unit)

    if parsed["is_blank"] and verbose:
        print(f"[diag][{qid}] model returned is_blank")
    elif top_score < retrieval_threshold and verbose:
        print(f"[diag][{qid}] retrieval blocked: top cosine={top_score:.3f} < threshold={retrieval_threshold:.3f}")
    is_blank = parsed["is_blank"] or (top_score < retrieval_threshold)

    if is_blank:
        return {
            "id": qid,
            "question": question,
            "answer": UNANSWERABLE_TEXT,
            "answer_value": BLANK,
            "answer_unit": answer_unit,
            "ref_id": BLANK,
            "ref_url": BLANK,
            "supporting_materials": BLANK,
            "explanation": BLANK,
        }

    ref_ids = parsed["ref_id"]
    urls = [docid_to_url[rid] for rid in ref_ids if rid in docid_to_url]
    explanation = explanation_phase_for_question(
        qid,
        question,
        parsed["answer"],
        parsed["supporting_materials"],
        generate,
        ref_ids=ref_ids,
        with_evidence_type=with_evidence_type,
    )
    return {
        "id": qid,
        "question": question,
        "answer": parsed["answer"],
        "answer_value": parsed["answer_value"],
        "answer_unit": answer_unit,
        "ref_id": ";".join(ref_ids) if ref_ids else BLANK,
        "ref_url": ";".join(urls) if urls else BLANK,
        "supporting_materials": parsed["supporting_materials"],
        "explanation": explanation,
    }


# ---------------------------------------------------------------------------
# Scoring (WattBot 2026 metric: 0.75 answer_value + 0.20 ref_id F1 + 0.05 is_NA)
# ---------------------------------------------------------------------------


EVIDENCE_FLAGS = ["Quote", "Table", "Figure", "Math", "is_NA"]


def evidence_flags(gt: pd.DataFrame):
    """Return (gt with 0/1 evidence-type columns, whether they were derived).

    The competition's solution file carries Quote/Table/Figure/Math/is_NA flags saying
    what kind of evidence each answer rests on, and Score.py breaks the error rate down
    by them. The 2026 train_QA.csv has them; an older file without them has a free-text `explanation`
    column says the same thing informally ("Quote", "Table 3", "Figure, Quote", or a
    worked calculation). When the flags are missing, derive them from that column so
    the breakdown still prints. A question can carry several flags.
    """
    if all(c in gt.columns for c in EVIDENCE_FLAGS):
        return gt, False
    gt = gt.copy()
    e = gt.get("explanation", pd.Series([""] * len(gt))).fillna("").astype(str)
    gt["Quote"] = e.str.contains(r"\bquote", case=False, regex=True).astype(int)
    gt["Table"] = e.str.contains(r"\btable", case=False, regex=True).astype(int)
    gt["Figure"] = e.str.contains(r"\bfig", case=False, regex=True).astype(int)
    gt["Math"] = e.str.contains(r"[=÷×→≈]|\d\s*/\s*\d|\bcalc|\bmath", case=False, regex=True).astype(int)   # a worked calculation
    gt["is_NA"] = gt["answer_value"].astype(str).str.strip().str.lower().eq(BLANK).astype(int)
    return gt, True


def compute_wattbot_score(
    train_qa_path: str = "train_QA.csv",
    preds_path: str = "wattbot_solutions.csv",
    n_examples: int = 10,
    **_legacy_kwargs,
) -> pd.DataFrame:
    """Score a predictions CSV against train_QA.csv with the competition's own metric.

    The headline number comes from `Score.score` in `Score.py`, the file the WattBot
    leaderboard runs (2026 rules: 0.75 answer_value, 0.20 citation F1, 0.05 clean
    refusal; answering an unanswerable question scores zero on answer_value). That
    file has to sit next to this module. This wrapper adds what the leaderboard does
    not print: a per-row score, the worst rows with the model's own explanation, and
    the returned DataFrame for further digging.
    """
    import Score   # the competition's metric, copied beside this module; imported here so the batch job does not need it

    gt, flags_derived = evidence_flags(pd.read_csv(train_qa_path))
    preds = pd.read_csv(preds_path)
    overall = Score.score(gt.copy(), preds.copy(), "id", verbose=False)   # the leaderboard number
    summary = Score.LAST_BREAKDOWN

    merged = gt.merge(preds, on="id", suffixes=("_gt", "_pred"))
    if merged.empty:
        raise ValueError("No overlapping ids between ground truth and predictions.")

    # The same per-row rubric Score.score applies, kept per row so the worst rows can be listed
    def _row(r):
        sol = {k: str(r[f"{k}_gt"]) for k in ("answer_value", "answer_unit", "ref_id")}
        sub = {k: str(r[f"{k}_pred"]) for k in ("answer_value", "answer_unit", "ref_id", "ref_url", "supporting_materials") if f"{k}_pred" in merged.columns}
        bits = Score.row_bits(sol, sub)
        return float(bits["val"]), Score.ref_prf(sol["ref_id"], sub["ref_id"])[2], float(bits["na"])

    merged["answer_score"], merged["ref_id_score"], merged["is_NA_score"] = zip(*(_row(r) for _, r in merged.iterrows()))
    merged["wattbot_score"] = 0.75 * merged["answer_score"] + 0.20 * merged["ref_id_score"] + 0.05 * merged["is_NA_score"]

    print(f"Rows compared: {len(merged)}")
    print(f"answer_value correct (weight 0.75): {merged['answer_score'].mean():.4f}")
    print(f"citation F1          (weight 0.20): {merged['ref_id_score'].mean():.4f}   (precision {summary['ref_precision']:.3f}, recall {summary['ref_recall']:.3f})")
    print(f"clean refusals       (weight 0.05): {merged['is_NA_score'].mean():.4f}")
    if summary.get("n_unanswerable"):
        n_na = summary["n_unanswerable"]
        n_answered = round(summary["false_answer_rate"] * n_na)
        print(f"unanswerable questions (correct response is is_blank): {n_na}; answered anyway: {n_answered} of {n_na}, each scoring zero on answer_value")
    print(f"WattBot score (Score.py, what the leaderboard reports): {overall:.4f}")
    per_type = {k: v for k, v in summary.get("per_type", {}).items() if k != "ALL"}
    if per_type:
        source = "flags derived from the explanation column" if flags_derived else "the competition's own flags"
        print(f"\nanswer_value correct by evidence type ({source}; a question can carry several):")
        for label, d in per_type.items():
            print(f"  {label:<11} {d['n'] - d['wrong']:>3}/{d['n']:<3} {100 - d['pct_wrong']:5.1f}% correct")

    incorrect = merged[merged["wattbot_score"] < 1.0]
    if not incorrect.empty and n_examples > 0:
        print(f"\nExamples of incorrect / partially correct responses (up to {n_examples} rows):\n")
        q_col = "question_gt" if "question_gt" in merged.columns else ("question" if "question" in merged.columns else None)
        for _, row in incorrect.sort_values("wattbot_score").head(n_examples).iterrows():
            print("-" * 80)
            print(f"id: {row['id']}")
            if q_col:
                print(f"Question: {row[q_col]}")
            print(f"GT answer_value:   {row['answer_value_gt']}")
            print(f"Pred answer_value: {row['answer_value_pred']}")
            print(f"GT ref_id:         {row['ref_id_gt']}")
            print(f"Pred ref_id:       {row['ref_id_pred']}")
            # The model's own account of its answer, when the predictions carry one (blank rows have none)
            for label, col in [("Model explanation", "explanation"), ("Cited passage", "supporting_materials")]:
                col = f"{col}_pred" if f"{col}_pred" in merged.columns else col
                val = row.get(col)
                if isinstance(val, str) and val and val != BLANK:
                    print(f"{label + ':':<19}{val[:400]}")
            print(
                f"answer_score: {row['answer_score']:.3f}, ref_id_score: {row['ref_id_score']:.3f}, "
                f"is_NA_score: {row['is_NA_score']:.3f}, wattbot_score: {row['wattbot_score']:.3f}"
            )
        print("-" * 80)
    return merged


# ---------------------------------------------------------------------------
# Evaluation subsets
# ---------------------------------------------------------------------------

ALL_EVIDENCE_FLAGS = ["Quote", "Table", "Figure", "Math", "is_NA", "CrossPaper", "Reconcile"]


def sample_questions_by_type(
    train_df: pd.DataFrame, per_type: int = 10, seed: int = 0, flags: Sequence[str] = ALL_EVIDENCE_FLAGS
) -> pd.DataFrame:
    """A stratified subset of train_QA.csv: up to `per_type` questions from each evidence type.

    The 2026 file flags each question by the evidence its answer rests on (Quote,
    Table, Figure, Math, is_NA, CrossPaper, Reconcile; a question can carry several).
    Drawing the same number from each flag keeps the rare types (17 Figure questions,
    5 unanswerable ones) in a short run, where the first N rows of the file would be
    almost all Quote. `per_type=0` returns the whole file. The seed makes the draw
    repeatable, so two pipelines scored on it are scored on the same questions.
    """
    if not per_type:
        return train_df
    present = [f for f in flags if f in train_df.columns]
    if not present:
        return train_df.head(per_type * len(flags))
    keep = set()
    for f in present:
        rows = train_df[train_df[f].fillna(0).astype(int) == 1]
        keep |= set(rows.sample(min(per_type, len(rows)), random_state=seed)["id"])
    return train_df[train_df["id"].isin(keep)].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Multimodal retrieval: page images for the table and figure questions
# ---------------------------------------------------------------------------

CAPTION_RE = re.compile(r"^\s*(Table|Fig\.?|Figure)\s*([A-Z]?\d+)", re.IGNORECASE | re.MULTILINE)


def pdfs_to_page_docs(metadata: pd.DataFrame, pdf_dir: str) -> List[Dict[str, Any]]:
    """Load each PDF under pdf_dir into page-level records (the Bedrock episode's Step 5).

    Each dict has keys: text, doc_id, title, url, page_num, page_label, total_pages.
    Needs pypdf (`pip install pypdf`); the corpus build script writes the same records to pages.jsonl.
    """
    from pypdf import PdfReader
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


def find_caption_pages(page_docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """The pages whose text carries a table or figure caption, with the captions found.

    pypdf keeps caption lines ("Table 3: ...", "Figure 2. ...") even where it loses the
    table's layout or the figure entirely, so a line starting with one of those words
    marks a page worth looking at as an image. Returns one record per such page:
    doc_id, page_num, page_label, captions (for example ["Figure 2", "Table 3"]).
    """
    out = []
    for p in page_docs:
        found = CAPTION_RE.findall(p.get("text", "") or "")
        if not found:
            continue
        captions = sorted({("Figure" if w.lower().startswith("fig") else "Table") + " " + n for w, n in found})
        out.append(
            {
                "doc_id": p["doc_id"],
                "page_num": int(p["page_num"]),
                "page_label": str(p.get("page_label", int(p["page_num"]) + 1)),
                "captions": captions,
            }
        )
    return out


def render_page_png(pdf_path: str, page_num: int, dpi: int = 100) -> bytes:
    """Render one page of a PDF to PNG bytes with pypdfium2 (`pip install pypdfium2`).

    100 dpi makes a letter page about 850 x 1100 pixels: captions and axis labels stay
    readable, and the file is well under the multimodal models' size limits.
    """
    import io

    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument(pdf_path)
    try:
        image = pdf[page_num].render(scale=dpi / 72).to_pil()
    finally:
        pdf.close()
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def embed_multimodal(
    client,
    model_id: str,
    text: Optional[str] = None,
    image_png: Optional[bytes] = None,
    purpose: str = "index",
    dimension: int = 1024,
) -> np.ndarray:
    """One vector for a text string or a PNG image from a Bedrock multimodal embedding model.

    `client` is a bedrock-runtime client in a region that has the model (us-east-1 for
    both, as of October 2026). Two request shapes are supported, chosen by model ID:
    Amazon Nova Multimodal Embeddings (`amazon.nova-2-multimodal-embeddings-v1:0`),
    whose body says whether the vector is for indexing or for a query and takes one
    input per call, and Titan Multimodal Embeddings G1 (`amazon.titan-embed-image-v1`),
    which takes `inputText` and/or `inputImage`. `purpose` is "index" for what is
    stored and "query" for the question. Vectors are L2-normalized so cosine similarity
    is a dot product.
    """
    import base64

    if text is None and image_png is None:
        raise ValueError("give text or image_png")
    if "nova" in model_id:
        params: Dict[str, Any] = {
            "embeddingPurpose": "GENERIC_INDEX" if purpose == "index" else "GENERIC_RETRIEVAL",
            "embeddingDimension": dimension,
        }
        if image_png is not None:
            params["image"] = {
                "format": "png",
                "source": {"bytes": base64.b64encode(image_png).decode()},
                "detailLevel": "DOCUMENT_IMAGE",   # a rendered page, not a photo
            }
        else:
            params["text"] = {"truncationMode": "END", "value": text}
        body: Dict[str, Any] = {
            "schemaVersion": "nova-multimodal-embed-v1",   # without it Nova validates the body as a chat request and asks for "messages"
            "taskType": "SINGLE_EMBEDDING",
            "singleEmbeddingParams": params,
        }
    else:
        body = {"embeddingConfig": {"outputEmbeddingLength": dimension}}
        if text is not None:
            body["inputText"] = text[:1000]   # Titan G1 reads at most 256 tokens of text
        if image_png is not None:
            body["inputImage"] = base64.b64encode(image_png).decode()
    response = client.invoke_model(
        modelId=model_id, contentType="application/json", accept="application/json", body=json.dumps(body)
    )
    out = json.loads(response["body"].read())
    vec = out["embeddings"][0]["embedding"] if "embeddings" in out else out["embedding"]
    v = np.asarray(vec, dtype="float32")
    return v / (np.linalg.norm(v) + 1e-12)


def image_blocks(images: Sequence[Tuple[str, bytes]]) -> List[Dict[str, Any]]:
    """`converse` content blocks for labeled PNG images: a text label, then the image, per page.

    `images` is a list of (label, png_bytes); the label (for example "luccioni2024, page 6")
    lets the model cite the paper the page came from.
    """
    blocks: List[Dict[str, Any]] = []
    for label, png in images:
        blocks.append({"text": f"[page image: {label}]"})
        blocks.append({"image": {"format": "png", "source": {"bytes": png}}})
    return blocks


def hit_at_k(retrieved_doc_ids: Sequence[Sequence[str]], gt_ref_ids: Sequence[Sequence[str]]) -> float:
    """Fraction of questions whose top-k retrieved documents include a ground-truth paper."""
    hits = [bool(set(r) & set(g)) for r, g in zip(retrieved_doc_ids, gt_ref_ids)]
    return float(np.mean(hits)) if hits else float("nan")


def drop_questions_without_papers(train_df: pd.DataFrame, available_doc_ids) -> pd.DataFrame:
    """Keep the questions whose cited papers are all in the corpus, plus the unanswerable ones.

    A paper whose PDF could not be downloaded when the corpus was built cannot be
    retrieved, so a question citing it can only score zero; scoring it would measure the
    download, not the pipeline. Prints what was dropped.
    """
    available = set(available_doc_ids)
    refs = train_df["ref_id"].apply(split_ref_ids)
    keep = refs.apply(lambda ids: all(r in available for r in ids))
    dropped = train_df.loc[~keep, "ref_id"].apply(split_ref_ids)
    missing = sorted({r for ids in dropped for r in ids if r not in available})
    if missing:
        print(f"Dropping {int((~keep).sum())} of {len(train_df)} questions: they cite papers not in the corpus {missing}")
    return train_df[keep].reset_index(drop=True)


def extract_figures_and_tables(pdf_path: str, doc_id: str, out_dir, dpi: int = 150,
                               min_width: float = 80, min_height: float = 36,
                               pages: Optional[Any] = None) -> List[Dict[str, Any]]:
    """Crop every table and figure out of one PDF with PyMuPDF (`pip install pymupdf`).

    Three detectors, all built into PyMuPDF: `find_tables` for tables (ruled or
    rule-less), `get_image_info` for raster figures, and `cluster_drawings` for vector
    figures, which groups the thousands of line and fill commands a plot is made of into
    one bounding box. Page-sized background fills and the rulings inside detected tables
    are dropped first, boxes smaller than `min_width` x `min_height` points (rules, logos)
    are ignored, and a box inside a larger one is ignored. Each box is paired with the
    nearest caption line starting "Table N" or "Figure N" (above for tables, below for
    figures) and the crop is widened to include that caption, so the crop carries the
    identifier a question will use. A caption with no detected box falls back to the
    whole page, so every table and figure the text mentions gets an image.

    `pages` limits the work: a list of page numbers (0-based), or a dict mapping a page
    number to the caption kinds found on it ({"table"}, {"figure"} or both), in which
    case table detection runs only on pages with a table caption and figure detection
    only on pages with a figure caption. `find_caption_pages` gives those pages.

    Returns one record per crop: doc_id, page_num, page_label, kind ("table" or
    "figure"), caption, png_path, whole_page. Crops are PNGs at `dpi` in `out_dir`.
    """
    import pymupdf
    from pathlib import Path

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    records: List[Dict[str, Any]] = []
    doc = pymupdf.open(pdf_path)
    for page in doc:
        if pages is not None and page.number not in pages:
            continue
        kinds = set(pages[page.number]) if isinstance(pages, dict) else {"table", "figure"}
        page_area = page.rect.get_area()
        tables = [pymupdf.Rect(t.bbox) for t in page.find_tables().tables] if "table" in kinds else []
        boxes: List[Tuple[str, Any]] = [("table", r) for r in tables]
        if "figure" in kinds:
            boxes += [("figure", pymupdf.Rect(info["bbox"])) for info in page.get_image_info()]
            drawings = [
                d for d in page.get_drawings()
                if d["rect"].get_area() < 0.5 * page_area and not any(t.contains(d["rect"]) for t in tables)
            ]
            if drawings:
                boxes += [("figure", r) for r in page.cluster_drawings(drawings=drawings)]
        boxes = [(k, r) for k, r in boxes if r.width >= min_width and r.height >= min_height and r.get_area() < 0.9 * page_area]
        boxes = [(k, r) for k, r in boxes if not any(o is not r and o.contains(r) for _, o in boxes)]

        captions = []
        for b in page.get_text("blocks"):
            m = CAPTION_RE.match(b[4])
            if m:
                kind = "figure" if m.group(1).lower().startswith("fig") else "table"
                captions.append({"rect": pymupdf.Rect(b[:4]), "kind": kind, "text": b[4].strip().split("\n")[0][:120], "used": False})

        def nearest_caption(kind, r):
            best, best_gap = None, 200.0   # a caption more than 200 points away is not this box's
            for c in captions:
                if c["kind"] != kind:
                    continue
                gap = (r.y0 - c["rect"].y1) if kind == "table" else (c["rect"].y0 - r.y1)   # table captions sit above, figure captions below
                if gap < -5:
                    gap = abs(gap) + 100   # on the wrong side: allowed, but only if nothing better
                if gap < best_gap:
                    best, best_gap = c, gap
            return best

        for i, (kind, r) in enumerate(boxes):
            cap = nearest_caption(kind, r)
            clip = r + (-6, -6, 6, 6)
            if cap is not None:
                cap["used"] = True
                clip |= cap["rect"]
            records.append(_save_crop(page, clip & page.rect, out_dir, doc_id, kind, cap["text"] if cap else "", i))
        for i, cap in enumerate(c for c in captions if not c["used"]):   # a caption whose box was not found: keep the page
            records.append(_save_crop(page, page.rect, out_dir, doc_id, cap["kind"], cap["text"], 100 + i))
    doc.close()
    return records


def _save_crop(page, clip, out_dir, doc_id: str, kind: str, caption: str, index: int) -> Dict[str, Any]:
    pix = page.get_pixmap(clip=clip, dpi=150)
    path = out_dir / f"{doc_id}_p{page.number:03d}_{kind}{index}.png"
    path.write_bytes(pix.tobytes("png"))
    return {
        "doc_id": doc_id,
        "page_num": page.number,
        "page_label": str(page.number + 1),
        "kind": kind,
        "caption": caption,
        "png_path": str(path),
        "whole_page": clip == page.rect,   # True for a caption whose table or figure no detector found
    }


def cited_captions(supporting_materials: Any) -> List[str]:
    """The tables and figures a train_QA.csv row names in its supporting_materials, as "Table 3" / "Figure 2".

    "Fig. 5, upper panel" and "Figure 2(d)" both become "Figure 2"; the text column is free-form, so
    this is the identifier only, matched against the caption on each crop.
    """
    text = str(supporting_materials or "")
    found = []
    for word, num in re.findall(r"\b(Table|Fig\.?|Figure)\s*([A-Z]?\d+)", text, flags=re.IGNORECASE):
        label = ("Figure" if word.lower().startswith("fig") else "Table") + " " + num
        if label not in found:
            found.append(label)
    return found


def crop_matches(crop: Dict[str, Any], doc_id: str, label: str) -> bool:
    """Whether a crop record is the named table or figure of the named paper."""
    if crop.get("doc_id") != doc_id:
        return False
    m = re.match(r"\s*(Table|Fig\.?|Figure)\s*([A-Z]?\d+)", crop.get("caption", ""), flags=re.IGNORECASE)
    if not m:
        return False
    crop_label = ("Figure" if m.group(1).lower().startswith("fig") else "Table") + " " + m.group(2)
    return crop_label == label


GenerateBatch = Callable[[Sequence[Tuple[str, str]], int], List[str]]


def run_batch_qa(
    rows: pd.DataFrame,
    embed_query: Callable[[str], np.ndarray],
    generate_batch: GenerateBatch,
    chunk_embeddings: np.ndarray,
    chunked_docs: List[Dict[str, Any]],
    docid_to_url: Dict[str, str],
    top_k: int = 8,
    retrieval_threshold: float = 0.25,
    batch_size: int = 8,
    log: Callable[[str], None] = print,
) -> List[Dict[str, Any]]:
    """`run_single_qa` over many rows, with the model calls made in batches.

    Same retrieval, prompts, parsing and unanswerable rule as `run_single_qa`, so the
    two produce the same submission rows; the difference is that `generate_batch`
    receives a list of (system_prompt, user_prompt) pairs and returns a list of texts,
    which lets a self-hosted model decode several prompts at once. One GPU decoding
    one prompt at a time is idle most of the time; eight at a time is several times
    faster for the same answers. Three passes: retrieve every question, answer them in
    batches, then explain the answered ones in batches.
    """
    prepared = []
    for _, row in rows.iterrows():
        qid, question = row["id"], row["question"]
        answer_unit = row.get("answer_unit", BLANK) if hasattr(row, "get") else BLANK
        if not isinstance(answer_unit, str) or not answer_unit.strip():
            answer_unit = BLANK
        retrieved = retrieve_top_k(embed_query(question), chunk_embeddings, chunked_docs, k=top_k)
        top_score = retrieved[0]["score"] if retrieved else 0.0
        prompt = build_answer_prompt(question, format_context_for_prompt(retrieved), answer_unit)
        prepared.append({"id": qid, "question": question, "answer_unit": answer_unit, "top_score": top_score, "prompt": prompt})
    log(f"Retrieved context for {len(prepared)} questions; answering in batches of {batch_size}")

    def in_batches(pairs: List[Tuple[str, str]], max_new_tokens: int, what: str) -> List[str]:
        out: List[str] = []
        for start in range(0, len(pairs), batch_size):
            out.extend(generate_batch(pairs[start:start + batch_size], max_new_tokens))
            log(f"  {what}: {min(start + batch_size, len(pairs))} / {len(pairs)}")
        return out

    raw_answers = in_batches([(ANSWER_SYSTEM_PROMPT, p["prompt"]) for p in prepared], 512, "answers")
    results: List[Dict[str, Any]] = []
    to_explain: List[Tuple[int, Dict[str, Any]]] = []
    for p, raw in zip(prepared, raw_answers):
        parsed = parse_answer_json(raw, qid=p["id"])
        is_blank = parsed["is_blank"] or (p["top_score"] < retrieval_threshold)
        if is_blank:
            results.append({"id": p["id"], "question": p["question"], "answer": UNANSWERABLE_TEXT, "answer_value": BLANK,
                            "answer_unit": p["answer_unit"], "ref_id": BLANK, "ref_url": BLANK,
                            "supporting_materials": BLANK, "explanation": BLANK})
            continue
        ref_ids = parsed["ref_id"]
        urls = [docid_to_url[rid] for rid in ref_ids if rid in docid_to_url]
        results.append({"id": p["id"], "question": p["question"], "answer": parsed["answer"], "answer_value": parsed["answer_value"],
                        "answer_unit": p["answer_unit"], "ref_id": ";".join(ref_ids) if ref_ids else BLANK,
                        "ref_url": ";".join(urls) if urls else BLANK, "supporting_materials": parsed["supporting_materials"],
                        "explanation": ""})
        to_explain.append((len(results) - 1, parsed))

    pairs = [(explanation_system_prompt(False),
              build_explanation_prompt(results[i]["question"], parsed["answer"], parsed["supporting_materials"], parsed["ref_id"], False))
             for i, parsed in to_explain]
    explanations = in_batches(pairs, 256, "explanations") if pairs else []
    for (i, _), text in zip(to_explain, explanations):
        results[i]["explanation"] = text.strip()
    return results

"""Shared helpers for the Badger Scribe fine-tuning episode.

Used by both `finetune_churro_lora.py` (SageMaker training job) and
`churro_transcribe_batch.py` (SageMaker processing job), so the prompt the
model is trained on is exactly the prompt it is asked to answer at inference
time. Keep this file next to those two scripts in the `code/` directory you
pass as `source_dir`.
"""

import re
import unicodedata

import pandas as pd
from PIL import Image

# Qwen2.5-VL turns every 28x28 pixel block into one visual token. The
# processor resizes each page so its pixel count stays under `max_pixels`,
# so this single number sets both image detail AND sequence length (and
# therefore GPU memory and time). 1280 blocks ~= 1 megapixel ~= 1280 tokens.
PATCH = 28 * 28
DEFAULT_MAX_PIXELS = 1280 * PATCH
DEFAULT_MIN_PIXELS = 256 * PATCH

INSTRUCTION = (
    "Transcribe all handwritten and printed text on this page exactly as "
    "written, in natural reading order. Keep the original spelling, "
    "capitalization, punctuation and abbreviations; do not modernize, "
    "translate, summarize or add commentary. Write one line of text per "
    "line on the page. Do not transcribe drawings, ruled lines or filler "
    "dashes."
)


def load_metadata(path: str) -> dict:
    """metadata.csv -> {doc_id: one-line catalog context}."""
    if not path:
        return {}
    md = pd.read_csv(path, keep_default_na=False)
    ctx = {}
    for _, r in md.iterrows():
        bits = [b for b in (r.get("title", ""), r.get("date", ""), r.get("language", "")) if b]
        ctx[r["doc_id"]] = f"{r.get('collection', '')}: " + "; ".join(bits)
    return ctx


def build_prompt(category: str, doc_context: str = "") -> str:
    """The text half of the user turn. The catalog record is cheap context that
    the competition organizers recommend feeding the model (names, dates and
    places are where recognition fails most)."""
    lang = "German (Kurrent script)" if category == "kade_letters" else "English"
    parts = [INSTRUCTION, f"Language: {lang}."]
    if doc_context:
        parts.append(f"Catalog record: {doc_context}")
    return "\n".join(parts)


def user_messages(prompt: str):
    """The user turn in the chat format Qwen-VL expects: an image placeholder
    (the processor replaces it with the image tokens) followed by the prompt text."""
    return [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]


def full_messages(prompt: str, answer: str):
    """The user turn plus the assistant's answer: a complete training example."""
    return user_messages(prompt) + [
        {"role": "assistant", "content": [{"type": "text", "text": answer}]}
    ]


def load_page_image(path: str, max_pixels: int = DEFAULT_MAX_PIXELS) -> Image.Image:
    """Open a page scan at roughly the resolution the model will see.

    The archive scans are ~2 MB JPEGs, several thousand pixels on a side.
    `draft` lets the JPEG decoder skip detail we are about to throw away,
    which keeps the 4-vCPU training instance from becoming the bottleneck.
    The processor still does the exact resize to a multiple of 28.
    """
    img = Image.open(path)
    side = int((max_pixels * 2) ** 0.5)  # decode at ~2x the final pixel budget
    img.draft("RGB", (side, side))
    return img.convert("RGB")


# ---------------------------------------------------------------------------
# A copy of the competition's text normalization and CER, so a job can print
# a score without shipping metric.py. Score official numbers with metric.py.
# ---------------------------------------------------------------------------
_TAG_RE = re.compile(r"</?(?:u|del|ins)>")


def normalize_text(text) -> str:
    """Make two transcriptions comparable before computing CER: treat missing as
    empty, unescape newlines, strip the markup tags the labels use, apply Unicode
    NFC (so 'ü' is one code point however it was typed), and collapse whitespace.
    The competition's metric does the same, so scores here match the leaderboard."""
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return ""
    text = str(text).replace("\\n", "\n")
    text = _TAG_RE.sub("", text)
    text = unicodedata.normalize("NFC", text)
    return " ".join(text.split())


def page_cer(prediction: str, reference: str) -> float:
    """Levenshtein / len(reference), capped at 1.0; `#` in the reference is a
    free wildcard, exactly as in the leaderboard metric."""
    pred, ref = normalize_text(prediction), normalize_text(reference)
    if not ref:
        return 0.0 if not pred else 1.0
    if not pred:
        return 1.0
    prev = list(range(len(ref) + 1))
    for i, cp in enumerate(pred, start=1):
        curr = [i]
        for j, cr in enumerate(ref, start=1):
            sub = 0 if (cr == "#" or cr == cp) else 1
            curr.append(min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + sub))
        prev = curr
    return min(1.0, prev[-1] / len(ref))


def macro_cer(df: pd.DataFrame) -> tuple:
    """df needs columns text_pred, text_ref, category. Returns (macro, per-category Series)."""
    df = df.copy()
    df["cer"] = [page_cer(p, r) for p, r in zip(df["text_pred"], df["text_ref"])]
    by_cat = df.groupby("category")["cer"].mean()
    return float(by_cat.mean()), by_cat

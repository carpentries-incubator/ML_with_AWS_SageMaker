"""Batch-transcribe Badger Scribe pages with CHURRO-3B, with or without a
LoRA adapter. Runs as a SageMaker Processing job.

Inputs (mounted under --input_dir, one ProcessingInput each):
    images/<page_id>.jpg           the scans
    data/pages.csv                 which pages to transcribe (page_id, doc_id, category[, text])
    data/metadata.csv              optional catalog context
    data/<page_ids_csv>            optional: restrict to these page_ids (e.g. the holdout list)
    adapter/model.tar.gz           optional: the training job's output
    model/                         optional: a staged copy of the base model; pass its path
                                   as --model_id so the job reads S3 instead of Hugging Face

Outputs (written to --output_dir):
    predictions.csv                page_id, text  -- the competition submission format
    scores.json                    CER per category over the rows of pages.csv that have a
                                   non-empty text column (test pages, whose text is empty,
                                   are transcribed but never scored)

Run it twice with the same pages: once without --adapter_tar (the base
model) and once with it. The difference is what your fine-tune bought.
"""

import argparse
import json
import os
import tarfile
import time

import pandas as pd
import torch

from badger_common import (
    DEFAULT_MAX_PIXELS,
    DEFAULT_MIN_PIXELS,
    build_prompt,
    load_metadata,
    load_page_image,
    macro_cer,
    user_messages,
)


def parse_args():
    """Flags arrive from the Processing job's arguments= list (a processor has no hyperparameters=)."""
    p = argparse.ArgumentParser()
    # A local directory works as a model id: SageMaker sets SM_CHANNEL_MODEL when a
    # "model" channel is mounted (training jobs); Processing jobs pass the path explicitly.
    p.add_argument("--model_id", type=str, default=os.environ.get("SM_CHANNEL_MODEL", "stanford-oval/churro-3B"))
    p.add_argument("--input_dir", type=str, default="/opt/ml/processing/input")
    p.add_argument("--output_dir", type=str, default="/opt/ml/processing/output")
    p.add_argument("--pages_csv", type=str, default="pages.csv", help="file name inside input_dir/data")
    p.add_argument("--page_ids_csv", type=str, default="", help="optional file inside input_dir/data; restrict to these page_ids")
    p.add_argument("--adapter_tar", type=str, default="", help="path to model.tar.gz from the training job; empty = base model")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--max_new_tokens", type=int, default=1024)
    p.add_argument("--max_pixels", type=int, default=DEFAULT_MAX_PIXELS)
    p.add_argument("--min_pixels", type=int, default=DEFAULT_MIN_PIXELS)
    p.add_argument("--use_metadata", type=int, default=1)
    p.add_argument("--max_pages", type=int, default=0, help="0 = all")
    p.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "auto"],
                   help="weights dtype; auto = bf16 if the GPU supports it, else fp16 (a Colab T4)")
    p.add_argument("--adapter_dir", type=str, default="", help="an unpacked adapter folder instead of --adapter_tar")
    return p.parse_args()


def main():
    """Read the page list, load the model (merging the adapter if given), transcribe
    in batches, write predictions.csv, and score the rows that have a reference."""
    args = parse_args()
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    data_dir = os.path.join(args.input_dir, "data")
    images_dir = os.path.join(args.input_dir, "images")
    pages = pd.read_csv(os.path.join(data_dir, args.pages_csv), keep_default_na=False)
    if args.page_ids_csv:
        keep = set(pd.read_csv(os.path.join(data_dir, args.page_ids_csv))["page_id"])
        pages = pages[pages["page_id"].isin(keep)]
    if args.max_pages > 0:
        pages = pages.head(args.max_pages)
    meta_path = os.path.join(data_dir, "metadata.csv")
    doc_ctx = load_metadata(meta_path) if (args.use_metadata and os.path.exists(meta_path)) else {}
    print(f"transcribing {len(pages)} pages")

    # Unpack the adapter first: the training job saved its processor next to the
    # adapter, and using that copy keeps the inference resize identical to training.
    adapter_dir = ""
    if args.adapter_tar:
        adapter_dir = "/tmp/adapter"
        os.makedirs(adapter_dir, exist_ok=True)
        with tarfile.open(args.adapter_tar) as tar:
            tar.extractall(adapter_dir, filter="data")
        print("adapter files:", sorted(os.listdir(adapter_dir)))

    if args.adapter_dir:
        adapter_dir = args.adapter_dir
    processor_src = adapter_dir if (adapter_dir and os.path.exists(os.path.join(adapter_dir, "preprocessor_config.json"))) else args.model_id
    print(f"loading processor from {processor_src}; model from {args.model_id}")
    processor = AutoProcessor.from_pretrained(processor_src, min_pixels=args.min_pixels, max_pixels=args.max_pixels)
    processor.tokenizer.padding_side = "left"  # batched generation needs left padding
    if args.dtype == "auto":
        args.dtype = "bf16" if torch.cuda.is_bf16_supported() else "fp16"
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_id, torch_dtype=(torch.bfloat16 if args.dtype == "bf16" else torch.float16), attn_implementation="sdpa"
    ).to("cuda")

    if adapter_dir:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_dir).merge_and_unload()
        print("LoRA adapter merged into base weights")
    model.eval()

    rows = pages.to_dict("records")
    preds = []
    t0 = time.time()
    for start in range(0, len(rows), args.batch_size):
        chunk = rows[start:start + args.batch_size]
        images = [load_page_image(os.path.join(images_dir, f"{r['page_id']}.jpg"), args.max_pixels) for r in chunk]
        texts = [
            processor.apply_chat_template(
                user_messages(build_prompt(r["category"], doc_ctx.get(r["doc_id"], ""))),
                tokenize=False, add_generation_prompt=True)
            for r in chunk
        ]
        enc = processor(text=texts, images=images, padding=True, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=args.max_new_tokens, do_sample=False, repetition_penalty=1.05)
        gen = out[:, enc["input_ids"].shape[1]:]   # generate() returns prompt + answer; keep only the answer
        preds.extend(t.strip() for t in processor.batch_decode(gen, skip_special_tokens=True))
        done = start + len(chunk)
        print(f"  {done}/{len(rows)} pages, {(time.time() - t0) / done:.1f} s/page")

    os.makedirs(args.output_dir, exist_ok=True)
    pages = pages.assign(text_pred=preds)
    pages[["page_id"]].assign(text=preds).to_csv(os.path.join(args.output_dir, "predictions.csv"), index=False)
    print(f"wrote predictions.csv ({len(pages)} rows) in {(time.time() - t0) / 60:.1f} min")

    # Score only rows that carry a reference. Test pages have an empty text
    # column; counting them would score every one of them as CER 1.0.
    if "text" in pages.columns:
        scored = pages[pages["text"].str.len() > 0]
        if len(scored) > 0:
            macro, by_cat = macro_cer(scored.rename(columns={"text": "text_ref"}))
            scores = {"adapter": bool(args.adapter_tar), "n_pages": int(len(pages)), "n_scored": int(len(scored)),
                      "macro_cer": round(macro, 4), "per_category": by_cat.round(4).to_dict()}
            print(json.dumps(scores, indent=2))
            with open(os.path.join(args.output_dir, "scores.json"), "w") as f:
                json.dump(scores, f, indent=2)


if __name__ == "__main__":
    main()

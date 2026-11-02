"""LoRA fine-tune of CHURRO-3B (a Qwen2.5-VL-3B fine-tune for historical
text) on the Badger Scribe train set. Runs as a SageMaker training job.

SageMaker passes hyperparameters as command-line flags and mounts the
`train` channel at $SM_CHANNEL_TRAIN. That directory must contain:

    train.csv              page_id, doc_id, text, category, label_source
    metadata.csv           optional catalog context, joined on doc_id
    images/<page_id>.jpg   one scan per page

What gets written to $SM_MODEL_DIR (SageMaker tars it into model.tar.gz):

    adapter_config.json, adapter_model.safetensors   the LoRA adapter
    processor files                                  the resize/tokenizer settings used in
                                                     training, so inference matches them
                                                     (the base weights are still needed)
    holdout_page_ids.csv                             pages held out of training
    eval_before.csv, eval_after.csv                  per-page CER on sampled holdout pages,
                                                     base model vs. fine-tuned (same pages)
    train_summary.json                               what this run did and scored

The base weights are never modified: a LoRA adapter is ~100 MB that sits on
top of the frozen 3B model. That is what makes this fit on a 24 GB GPU and
finish in about half an hour.

If a `model` channel is mounted (a staged copy of the base weights in S3),
SageMaker sets $SM_CHANNEL_MODEL and the script loads from that directory
instead of pulling from Hugging Face.
"""

import argparse
import json
import os
import random
import time

# Silence the tokenizers fork warning from the dataloader workers.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import pandas as pd
import torch
from torch.utils.data import Dataset

from badger_common import (
    DEFAULT_MAX_PIXELS,
    DEFAULT_MIN_PIXELS,
    build_prompt,
    full_messages,
    load_metadata,
    load_page_image,
    macro_cer,
    page_cer,
    user_messages,
)

MAX_TRAINABLE_FRACTION = 0.05   # a LoRA adapter is well under this; more means the base weights are not frozen


def parse_args():
    """Every knob the notebook can set reaches the script as a command-line flag;
    SageMaker turns the estimator's hyperparameters dict into exactly these flags."""
    p = argparse.ArgumentParser()
    # --- what to train ---
    # A local directory works as a model id for from_pretrained; SageMaker sets
    # SM_CHANNEL_MODEL when the estimator is given a "model" channel.
    p.add_argument("--model_id", type=str, default=os.environ.get("SM_CHANNEL_MODEL", "stanford-oval/churro-3B"))
    p.add_argument("--epochs", type=float, default=2.0)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--batch_size", type=int, default=1, help="pages per step per GPU")
    p.add_argument("--grad_accum", type=int, default=8, help="effective batch = batch_size * grad_accum")
    p.add_argument("--max_pixels", type=int, default=DEFAULT_MAX_PIXELS)
    p.add_argument("--min_pixels", type=int, default=DEFAULT_MIN_PIXELS)
    p.add_argument("--use_metadata", type=int, default=1)
    # --- how much of the data ---
    p.add_argument("--categories", type=str, default="", help="comma-separated subset, e.g. survey_notes; empty = all")
    p.add_argument("--holdout_frac", type=float, default=0.15, help="fraction of pages per category kept for scoring")
    p.add_argument("--max_train_pages", type=int, default=0, help="0 = all; set small (e.g. 20) for a smoke test")
    p.add_argument("--eval_pages", type=int, default=20, help="holdout pages transcribed to report CER (0 = skip)")
    p.add_argument("--eval_before", type=int, default=1, help="1 = also score the base model on the same pages before training")
    p.add_argument("--curve_pages", type=int, default=8, help="holdout pages scored at the end of every epoch for the CER curve (0 = skip)")
    p.add_argument("--max_new_tokens", type=int, default=1024)
    p.add_argument("--seed", type=int, default=42)
    # Precision. SageMaker's A10G/L4/L40S GPUs run bf16, the default. A Colab T4 has no bf16, so
    # --dtype auto picks fp16 there, and --load_in_4bit 1 (QLoRA) keeps the frozen weights in 4-bit
    # so the whole run fits a 16 GB card.
    p.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "auto"],
                   help="compute dtype; auto = bf16 if the GPU supports it, else fp16")
    p.add_argument("--load_in_4bit", type=int, default=0,
                   help="1 = load the frozen base weights in 4-bit (QLoRA, needs bitsandbytes); 0 = 16-bit")
    # --- SageMaker paths ---
    p.add_argument("--train_dir", type=str, default=os.environ.get("SM_CHANNEL_TRAIN", "/opt/ml/input/data/train"))
    p.add_argument("--model_dir", type=str, default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
    p.add_argument("--output_data_dir", type=str, default=os.environ.get("SM_OUTPUT_DATA_DIR", "/opt/ml/output/data"))
    # SageMaker creates /opt/ml/checkpoints when the estimator sets checkpoint_s3_uri, and syncs it
    # to S3 while the job runs. Anything saved there survives the job being killed or interrupted.
    p.add_argument("--checkpoint_dir", type=str, default="/opt/ml/checkpoints")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def split_pages(df: pd.DataFrame, holdout_frac: float, seed: int):
    """Stratified page-level holdout. NOTE: the competition's hidden test set is
    split by *document*, so a page-level holdout (same writer in both halves)
    gives an optimistic CER. It is good enough to see whether training helped."""
    rng = random.Random(seed)
    hold = []
    for cat, grp in df.groupby("category"):
        ids = list(grp["page_id"])
        rng.shuffle(ids)
        n = max(1, int(round(len(ids) * holdout_frac))) if holdout_frac > 0 else 0
        hold.extend(ids[:n])
    hold = set(hold)
    return df[~df["page_id"].isin(hold)].copy(), df[df["page_id"].isin(hold)].copy()


class PageDataset(Dataset):
    """One training example per page: the page image, the instruction prompt
    built for its category, and the reference transcription as the answer.
    The Collator below turns a batch of these into tensors."""

    def __init__(self, df, images_dir, doc_ctx, max_pixels):
        self.rows = df.to_dict("records")
        self.images_dir = images_dir
        self.doc_ctx = doc_ctx
        self.max_pixels = max_pixels

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        prompt = build_prompt(r["category"], self.doc_ctx.get(r["doc_id"], ""))
        image = load_page_image(os.path.join(self.images_dir, f"{r['page_id']}.jpg"), self.max_pixels)
        return {"image": image, "prompt": prompt, "answer": r["text"]}


class Collator:
    """Tokenize image + prompt + transcription, and mask everything except the
    transcription out of the loss. The model is only graded on what it writes
    as the assistant, not on reproducing the prompt."""

    def __init__(self, processor):
        self.processor = processor
        self.processor.tokenizer.padding_side = "right"

    def __call__(self, batch):
        images = [b["image"] for b in batch]
        full_texts, prompt_texts = [], []
        for b in batch:
            full_texts.append(self.processor.apply_chat_template(
                full_messages(b["prompt"], b["answer"]), tokenize=False, add_generation_prompt=False))
            prompt_texts.append(self.processor.apply_chat_template(
                user_messages(b["prompt"]), tokenize=False, add_generation_prompt=True))

        enc = self.processor(text=full_texts, images=images, padding=True, return_tensors="pt")
        labels = enc["input_ids"].clone()
        labels[enc["attention_mask"] == 0] = -100
        for i, (ptxt, img) in enumerate(zip(prompt_texts, images)):
            # Run the prompt alone through the processor to learn how many
            # tokens it occupies once the image placeholder is expanded.
            n_prompt = self.processor(text=[ptxt], images=[img], return_tensors="pt")["input_ids"].shape[1]
            labels[i, :n_prompt] = -100
        enc["labels"] = labels
        return enc


# ---------------------------------------------------------------------------
# Evaluation (greedy transcription of a few holdout pages)
# ---------------------------------------------------------------------------
@torch.no_grad()
def transcribe(model, processor, image, prompt, max_new_tokens):
    """Transcribe one page: build the user turn (image + prompt), generate greedily,
    and return only the newly generated text (the input tokens are sliced off)."""
    text = processor.apply_chat_template(user_messages(prompt), tokenize=False, add_generation_prompt=True)
    enc = processor(text=[text], images=[image], return_tensors="pt").to(model.device)
    out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, repetition_penalty=1.05)
    gen = out[0, enc["input_ids"].shape[1]:]
    return processor.decode(gen, skip_special_tokens=True).strip()


def evaluate(model, processor, df, images_dir, doc_ctx, max_pixels, n_pages, max_new_tokens, seed, label, out_dir):
    """Transcribe a fixed sample of holdout pages and report CER. Called with
    the same seed before and after training, so both scores are on the same
    pages. Writes eval_<label>.csv with one row per page.

    The sample is the first n of a seeded shuffle of the sorted page ids, so
    it depends only on the holdout set and the seed, not on row order or on
    n: a 40-page sample contains the 15-page sample by construction. That is
    what lets the instructor's full run be compared against a student's."""
    if n_pages <= 0 or len(df) == 0:
        return None
    ids = sorted(df["page_id"])
    random.Random(seed).shuffle(ids)
    n = min(n_pages, len(ids))
    sample = df.set_index("page_id").loc[ids[:n]].reset_index()
    was_training = model.training
    model.eval()
    processor.tokenizer.padding_side = "left"
    preds = []
    t0 = time.time()
    for r in sample.to_dict("records"):
        img = load_page_image(os.path.join(images_dir, f"{r['page_id']}.jpg"), max_pixels)
        preds.append(transcribe(model, processor, img, build_prompt(r["category"], doc_ctx.get(r["doc_id"], "")), max_new_tokens))
    secs = time.time() - t0
    sample = sample.assign(text_pred=preds).rename(columns={"text": "text_ref"})
    macro, by_cat = macro_cer(sample)
    sample["cer"] = [page_cer(p, r) for p, r in zip(sample["text_pred"], sample["text_ref"])]
    sample[["page_id", "doc_id", "category", "cer", "text_ref", "text_pred"]].to_csv(
        os.path.join(out_dir, f"eval_{label}.csv"), index=False)
    print(f"[eval {label}] {len(sample)} holdout pages in {secs:.0f}s ({secs / len(sample):.1f} s/page) -> macro CER {macro:.4f}")
    print(by_cat.to_string())
    processor.tokenizer.padding_side = "right"
    if was_training:
        model.train()
    return {"n_pages": int(len(sample)), "macro_cer": round(macro, 4),
            "per_category": by_cat.round(4).to_dict(), "seconds_per_page": round(secs / len(sample), 1)}


# ---------------------------------------------------------------------------
def main():
    """The whole job, in order: split pages, load the base model, attach LoRA,
    score the base model on the holdout, train (saving a checkpoint and a CER point
    per epoch), save the adapter, score again, write the summary and the curves."""
    args = parse_args()
    torch.manual_seed(args.seed)
    from peft import LoraConfig, get_peft_model
    from transformers import (
        AutoProcessor,
        Qwen2_5_VLForConditionalGeneration,
        Trainer,
        TrainerCallback,
        TrainingArguments,
    )

    images_dir = os.path.join(args.train_dir, "images")
    df = pd.read_csv(os.path.join(args.train_dir, "train.csv"), keep_default_na=False)
    meta_path = os.path.join(args.train_dir, "metadata.csv")
    doc_ctx = load_metadata(meta_path) if (args.use_metadata and os.path.exists(meta_path)) else {}

    # Split over ALL pages first (so the holdout is the same for every run,
    # whatever subset is trained on), then filter to the requested categories.
    train_df, hold_df = split_pages(df, args.holdout_frac, args.seed)
    if args.categories:
        cats = [c.strip() for c in args.categories.split(",") if c.strip()]
        train_df, hold_df = train_df[train_df.category.isin(cats)], hold_df[hold_df.category.isin(cats)]
    if args.max_train_pages > 0:
        train_df = train_df.sample(n=min(args.max_train_pages, len(train_df)), random_state=args.seed)
    print(f"pages: {len(df)} total -> {len(train_df)} train / {len(hold_df)} holdout")
    print(train_df["category"].value_counts().to_string())

    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.output_data_dir, exist_ok=True)
    hold_df[["page_id", "doc_id", "category"]].to_csv(os.path.join(args.model_dir, "holdout_page_ids.csv"), index=False)

    # --- model ---
    if args.dtype == "auto":
        args.dtype = "bf16" if torch.cuda.get_device_capability(0)[0] >= 8 else "fp16"   # bf16 hardware is Ampere (8.0) and newer; a T4 is 7.5
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    print(f"loading {args.model_id} in {args.dtype}" + (", frozen weights in 4-bit" if args.load_in_4bit else ""))
    processor = AutoProcessor.from_pretrained(args.model_id, min_pixels=args.min_pixels, max_pixels=args.max_pixels)
    if args.load_in_4bit:
        from transformers import BitsAndBytesConfig
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.model_id, attn_implementation="sdpa", device_map={"": 0},
            quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                   bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype),
        )
        from peft import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True,
                                                gradient_checkpointing_kwargs={"use_reentrant": False})
    else:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.model_id, torch_dtype=dtype, attn_implementation="sdpa"
        ).to("cuda")

    # LoRA on the language model's attention + MLP projections only. The
    # vision tower stays frozen: its module names contain "visual", which
    # the negative lookahead excludes.
    lora = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=r"^(?!.*visual).*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$",
    )
    model = get_peft_model(model, lora)
    model.enable_input_require_grads()  # needed for gradient checkpointing with frozen embeddings
    if args.dtype == "fp16":
        # fp16 mixed precision keeps a 32-bit copy of what it trains; the adapter must start in 32-bit
        for p_ in model.parameters():
            if p_.requires_grad:
                p_.data = p_.data.float()
    model.print_trainable_parameters()

    # Two checks before anything trains. A LoRA run is what was estimated and
    # paid for; if the adapter landed on the vision tower, or if most of the
    # model is trainable, this is a different (and far more expensive) job.
    trainable, total = model.get_nb_trainable_parameters()   # PEFT's count; it corrects for 4-bit weights, which numel() would halve
    in_vision = [n for n, p in model.named_parameters() if p.requires_grad and "visual" in n]
    if in_vision:
        raise RuntimeError(f"{len(in_vision)} trainable tensors are in the vision tower (e.g. {in_vision[0]}); "
                           "target_modules should exclude 'visual'")
    if trainable / total > MAX_TRAINABLE_FRACTION:
        raise RuntimeError(f"{100 * trainable / total:.1f}% of the model is trainable; a LoRA run should be a few "
                           f"percent at most (cap {100 * MAX_TRAINABLE_FRACTION:.0f}%). Check target_modules and --lora_r")
    print(f"safeguards: adapter is {trainable:,} of {total:,} parameters ({100 * trainable / total:.2f}%), vision tower frozen")

    # --- checkpoints: where they go, and whether we are resuming ---
    ckpt_dir = args.checkpoint_dir if os.path.isdir(args.checkpoint_dir) \
        else os.path.join(args.output_data_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    from transformers.trainer_utils import get_last_checkpoint
    last_ckpt = get_last_checkpoint(ckpt_dir)
    print(f"checkpoints -> {ckpt_dir}; resuming from {last_ckpt or 'nothing (fresh run)'}")

    # --- score the base model on the holdout sample BEFORE training ---
    # (the adapter is freshly initialized to zero, so this is the base model)
    eval_before = None
    before_json = os.path.join(ckpt_dir, "eval_before.json")
    if last_ckpt and os.path.exists(before_json):
        with open(before_json) as f:          # resumed run: the base-model score was already computed
            eval_before = json.load(f)
    elif args.eval_before:
        eval_before = evaluate(model, processor, hold_df, images_dir, doc_ctx, args.max_pixels,
                               args.eval_pages, args.max_new_tokens, args.seed, "before", args.model_dir)
        with open(before_json, "w") as f:
            json.dump(eval_before, f)

    # --- the curve: score a few holdout pages at the end of every epoch ---
    # The first curve_pages of the same seeded shuffle evaluate() uses, so they are a
    # subset of the before/after sample and every point is on the same pages.
    curve = []
    if eval_before is not None and args.curve_pages > 0:
        ids = sorted(hold_df["page_id"]); random.Random(args.seed).shuffle(ids)
        sub = pd.read_csv(os.path.join(args.model_dir, "eval_before.csv"), keep_default_na=False)
        sub = sub[sub.page_id.isin(ids[:args.curve_pages])]
        if len(sub):
            curve.append({"epoch": 0, "macro_cer": round(macro_cer(sub)[0], 4)})

    class CurveCallback(TrainerCallback):
        def on_epoch_end(self, targs_, state, control, model=None, **kw):
            if args.curve_pages <= 0:
                return
            ep = int(round(state.epoch))
            r = evaluate(model, processor, hold_df, images_dir, doc_ctx, args.max_pixels,
                         args.curve_pages, args.max_new_tokens, args.seed, f"epoch{ep}", args.model_dir)
            if r:
                curve.append({"epoch": ep, "macro_cer": r["macro_cer"]})

    # --- train ---
    # transformers 5 dropped warmup_ratio and reads a warmup_steps value below 1 as a ratio instead
    import dataclasses
    warmup = {"warmup_ratio": 0.03} if "warmup_ratio" in {f.name for f in dataclasses.fields(TrainingArguments)} else {"warmup_steps": 0.03}
    targs = TrainingArguments(
        output_dir=ckpt_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        **warmup,
        weight_decay=0.0,
        bf16=(args.dtype == "bf16"),
        fp16=(args.dtype == "fp16"),
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=5,
        save_strategy="epoch",            # adapter + optimizer state at the end of every epoch
        save_total_limit=1,               # keep only the latest; the adapter is ~100 MB
        report_to="none",
        remove_unused_columns=False,
        dataloader_num_workers=2,
        seed=args.seed,
    )
    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=PageDataset(train_df, images_dir, doc_ctx, args.max_pixels),
        data_collator=Collator(processor),
        callbacks=[CurveCallback()],
    )
    t0 = time.time()
    result = trainer.train(resume_from_checkpoint=last_ckpt)   # None on a fresh run
    train_secs = time.time() - t0
    print(f"training finished in {train_secs / 60:.1f} min; final loss {result.training_loss:.4f}")

    # --- save adapter + processor, and the curves ---
    model.save_pretrained(args.model_dir)
    processor.save_pretrained(args.model_dir)
    loss_curve = [{"step": h["step"], "epoch": round(h["epoch"], 3), "loss": h["loss"]}
                  for h in trainer.state.log_history if "loss" in h]
    with open(os.path.join(args.model_dir, "train_log.json"), "w") as f:
        json.dump({"loss": loss_curve, "holdout_cer_by_epoch": curve,
                   "curve_pages": args.curve_pages, "note": "epoch 0 is the base model; the curve covers the epochs this job ran"}, f, indent=2)

    # --- score the fine-tuned model on the SAME holdout pages ---
    eval_after = evaluate(model, processor, hold_df, images_dir, doc_ctx, args.max_pixels,
                          args.eval_pages, args.max_new_tokens, args.seed, "after", args.model_dir)

    summary = {
        "model_id": args.model_id,
        "categories": args.categories or "all",
        "train_pages": int(len(train_df)),
        "holdout_pages": int(len(hold_df)),
        "epochs": args.epochs,
        "effective_batch": args.batch_size * args.grad_accum,
        "max_pixels": args.max_pixels,
        "lora_r": args.lora_r,
        "curve_pages": args.curve_pages,
        "train_minutes": round(train_secs / 60, 1),
        "final_loss": round(float(result.training_loss), 4),
        "holdout_eval_before": eval_before,
        "holdout_eval_after": eval_after,
    }
    with open(os.path.join(args.model_dir, "train_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

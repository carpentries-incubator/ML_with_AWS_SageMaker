# LoRA fine-tuning on a free GPU: teaching a 3B model to read 1930s German handwriting

[Open this notebook in Google Colab](https://colab.research.google.com/github/carpentries-incubator/ML_with_AWS_SageMaker/blob/main/colab/LoRA-finetuning-on-a-free-GPU.ipynb)

This notebook fine-tunes CHURRO-3B, a 3-billion-parameter vision-language model, to transcribe handwritten German letters, on the free GPU that Google Colab provides. It costs nothing. The whole run, with the model scored before and after, takes about an hour on Colab's T4.

What you will do, in order:

1. Check the GPU and install three libraries.
2. Download the data, 295 labelled pages of 1930s letters in Kurrent script, and look at a page.
3. Measure how many tokens a page is, because that number sets the training time.
4. Work out how many parameters LoRA will train, from the model's own config file.
5. Run a two-minute smoke test, then the real fine-tune, with checkpoints on Google Drive so a disconnect costs one epoch and not the run.
6. Compare the base model and the fine-tuned model on the same held-out pages, read the learning curve, and look at a page the fine-tune fixed.
7. Optionally, transcribe the competition's test pages and make a Kaggle submission.

You need a Google account, and before running anything, a GPU runtime. In Colab's menu choose **Runtime → Change runtime type**, pick **T4 GPU**, and save. The free tier gives a T4 with 16 GB of GPU memory. Sessions end after about 12 hours, or 90 minutes idle, and a GPU is not always available at busy times. Nothing here needs more than that, and the checkpoints below are what make the limits harmless.

## 1. The GPU

`nvidia-smi` names the card and its memory. The second line matters for what comes later. The T4 cannot compute in bf16, the 16-bit format that newer GPUs use for training, so this notebook runs in fp16 and keeps the frozen model in 4-bit. The training script picks that up on its own from the `--dtype auto` flag.

```python
!nvidia-smi --query-gpu=name,memory.total --format=csv
import torch
print("bf16 supported:", torch.cuda.is_bf16_supported(), "(False on a T4, which is why we use fp16 and 4-bit weights below)")
```

Install the three libraries the training script needs beyond what Colab ships. `transformers` loads the model, `peft` adds the LoRA adapter, and `bitsandbytes` stores the frozen weights in 4-bit.

```python
%pip install -q "transformers>=4.56" "peft>=0.14" bitsandbytes accelerate
```

Mount Google Drive. The training script saves a checkpoint at the end of every epoch, and if those live on Drive, a Colab disconnect loses at most one epoch. Rerun the training cell and it resumes from the last checkpoint.

```python
from google.colab import drive
drive.mount("/content/drive")

import os
work = "/content/badger-scribe"                                   # data, code and outputs for this session; gone when the session ends
ckpt_dir = "/content/drive/MyDrive/badger-scribe-checkpoints"     # on Drive, so it survives a disconnect
os.makedirs(work, exist_ok=True); os.makedirs(ckpt_dir, exist_ok=True)
%cd {work}
```

## 2. The code and the data

The training script, the scoring code and a batch transcriber come from the lesson this notebook is adapted from, a Carpentries lesson on machine learning on AWS. Only its `scripts/` folder is needed here.

```python
!git clone -q --depth 1 https://github.com/carpentries-incubator/ML_with_AWS_SageMaker.git lesson
!cp lesson/scripts/badger_common.py lesson/scripts/finetune_churro_lora.py lesson/scripts/churro_transcribe_batch.py .
!ls *.py
```

The data is the German-letter part of [Badger Scribe](https://www.kaggle.com/competitions/badger-scribe), a Kaggle competition built from UW Digital Collections material. The letters are the Anna Seifert correspondence held by the Max Kade Institute, 191 letters and postcards written between 1932 and 1942 by one woman to her son, in Kurrent, the old German handwriting taught in schools until the 1940s. Its letterforms look nothing like today's cursive. An *e* is two short strokes, an *s* has three shapes, and most Germans alive today cannot read it. The Institute's Kurrent Transcription Group transcribed every page by hand. The lesson publishes the data as a GitHub release, three zip files of about 130 MB in all.

```python
release = "https://github.com/carpentries-incubator/ML_with_AWS_SageMaker/releases/download/badger-scribe-data-v1"
for part in ["labels", "images-train", "images-test"]:
    !curl -sSL -o {part}.zip {release}/badger-scribe-v1-{part}.zip && unzip -q -o {part}.zip -d data && rm {part}.zip
!ls data
!ls data/images | wc -l
```

`train.csv` has one row per labelled page, with the page's id, the letter it belongs to, and the transcription. The test pages have no labels. Kaggle keeps those.

```python
import pandas as pd
train_df = pd.read_csv("data/train.csv", keep_default_na=False)   # keep_default_na=False: an empty transcription stays "" instead of becoming NaN
test_df = pd.read_csv("data/test.csv", keep_default_na=False)
print("labelled training pages:", len(train_df), " test pages:", len(test_df))
print(train_df.groupby(["doc_id"]).size().rename("pages per letter bundle"))
train_df.head(3)
```

Look at one page and its transcription. Verbatim means the writer's spelling and punctuation, umlauts and ß as written, Kurrent letterforms mapped to their Latin equivalents, and `#` for a character the transcriber could not read.

```python
from IPython.display import Image, display
row = train_df.iloc[3]
print(row.page_id, "|", row.doc_id)
print(row.text[:600])
display(Image(filename=f"data/images/{row.page_id}.jpg", width=500))
```

## 3. Why fine-tune, and why LoRA

CHURRO-3B already reads old handwriting. Its authors fully fine-tuned Qwen2.5-VL-3B on about 100,000 pages of historical documents and report better accuracy than Gemini 2.5 Pro on historical text recognition ([paper](https://arxiv.org/abs/2509.19768)). It has never seen this writer's hand or this archive's transcription conventions, and no prompt can teach it those. Fine-tuning can. We show it pages with their transcriptions, it computes how wrong its output was, and gradient descent nudges the weights toward our data. Because it already reads Kurrent-like scripts, a few hundred pages are enough.

How to fine-tune is decided by the GPU's memory, and the arithmetic is short. Every parameter the training updates needs 16 bytes of GPU memory, the weight in 16-bit, its gradient, a 32-bit copy of the weight, and the two running averages the Adam optimizer keeps. A parameter that is only run, not updated, needs 2 bytes in 16-bit or half a byte in 4-bit.

| How the 3.75B parameters are used | Bytes per parameter | GPU memory | Fits a 16 GB T4? |
|---|---|---|---|
| **Full fine-tune**, every weight updated | 16 | 60 GB | No. That is a four-GPU machine. |
| **LoRA**, weights frozen in 16-bit, a small adapter trained | 2, plus the adapter | about 12 GB with activations | Only just, and the T4 cannot do bf16. |
| **QLoRA**, weights frozen in 4-bit, the same adapter | 0.5, plus the adapter | about 6 GB with activations | Yes, with room. |

***LoRA***, low-rank adaptation, leaves every weight of the model frozen and trains new small matrices added beside the large ones. For a weight matrix W of shape d × d it adds two thin matrices, B (d × r) and A (r × d), with a rank r of 8 to 64, and the layer computes W·x + B·A·x. Only A and B get gradients, so the 16-bytes-per-parameter cost applies to a few tens of millions of parameters instead of billions. The adapter that comes out is about 100 MB and is loaded on top of the base model at inference. ***QLoRA*** is the same with the frozen weights stored in 4-bit, which is what puts this run on a T4. Nothing about what the model learns changes. The arithmetic the training does is the same, on a compressed copy of the weights.

## 4. How many tokens is a page?

Training time is proportional to how many tokens the model processes, and a page image is tokens too. The model resizes the image to at most `max_pixels` and turns every 28 × 28 block of pixels into one token. The transcription becomes pieces of words, more pieces for German and for old spellings than for English. The count depends on the model's own tokenizer and image processor, so measure it with them. Loading the processor downloads a few megabytes of configuration, not the 7.5 GB of weights.

```python
from transformers import AutoProcessor
import badger_common as bc

processor = AutoProcessor.from_pretrained("stanford-oval/churro-3B", min_pixels=bc.DEFAULT_MIN_PIXELS, max_pixels=bc.DEFAULT_MAX_PIXELS)

image = bc.load_page_image(f"data/images/{row.page_id}.jpg", bc.DEFAULT_MAX_PIXELS)   # resized the way the training script resizes it
prompt = bc.build_prompt("kade_letters")
full = processor.apply_chat_template(bc.full_messages(prompt, row.text), tokenize=False)
enc = processor(text=[full], images=[image], return_tensors="pt")
n_total = enc["input_ids"].shape[1]
n_prompt = processor(text=[processor.apply_chat_template(bc.user_messages(prompt), tokenize=False, add_generation_prompt=True)],
                     images=[image], return_tensors="pt")["input_ids"].shape[1]
n_image = int((enc["image_grid_thw"].prod(dim=1) // 4).sum())   # 14-pixel patches merged 2x2: one token per 28x28 block

print(f"page {row.page_id}: {image.size[0]} x {image.size[1]} px after resizing -> {n_image} image tokens")
print(f"instruction prompt, the same on every page: {n_prompt - n_image} tokens")
print(f"transcription: {len(row.text.split())} words -> {n_total - n_prompt} tokens")
print(f"tokens the model sees for this page: {n_total}")
```

The image is close to the 1,280-token cap whatever the scan's original size, because every scan is larger than a megapixel and gets shrunk to it. The transcription comes out near two tokens per word, where English text is about 1.3. One page is a point, not an average, so count the text tokens of every page with the tokenizer alone, which takes seconds.

```python
text_tokens = train_df.text.map(lambda t: len(processor.tokenizer(t)["input_ids"]))
avg_tokens_per_page = n_image + (n_prompt - n_image) + text_tokens.mean()
print(f"text tokens per page: mean {text_tokens.mean():.0f}, min {text_tokens.min()}, max {text_tokens.max()}")
print(f"average tokens per page: about {avg_tokens_per_page:.0f}")
print(f"tokens per pass over the training pages: about {avg_tokens_per_page * len(train_df) * 0.85 / 1e6:.2f} million")   # 85%: the script holds 15% of pages out for scoring
```

That last number, times the number of passes, is the work the GPU does. Halve `max_pixels` and the image tokens halve with it, and so does the training time, at some cost in accuracy on small handwriting.

## 5. How many parameters will train?

The adapter rank decides how many parameters LoRA trains, and that count is worth knowing before the run for two reasons. The script refuses to train if more than 5% of the model turns out to be trainable, because that would mean the base weights were not frozen, and the number appears in the training log as `trainable params`, so working it out now tells you whether the run is doing what you asked.

LoRA attaches to seven weight matrices in every layer of the language model, the four attention projections and the three in the MLP. An adapter of rank r on a matrix of shape in × out adds r × (in + out) parameters. Add that up over the seven, multiply by the number of layers, and you have the count. The widths come from the model's `config.json`, which for any model on Hugging Face is under **Files and versions** on the model page.

```python
import json
from huggingface_hub import hf_hub_download

with open(hf_hub_download("stanford-oval/churro-3B", "config.json")) as f:
    cfg = json.load(f)
text = cfg.get("text_config", cfg)            # newer configs keep the language model's sizes under text_config

layers = text["num_hidden_layers"]            # how many times the seven matrices repeat
hidden = text["hidden_size"]                  # the width of the model's main vector
inter = text["intermediate_size"]             # the MLP's wider middle layer
kv = hidden * text["num_key_value_heads"] // text["num_attention_heads"]   # k_proj and v_proj are narrower: several heads share one key and value

projections = {                               # (in, out) of each matrix LoRA attaches to, in one layer
    "q_proj": (hidden, hidden), "k_proj": (hidden, kv), "v_proj": (hidden, kv), "o_proj": (hidden, hidden),
    "gate_proj": (hidden, inter), "up_proj": (hidden, inter), "down_proj": (inter, hidden),
}
total_params = 3.75e9                         # the whole model, vision tower included
for lora_r in (8, 16, 32):
    trainable = sum(lora_r * (i + o) for i, o in projections.values()) * layers
    print(f"rank {lora_r:>2}: {trainable / 1e6:5.1f}M trainable parameters, {100 * trainable / total_params:.2f}% of the model")
```

Rank 16, used below, trains about 30 million parameters, under 1% of the model. Their gradients and optimizer state at 16 bytes each are about half a gigabyte, which is why the run fits beside the frozen weights.

## 6. The fine-tune

### Read the script first

Open `finetune_churro_lora.py` in Colab's file browser (the folder icon on the left) before you run it. It is about 400 lines. The parts to find: `split_pages`, which holds 15% of the pages out of training with a fixed seed so every run scores the same pages; `LoraConfig`, whose `target_modules` pattern names the seven projections and excludes the vision tower; the two safeguard checks right after `get_peft_model`; and `evaluate`, which transcribes holdout pages and computes the character error rate (CER), edits per reference character, which is the competition's metric.

### A smoke test

Every setting below reaches the script as a command-line flag. Before the real run, a two-minute version, 20 pages for one pass with three pages scored, to catch a bad path or a missing library while it is cheap. It also downloads the 7.5 GB of weights, once, into Colab's Hugging Face cache.

```python
!python finetune_churro_lora.py \
    --train_dir data --model_dir output/smoke --output_data_dir output/smoke-data --checkpoint_dir output/smoke-ckpt \
    --dtype auto --load_in_4bit 1 \
    --max_train_pages 20 --epochs 1 --eval_pages 3 --curve_pages 0 --grad_accum 4
```

Look for four lines in the output. `pages: 295 total -> 20 train / 44 holdout`, the split, with the 20 training pages this test asked for and the 44 holdout pages every run scores. `loading stanford-oval/churro-3B in fp16, frozen weights in 4-bit`, the precision chosen for this GPU. `trainable params: ...`, which should be about 30 million, the number from section 5. And `safeguards: adapter is ... (0.80%), vision tower frozen`, the two checks passing.

### The real run

The same script on all 251 training pages for two passes, scoring 15 holdout pages before and after and 8 of them at the end of each pass. On a T4 expect about 25 to 40 minutes of training and about 30 minutes of scoring, an hour or so in all. The checkpoints go to Drive. If Colab disconnects, run this cell again with the same `run_name` and it resumes from the last completed epoch.

```python
run_name = "churro-lora-2ep"          # change it to start a fresh run instead of resuming one
os.makedirs(f"{ckpt_dir}/{run_name}", exist_ok=True)   # the script writes checkpoints here only if the folder exists

!python finetune_churro_lora.py \
    --train_dir data --model_dir output/{run_name} --output_data_dir output/{run_name}-data --checkpoint_dir {ckpt_dir}/{run_name} \
    --dtype auto --load_in_4bit 1 \
    --epochs 2 --lr 2e-4 --lora_r 16 --batch_size 1 --grad_accum 4 \
    --eval_pages 15 --eval_before 1 --curve_pages 8
```

While it runs, the log shows the base model's CER on the holdout pages first, then the Trainer's loss every five steps, then a CER point at the end of each epoch, then the fine-tuned model's CER on the same pages.

## 7. What changed

The script wrote everything into the output folder: the adapter, the per-page scores before and after, the curves and a summary.

```python
out = f"output/{run_name}"
print(sorted(os.listdir(out)))
summary = json.load(open(f"{out}/train_summary.json"))
print(json.dumps(summary, indent=2))
```

Both evaluations ran on the same 15 held-out pages, so the difference is the fine-tune and nothing else.

```python
before = pd.read_csv(f"{out}/eval_before.csv", keep_default_na=False)
after = pd.read_csv(f"{out}/eval_after.csv", keep_default_na=False)

comparison = before[["page_id", "cer"]].merge(after[["page_id", "cer"]], on="page_id", suffixes=("_base", "_lora"))
comparison["gain"] = comparison.cer_base - comparison.cer_lora   # positive: the fine-tune helped on that page
print(comparison.sort_values("gain", ascending=False).round(3).to_string(index=False))
print(f"\nmean CER  base: {comparison.cer_base.mean():.3f}   fine-tuned: {comparison.cer_lora.mean():.3f}")
```

Look at the page that improved most. The fine-tuned model should have picked up the conventions, umlauts and ß as written, no modernised spelling, Kurrent *s* and *e* read correctly, as well as this writer's hand.

```python
best = comparison.sort_values("gain", ascending=False).iloc[0].page_id
b, a = before.set_index("page_id"), after.set_index("page_id")
print("REFERENCE\n", b.loc[best, "text_ref"][:500])
print("\nBASE\n", b.loc[best, "text_pred"][:500])
print("\nFINE-TUNED\n", a.loc[best, "text_pred"][:500])
display(Image(filename=f"data/images/{best}.jpg", width=500))
```

Before and after says whether the fine-tune helped. The curve says whether two passes was the right amount. The script recorded the training loss every five steps and scored eight holdout pages at the end of each pass, with the base model as epoch 0.

```python
import matplotlib.pyplot as plt

log = json.load(open(f"{out}/train_log.json"))
loss = pd.DataFrame(log["loss"]); curve = pd.DataFrame(log["holdout_cer_by_epoch"])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.8))
ax1.plot(loss.epoch, loss.loss, marker="."); ax1.set(xlabel="epoch", ylabel="training loss", title="Training loss")
ax2.plot(curve.epoch, curve.macro_cer, marker="o"); ax2.set(xlabel="epoch (0 = base model)", ylabel="holdout CER", title=f"Holdout CER on {log['curve_pages']} pages")
ax2.set_xticks(curve.epoch); plt.tight_layout(); plt.show()
print(curve.to_string(index=False))
```

Read the right-hand plot, not the left. Training loss almost always keeps falling, because it is measured on the pages the model is being fitted to. The holdout CER is the number that matters, and it has three shapes. Still falling at the last epoch means another pass would probably help. Flat means the data has given what it can. Rising while the loss still falls means the model is memorising the training pages, so fewer passes or more data. Eight pages is a noisy estimate, so read the trend, not the third decimal.

To try a change, give the run a new name and relaunch section 6's real run. Four passes instead of two, a rank of 32, or `--max_pixels 2560*28*28` for sharper images at twice the time, each changes one thing, and the before-and-after table says what it bought.

## 8. Optional: a Kaggle submission

The batch transcriber uses the adapter to transcribe any list of pages. Here it writes a prediction for each of the 141 test pages, in the competition's submission format, which takes about 40 minutes on a T4. The competition allows two submissions a day, and this step is the only one that needs a Kaggle account.

```python
sub_dir = "submission"
os.makedirs(f"{sub_dir}/data", exist_ok=True)
test_df[["page_id", "doc_id", "category"]].assign(text="").to_csv(f"{sub_dir}/data/pages.csv", index=False)
!cp data/metadata.csv {sub_dir}/data/ && ln -sfn {work}/data/images {sub_dir}/images

!python churro_transcribe_batch.py \
    --model_id stanford-oval/churro-3B --adapter_dir output/{run_name} --dtype auto \
    --input_dir {sub_dir} --output_dir {sub_dir}/out --pages_csv pages.csv --batch_size 2

preds = pd.read_csv(f"{sub_dir}/out/predictions.csv", keep_default_na=False)
submission = test_df[["page_id"]].merge(preds, on="page_id", how="left").fillna({"text": ""})
assert len(submission) == len(test_df) and submission.text.str.len().gt(0).all(), "every test page needs a prediction"
submission.to_csv("submission_churro_lora.csv", index=False)
print("wrote submission_churro_lora.csv,", len(submission), "rows; download it from the file browser and upload it on the competition's Submit page")
```

## What to take away

- Fine-tuning is training that starts from a trained model, so hundreds of examples do what millions would from scratch.
- The GPU's memory decides the method. A full fine-tune of a 3B model needs 60 GB. LoRA trains under 1% of the parameters and needs 12 GB. QLoRA stores the frozen weights in 4-bit and needs about 6 GB, which is what fits a free T4.
- Measure before you train. Tokens per page set the time, the adapter rank sets the trainable parameters, and both come from the model's own processor and config file.
- Smoke-test on a slice first, checkpoint every epoch, and compare the model before and after on the same held-out pages with the task's own metric. The holdout curve, not a rule of thumb, says how many passes were worth it.
- The same run on a rented cloud GPU costs about a dollar and finishes in half the time. The lesson this notebook comes from shows how to run it as a SageMaker job, with the cost estimated before the launch and read back after it.

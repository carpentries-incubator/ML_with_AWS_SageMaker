# LoRA fine-tuning on CHTC

> **In development.** This guide has not been run end to end on CHTC yet. The submit files follow CHTC's documentation, and the first tested draft is planned for 1 December 2026. If something fails before then, [open an issue](https://github.com/carpentries-incubator/ML_with_AWS_SageMaker/issues) and attach the job's `.log` and `.err` files.

This guide runs the fine-tuning episode of the [ML on AWS lesson](https://carpentries-incubator.github.io/ML_with_AWS_SageMaker/09_Finetuning-BadgerScribe.html) on UW-Madison's [Center for High Throughput Computing](https://chtc.cs.wisc.edu/) (CHTC) instead of on AWS. The model, the data, the training script and the LoRA adapter it produces are the same as in the episode. What changes is where the GPU comes from and how the job is described.

CHTC is free for UW-Madison researchers. It runs HTCondor, a job scheduler. You describe a job in a text file and submit it. HTCondor finds a machine with the resources the job asked for, copies your files to it, runs your program, and copies the results back. Nothing is billed. The costs are the time a job waits in the queue and the work of packaging a job so it runs without you on a machine you never log in to.

Getting an account takes a few days. Building the container and staging the model weights takes an afternoon the first time. After that, each fine-tuning run is about 20 minutes of GPU time plus its wait in the queue.

The files in this folder:

| File | What it is |
|---|---|
| [`churro.def`](churro.def) | The recipe for the software container, PyTorch 2.8 plus the Python packages the training script imports |
| [`build.sub`](build.sub) | The interactive job that builds the container |
| [`fetch_model.sub`](fetch_model.sub), [`fetch_model.sh`](fetch_model.sh) | A small job that downloads the 7.5 GB of CHURRO-3B weights into your staging space |
| [`finetune.sub`](finetune.sub), [`finetune.sh`](finetune.sh) | The GPU job, and the script it runs inside the container |
| [`read_results.py`](read_results.py) | Reads what the job sends back and prints the before-and-after comparison |

The training script, [`finetune_churro_lora.py`](../../scripts/finetune_churro_lora.py), and its helper [`badger_common.py`](../../scripts/badger_common.py) come from the lesson's [`scripts/`](../../scripts) folder unchanged.

## Three places to run this fine-tune

The same fine-tune runs in three places. This guide is the CHTC column. The others are the [lesson's episode](https://carpentries-incubator.github.io/ML_with_AWS_SageMaker/09_Finetuning-BadgerScribe.html) on AWS and a [Colab notebook](https://colab.research.google.com/github/carpentries-incubator/ML_with_AWS_SageMaker/blob/main/finetune_extras/colab/LoRA-finetuning-on-a-free-GPU.ipynb).

| | [AWS SageMaker](https://carpentries-incubator.github.io/ML_with_AWS_SageMaker/09_Finetuning-BadgerScribe.html) | [Google Colab](https://colab.research.google.com/github/carpentries-incubator/ML_with_AWS_SageMaker/blob/main/finetune_extras/colab/LoRA-finetuning-on-a-free-GPU.ipynb) | [CHTC](https://github.com/carpentries-incubator/ML_with_AWS_SageMaker/blob/main/finetune_extras/chtc/LoRA-finetuning-on-CHTC.md) |
|---|---|---|---|
| Who can use it | Anyone with an AWS account and a card or credits | Anyone with a Google account | UW-Madison researchers and students, with a faculty sponsor |
| Cost of this run | About `$0.75` | Free | Free |
| GPU | A10G, 24 GB, bf16 | T4, 16 GB, no bf16, so the frozen weights go to 4-bit | L40S, A100, H100 or H200, 48 to 141 GB, bf16; NVLink on the A100, H100 and H200 servers |
| Time for this run | 33 minutes, starts in a minute | About an hour and a half, starts at once if a GPU is free | About 20 minutes, after a queue wait of minutes to hours |
| Largest LoRA fine-tune | No hard limit, only cost. 7B on the lesson's `$1.41`/h card, 70B at `$13`/h | about 7B, in 4-bit | about 50B in bf16 on an H200, a 70B in 4-bit on one 80 GB card, larger across a server's cards |
| Largest full fine-tune | No hard limit, only cost. 3B at `$7`/h, 70B at `$127`/h | none, a 3B needs 52 GB | a 7B on one H200, a 70B across the H100 or H200 servers' cards |
| What you learn beyond the fine-tune | Estimating and reading a bill, tags, job guard rails | Fitting a model into a small GPU | Describing a job for a scheduler, containers, staging data |

For fine-tuning, CHTC is the first choice for UW-Madison researchers. It is free and its GPUs are large. Colab is for practicing LoRA with nothing to set up. The cloud has the newest and fastest GPUs, which a fine-tune on a large dataset sometimes needs, and it bills by the hour, so use it when the other two cannot do the job.

None of the three runtime limits is a wall. The training script writes a checkpoint at the end of every epoch and resumes from it, so a run longer than a CHTC job lane, a Colab session or an AWS runtime cap is several runs that each continue where the last one stopped. The submit file here already tells HTCondor to carry the checkpoints between runs. Section 7 explains those lines.

## 1. Get an account

CHTC accounts are for research and approved courses. You need an active NetID and a faculty sponsor, usually your PI or the course instructor.

1. Fill in the [account request form](https://userapp.chtc.wisc.edu/forms/user-applications/create/). Describe the work: fine-tuning a 3-billion-parameter vision-language model on a few hundred page images, one GPU with about 20 GB of memory, runs under an hour. The description decides which resources your account is set up for, so say that you need a GPU.
2. An automated email from chtc@cs.wisc.edu arrives within hours. A facilitator follows up within two or three business days, and a new research group gets a meeting first. If you hear nothing after three business days, email them.
3. At the meeting, or in the form, ask for two things beyond the default account: access to the GPU Lab, and a directory in `/staging`. This guide needs both.

Your welcome email names the access point you log in to.

## 2. Log in

CHTC's HTC system has two access points, `ap2001.chtc.wisc.edu` and `ap2002.chtc.wisc.edu`. Use the one in your welcome email. You need to be on the campus network or the VPN, and Duo.

```bash
ssh <netid>@ap2001.chtc.wisc.edu
```

The access point is where you keep files and submit jobs. Jobs run on other machines. Nothing in this guide runs Python on the access point except the small results reader at the end. Once a job is submitted you can log out. It runs and returns its output on its own.

## 3. Where files live

There are two places for files, and size decides which one.

| | `/home/<netid>` | `/staging/<n>/<netid>` |
|---|---|---|
| For | Many small files, under 1 GB each | Few large files, over 1 GB each |
| Quota | 40 GB | 100 GB, 1,000 items |
| Submit jobs from here | Yes | No |
| A job reads a file from here with | `transfer_input_files = file.txt` | `transfer_input_files = osdf:///chtc/staging/<n>/<netid>/file` |

`<n>` is the first letter of your NetID, so `bbadger` has `/staging/b/bbadger`. This guide keeps the code, the data archive and the submit files in `/home`, and the container and the model weights in `/staging`.

A job does not read either location directly. HTCondor copies the files the submit file lists into a scratch directory on the execute machine, the program runs there, and HTCondor copies back the files the submit file names as output. Every line in the submit files follows from that.

## 4. Get the code and the data

In your home directory, clone the lesson and copy the two Python files and this folder's job files into a working directory.

```bash
cd ~
rm -rf ML_with_AWS_SageMaker && git clone -q --depth 1 https://github.com/carpentries-incubator/ML_with_AWS_SageMaker.git   # a fresh copy, so a rerun gets the current scripts
mkdir -p badger-scribe && cd badger-scribe
cp ../ML_with_AWS_SageMaker/scripts/finetune_churro_lora.py ../ML_with_AWS_SageMaker/scripts/badger_common.py .
cp ../ML_with_AWS_SageMaker/finetune_extras/chtc/* .
ls
```

The data is the German-letter part of [Badger Scribe](https://www.kaggle.com/competitions/badger-scribe), a Kaggle competition built from UW Digital Collections material. The lesson publishes it as a GitHub release, three zip files of about 130 MB in all. Download them, unpack them into one folder, and pack that folder as one archive. HTCondor moves one file faster than 436 image files.

```bash
release=https://github.com/carpentries-incubator/ML_with_AWS_SageMaker/releases/download/badger-scribe-data-v1
for part in labels images-train images-test; do
  curl -sSL -o $part.zip $release/badger-scribe-v1-$part.zip && unzip -q -o $part.zip -d data && rm $part.zip
done
ls data; ls data/images | wc -l
tar czf badger-scribe-data.tar.gz data
ls -lh badger-scribe-data.tar.gz     # about 130 MB, under a gigabyte, so it can stay in /home and transfer from there
```

`train.csv` has one row per labeled page, with the page's id, the letter it belongs to and the transcription. The training script holds 15% of the pages out of training, with a fixed seed, and scores them before and after. The lesson's episode describes the data and shows a page. Here the pages go straight into the job.

## 5. Build the container

A job runs on a machine with an operating system and little else. The software comes along in a container, a single file holding a Linux environment with PyTorch and the packages the script imports. [`churro.def`](churro.def) is its recipe. It starts from the public PyTorch 2.8 image with CUDA 12.8, the same versions as the lesson's SageMaker container, and installs `transformers`, `peft` and a few small packages on top.

Containers are built in an interactive job on a build machine, not on the access point. [`build.sub`](build.sub) describes that job. Submit it with `-i`, wait for the prompt to change to the build machine, and run the build there.

```bash
condor_submit -i build.sub
# wait; the prompt changes when a build machine is yours
apptainer build churro.sif churro.def
apptainer shell -e churro.sif            # test it: the prompt changes again, into the container
python -c "import torch, transformers, peft; print(torch.__version__, transformers.__version__, peft.__version__)"
exit                                     # leave the container
exit                                     # leave the build job; churro.sif comes back with you
```

The build downloads several gigabytes and takes 10 to 20 minutes. If it dies, the usual cause is disk, and the fix is a larger `request_disk` in `build.sub`. The result, `churro.sif`, is a few gigabytes, so it goes to staging, where every job reads it from.

```bash
mv churro.sif /staging/<n>/<netid>/
```

## 6. Stage the model

The base weights are 7.5 GB, too big for `/home` and too slow to download for every run. [`fetch_model.sub`](fetch_model.sub) is a small CPU job that downloads them from Hugging Face once, inside the container you just built, packs them as one tar file, and writes the tar to your staging directory. Read both files first. The submit file's `transfer_output_remaps` line sends the job's output somewhere other than the submit directory, and the shell script is what the job runs.

Edit `NETID` at the top of `fetch_model.sub`, then submit and watch.

```bash
condor_submit fetch_model.sub
condor_q                                 # your jobs in the queue
condor_tail -f <jobid>                   # the job's output as it runs; Ctrl-C to stop watching
ls -lh /staging/<n>/<netid>/             # when the job is done: churro.sif and churro-3B.tar
```

A job's history is in its `.log` file. Open `fetch_model.log` when the job finishes and read the events in order: submitted, started on a machine, files transferred, finished, and how much memory and disk it used against what it asked for. The `.out` file has what the script printed, and the `.err` file has anything it wrote to standard error.

## 7. Read the job before you submit it

Open [`finetune.sub`](finetune.sub) and [`finetune.sh`](finetune.sh) together. The submit file describes the machine and the files, which the lesson's estimator did on SageMaker. The shell script runs the training script. Every setting has a comment on the line above it. HTCondor reads a `#` at the end of a setting's line as part of the value, so a comment never shares a line with a setting in a submit file. These are the ones to understand.

**The arguments line** holds the script's command-line flags, the same hyperparameters as the lesson's SageMaker job: two passes, a LoRA rank of 16, 15 holdout pages scored before and after, and 8 of them scored at the end of every pass. To change a run, change this line, not the script.

**What one training step does.** The Trainer takes one page at a time, since `--batch_size` is 1. It runs the page image, the instruction prompt and the human transcription through the model, and measures the loss on the transcription tokens only, since the collator masks the prompt and image tokens out of it. The gradient of that loss flows back through the frozen model into the adapter matrices, the only parameters with gradients. After `--grad_accum` pages, four here, the summed gradient makes one AdamW update to the adapter, so a pass over 251 pages is about 63 updates and two passes about 126. The learning rate ramps up over the first 3% of updates and then falls along a cosine curve to zero at the last one. Every update costs a forward and a backward pass over about 1,500 tokens.

| Setting | Value here | What it does | What changing it costs or buys |
|---|---|---|---|
| `--epochs` | 2 | Passes over the training pages. Training time scales with it, and the holdout curve at the end of each pass says whether the next one helped. | A third pass is 50% more training time. The homework runs four. |
| `--lr` | 2e-4 | The size of each update to the adapter. LoRA uses about ten times a full fine-tune's rate (~1e-5), because the adapter starts at zero and has few parameters. | Too high and the loss climbs or jumps around. Too low and two passes do not get far. |
| `--lora_r` | 16 | The adapter rank, the inner width of the two thin matrices. The trainable parameter count scales with it, about 30 million at 16. | 32 doubles the trainable parameters and the adapter's memory, and on a few hundred pages tends to overfit sooner. |
| `--lora_alpha` | 32 (default) | Scales the adapter's output by alpha ÷ r, 2 here. The script's default, not set in the run. | Keep alpha at twice r when changing r, or the effective learning rate of the adapter changes with it. |
| `--lora_dropout` | 0.05 (default) | Dropout on the adapter's input, a little regularization against a few hundred pages. | Rarely worth changing. |
| `--batch_size` | 1 | Pages per forward pass per GPU. A page is about 1,500 tokens with its image, so one page at a time is what fits beside the model on a 24 GB card. | 2 needs roughly twice the activation memory and does not fit the lesson's card. |
| `--grad_accum` | 4 | Pages whose gradients are summed before one update. The effective batch is `batch_size` × `grad_accum`. | 8 halves the number of updates per pass, each a little smoother. The script's default is 8; the lesson uses 4 so that 251 pages give enough updates in two passes. |
| `--max_pixels` | 1,003,520 (1280 × 28 × 28) | The page is resized to at most this many pixels, and every 28 × 28 block is one token, so about 1,280 image tokens a page. | Halving it halves the image tokens and the training time, and loses detail on small handwriting. Doubling it doubles both. |
| `--eval_pages` | 15 | Holdout pages transcribed and scored, before training when `eval_before` is 1 and again after. | Scoring is 20 to 70 seconds a page, so 15 pages is 5 to 15 minutes each time. 40 pages, the homework's choice, gives a steadier number at three times the scoring time. |
| `--eval_before` | 1 | Score the base model on the same pages before training, so the before-and-after comparison is on identical pages. | 0 skips it and saves the time. Then there is nothing to compare against. |
| `--curve_pages` | 8 | Of the 15 holdout pages, the ones scored again at the end of every pass, which draws the holdout CER curve. | 0 skips the curve. More pages make the curve steadier and each pass longer. |
| `--holdout_frac` | 0.15 (default) | The fraction of pages, within each category, held out of training with a fixed seed. 44 of 295 here. | Changing it changes which pages are scored, so runs with different values are not comparable. |
| `--seed` | 42 (default) | Fixes the holdout split, the evaluation sample and the training order, so every run scores the same pages. |  |
| `--max_new_tokens` | 1024 (default) | The most tokens the model may generate when transcribing a page during scoring. | A page is about 200 text tokens. The cap matters only when the model rambles, which the base model does on pages it cannot read. |
| `--dtype` | bf16 (default) | The number format for the computation. Every CHTC card from the A100 up supports bf16, so the default holds. | `auto` picks fp16 on an older card, and `--load_in_4bit 1` stores the frozen weights in 4-bit if a card is too small, which none of these are. |

**`transfer_input_files`** names everything the job receives: the two Python files and the data archive from `/home`, and the model tar from staging through its `osdf:///` address. The script sees nothing else. `finetune.sh` unpacks the two archives in the job's scratch directory before training starts.

**`transfer_output_files`** names what comes back, the adapter archive the shell script packs and the folder of scores and curves. HTCondor returns new files in the job's top-level directory by default, but not folders, so the folder has to be listed.

![GPU memory per parameter: inference, a LoRA fine-tune, and a full fine-tune](https://carpentries-incubator.github.io/ML_with_AWS_SageMaker/fig/memory-per-parameter.svg)

*GPU memory per parameter, the lesson's memory rule.*

**The GPU request** turns the lesson's memory rule into a resource request. The rule gives about 12 GB for a 3B LoRA run, so `gpus_minimum_memory = 20000` asks for a card with at least 20 GB, with room for the before-and-after scoring. `gpus_minimum_capability = 8.0` asks for an Ampere or newer GPU, which bf16 needs. On a card this size there is no reason to compress anything. The frozen base weights stay in bf16 and the adapter trains in bf16, as in the lesson's SageMaker job, so the script's default precision is right here and the Colab version's 4-bit setting is not needed.

**`+GPUJobLength = "short"`** is the lane for jobs under 12 hours. The lesson's SageMaker job took 32 minutes on an A10G including the scoring, and CHTC's GPUs are faster. Short jobs can use two thirds of the GPU Lab's cards, medium jobs (under 24 hours) one third, and long jobs (up to seven days) at most four GPUs per user. Ask for the shortest lane the job fits and it starts sooner.

**`+is_resumable = true`** lets the job run on CHTC's backfill GPUs as well as the GPU Lab's. Research groups own GPU servers at CHTC, and when the owners are not using them, other jobs may run there. A backfill job can be interrupted when the owner needs the card, so CHTC asks for this setting on jobs that finish within a few hours or checkpoint at least that often. This job does both. Backfill is most of CHTC's GPU capacity and CHTC asks everyone to use it wherever a job can, so leave the line in.

**`checkpoint_exit_code = 85` and `transfer_checkpoint_files = output/ckpt`** limit what an interruption costs to one epoch. The training script writes a checkpoint to `output/ckpt` at the end of every epoch, and when it starts and finds one there, it resumes from it. `finetune.sh` runs the script under a two-hour timeout and exits with code 85 when the time is up. HTCondor treats exit code 85 as a request to save the checkpoint files and put the job back in the queue, and the next run starts from the checkpoint. An interruption on a backfill GPU is handled the same way, from the checkpoint files saved at the last exit 85. This run finishes in 20 minutes and never reaches the timeout, and a longer run on your own data gets the protection without any change. The setup follows CHTC's checkpointing guide, which also explains how to change the timeout.

**`request_memory` and `request_disk`** are the machine's RAM and scratch disk, not the GPU's. Loading 7.5 GB of weights peaks CPU memory at about twice that, so 32 GB. The disk holds the model tar and its unpacked copy, the data, the container's working files and the outputs, so 40 GB. A job that exceeds its request is held or killed, and the `.log` file says by how much. Raise the number and resubmit.

To see which GPUs exist and which are free now:

```bash
condor_status -compact -constraint 'TotalGpus > 0'
condor_status -constraint 'Gpus > 0' -af Machine GPUs_DeviceName GPUs_GlobalMemoryMb GPUs_Capability | sort | uniq -c
```

## 8. Submit the job and watch it run

Edit `NETID` in `finetune.sub`, then:

```bash
condor_submit finetune.sub
condor_q
```

The job sits idle until a GPU matching the request is free, which takes minutes at a quiet time and longer when the lab is busy. If it idles for more than an hour, `condor_q -better-analyze <jobid>` says how many machines match each requirement, which shows whether the request is one nothing can satisfy.

Once it runs, `condor_tail -f <jobid>` shows the script's output. The lines to find are the ones the lesson's episode reads in the SageMaker log, in this order:

- `pages: 295 total -> 251 train / 44 holdout`, the split.
- `trainable params: ... || trainable%: 0.80`, about 30 million parameters, under 1% of the model, which the lesson's episode works out from the model's config file.
- `safeguards: adapter is ... vision tower frozen`, the script's two checks passing.
- `[eval before] 15 holdout pages ... macro CER ...`, the base model's score.
- The Trainer's loss every five steps, an `[eval epoch1]` line at the end of each pass, then `[eval after]` on the same 15 pages.

On an A100 or an L40 expect about 5 minutes of training and 10 to 15 minutes of scoring, half the SageMaker job's time on its A10G. When `condor_q` shows nothing, the job is done and `adapter.tar.gz` and `output/data` are in your working directory.

## 9. Read the results

[`read_results.py`](read_results.py) unpacks the archive, prints the summary and the per-page character error rate before and after on the same 15 held-out pages, shows the page that improved most, and saves the loss and holdout-CER curves as a picture. It needs pandas and matplotlib. Run it inside the container on the access point, which is light enough for that, or copy `adapter.tar.gz` to your own machine.

```bash
apptainer exec /staging/<n>/<netid>/churro.sif python read_results.py adapter.tar.gz
```

Read the output the way the lesson's episode reads the SageMaker run. The mean CER before and after on the same pages measures the fine-tune's effect and nothing else. The page that improved most shows what was learned: this writer's hand, and the archive's conventions, such as umlauts and ß as written and Kurrent letterforms read correctly. The holdout curve, not the training loss, says whether two passes was the right number. If it is still falling, another pass would help. If it is flat, stop. If it rises while the loss keeps falling, the model is memorizing the training pages.

### How many passes to plan

The number of passes is fixed before a run starts, because the cost cap and the learning-rate schedule are both laid out over it. The cosine schedule decays to zero at the planned end, so a run cannot be extended by a pass without changing the schedule it already followed. The common practice is not to guess a number but to pick a budget, score the holdout at the end of every pass inside it, and keep the best pass.

- **A few hundred to a few thousand examples**, as here: plan 3 to 5 passes. One pass rarely settles, and ten usually memorizes the training pages.
- **Tens of thousands of examples and up**: 1 to 3 passes. A second pass over that much data often helps less than it costs.
- **Keep the best checkpoint, not the last.** The script saves one per pass. Hugging Face's `Trainer` can reload the best automatically with `load_best_model_at_end=True`, when `save_strategy` and `eval_strategy` are both `epoch`.
- **Early stopping**, ending the run when the holdout has not improved for a set number of passes, is `EarlyStoppingCallback` in `Trainer`. It is used less with large models than with small ones, because each evaluation costs minutes of GPU time and because stopping early leaves the cosine schedule unfinished. Most teams plan the budget and pick the best pass afterwards.

Here a pass is a few minutes of GPU time and costs nothing, so the budget is set by the queue and the job lane, not by a bill. Four passes with the scoring is still a short-lane job. The epoch count is on the submit file's arguments line.

## 10. What CHTC's GPUs can do

Nothing here is billed, so what matters is which GPUs are in the pool and how long you may hold one, because that sets the model sizes you can work with. The inventory changes as cards are added, so read it from the scheduler, which lists every GPU type with its memory:

```bash
condor_status -constraint 'Gpus > 0' -af GPUs_DeviceName GPUs_GlobalMemoryMb | sort | uniq -c
```

CHTC publishes the inventory behind that command on its [GPU jobs page](https://chtc.cs.wisc.edu/uw-research-computing/gpu-jobs), with the count of each card and whether it is in the open pool or backfill. As of late 2026 the open pool has 34 H200s with 141 GB, 27 L40s and 22 L40Ss with 48 GB, 24 A100s with 80 or 40 GB, 12 RTX PRO 6000s with 96 GB, 8 H100s with 80 GB, and a few dozen older cards with 11 to 24 GB. The backfill pool, the GPUs research groups own, is larger, and `+is_resumable = true` in the submit file is what lets a job use it. The lanes and the checkpointing protocol are on the same page and on CHTC's [checkpointing guide](https://chtc.cs.wisc.edu/uw-research-computing/checkpointing).

Two things about the cards matter for fine-tuning beyond their memory. Every card from the A100 up has capability 8.0 or newer, so bf16 training works on all of them. And the cards in one server are connected in one of two ways. In the A100, H100 and H200 servers the GPUs are mounted on one board and joined by NVLink, NVIDIA's direct GPU-to-GPU link, which is how data-center AI servers are built and what a multi-GPU job wants. The L40, L40S and RTX PRO 6000 are plug-in cards that talk through the PCIe bus, which is several times slower.

The lesson's memory rules turn GPU memory into model sizes. Inference needs 2 bytes per parameter in bf16. A LoRA fine-tune needs about 2.5 with the frozen weights in bf16, or about 1 with them in 4-bit, the QLoRA setup the Colab version uses. A full fine-tune needs 16. Each needs a few gigabytes of working room on top.

| GPU memory | Cards | Inference, bf16 | LoRA fine-tune | Full fine-tune |
|---|---|---|---|---|
| 40 GB | A100 40 GB, A100 MIG slices | up to about 15B | up to about 12B, about 35B in 4-bit | nothing; a 3B model needs 52 GB |
| 48 GB | L40, L40S | up to about 20B | up to about 15B, about 40B in 4-bit | nothing |
| 80 GB | A100 80 GB, H100 | up to about 35B, or a 70B in 4-bit | up to about 27B, about 70B in 4-bit | a 3B model, just |
| 96 GB | RTX PRO 6000 | up to about 40B | up to about 35B, a 70B in 4-bit | a 3B model |
| 141 GB | H200 | up to about 65B | up to about 50B, a 70B in 4-bit with room | a 3B model easily, a 7B model, just |
| several cards in one server, model sharded across them | the A100, H100 and H200 servers, joined by NVLink | a 70B in bf16 | a 70B in bf16 | a 7B on two 80 GB cards, a 70B on many |

The last row is for a model that does not fit one card, and how well it works depends on the server. On the A100, H100 and H200 servers the cards share NVLink and a sharded job runs at close to the card count. On the L40S and RTX PRO 6000 servers the cards share a PCIe bus, and the lesson's PyTorch episode estimated a speedup of 3.2 on four cards for a full fine-tune's gradient exchange over PCIe, less for a smaller batch.

Stay on one card whenever the memory rule says the job fits. This job needs 12 GB, so every card from the A100 up holds it with room to spare. The submit file's 20 GB minimum and capability 8.0 match all of those, so the job starts soonest. A 7B LoRA, the largest the lesson's cost section considers on one card, needs 22 GB and also fits everywhere here. A full fine-tune of CHURRO itself, which the lesson prices at `$6,300` on AWS, needs the 80 GB and larger cards and costs nothing here except its wait in the queue.

How long you may hold a GPU is the job-length lane. Short jobs, under 12 hours, may use two thirds of the lab's cards at once. Medium jobs, under 24 hours, one third. Long jobs, up to seven days, at most four GPUs per user. This run takes about 20 minutes, so it goes in the short lane and competes for the most cards. A seven-day job holds a card nobody else can use, which is why the long lane is narrow and why checkpointing matters once a run is long.

What a run used is in the scheduler's history, the same numbers the lesson read from a job's billable seconds on AWS. Use them to fit the next request to the job:

```bash
condor_history <netid> -limit 3 -af:h ClusterId RemoteWallClockTime MemoryUsage DiskUsage MachineAttrGPUs_DeviceName0
```

## 11. Scale up

The lesson's rules for a bigger run hold here, in CHTC's terms.

- **A longer run goes in a longer lane.** The homework's four-pass run is still short. A run of many hours needs `+GPUJobLength = "medium"` or `"long"`, and the long lane caps a user at four GPUs.
- **Checkpointing is already on.** The submit file's checkpoint lines and the wrapper's two-hour timeout carry the last epoch's checkpoint across an interruption or a lane limit, so a run of many hours is several two-hour runs that each resume. Watch the epoch length. The script checkpoints at the end of an epoch, so an epoch longer than the timeout never saves and the job repeats its first two hours forever. Keep an epoch under about an hour with `--max_train_pages`, or raise the timeout in `finetune.sh`. CHTC's guide caps it at ten hours.
- **Test on a slice first.** Add `--max_train_pages 20 --epochs 1 --eval_pages 3 --curve_pages 0` to the arguments line for a ten-minute run before the full one. A bug costs ten minutes of a free GPU here instead of a dollar, and the queue time is the same either way.
- **Stay on one GPU unless the model does not fit it.** Almost every fine-tune a research group runs fits one card here: LoRA on anything up to about 27B with the frozen weights in bf16 on an 80 GB card, about 50B on an H200, and up to a 70B with them in 4-bit. One card is also the request that starts soonest. A second GPU does not make a job that fits one card cheaper, since nothing is billed, and a job that asks for two waits for a server with two free. When a model does not fit one card, `request_gpus = 2` or more and the training script's `Trainer` with `fsdp=` shards the weights, gradients and optimizer state across the cards, which the lesson's PyTorch episode explains. Ask for the A100, H100 or H200 servers for that, where the cards share NVLink. On the L40S and RTX PRO 6000 servers they share PCIe and the split loses more. `condor_status` names the server a card is in, and CHTC's GPU table says which cards those are. Test on a slice before a long run.
- **Transcribing the test pages** uses the lesson's `churro_transcribe_batch.py` with `--adapter_dir` pointing at the unpacked adapter, over the 141 unlabeled test pages. The Colab version of this lesson, in the folder beside this one, has that step written out, and the same arguments work in a second GPU job here. The result is not a Kaggle submission, because the lesson's data is the German-letter part of the competition and a submission needs all 208 pages of its test set, the 141 letters plus 67 pages of surveyors' notes and account books. For that, fine-tune on the competition's own `train.csv` and images, 619 pages, which the script takes unchanged, and transcribe its `test.csv`.

### Exercise: count the page for another model

The lesson's episode and the Colab notebook measure how many tokens a page costs CHURRO, about 1,500, with the model's own processor. That count is what the training time follows, and the method works for any model. Pick another open vision-language model and count what the same page costs there. `llava-hf/llava-1.5-7b-hf` is not gated and its processor downloads in seconds. Before running anything, read its model card and predict two things: how many image tokens a page will take, and whether `max_pixels` will change that. Then load its processor, build the message with its chat template, the image, the instruction and the transcription, and read the `input_ids` length. Run the text without the image to separate the two. The count needs no GPU. Run it inside the container on the access point, from the `badger-scribe` directory with the data unpacked, with `apptainer exec /staging/<n>/<netid>/churro.sif python count_tokens.py`, or on your own machine.

<details><summary>Solution</summary>

```python
from transformers import AutoProcessor
import pandas as pd
import badger_common as bc

row = pd.read_csv("data/train.csv", keep_default_na=False).iloc[0]
image = bc.load_page_image(f"data/images/{row.page_id}.jpg", bc.DEFAULT_MAX_PIXELS)
prompt = bc.build_prompt(row.category)

llava = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")
messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]},
            {"role": "assistant", "content": [{"type": "text", "text": row.text}]}]
text = llava.apply_chat_template(messages, tokenize=False)
n_llava = llava(text=[text], images=[image], return_tensors="pt")["input_ids"].shape[1]
n_llava_text = len(llava.tokenizer(text.replace("<image>", ""))["input_ids"])
print(f"LLaVA 1.5: {n_llava} tokens for the page, {n_llava - n_llava_text} of them for the image; CHURRO: about 1,500")
```

LLaVA 1.5 resizes every image to 336 × 336 pixels and cuts it into 24 × 24 patches of 14 pixels, 576 image tokens however large the page, so `max_pixels` changes nothing for it. The page comes to about 900 tokens, 576 for the image and the rest for the instruction and the German text, which its Llama tokenizer splits into a different number of pieces than Qwen's. Fewer tokens is a cheaper step. It is not a better model for this task. A 336-pixel image of a letter page is too small to read the handwriting, which is why CHURRO spends 1,280 tokens on a megapixel. The token count is cost. Accuracy has to be measured.

</details>

## 12. When a job fails

Every job leaves three files in the directory it was submitted from, named by the submit file's `log`, `output` and `error` lines. For the fine-tuning job they are `finetune_<cluster>.log`, `.out` and `.err`, where the cluster number is what `condor_submit` printed. Read them in this order.

| File | What is in it | Read it when |
|---|---|---|
| `.log` | HTCondor's record of the job: submitted, matched to a machine, files transferred, started, evicted or held, finished, with the memory and disk it used against what it requested | Always first. It says whether the job ran at all, and why it stopped |
| `.err` | What the script wrote to standard error: Python tracebacks, CUDA errors, warnings from the libraries | The job ran and ended with a non-zero exit code |
| `.out` | What the script printed: the page split, the trainable parameter line, the safeguards, the scores and the Trainer's loss lines | The job ran, to see how far it got, and the results of a job that finished |

`condor_tail -f <jobid>` shows the `.out` file of a running job. A job that is still in the queue has no `.out` or `.err` yet, only the `.log`. On the shared GPU machines `condor_ssh_to_job` is not available, so the three files and `condor_tail` are the way in.

The symptoms and where they point:

| What you see | Where to look | What it usually is |
|---|---|---|
| `condor_q` shows the job idle for over an hour | `condor_q -better-analyze <jobid>`, which counts the machines that match each requirement | A requirement nothing satisfies, or a busy lab. Zero matches on `gpus_minimum_memory` or `gpus_minimum_capability` means the number is wrong; a few matches means wait, or ask for a shorter lane |
| `condor_q` shows the job held (`H`) | `condor_q -hold <jobid>`, or `condor_q <jobid> -af HoldReason`, and the `.log` | Memory or disk over the request, a missing input file, or a bad path in `transfer_input_files`. The reason names the number or the file. Fix the submit file and `condor_submit` again; a held job does not resume on its own |
| The job ran for a minute and ended | `.err`, then the `.log` for the exit code | The script failed before training: a missing module means the container is wrong, a `FileNotFoundError` means the tar or the data archive did not unpack where the script looks, a `RuntimeError` from the safeguards means the adapter landed where it should not |
| `CUDA out of memory` in `.err` | `.err`, and the `.log` for which GPU it ran on | The card is smaller than the request assumed, or `--max_pixels` is set high. Raise `gpus_minimum_memory`, or lower `--max_pixels` on the arguments line |
| The job ended at exactly two hours and was requeued | `.log` shows the exit code 85 and a new start | That is the checkpoint timeout working, not a failure. The next run resumes from `output/ckpt`. If it happens on every run, an epoch is longer than the timeout, which section 11 covers |
| The job finished but `adapter.tar.gz` is missing | `.out` for the last lines, `.err` for a traceback in the scoring | The script failed after training, in `[eval after]`, or the wrapper's `tar` step. The checkpoint in `output/ckpt` is not returned, so rerun with the fix |
| `condor_history` shows a job you do not remember | `condor_history <netid> -limit 10` | Jobs finish and leave the queue; the history is where they went |

To try a command by hand on a GPU machine, submit the fine-tuning job interactively, `condor_submit -i finetune.sub`, which gives a shell on a matching machine with the files transferred and the container running. Run `finetune.sh` there a line at a time. The lane and the GPU request apply as for a batch job, so the wait is the same.

CHTC's facilitators answer questions at chtc@cs.wisc.edu, and the [get help page](https://chtc.cs.wisc.edu/uw-research-computing/get-help) has office hours. Send the three files with the question.

## 13. Clean up

Jobs end on their own and nothing bills, but staging has a quota and the build machine is shared. When you are done with a project:

```bash
condor_q                                        # nothing of yours should be left idle or held
condor_rm <jobid>                               # if something is
rm /staging/<n>/<netid>/churro-3B.tar           # 7.5 GB you can download again; keep churro.sif if you will fine-tune again
rm -rf ~/badger-scribe/output ~/badger-scribe/results   # outputs you have copied elsewhere
```

Keep the adapter, about 100 MB. With the base weights from Hugging Face it reproduces the fine-tuned model anywhere.

# LoRA fine-tuning on CHTC: the same job on UW-Madison's free GPUs

> **In development.** This guide has not yet been run end to end on CHTC. The submit files follow CHTC's current documentation, but expect rough edges until the first draft is complete, planned for 1 December 2026. If you try it before then and something fails, [open an issue](https://github.com/carpentries-incubator/ML_with_AWS_SageMaker/issues) with the job's `.log` and `.err` files.

This is the fine-tuning episode of the [ML on AWS lesson](https://carpentries-incubator.github.io/ML_with_AWS_SageMaker/08_Finetuning-BadgerScribe.html) run on the [Center for High Throughput Computing](https://chtc.cs.wisc.edu/) (CHTC) instead of on AWS. Same model, CHURRO-3B. Same data, 295 pages of 1930s German letters in Kurrent script. Same training script and the same LoRA adapter at the end. What changes is who provides the GPU and how a job is described.

CHTC is free to UW-Madison researchers and should be the first place a campus project looks for a GPU. It runs HTCondor, a job scheduler. You describe a job in a text file, hand it to the scheduler, and it finds a machine, copies your files there, runs your program and copies the results back. Nobody is billed. What you pay with is the time your job waits in the queue, and the care it takes to package a job so it runs unattended on a machine you never log in to. This guide goes from requesting an account to reading the fine-tuned model's scores. It takes a few days of calendar time for the account, an afternoon for the first container and model staging, and about an hour per fine-tuning run after that.

The files in this folder are everything the job needs:

| File | What it is |
|---|---|
| `churro.def` | The recipe for the software container, PyTorch 2.8 plus the Python packages the training script imports |
| `build.sub` | The interactive job that builds the container |
| `fetch_model.sub`, `fetch_model.sh` | A small job that downloads the 7.5 GB of CHURRO-3B weights once into your staging space |
| `finetune.sub`, `finetune.sh` | The GPU job, and the script it runs inside the container |
| `read_results.py` | Reads what the job sends back and prints the before-and-after comparison |

The training script itself, `finetune_churro_lora.py`, and its helper `badger_common.py` come from the lesson's `scripts/` folder and are not changed.

## Three places to run this fine-tune

The same fine-tune runs in three places. This guide is the CHTC column. The others are the [lesson's episode](https://carpentries-incubator.github.io/ML_with_AWS_SageMaker/08_Finetuning-BadgerScribe.html) on AWS and a [Colab notebook](https://colab.research.google.com/github/carpentries-incubator/ML_with_AWS_SageMaker/blob/main/finetune_extras/colab/LoRA-finetuning-on-a-free-GPU.ipynb).

| | [AWS SageMaker](https://carpentries-incubator.github.io/ML_with_AWS_SageMaker/08_Finetuning-BadgerScribe.html) | [Google Colab](https://colab.research.google.com/github/carpentries-incubator/ML_with_AWS_SageMaker/blob/main/finetune_extras/colab/LoRA-finetuning-on-a-free-GPU.ipynb) | [CHTC](https://github.com/carpentries-incubator/ML_with_AWS_SageMaker/tree/main/finetune_extras/chtc) |
|---|---|---|---|
| Who can use it | Anyone with an AWS account and a card or credits | Anyone with a Google account | UW-Madison researchers and students, with a faculty sponsor |
| Cost of this run | About `$0.75` | Free | Free |
| GPU | A10G, 24 GB, bf16 | T4, 16 GB, no bf16, so the frozen weights go to 4-bit | L40S, A100, H100 or H200, 48 to 141 GB, bf16; NVLink on the A100, H100 and H200 servers |
| Time for this run | 33 minutes, starts in a minute | About an hour, starts at once if a GPU is free | About 20 minutes, after a queue wait of minutes to hours |
| Largest LoRA fine-tune | No hard limit, only cost. 7B on the lesson's `$1.41`/h card, 70B at `$13`/h | about 7B, in 4-bit | about 50B in bf16 on an H200, a 70B in 4-bit on one 80 GB card, larger across a server's cards |
| Largest full fine-tune | No hard limit, only cost. 3B at `$7`/h, 70B at `$127`/h | none, a 3B needs 60 GB | a 7B on one H200, a 70B across the H100 or H200 servers' cards |
| What you learn beyond the fine-tune | Estimating and reading a bill, tags, job guard rails | Fitting a model into a small GPU | Describing a job for a scheduler, containers, staging data |

For fine-tuning, CHTC is the first choice for anyone at UW-Madison. It is free, its GPUs are large, and the queue is the only price. Colab is for practising LoRA with nothing to set up. The cloud has the newest and fastest GPUs, and a fine-tune on a large dataset sometimes needs them, but it bills by the hour, so it is the choice when the other two cannot do the job, not the default.

None of the three runtime limits is a wall. The training script writes a checkpoint at the end of every epoch and resumes from it, so a run longer than a CHTC job lane, a Colab session or an AWS runtime cap is several runs that each pick up where the last one stopped. Section 11 says how to make HTCondor carry the checkpoints between runs.

## 1. Get an account

CHTC accounts are for research and approved courses. You need an active NetID and a faculty sponsor, usually your PI or the course instructor.

1. Fill in the [account request form](https://userapp.chtc.wisc.edu/forms/user-applications/create/). Describe the work as it is: fine-tuning a 3-billion-parameter vision-language model on a few hundred page images, one GPU with about 20 GB of memory, runs of about an hour. The description decides which resources you are set up for, so saying "GPU" here matters.
2. An automated email from chtc@cs.wisc.edu arrives within hours, and a facilitator follows up within two or three business days. A new research group gets a meeting first. If you hear nothing after three business days, email them.
3. At the meeting, or in the form, ask for two things beyond the default account: access to the GPU Lab, and a directory in `/staging`, which this guide needs for the model weights and the container.

Your welcome email names the access point you log in to.

## 2. Log in

CHTC's HTC system has two access points, `ap2001.chtc.wisc.edu` and `ap2002.chtc.wisc.edu`. Use the one in your welcome email. You need the campus network or the VPN, and Duo.

```bash
ssh <netid>@ap2001.chtc.wisc.edu
```

The access point is where you keep files and submit jobs. It is not where work runs. Nothing in this guide runs Python on the access point except the small results reader at the end, and even that can run elsewhere. Once a job is submitted you can log out, and it runs and returns its output without you.

## 3. Where files live

Two places, chosen by size.

| | `/home/<netid>` | `/staging/<n>/<netid>` |
|---|---|---|
| For | Many small files, under 1 GB each | Few large files, over 1 GB each |
| Quota | 40 GB | 100 GB, 1,000 items |
| Submit jobs from here | Yes | No |
| A job reads a file from here with | `transfer_input_files = file.txt` | `transfer_input_files = osdf:///chtc/staging/<n>/<netid>/file` |

`<n>` is the first letter of your NetID, so `bbadger` has `/staging/b/bbadger`. This guide puts the code, the data archive and the submit files in `/home`, and the container and the model weights in `/staging`. A job does not read either location directly. HTCondor copies what the submit file lists into the job's own scratch directory on the execute machine, the program runs there, and HTCondor copies back what the submit file names as output. That is the whole model of a job, and every line below follows from it.

## 4. The code and the data

In your home directory, clone the lesson and copy the two Python files and this folder's job files into a working directory.

```bash
cd ~
rm -rf ML_with_AWS_SageMaker && git clone -q --depth 1 https://github.com/carpentries-incubator/ML_with_AWS_SageMaker.git   # a fresh copy, so a rerun gets the current scripts
mkdir -p badger-scribe && cd badger-scribe
cp ../ML_with_AWS_SageMaker/scripts/finetune_churro_lora.py ../ML_with_AWS_SageMaker/scripts/badger_common.py .
cp ../ML_with_AWS_SageMaker/finetune_extras/chtc/* .
ls
```

The data is the German-letter part of [Badger Scribe](https://www.kaggle.com/competitions/badger-scribe), a Kaggle competition built from UW Digital Collections material. The lesson publishes it as a GitHub release, three zip files of about 130 MB in all. Download them, unpack into one folder, and pack that folder as one archive. One archive instead of 436 image files is the rule for job inputs, since HTCondor moves one file faster than hundreds.

```bash
release=https://github.com/carpentries-incubator/ML_with_AWS_SageMaker/releases/download/badger-scribe-data-v1
for part in labels images-train images-test; do
  curl -sSL -o $part.zip $release/badger-scribe-v1-$part.zip && unzip -q -o $part.zip -d data && rm $part.zip
done
ls data; ls data/images | wc -l
tar czf badger-scribe-data.tar.gz data
ls -lh badger-scribe-data.tar.gz     # about 130 MB: under a gigabyte, so it can stay in /home and transfer from here
```

`train.csv` has one row per labelled page, with its id, the letter it belongs to and the transcription. The training script holds 15% of the pages out of training, with a fixed seed, and scores them before and after. The lesson's episode has the data's story and a page to look at. Here the pages go straight into the job.

## 5. Build the container

A job runs on a machine with an operating system and little else. The software comes along in a container, a file holding a complete Linux environment with PyTorch and the packages the script imports. `churro.def` is its recipe. It starts from the public PyTorch 2.8 image with CUDA 12.8, the same versions the lesson's SageMaker container had, and installs `transformers`, `peft` and a few small packages on top.

Containers are built in an interactive job, on a build machine, not on the access point. `build.sub` describes that job. Submit it with `-i`, wait for the prompt to change to the build machine, and run the build there.

```bash
condor_submit -i build.sub
# ... wait; the prompt changes when a build machine is yours ...
apptainer build churro.sif churro.def
apptainer shell -e churro.sif            # test it: the prompt changes again, into the container
python -c "import torch, transformers, peft; print(torch.__version__, transformers.__version__, peft.__version__)"
exit                                     # leave the container
exit                                     # leave the build job; churro.sif comes back with you
```

The build pulls several gigabytes and takes 10 to 20 minutes. If it dies, the usual cause is disk, and the fix is a larger `request_disk` in `build.sub`. The result, `churro.sif`, is a few gigabytes, so it goes to staging, where every job reads it from.

```bash
mv churro.sif /staging/<n>/<netid>/
```

## 6. Stage the model

The base weights are 7.5 GB, too big for `/home` and too slow to download again for every run. `fetch_model.sub` is a small CPU job, no GPU, that downloads them from Hugging Face once, inside the container you just built, packs them as one tar, and writes the tar straight to your staging directory. Read the two files first. The submit file's `transfer_output_remaps` line is what sends a job's output somewhere other than the submit directory, and the shell script is what the job runs.

Edit `NETID` at the top of `fetch_model.sub`, then submit and watch.

```bash
condor_submit fetch_model.sub
condor_q                                 # your jobs in the queue
condor_tail -f <jobid>                   # the job's output as it runs; Ctrl-C to stop watching
ls -lh /staging/<n>/<netid>/             # when the job is done: churro.sif and churro-3B.tar
```

A job's life is in its `.log` file. Open `fetch_model.log` when it finishes and read the events in order: submitted, started on a machine, files transferred, finished, and how much memory and disk it used against what it asked for. The `.out` file has what the script printed, and `.err` anything it wrote to standard error.

## 7. Read the job before you submit it

Open `finetune.sub` and `finetune.sh` together. The submit file is the part of the lesson's estimator that described the machine and the files. The shell script is the part that ran the training script. Every setting has a comment, and these are the ones to understand.

**The arguments line** is the script's command-line flags, the same hyperparameters the lesson's SageMaker job set, two passes, a LoRA rank of 16, 15 holdout pages scored before and after, 8 of them at the end of every pass. Change a run here, not in the script.

**`transfer_input_files`** names everything the job gets, the two Python files and the data archive from `/home`, and the model tar from staging through its `osdf:///` address. The script sees nothing else. `finetune.sh` unpacks the two archives in the job's scratch directory before training starts.

**`transfer_output_files`** names what comes back, the adapter archive the shell script packs and the folder of scores and curves. HTCondor returns new files in the job's top-level directory by default, but not folders, so the folder is listed.

![GPU memory per parameter: inference, a LoRA fine-tune, and a full fine-tune](https://carpentries-incubator.github.io/ML_with_AWS_SageMaker/fig/memory-per-parameter.svg)

*GPU memory per parameter: inference, a LoRA fine-tune, and a full fine-tune. The lesson's memory rule, which the GPU request below is read from.*

**The GPU request** is where the lesson's cost estimate becomes a resource request. The lesson's memory rule gives about 12 GB for a 3B LoRA run, so `gpus_minimum_memory = 20000` asks for a card with at least 20 GB, with room for the before-and-after scoring. `gpus_minimum_capability = 8.0` asks for an Ampere or newer GPU, which is what bf16 needs. On a card this size there is no reason to compress anything. The frozen base weights stay in bf16 and the adapter trains in bf16, as in the lesson's SageMaker job, which is why the script's default precision is right here and the Colab version's 4-bit setting is not needed. The lesson's time rule gave 12 minutes of training on an A10G and the measured job was 32 minutes with its scoring, and CHTC's GPUs are faster, so `+GPUJobLength = "short"`, the lane for jobs under 12 hours. Short jobs can use two thirds of the GPU Lab's cards, medium jobs (under 24 hours) one third, and long jobs (up to seven days) at most four GPUs per user. Ask for the shortest lane the job fits, and it starts sooner.

**`request_memory` and `request_disk`** are the machine's RAM and scratch disk, not the GPU's. Loading 7.5 GB of weights peaks CPU memory at about twice that, so 32 GB. The disk holds the model tar and its unpacked copy, the data, the container's working files and the outputs, so 40 GB. A job that exceeds its request is held or killed, and the `.log` file says by how much, so a first run that is held is corrected by raising the number and resubmitting.

To see which GPUs exist and what is free right now:

```bash
condor_status -compact -constraint 'TotalGpus > 0'
condor_status -constraint 'Gpus > 0' -af Machine GPUs_DeviceName GPUs_GlobalMemoryMb GPUs_Capability | sort | uniq -c
```

## 8. Submit, and watch it run

Edit `NETID` in `finetune.sub`, then:

```bash
condor_submit finetune.sub
condor_q
```

The job sits idle until a GPU matching the request is free. Minutes at a quiet time, longer when the lab is busy. If it idles for more than an hour, `condor_q -better-analyze <jobid>` says how many machines match each requirement, which is how you find a request that nothing can satisfy.

Once it runs, `condor_tail -f <jobid>` shows the script's output. The lines to find are the same ones the lesson's episode reads in the SageMaker log, in this order:

- `pages: 295 total -> 251 train / 44 holdout`, the split.
- `trainable params: ... || trainable%: 0.80`, about 30 million parameters, under 1% of the model, which the lesson's episode works out from the model's config file.
- `safeguards: adapter is ... vision tower frozen`, the script's two checks passing.
- `[eval before] 15 holdout pages ... macro CER ...`, the base model's score.
- The Trainer's loss every five steps, a `[eval epoch1]` line at the end of each pass, then `[eval after]` on the same 15 pages.

On an A100 or an L40 expect about 5 minutes of training and 10 to 15 minutes of scoring, half the SageMaker job's time on its A10G. When `condor_q` shows nothing, the job is done and `adapter.tar.gz` and `output/data` are in your working directory.

## 9. What changed

`read_results.py` unpacks the archive and prints the summary, the per-page character error rate before and after on the same 15 held-out pages, the page that improved most, and saves the loss and holdout-CER curves as a picture. It needs pandas and matplotlib. Run it inside the container on the access point, which is light enough for there, or copy `adapter.tar.gz` to your own machine.

```bash
apptainer exec /staging/<n>/<netid>/churro.sif python read_results.py adapter.tar.gz
```

Read it the way the lesson's episode reads the SageMaker run. The mean CER before and after on the same pages is the fine-tune's effect and nothing else. The page that improved most shows what was learned, this writer's hand and the archive's conventions, umlauts and ß as written, Kurrent letterforms read correctly. The holdout curve, not the training loss, says whether two passes was the right amount. Still falling means another pass would help, flat means stop, rising while the loss falls means the model is memorising the training pages.

## 10. What CHTC gives you for free

Nothing here is billed. What matters instead is what hardware is in the pool and how long you may hold it, because that sets the model sizes you can work with. The GPU Lab's inventory changes as cards are added, so read it from the scheduler, which lists every GPU type with its memory:

```bash
condor_status -constraint 'Gpus > 0' -af GPUs_DeviceName GPUs_GlobalMemoryMb | sort | uniq -c
```

CHTC publishes the inventory behind that command, and as of late 2026 the open pool is, in order of how many there are: 34 H200s with 141 GB, 27 L40s and 22 L40Ss with 48 GB, 24 A100s at 80 or 40 GB, 12 RTX PRO 6000s with 96 GB, 8 H100s with 80 GB, and a few dozen older cards with 11 to 24 GB. A second, larger backfill pool runs jobs that can be interrupted, which is what `+is_resumable = true` asks for. Two things about the cards matter for fine-tuning beyond their memory. Their generation, which the data file calls `hardware_generation` and HTCondor calls capability, is 8.0 or newer for every card from the A100 up, so bf16 training works on all of them. And how the cards in one server are wired together. In the A100, H100 and H200 servers the GPUs are soldered onto one board and joined by NVLink, NVIDIA's direct GPU-to-GPU link, which is the build data-center AI servers use and what a multi-GPU job wants. The L40, L40S and RTX PRO 6000 are ordinary plug-in cards that talk through the PCIe bus, several times slower.

The lesson's memory rules turn the memory into model sizes. Inference needs 2 bytes per parameter in bf16, a LoRA fine-tune about 2.5 with the frozen weights in bf16 or about 1 with them in 4-bit (the QLoRA setup the Colab version uses), and a full fine-tune 16, plus a few gigabytes of working room:

| GPU memory | Cards | Inference, bf16 | LoRA fine-tune | Full fine-tune |
|---|---|---|---|---|
| 40 GB | A100 40 GB, A100 MIG slices | up to about 15B | up to about 12B, about 35B in 4-bit | nothing; a 3B model needs 60 GB |
| 48 GB | L40, L40S | up to about 20B | up to about 15B, about 40B in 4-bit | nothing |
| 80 GB | A100 80 GB, H100 | up to about 35B, or a 70B in 4-bit | up to about 27B, about 70B in 4-bit | a 3B model, just |
| 96 GB | RTX PRO 6000 | up to about 40B | up to about 35B, a 70B in 4-bit | a 3B model |
| 141 GB | H200 | up to about 65B | up to about 50B, a 70B in 4-bit with room | a 3B model easily, a 7B model, just |
| several cards in one server, model sharded across them | the A100, H100 and H200 servers, joined by NVLink | a 70B in bf16 | a 70B in bf16 | a 7B on two 80 GB cards, a 70B on many |

The last row is for a model that does not fit one card, and how well it works depends on the server. On the A100, H100 and H200 servers the cards share NVLink and a sharded job runs close to the card count. On the L40S and RTX PRO 6000 servers the cards share a PCIe bus, and the lesson's PyTorch episode estimated 3.2 of an ideal 4 for a full fine-tune's gradient exchange over PCIe, less for a smaller batch. Stay on one card whenever the memory rule says the job fits. This job needs 12 GB, so any card in the pool holds it with room to spare, and asking for a 20 GB minimum, as the submit file does, matches every one of them and starts soonest. A 7B LoRA, the largest the lesson's cost section considers on one card, is 22 GB and also fits everywhere here. The 80 GB and larger cards are what a full fine-tune of CHURRO itself would take, and the lesson's `$6,300` for that run on AWS is zero dollars here, in exchange for its place in the queue.

How long you may hold a GPU is the job-length lane. Short jobs, under 12 hours, may use two thirds of the lab's cards at once. Medium jobs, under 24 hours, one third. Long jobs, up to seven days, at most four GPUs per user. This run is an hour, so it goes in the short lane and competes for the most cards. A seven-day job holds a card nobody else can use, which is why the long lane is narrow and why checkpointing, in the next section, matters once a run is long.

What a run used is in the scheduler's history, the same numbers the lesson read from a job's billable seconds on AWS, here read for fitting the next request to the job:

```bash
condor_history <netid> -limit 3 -af:h ClusterId RemoteWallClockTime MemoryUsage DiskUsage MachineAttrGPUs_DeviceName0
```

## 11. Scaling up, and the same guard rails

The lesson's rules for a bigger run hold here, in CHTC's terms.

- **A longer run goes in a longer lane.** The homework's four-pass run is still short. A run of many hours is `+GPUJobLength = "medium"` or `"long"`, and the long lane caps a user at four GPUs.
- **Checkpoint anything long.** The training script already writes a checkpoint at the end of every epoch to `output/ckpt` and resumes from it when restarted. HTCondor can carry those checkpoints across an interruption if the job tells it to. Add `checkpoint_exit_code = 85` and `transfer_checkpoint_files = output/ckpt` to the submit file, and have the wrapper exit with code 85 after each epoch's checkpoint instead of running to the end. CHTC's checkpointing guide has the protocol, and jobs that checkpoint every few hours can add `+is_resumable = true` to run on backfill GPUs too. For a one-hour run it is not worth the setup, which is the same call the lesson made.
- **Test on a slice first.** Add `--max_train_pages 20 --epochs 1 --eval_pages 3 --curve_pages 0` to the arguments line for a ten-minute run before the real one. A bug costs ten minutes of a free GPU here instead of a dollar, but the queue time is the same, so the habit still pays.
- **Stay on one GPU unless the model does not fit it.** Almost every fine-tune a research group runs fits one card here, LoRA on anything up to about 27B with the frozen weights in bf16 on an 80 GB card, about 50B on an H200, and up to a 70B with them in 4-bit, and one card is the request that starts soonest. A second GPU does not make a job that fits one card cheaper, since nothing is billed, and a job that asks for two waits for a server with two free. When a model does not fit one card, `request_gpus = 2` or more and the training script's `Trainer` with `fsdp=` shards the weights, gradients and optimizer state across them, which the lesson's PyTorch episode explains. Ask for the A100, H100 or H200 servers for that, where the cards share NVLink. On the L40S and RTX PRO 6000 servers they share PCIe and the split loses more. `condor_status` names the server a card is in, and CHTC's GPU table says which cards those are. Test that it works on a slice before a long run.
- **A Kaggle submission** uses the lesson's `churro_transcribe_batch.py` with `--adapter_dir` pointing at the unpacked adapter, over the 141 test pages. The Colab version of this lesson, in the folder beside this one, has that step written out, and the same arguments work in a second GPU job here.

## 12. Clean up

Jobs end on their own, and nothing bills, but staging has a quota and the build machine is shared. When you are done with a project:

```bash
condor_q                                        # nothing of yours should be left idle or held
condor_rm <jobid>                               # if something is
rm /staging/<n>/<netid>/churro-3B.tar           # 7.5 GB you can download again; keep churro.sif if you will fine-tune again
rm -rf ~/badger-scribe/output ~/badger-scribe/results   # outputs you have copied elsewhere
```

What to keep is the adapter, about 100 MB, which with the base weights from Hugging Face reproduces the fine-tuned model anywhere.

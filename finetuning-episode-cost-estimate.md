---
title: 'Fine-tuning episode: cost, capacity and setup'
---

Planning notes for running the [Fine-tuning a Vision-Language Model with Training Jobs](../episodes/08_Finetuning-BadgerScribe.md) episode (Badger Scribe, CHURRO-3B LoRA) with a class of about 50 in a 2-hour slot (3 with the bonus exercises) (quotas below are sized for 100). Prices are on-demand SageMaker rates from the AWS price list published 2026-10-16 (`us-east-1` and the class account's `us-east-2` are the same for these instance types); re-check them before the term, since `ml.g5.xlarge` has already moved from `$1.21` to `$1.41` since the RAG episodes were written.

## The in-class design

In class, every student runs one LoRA fine-tuning job of about 33 minutes: it scores 15 held-out German-letter (`kade_letters`) pages with the base model, trains on all 251 training pages (295 minus a 44-page holdout) for 2 passes, scores 8 holdout pages after each pass for a curve, scores the same 15 pages again, and saves the adapter. The lesson ships only the German letters (295 train, 141 test pages) because every label is human-transcribed (the competition's survey and account-book labels are mostly machine drafts), because they are most of the hidden test set, and because the scans are smaller. Before/after lives inside that one job, so there are no separate inference jobs during class. The full run, the Processing jobs over the test set and the optional Kaggle submission are homework.

Launches use `wait=True` and an explicit job name with the launch time in it (`badger-<user>-churro-lora-2ep-<MMDD-HHMM>`), so a rerun gets a new name instead of failing with `ResourceInUse`; `wait=True` is the double-launch guard, since the cell blocks until the job ends. Interrupting the cell does not stop the job; the episode says so in a callout, and you should say it out loud.

## What one student spends

| Item | Instance | Billable time | Cost |
|---|---|---|---|
| Controller notebook (CPU) | `ml.t3.medium` (`$0.05`/h) | 3 h | `$0.15` |
| In-class LoRA fine-tune (251 German-letter pages × 2 passes, 15 pages scored before and after, 8-page curve) | `ml.g5.xlarge` (`$1.41`/h) | ~32 min | `$0.75` |
| **In class** | | **~0.5 GPU-hours** | **~`$0.90`** |
| Bonus Exercise 1, if anyone does it: the class job at 2 megapixels | `ml.g5.xlarge` | ~60 min | `$1.40` |
| Homework: four-pass LoRA run (251 pages × 4 passes, 40 pages scored before and after) | `ml.g5.xlarge` | ~65 min | `$1.50` |
| Homework: base and LoRA fine-tuned over holdout + test (~185 pages, two Processing jobs) | `ml.g5.xlarge` | ~15 min each | `$0.70` |
| **In class + bonus + homework** | | **~3.1 GPU-hours** | **~`$4.50`** |

Assumptions behind the training lines: CHURRO-3B in bf16 with LoRA rank 16, gradient checkpointing, one page per step, `max_pixels` of 1280 × 28 × 28 (about one megapixel, ~1,300 visual tokens plus ~300 text tokens per page), and, measured on the class job on 2026-10-21, 1.45 s per page-pass of training, 17–21 s per page for greedy generation on an A10G, about 5 minutes of billed startup (container pull, ~130 MB of images, 7.5 GB of weights copied from the shared bucket as a `model` channel) and under a minute of unbilled provisioning. The 2-megapixel and homework rows are scaled from those rates. Every job is capped (`max_run` of 65 min on the class job, 130 min on the homework run, 45 min on each Processing job), so the worst case from the scripted steps is about `$7` per student.

## What the class spends

| Scenario | Per student | 50 students (double for 100) |
|---|---|---|
| In class, one clean pass each | `$0.90` | `$45` |
| In class, realistic (one extra relaunch each) | ~`$1.65` | ~`$85` |
| In class, two launches each at the `max_run` cap | ~`$3.20` | ~`$160` |
| Homework, if everyone does it | ~`$2.20` | ~`$110` |
| **Budget to request** (realistic class + homework + 50%) | | **`$300`** |

For comparison, teaching the same material on `ml.g5.xlarge` *notebook* instances costs about the same during class (`$1.41` × 3 h × 50 = `$212`) but adds the forgotten-instance risk: each one left running costs `$34`/day, `$237`/week, `$1,029`/month. Two forgotten instances over a long weekend exceed the entire jobs-based budget above. That difference is the point of the episode, and it is worth saying out loud to the class.

## Suggested timeline (3 h)

| Time | What |
|---|---|
| 0:00–0:30 | Why fine-tune; the cost table; setup cells (CSV download, token counting, guard rails) |
| 0:30–0:40 | Everyone launches the class job (`wait=True`) |
| 0:40–1:15 | While it runs (~33 min), in the console: job page, `max_run`, tags, CloudWatch log, the checkpoint folder in S3; forgotten-instance arithmetic; quotas |
| 1:15–1:45 | Step 2: bill vs. estimate, before/after CER, the curves, a page side by side; the holdout callouts |
| 1:45–2:00 | Step 3 totals; walk through the homework section; cleanup |

The bonus exercises are not scheduled. With a third hour, Exercise 1 (the class job at 2 megapixels) is about an hour of GPU time, so launch it first and discuss Exercise 2 while it runs.

If the jobs are still queueing at 0:50 (capacity), teach the console and the cost arithmetic from your own job's page, run before class.

## Things that will go wrong unless arranged ahead of time

**Service quotas.** This is the item most likely to sink the session. In Service Quotas → Amazon SageMaker, per region:

- `ml.g5.xlarge for training job usage`: at least the number of students launching at once. Request **100**.
- `ml.g5.xlarge for processing job usage`: only needed for the homework, where each student launches two jobs at once. Request **200**.
- `Maximum number of instances across all training jobs`: check it is not lower than the above.
- If you want `ml.g6e.xlarge` for Exercise 2, request 100 of those too.

New accounts often have these at 0 to 4. Quota increases for GPU instances can take days and sometimes a back-and-forth with support, so file them **three to four weeks** before class.

**Capacity.** Fifty `ml.g5.xlarge` launches within the same minute in one region can fail with `InsufficientCapacity` even when the quota allows it. Mitigations, in order: launch the class jobs in waves (table by table), and have `ml.g6.xlarge` (L4, 24 GB, `$1.13`/h; somewhat slower on this workload, with half the memory bandwidth, but still well within `max_run`) ready as a documented fallback instance type.

**Model download.** The base weights (`stanford-oval/churro-3B`, ~7.5 GB) are *not* pulled from Hugging Face by the jobs. They are staged once in the shared bucket under `badger-scribe/model/churro-3B/` (step on the [Shared data bucket](shared-data-bucket.md) page) and every estimator passes that prefix as a `model` channel; the Processing jobs mount the same prefix and pass it as `--model_id`. SageMaker copies it from S3 in ~1–2 minutes per job, and a hundred jobs starting in the same minute hit S3, which is built for that, rather than one Hugging Face repository. The scripts fall back to the Hub only when no `model` channel is given (self-paced learners who skipped the staging step), in which case the instance needs internet access (default outside a VPC). The CHURRO weights carry the Qwen research license (non-commercial), which is fine for a class.

**Warm pools.** `keep_alive_period_in_seconds` on the estimator keeps the instance for the next job, which would let a relaunch skip the startup overhead. It needs its own quota (`ml.g5.xlarge for training warm pool usage`) and it bills for the idle time at the full rate, so for a class it adds a way to leave meters running. Leave it off.

## Before class

### 1. Stage the data in the shared read-only bucket

Students do not create a bucket or upload anything for this episode. The data lives as a release on the lesson repository (`badger-scribe-data-v1`, ~130 MB) and is staged once per term under `badger-scribe/` in the shared bucket, next to the WattBot data. The CHURRO-3B weights (~7.5 GB) are staged once under `badger-scribe/model/churro-3B/` in the same bucket, from a laptop or a notebook instance rather than CloudShell. The bucket policy, layout and exact commands are on the [Shared data bucket: staging before each term](shared-data-bucket.md) page.

### 2. The rest

- Before anything launches, check that the container the episode asks for exists in the class region (a wrong version combination fails at `fit()` with a confusing error):

  ```python
  import sagemaker
  region = sagemaker.Session().boto_region_name   # us-east-2 for the class account
  sagemaker.image_uris.retrieve("huggingface", region, version="4.56", base_framework_version="pytorch2.8",
                                py_version="py312", image_scope="training", instance_type="ml.g5.xlarge")
  ```

- Run the mini job once too, so you have seen the log students will see and know where `max_run` lands.
- Set a Budgets alert on the class account at the planned total, and a Cost Explorer filter on the `Project` tag students use in `make_job_tags`.
- Have the fallback instance type on a slide: `instance_type = "ml.g6.xlarge"`.
- Tell students to stop the controller notebook at the end. It is five cents an hour, which is why nobody notices it.

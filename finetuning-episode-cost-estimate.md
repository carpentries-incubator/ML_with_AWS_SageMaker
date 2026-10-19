---
title: 'Fine-tuning episode: cost, capacity and setup'
---

Planning notes for running the *Fine-tuning a Vision-Language Model with Training Jobs* episode (Badger Scribe, CHURRO-3B LoRA) with a class of about 50 in a 3-hour slot (quotas below are sized for 100). Prices are on-demand SageMaker rates from the AWS price list published 2026-10-16 (`us-east-1` and the class account's `us-east-2` are the same for these instance types); re-check them before the term, since `ml.g5.xlarge` has already moved from $1.21 to $1.41 since the RAG episodes were written.

## The in-class design

The full 295-page fine-tune takes about 45 minutes, which does not fit a 3-hour class once setup and discussion are added. In class, every student runs one ~25-minute job instead: it scores 15 held-out German-letter (`kade_letters`) pages with the base model, trains on 120 German-letter pages for 2 epochs, scores the same 15 pages again, and saves the adapter. The lesson ships only the German letters (295 train, 141 test pages) because every label is human-transcribed (the competition's survey and account-book labels are mostly machine drafts), because they are most of the hidden test set, and because the scans are smaller. Before/after lives inside that one job, so there are no separate inference jobs during class. The full run, the Processing jobs over the test set and the optional Kaggle submission are homework.

Students compare their mini adapter against a full run **you** ran before class (see "Before class" below). That is what makes the "powerful when done right" point without anyone waiting an hour.

Launches use `wait=True` and an explicit `job_name=f"badger-{username}-mini-v1"`. The explicit name is the double-launch guard: SageMaker rejects a second job with the same name, so a queued Shift+Enter fails harmlessly. Interrupting the cell does not stop the job; the episode says so in a callout, and you should say it out loud.

## What one student spends

| Item | Instance | Billable time | Cost |
|---|---|---|---|
| Controller notebook (CPU) | `ml.t3.medium` ($0.05/h) | 3 h | $0.15 |
| In-class mini fine-tune (120 German-letter pages × 2 epochs, 15 pages scored before and after) | `ml.g5.xlarge` ($1.41/h) | ~25 min | $0.60 |
| Exercise 1: same job at 2 megapixels | `ml.g5.xlarge` | ~35 min | $0.80 |
| **In class** | | **~1 GPU-hour** | **~$1.55** |
| Homework: full fine-tune (251 pages × 2 epochs, 40 pages scored before and after) | `ml.g5.xlarge` | ~45 min | $1.05 |
| Homework: base and fine-tuned over holdout + test (~185 pages, two Processing jobs) | `ml.g5.xlarge` | ~15 min each | $0.70 |
| **In class + homework** | | **~2.3 GPU-hours** | **~$3.30** |

Assumptions behind the training lines: CHURRO-3B in bf16 with LoRA rank 16, gradient checkpointing, one page per step, `max_pixels` of 1280 × 28 × 28 (about one megapixel, ~1,300 visual tokens plus ~300 text tokens per page), about 2.5 s per page-pass and ~10 s per page for greedy generation on an A10G, plus ~10 minutes of fixed startup (container pull, ~130 MB of images, 7.5 GB of weights copied from the shared bucket as a `model` channel, which adds ~1–2 minutes of S3 download per job instead of a Hugging Face pull). These are estimates, not measurements; a 2x miss either way would not be surprising until you have timed a run. Every job is capped (`max_run` of 45 min on the mini job, 90 min on the full run, 45 min on each Processing job), so the worst case from the scripted steps is about $6 per student.

## What the class spends

| Scenario | Per student | 50 students (double for 100) |
|---|---|---|
| In class, one clean pass each (mini job + Exercise 1) | $1.55 | $80 |
| In class, realistic (one extra relaunch each) | ~$2.20 | ~$110 |
| In class, everything hits its `max_run` cap | ~$3.20 | ~$160 |
| Homework, if everyone does it | ~$1.75 | ~$90 |
| **Budget to request** (realistic class + homework + 50%) | | **$400** |

For comparison, teaching the same material on `ml.g5.xlarge` *notebook* instances costs about the same during class ($1.41 × 3 h × 50 = $212) but adds the forgotten-instance risk: each one left running costs $34/day, $237/week, $1,029/month. Two forgotten instances over a long weekend exceed the entire jobs-based budget above. That difference is the point of the episode, and it is worth saying out loud to the class.

## Suggested timeline (3 h)

| Time | What |
|---|---|
| 0:00–0:25 | Why fine-tune; the cost-scaling table; setup cells (CSV download, scripts, tags) |
| 0:25–0:35 | Everyone launches the mini job (`wait=True`) |
| 0:35–1:00 | While it runs, in the console: job page, `max_run`, tags, CloudWatch log; forgotten-instance arithmetic; quotas |
| 1:00–1:25 | Step 2: bill vs. estimate, before/after CER, look at a page |
| 1:25–1:50 | Step 3: compare against your full run; holdout-optimism callout |
| 1:50–2:30 | Exercise 1 (relaunch at 2 MP; ~35 min, so launch first and discuss Exercises 2 and 3 while it runs) |
| 2:30–3:00 | Step 4 totals; walk through the homework section; cleanup |

If the mini jobs are still queueing at 0:45 (capacity), move Step 3 forward: the comparison against your full run needs no student job.

## Things that will go wrong unless arranged ahead of time

**Service quotas.** This is the item most likely to sink the session. In Service Quotas → Amazon SageMaker, per region:

- `ml.g5.xlarge for training job usage`: at least the number of students launching at once. Request **100**.
- `ml.g5.xlarge for processing job usage`: only needed for the homework, where each student launches two jobs at once. Request **200**.
- `Maximum number of instances across all training jobs`: check it is not lower than the above.
- If you want `ml.g6e.xlarge` for Exercise 2, request 100 of those too.

New accounts often have these at 0 to 4. Quota increases for GPU instances can take days and sometimes a back-and-forth with support, so file them **three to four weeks** before class.

**Capacity.** Fifty `ml.g5.xlarge` launches within the same minute in one region can fail with `InsufficientCapacity` even when the quota allows it. Mitigations, in order: launch the mini jobs in waves (table by table), and have `ml.g6.xlarge` (L4, 24 GB, $1.13/h; somewhat slower on this workload, with half the memory bandwidth, but still well within `max_run`) ready as a documented fallback instance type.

**Model download.** The base weights (`stanford-oval/churro-3B`, ~7.5 GB) are *not* pulled from Hugging Face by the jobs. They are staged once in the shared bucket under `badger-scribe/model/churro-3B/` (step on the *Shared data bucket* page) and every estimator passes that prefix as a `model` channel; the Processing jobs mount the same prefix and pass it as `--model_id`. SageMaker copies it from S3 in ~1–2 minutes per job, and a hundred jobs starting in the same minute hit S3, which is built for that, rather than one Hugging Face repository. The scripts fall back to the Hub only when no `model` channel is given (self-paced learners who skipped the staging step), in which case the instance needs internet access (default outside a VPC). The CHURRO weights carry the Qwen research license (non-commercial), which is fine for a class.

**Warm pools.** `keep_alive_period_in_seconds` on the estimator keeps the instance for the next job, which would let a relaunch skip the startup overhead. It needs its own quota (`ml.g5.xlarge for training warm pool usage`) and it bills for the idle time at the full rate, so for a class it adds a way to leave meters running. Leave it off.

**Spot.** Managed spot training saves up to 70% but 50 simultaneous spot requests will see interruptions and capacity waits, and the script does not checkpoint. Keep it as the exercise it is.

## Before class

### 1. Stage the data in the shared read-only bucket

Students do not create a bucket or upload anything for this episode. The data lives as a release on the lesson repository (`badger-scribe-data-v1`, ~130 MB) and is staged once per term under `badger-scribe/` in the shared bucket, next to the WattBot data. The CHURRO-3B weights (~7.5 GB) are staged once under `badger-scribe/model/churro-3B/` in the same bucket, from a laptop or a notebook instance rather than CloudShell. The bucket policy, layout and exact commands are on the *Shared data bucket: staging before each term* page.

### 2. Run the full fine-tune yourself and stage its output

Run the episode's homework estimator (all 251 training pages, 2 epochs, `eval_pages` 40) under your own account a day or two before class. Then copy three files from its `model.tar.gz` into the shared bucket, where Step 3 of the episode expects them:

- `badger-scribe/instructor/full-run/train_summary.json`
- `badger-scribe/instructor/full-run/eval_before.csv`
- `badger-scribe/instructor/full-run/eval_after.csv`

The holdout split is seeded and the evaluation sample is the first *n* of one seeded shuffle of the sorted page ids, so your 40 evaluation pages include the students' 15 pages by construction, as long as everyone keeps `seed` and `holdout_frac` at their defaults and trains from the same data release. While you are at it, note the job's `BillableTimeInSeconds` and `seconds_per_page` and replace the time estimates in the episode's Step 0 tables with measured ones. The episode tells students to compare their bill to the estimate; the estimate should be a real number.

### 3. The rest

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

---
title: 'Shared data bucket: staging before each term'
---

Learners create exactly one bucket of their own (S3 episode) and write every output there. Everything larger than the Titanic CSVs lives in **one shared, read-only bucket** that instructors stage before class, so the class holds one copy of each dataset instead of fifty. This page is the whole staging procedure; the episodes assume the layout below.

## Layout

```
s3://<shared-bucket>/
    wattbot/
        metadata.csv        from data/data.zip in the lesson repo
        train_QA.csv        from data/data.zip
        corpus.zip          built once with scripts/build_wattbot_corpus.py
    badger-scribe/
        train.csv  test.csv  metadata.csv  metric.py
        sample_submission.csv  transcription_conventions.md
        images/<page_id>.jpg
        model/churro-3B/       the CHURRO-3B weights, copied once from Hugging Face (~7.5 GB)
        instructor/full-run/   train_summary.json, eval_before.csv, eval_after.csv
```

Learners set `shared_bucket = "<this bucket's name>"` in the first cell of the RAG and fine-tuning episodes. Tell them the name at the start of each session.

## 1. Create the bucket and make it readable by the student role

Create the bucket in the class region (same naming rules as the S3 episode; something like `ml-marathon-shared`), with the three cost-allocation tags. Then attach this read-only policy (Permissions → Bucket policy), editing the bucket name and replacing `<account-id>` and the role names with your account's:

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Principal": {
                "AWS": [
                    "arn:aws:iam::<account-id>:role/ml-sagemaker-use",
                    "arn:aws:iam::<account-id>:role/ml-sagemaker-bedrock-use"
                ]
            },
            "Action": [
                "s3:GetObject",
                "s3:ListBucket"
            ],
            "Resource": [
                "arn:aws:s3:::ml-marathon-shared",
                "arn:aws:s3:::ml-marathon-shared/*"
            ]
        }
    ]
}
```

`s3:ListBucket` is required: a training job's input channel lists the prefix before downloading it. No `PutObject`: learners never write here.

## 2. Stage WattBot (RAG episodes)

From a machine with Python and the AWS CLI (CloudShell works), in a clone of the lesson repo:

```bash
unzip -o data/data.zip -d /tmp/lessondata
pip install pandas requests
python scripts/build_wattbot_corpus.py --metadata /tmp/lessondata/data/wattbot/metadata.csv --out /tmp/lessondata/wattbot/corpus.zip
cp /tmp/lessondata/data/wattbot/metadata.csv /tmp/lessondata/data/wattbot/train_QA.csv /tmp/lessondata/wattbot/
aws s3 sync /tmp/lessondata/wattbot/ s3://<shared-bucket>/wattbot/
```

The corpus script downloads every PDF listed in `metadata.csv` and reports any URL that failed; decide whether to drop or fix those before staging. The corpus only needs rebuilding when `metadata.csv` changes.

## 3. Stage Badger Scribe (fine-tuning episode)

Run this from a laptop or a notebook instance, or from CloudShell for the ~130 MB Badger Scribe release only (CloudShell's home directory is 1 GB, which is why the loop handles one asset at a time in `/tmp`: download, unzip, sync, delete, so nothing larger than one zip is ever on disk). From the lesson's data release (see *Badger Scribe data: publishing a release* for how versions are published):

```bash
V=1
R=https://github.com/carpentries-incubator/ML_with_AWS_SageMaker/releases/download/badger-scribe-data-v$V
for a in labels images-train images-test; do
  rm -rf /tmp/bs && mkdir -p /tmp/bs
  curl -L -o /tmp/bs/$a.zip $R/badger-scribe-v$V-$a.zip
  unzip -q -o /tmp/bs/$a.zip -d /tmp/bs/badger-scribe && rm /tmp/bs/$a.zip
  aws s3 sync /tmp/bs/badger-scribe/ s3://<shared-bucket>/badger-scribe/
  rm -rf /tmp/bs
done
```

Then run the full fine-tune once (homework estimator in the episode) and copy `train_summary.json`, `eval_before.csv`, `eval_after.csv` from its `model.tar.gz` to `s3://<shared-bucket>/badger-scribe/instructor/full-run/`.

## 4. Stage the CHURRO-3B weights (fine-tuning episode)

Every training and processing job in the episode reads the base model from the bucket as a `model` channel, so that 100 jobs starting in the same minute copy it from S3 instead of each pulling 7.5 GB from Hugging Face. Do this once per bucket, from a laptop or a notebook instance with at least 10 GB free (not CloudShell):

```bash
pip install huggingface_hub
huggingface-cli download stanford-oval/churro-3B --local-dir churro-3B
aws s3 sync churro-3B/ s3://<shared-bucket>/badger-scribe/model/churro-3B/
rm -rf churro-3B
```

About 7.5 GB; it only needs redoing if the Hub repository changes or the bucket is rebuilt.

## 5. Keep it, or rebuild it

The end-of-workshop cleanup (`instructors/instructor-notes.md`, aws-nuke) deletes every S3 bucket in the account unless excluded. Either add the shared bucket to the nuke config's exclude filter, or accept that it goes and redo steps 1 to 4 next term; the data steps take about fifteen minutes and the model copy as long as a 7.5 GB download takes where you are. Storage is ~130 MB for Badger Scribe plus the WattBot corpus, and ~7.5 GB for the model copy: well under a dollar a month.

## Self-paced learners

Anyone working outside the shared account does steps 1, 3 and 4 (and 2 for the RAG episodes) into a bucket of their own and sets `shared_bucket` to its name. The episodes say so where it matters.

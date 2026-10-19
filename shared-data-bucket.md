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

From the lesson's data release (see *Badger Scribe data: publishing a release* for how versions are published):

```bash
V=1
R=https://github.com/carpentries-incubator/ML_with_AWS_SageMaker/releases/download/badger-scribe-data-v$V
mkdir -p badger-scribe && cd badger-scribe
for a in labels images-train images-test; do
  curl -L -o $a.zip $R/badger-scribe-v$V-$a.zip && unzip -q -o $a.zip && rm $a.zip
done
cd .. && aws s3 sync badger-scribe/ s3://<shared-bucket>/badger-scribe/
```

Then run the full fine-tune once (homework estimator in the episode) and copy `train_summary.json`, `eval_before.csv`, `eval_after.csv` from its `model.tar.gz` to `s3://<shared-bucket>/badger-scribe/instructor/full-run/`.

## 4. Keep it, or rebuild it

The end-of-workshop cleanup (`instructors/instructor-notes.md`, aws-nuke) deletes every S3 bucket in the account unless excluded. Either add the shared bucket to the nuke config's exclude filter, or accept that it goes and redo steps 1 to 3 next term; both take about fifteen minutes. Storage is ~150 MB for Badger Scribe plus the WattBot corpus, a few cents a month.

## Self-paced learners

Anyone working outside the shared account does the same three steps into a bucket of their own and sets `shared_bucket` to its name. The episodes say so where it matters.

---
title: 'Shared data bucket: staging before each term'
---

Learners create exactly one bucket of their own (S3 episode) and write every output there. Everything larger than the Titanic CSVs lives in **one shared, read-only bucket** that instructors stage before class, so the class holds one copy of each dataset instead of fifty. This page is the whole staging procedure; the episodes assume the layout below.

The ready-to-paste version is the *From CloudShell* section at the end of this page; the sections before it explain each step.

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

From a machine with Python and the AWS CLI (CloudShell works). Set the bucket name, clone the lesson repository, build the corpus from the data it ships, and sync:

```bash
BUCKET=ml-marathon-shared
pip install -q --user pandas requests
rm -rf /tmp/lessondata && mkdir -p /tmp/lessondata && cd /tmp/lessondata
git clone -q --depth 1 https://github.com/carpentries-incubator/ML_with_AWS_SageMaker.git lesson
unzip -q -o lesson/data/data.zip -d data
mkdir -p wattbot
cp data/data/wattbot/metadata.csv data/data/wattbot/train_QA.csv wattbot/
python3 lesson/scripts/build_wattbot_corpus.py --metadata wattbot/metadata.csv --out wattbot/corpus.zip
aws s3 sync wattbot/ s3://$BUCKET/wattbot/
```

The corpus script downloads every PDF listed in `metadata.csv` and reports any URL that failed; decide whether to drop or fix those before staging. The corpus only needs rebuilding when `metadata.csv` changes.

## 3. Stage Badger Scribe (fine-tuning episode)

Run this from a laptop or a notebook instance, or from CloudShell for the ~130 MB Badger Scribe release only (CloudShell's home directory is 1 GB, which is why the loop handles one asset at a time in `/tmp`: download, unzip, sync, delete, so nothing larger than one zip is ever on disk). From the lesson's data release (see *Badger Scribe data: publishing a release* for how versions are published):

```bash
BUCKET=ml-marathon-shared
V=1
R=https://github.com/carpentries-incubator/ML_with_AWS_SageMaker/releases/download/badger-scribe-data-v$V
for a in labels images-train images-test; do
  rm -rf /tmp/bs && mkdir -p /tmp/bs
  curl -sSL -o /tmp/bs/$a.zip $R/badger-scribe-v$V-$a.zip
  unzip -q -o /tmp/bs/$a.zip -d /tmp/bs/badger-scribe && rm /tmp/bs/$a.zip
  aws s3 sync /tmp/bs/badger-scribe/ s3://$BUCKET/badger-scribe/
  rm -rf /tmp/bs
done
```

Then run the full fine-tune once (homework estimator in the episode) and copy `train_summary.json`, `eval_before.csv`, `eval_after.csv` from its `model.tar.gz` to `s3://$BUCKET/badger-scribe/instructor/full-run/`.

## 4. Stage the CHURRO-3B weights (fine-tuning episode)

Every training and processing job in the episode reads the base model from the bucket as a `model` channel, so that 100 jobs starting in the same minute copy it from S3 instead of each pulling 7.5 GB from Hugging Face. Do this once per bucket, from a laptop or a notebook instance with at least 10 GB free (not CloudShell):

```bash
BUCKET=ml-marathon-shared
pip install -q huggingface_hub
huggingface-cli download stanford-oval/churro-3B --local-dir churro-3B
aws s3 sync churro-3B/ s3://$BUCKET/badger-scribe/model/churro-3B/
rm -rf churro-3B
```

From CloudShell, where there is no room to download it, use Block 3 at the end of this page instead: it streams each file from Hugging Face straight into S3.

About 7.5 GB; it only needs redoing if the Hub repository changes or the bucket is rebuilt.

## The whole procedure, from CloudShell

The learner execution role on a notebook instance can create the bucket but is refused on tagging, bucket policy and uploads (the bucket policy deliberately grants it read only). So the staging is done as your own SSO identity in **CloudShell** (console toolbar, same region). Three blocks, pasted in order. CloudShell's home is 1 GB, which is why the work happens one asset at a time under `/tmp` and the model is streamed rather than downloaded.

**Block 1: bucket, tags, public access, policy.** Fill in the account id.

```bash
BUCKET=ml-marathon-shared
REGION=us-east-2
ACCOUNT_ID=123456789012
aws s3api head-bucket --bucket $BUCKET 2>/dev/null || aws s3api create-bucket --bucket $BUCKET --region $REGION --create-bucket-configuration LocationConstraint=$REGION
aws s3api put-bucket-tagging --bucket $BUCKET --tagging 'TagSet=[{Key=Project,Value=Workshop},{Key=Name,Value=Everyone},{Key=Purpose,Value=Storage}]'
aws s3api put-public-access-block --bucket $BUCKET --public-access-block-configuration BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true
cat > /tmp/policy.json <<JSON
{"Version": "2012-10-17", "Statement": [{"Effect": "Allow",
  "Principal": {"AWS": ["arn:aws:iam::$ACCOUNT_ID:role/ml-sagemaker-use", "arn:aws:iam::$ACCOUNT_ID:role/ml-sagemaker-bedrock-use"]},
  "Action": ["s3:GetObject", "s3:ListBucket"],
  "Resource": ["arn:aws:s3:::$BUCKET", "arn:aws:s3:::$BUCKET/*"]}]}
JSON
aws s3api put-bucket-policy --bucket $BUCKET --policy file:///tmp/policy.json
```

**Block 2: WattBot and Badger Scribe.**

```bash
REPO=https://github.com/carpentries-incubator/ML_with_AWS_SageMaker
V=1   # Badger Scribe data release version
rm -rf /tmp/stage && mkdir -p /tmp/stage && cd /tmp/stage
git clone -q --depth 1 $REPO lesson
pip install -q --user pandas requests huggingface_hub
unzip -q -o lesson/data/data.zip -d data
mkdir -p wattbot && cp data/data/wattbot/metadata.csv data/data/wattbot/train_QA.csv wattbot/
python3 lesson/scripts/build_wattbot_corpus.py --metadata wattbot/metadata.csv --out wattbot/corpus.zip
aws s3 sync wattbot/ s3://$BUCKET/wattbot/
R=$REPO/releases/download/badger-scribe-data-v$V
for a in labels images-train images-test; do
  rm -rf bs && mkdir -p bs
  curl -sSL -o bs/$a.zip $R/badger-scribe-v$V-$a.zip
  unzip -q -o bs/$a.zip -d bs/badger-scribe && rm bs/$a.zip
  aws s3 sync bs/badger-scribe/ s3://$BUCKET/badger-scribe/
  rm -rf bs
done
```

**Block 3: CHURRO-3B weights**, streamed from Hugging Face into S3 file by file (about 7.5 GB, ten minutes or so, nothing written to disk).

```bash
python3 - <<'EOF'
import boto3, requests
from huggingface_hub import HfApi, hf_hub_url
s3 = boto3.client("s3")
bucket, model_id, prefix = "ml-marathon-shared", "stanford-oval/churro-3B", "badger-scribe/model/churro-3B"
for f in [f for f in HfApi().list_repo_files(model_id) if not f.startswith(".")]:
    with requests.get(hf_hub_url(model_id, f), stream=True, timeout=60) as r:
        r.raise_for_status(); r.raw.decode_content = True
        s3.upload_fileobj(r.raw, bucket, f"{prefix}/{f}")
    print("uploaded", f)
EOF
```

**Check, then clean up.**

```bash
aws s3 ls s3://$BUCKET/ --recursive --summarize | tail -2
aws s3 ls s3://$BUCKET/wattbot/
aws s3 ls s3://$BUCKET/badger-scribe/
aws s3 ls s3://$BUCKET/badger-scribe/model/churro-3B/
cd / && rm -rf /tmp/stage
```

Still by hand: run the fine-tuning episode's homework estimator once and copy `train_summary.json`, `eval_before.csv` and `eval_after.csv` from its `model.tar.gz` to `s3://ml-marathon-shared/badger-scribe/instructor/full-run/`.

## 5. Keep it, or rebuild it

The end-of-workshop cleanup (`instructors/instructor-notes.md`, aws-nuke) deletes every S3 bucket in the account unless excluded. Either add the shared bucket to the nuke config's exclude filter, or accept that it goes and redo steps 1 to 4 next term; the data steps take about fifteen minutes and the model copy as long as a 7.5 GB download takes where you are. Storage is ~130 MB for Badger Scribe plus the WattBot corpus, and ~7.5 GB for the model copy: well under a dollar a month.

## Self-paced learners

Anyone working outside the shared account does steps 1, 3 and 4 (and 2 for the RAG episodes) into a bucket of their own and sets `shared_bucket` to its name. The episodes say so where it matters.

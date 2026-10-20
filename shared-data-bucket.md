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

## Where to run this

Everything below runs in **CloudShell** (console toolbar, with the region set to the class region), where you act as your own SSO identity. It cannot be done from a notebook instance: the learner execution role there can create a bucket but is refused on tagging, the bucket policy and every upload, because the policy in step 2 deliberately gives it read-only access.

Paste the blocks in order in one CloudShell tab, so the variables set in step 1 are still there for the later steps. CloudShell's home directory is 1 GB, which is why the data steps work one asset at a time under `/tmp` and the model is streamed rather than downloaded. The whole thing takes about ten minutes; a run on 2026-10-20 ended with 461 objects and 7.7 GB in the bucket.

## 1. Create the bucket

Same naming rules as the S3 episode, in the class region, with the three cost-allocation tags, and all public access blocked. The account ID is read from your identity; the `echo` line is there so you can check it is the class account before anything is created.

```bash
BUCKET=ml-marathon-shared
REGION=us-east-2
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
echo $ACCOUNT_ID
aws s3api create-bucket --bucket $BUCKET --region $REGION --create-bucket-configuration LocationConstraint=$REGION
aws s3api put-bucket-tagging --bucket $BUCKET --tagging 'TagSet=[{Key=Project,Value=Workshop},{Key=Name,Value=Everyone},{Key=Purpose,Value=Storage}]'
aws s3api put-public-access-block --bucket $BUCKET --public-access-block-configuration BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true
```

If the bucket already exists (created in the console, or left from last term), `create-bucket` reports `BucketAlreadyOwnedByYou` and nothing else changes; the other two lines are safe to re-run.

## 2. Make it readable by the learner roles

Learners reach the bucket through two execution roles: `ml-sagemaker-use` on notebook instances and training jobs, and `ml-sagemaker-bedrock-use` in the Bedrock episode. This policy lets both read and list, and nothing else. `s3:ListBucket` is required because a training job's input channel lists the prefix before downloading it. No `PutObject`: learners never write here.

```bash
cat > /tmp/policy.json <<JSON
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Principal": {
                "AWS": [
                    "arn:aws:iam::$ACCOUNT_ID:role/ml-sagemaker-use",
                    "arn:aws:iam::$ACCOUNT_ID:role/ml-sagemaker-bedrock-use"
                ]
            },
            "Action": ["s3:GetObject", "s3:ListBucket"],
            "Resource": ["arn:aws:s3:::$BUCKET", "arn:aws:s3:::$BUCKET/*"]
        }
    ]
}
JSON
aws s3api put-bucket-policy --bucket $BUCKET --policy file:///tmp/policy.json
aws s3api get-bucket-policy --bucket $BUCKET --output text
```

The same JSON can be pasted into the console instead (bucket → Permissions → Bucket policy), with the account ID and bucket name written in. `MalformedPolicy: Invalid principal` means an ARN in the `Principal` list does not exist: a wrong account ID, or a role name that differs from what IAM has (`aws iam list-roles --query 'Roles[].RoleName'` lists them).

## 3. Stage WattBot (RAG episodes)

Clone the lesson repository, build the corpus from the data it ships, and sync:

```bash
rm -rf /tmp/stage && mkdir -p /tmp/stage && cd /tmp/stage
pip install -q pandas requests huggingface_hub
git clone -q --depth 1 https://github.com/carpentries-incubator/ML_with_AWS_SageMaker.git lesson
unzip -q -o lesson/data/data.zip -d data
mkdir -p wattbot
cp data/data/wattbot/metadata.csv data/data/wattbot/train_QA.csv wattbot/
python3 lesson/scripts/build_wattbot_corpus.py --metadata wattbot/metadata.csv --out wattbot/corpus.zip
aws s3 sync wattbot/ s3://$BUCKET/wattbot/
```

The corpus script downloads every PDF listed in `metadata.csv` and reports any URL that failed; decide whether to drop or fix those before staging. The corpus only needs rebuilding when `metadata.csv` changes.

## 4. Stage Badger Scribe (fine-tuning episode)

The data comes from the lesson's own GitHub release (see *Badger Scribe data: publishing a release* for how versions are published). The loop handles one asset at a time: download, unzip, sync, delete, so nothing larger than one zip is ever on disk.

```bash
V=1   # Badger Scribe data release version
R=https://github.com/carpentries-incubator/ML_with_AWS_SageMaker/releases/download/badger-scribe-data-v$V
for a in labels images-train images-test; do
  rm -rf /tmp/stage/bs && mkdir -p /tmp/stage/bs
  curl -sSL -o /tmp/stage/bs/$a.zip $R/badger-scribe-v$V-$a.zip
  unzip -q -o /tmp/stage/bs/$a.zip -d /tmp/stage/bs/badger-scribe && rm /tmp/stage/bs/$a.zip
  aws s3 sync /tmp/stage/bs/badger-scribe/ s3://$BUCKET/badger-scribe/
  rm -rf /tmp/stage/bs
done
```

## 5. Stage the CHURRO-3B weights (fine-tuning episode)

Every training and processing job in the episode reads the base model from the bucket as a `model` channel, so that 100 jobs starting in the same minute copy it from S3 instead of each pulling 7.5 GB from Hugging Face. This streams each file from Hugging Face straight into S3, nothing written to disk: about 7.5 GB, which took under two minutes from CloudShell.

```bash
python3 - <<EOF
import boto3, requests
from huggingface_hub import HfApi, hf_hub_url
s3 = boto3.client("s3")
bucket, model_id, prefix = "$BUCKET", "stanford-oval/churro-3B", "badger-scribe/model/churro-3B"
for f in [f for f in HfApi().list_repo_files(model_id) if not f.startswith(".")]:
    with requests.get(hf_hub_url(model_id, f), stream=True, timeout=60) as r:
        r.raise_for_status(); r.raw.decode_content = True
        s3.upload_fileobj(r.raw, bucket, f"{prefix}/{f}")
    print("uploaded", f)
EOF
```

On a machine with 10 GB free, the plain download works too:

```bash
huggingface-cli download stanford-oval/churro-3B --local-dir churro-3B
aws s3 sync churro-3B/ s3://$BUCKET/badger-scribe/model/churro-3B/
rm -rf churro-3B
```

It only needs redoing if the Hub repository changes or the bucket is rebuilt.

## 6. Check, then clean up

```bash
aws s3 ls s3://$BUCKET/ --recursive --summarize | tail -2
aws s3 ls s3://$BUCKET/wattbot/
aws s3 ls s3://$BUCKET/badger-scribe/
aws s3 ls s3://$BUCKET/badger-scribe/model/churro-3B/
cd / && rm -rf /tmp/stage
```

Expect three files under `wattbot/`, the CSVs plus `images/` under `badger-scribe/`, and the config, tokenizer and safetensors files under `model/churro-3B/`.

## 7. The instructor's full run

Still by hand: run the fine-tuning episode's homework estimator once and copy `train_summary.json`, `eval_before.csv` and `eval_after.csv` from its `model.tar.gz` to `s3://$BUCKET/badger-scribe/instructor/full-run/`.

## 8. Keep it, or rebuild it

The end-of-workshop cleanup (`instructors/instructor-notes.md`, aws-nuke) deletes every S3 bucket in the account unless excluded. The committed `nuke-config.yml` excludes `ml-marathon-shared`; if you named the bucket differently, change the filter there or accept that it goes and redo steps 1 to 6 next term. Storage is ~130 MB for Badger Scribe plus the WattBot corpus, and ~7.5 GB for the model copy: well under a dollar a month.

## Self-paced learners

Anyone working outside the shared account does steps 1, 4 and 5 (and 3 for the RAG episodes) into a bucket of their own and sets `shared_bucket` to its name. The episodes say so where it matters.

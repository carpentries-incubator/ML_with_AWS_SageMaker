---
title: 'Instructor Notes'
name: Zain Waseem & Chris Endemann
---

::::::::::::::::::::::::::::::::::::: callout

## All instructor pages

These live under the **Extras** dropdown, which the site shows only when the **Instructor View** toggle (top bar) is on.

- [Account preparation checklist (quotas, Bedrock, tags)](aws-account-prep.html): before the term. As of October 2026 the notebook role `ml-sagemaker-use` carries two inline policies set from CloudShell: `BedrockAccess`, so any instance can call Bedrock, and `WorkshopGuardRails`, which refuses jobs with missing or blank tags or with a `Project` or `Purpose` off the lists in `scripts/guardrails.json`, instance types outside the lesson's list, caps over 48 hours, endpoints and EC2. Both are on that page with the command that applies them; the guard rails are not yet on the learner permission set.
- [Shared data bucket: staging before each term](shared-data-bucket.html): create and fill `ml-marathon-shared` from CloudShell.
- [Badger Scribe data: publishing a release](badger-scribe-data-release.html): only when the competition data changes.
- [Fine-tuning episode: cost, capacity and setup](finetuning-episode-cost-estimate.html)
- This page: cleaning the account with aws-nuke after the workshop. The config it uses is [instructors/nuke-config.yml](https://github.com/carpentries-incubator/ML_with_AWS_SageMaker/blob/main/instructors/nuke-config.yml) in the repository.

::::::::::::::::::::::::::::::::::::::::::::::::

## How the workshop is taught

The first part is taught in lockstep: the overview, S3, the notebook instance, reading and writing S3, and the first training episode, through the first job launch and its bill. Everyone does the same thing at the same time, because everything after depends on it (a bucket, a running instance, the cloned repository, and the habit of estimating a job and then reading its billable seconds).

After that the lesson opens up. Learners pick the episodes closest to their own work, the PyTorch training episode, the fine-tune, the tuning episode, the Bedrock RAG episode or the Processing-jobs one, and work through them with instructors circulating. The prefilled notebooks make this possible: each episode runs on its own after the first part, and the expensive steps are staged in the shared bucket so a learner who starts late still finishes. The Bedrock episode is the one to steer everyone through at some point, since it is the cheapest route to a working pipeline and the one most projects will use.

This format worked for the 2026 Marathon class in three hours. It needs participants who are willing to ask questions and work through things on their own, which is the audience the lesson assumes: intermediate, comfortable in Python and Jupyter, knowing what training, tuning, fine-tuning and RAG are, with no AWS experience required. With a group that expects to be walked through every cell, keep the lockstep going through the PyTorch episode and the Bedrock episode and leave the rest as homework.

## Cleaning the account with aws-nuke

Hello all! In this document, we will show you how to run an AWS Nuke script within the ML-Marathon Sagemaker server. As you progress through the Sagemaker
workshop, towards the end of the workshop you'll be tasked to remove and empty your notebooks \& buckets. However, depending on the size of your data, it can take a long time to remove them altogether.

This is where the AWS Nuke script comes in handy, instead of removing your notebooks and buckets one-by-one, the script automates this process and removes
it for you!

Here are the following steps on how to launch and run the AWS-Nuke Script:

### 1\. Open CloudShell in the correct AWS account

1. Sign in to the AWS Console for the account you intend to clean.
2. In the top-right Region, select the Region where you want CloudShell storage
to live. In this case, we will choose US-East-2 (OHIO) since that's where we
store our servers.
3. Open CloudShell from the console toolbar.
4. Verify exactly which account/role you are using:

```bash
aws sts get-caller-identity
```

### 2\. Get the config file into CloudShell

**aws-nuke needs a config file in .yml or .yaml format or else your script will FAIL.** The one for this workshop is committed in the lesson repository at [instructors/nuke-config.yml](https://github.com/carpentries-incubator/ML_with_AWS_SageMaker/blob/main/instructors/nuke-config.yml). Two things it deliberately does *not* delete: the shared data bucket (the `filters` block) and Bedrock application inference profiles, which learners remove themselves in the cleanup episode. Everything else the lesson creates (training, processing and tuning jobs) has already ended and holds no billable resource.

Download it straight into CloudShell and check it:

```bash
curl -fsSL -o "$HOME/nuke-config.yml" https://raw.githubusercontent.com/carpentries-incubator/ML_with_AWS_SageMaker/main/instructors/nuke-config.yml
cat "$HOME/nuke-config.yml"
```

If you need to change it (a different account ID, another bucket to keep), edit the file in CloudShell with `nano "$HOME/nuke-config.yml"`, or edit a copy on your own machine and upload it with **"Actions" > "Upload file"**. Use space indenting, not tabs: YAML is sensitive to this and the script will fail on tabs.

### 3\. Download and install the linux version

AWS CloudShell uses Linux as their base distribution. So you will need to download
the linux version of AWS Nuke from Github. You should download only from the
official ekristen/aws-nuke releases, not use any local version.

Link: https://github.com/ekristen/aws-nuke/releases

Run the following:

```bash
mkdir -p "$HOME/bin"
cd "$HOME"
curl -fL -o aws-nuke-v3.67.0-linux-amd64.tar.gz \
  https://github.com/ekristen/aws-nuke/releases/download/v3.67.0/aws-nuke-v3.67.0-linux-amd64.tar.gz
tar -xzf aws-nuke-v3.67.0-linux-amd64.tar.gz
mv aws-nuke "$HOME/bin/aws-nuke"
chmod 700 "$HOME/bin/aws-nuke"
export PATH="$HOME/bin:$PATH"
aws-nuke --version
```

To make aws-nuke stay on your PATH in later CloudShell sessions:

```bash
echo 'export PATH="$HOME/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

Reminder: CloudShell is based on Amazon Linux, and software saved in `$HOME`
persists; software installed elsewhere may disappear after the session ends.

### 4\. DO a dry run first

It is important that you first run a dry-run so that you can verify which
files you want to remove before executing the actual script.

Here is the script to do so:

```bash
aws-nuke run --config "$HOME/nuke-config.yml"
```

This should list resources it would remove, without deleting them. Review the
output carefully, especially S3 buckets, IAM-related resources, and the displayed
account identity.

### 5\. Only after review: actual deletion

Only if the dry-run output is correct and this is truly the intended non-production account:

```bash
aws-nuke run --config "$HOME/nuke-config.yml" --no-dry-run
```

Do not add `--no-prompt` / `--force` on your first real run; leave the confirmation
prompt enabled as an additional safeguard.

### If the dry run stops on "account doesn't have an alias"

Oftentimes when running the dry run, you will run into an error stating:

```
WARN[0000] 'targets' is deprecated. Please use 'includes' instead.
aws-nuke - v3.67.0 - 9cc98ccaed6303dac29001642879320ab584d41a
FATA[0000] specified account doesn't have an alias. For safety reasons you need
to specify an account alias. Your production account should contain the term 'prod'
```

Do not panic, this isn't related to CloudShell or credentials. Rather, it connected to AWS successfully but stopped abruptly since the account you are trying to target has **no AWS account alias.** By default, aws-nuke requires
an alias so you must visually confirm a recognizable account name before proceeding. It also blocks aliases containing prod by default.

Recommended: create a non-production account alias.
First, verify the exact account:

```bash
aws sts get-caller-identity
```

Then add an alias in the AWS Console:

1. Go to IAM
2. Open Dashboard
3. Under AWS Account, choose Create Alias
4. Set a clear, unique non-production alias, e.g. mycompany-sandbox or training-cleanup
5. Do not use a name containing prod, production, or live.

Then run the dry-run again:

```bash
aws-nuke run --config "$HOME/nuke-config.yml"
```

Congratulations, you have reached the end of this tutorial! By now, all scripts
should work and the specified notebooks/buckets you wanted to clean should
now be all empty!

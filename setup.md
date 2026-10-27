---
title: Setup
---

## Setup (Complete Before The Workshop)
Before the workshop:

1. **GitHub Account (optional)** – Only needed for the optional episode on pushing your own code to GitHub.
2. **AWS Access** – Use a **shared AWS account** (UW-Madison) or an AWS account of your own.
3. **Data** – Download one small zip.
4. **Glossary** — Skim the workshop glossary.
5. **Episodes 1 to 4 first** — They create the bucket and notebook instance every later episode uses.

Details on each step are outlined below.

### 1. GitHub Account (optional)
Only needed for the optional episode on pushing your notebooks to GitHub. [Sign up](https://github.com/) if you plan to do it.

### 2. AWS Account 
There are two ways to get access to AWS for this lesson. Please wait for a pre-workshop email from the instructor to confirm which option to choose.

::::::::::::::::::::::::::::::::::::: tab

### Shared account (UW-Madison)

If you are attending this lesson at UW-Madison, the instructors will provide a shared AWS account for all attendees. You do not need to set up your own AWS account. What to expect:

* Before the workshop, the instructor adds you to the shared account and emails you a **sign-in link**. It opens the AWS console directly in the workshop account and role after you sign in with your UW–Madison NetID. Bookmark it; it is the link you will use every session. The link is sent by email rather than posted here on purpose: treat it like any other sign-in page and only use the one your instructor sent.
* **Verify your access before the workshop**: open the link and sign in. If you see a "no access" or "not assigned" message, email the instructor with your username. Check that the account name and region in the top-right of the console match the email.
* Three rules on the shared account:
    * Stick to the lesson materials, so that expensive pipelines (e.g., training/tuning LLMs) do not lead to high costs and wasted credits.
    * Do not use any tools we do not explicitly cover without discussing with the instructors first.
    * Stop (and at the end of the workshop, delete) every notebook instance you create. A GPU notebook left running costs about `$60` a day.
* The account enforces a short list of rules on every launch (the three cost tags with values from fixed lists, the lesson's instance types, a 48-hour cap, no endpoints). A job that breaks one is refused with an `AccessDenied` message that does not say why; [If a Launch Is Refused](refused-launch.md) does, and the lesson's helper module can tell you before and after the launch. Check there before emailing.

### Your own AWS account

If you are not attending through the shared account, follow these steps:

1. Create an account at [aws.amazon.com](https://aws.amazon.com/) on the **paid plan**. Since July 2025 new accounts start on a free plan limited to a set of free-tier services, which does not cover GPU instances.
2. Everything in the workshop is billed. Following the materials end to end costs about **`$10`**: the Titanic episodes a few dollars, the Bedrock episode cents, the Processing Jobs episode under a dollar, the fine-tuning episode about `$0.75` in class plus about `$1.50` for the optional homework. The one way to spend much more is to leave a GPU notebook instance running: about `$60` a day. Set a [Budgets alert](https://docs.aws.amazon.com/cost-management/latest/userguide/budgets-managing-costs.html) at `$50` before you start.
3. **Request GPU quotas three to four weeks before the workshop.** New accounts have a quota of 0 for GPU instances, and increases can take weeks. In the console, go to **Service Quotas → Amazon SageMaker** and request: `ml.g5.xlarge for training job usage` (1), `ml.g4dn.xlarge for training job usage` (1), `ml.g5.xlarge for processing job usage` (2), and `ml.g6e.xlarge for notebook instance usage` (1, optional GPU-notebook episode only). A justification like "self-paced machine learning workshop" is enough.
4. Check under **Amazon Bedrock → Model access** that the RAG episode's models, DeepSeek V3 and Titan Text Embeddings V2, show as available in your region. Bedrock enables its serverless models automatically; only Anthropic's need a use-case form.
5. Stage the shared datasets yourself, since there is no instructor bucket: the WattBot files for the RAG episodes, and the Badger Scribe data and the CHURRO-3B weights for the fine-tuning episode. The commands are in the lesson's instructor page *Shared data bucket: staging before each term*; set `shared_bucket` in those episodes to your own bucket.

:::::::::::::::::::::::::::::::::::::

Once your AWS account is set up, log in to the **AWS Management Console**. For the shared account that means the sign-in link from your instructor's email, of the form `https://<org>.awsapps.com/start/#/console?account_id=...&role_name=...`; for your own account, the regular console sign-in at `https://console.aws.amazon.com/`.

### 3. Download the Data

1. Download [data.zip](https://raw.githubusercontent.com/carpentries-incubator/ML_with_AWS_SageMaker/main/data/data.zip) (Right-click -> Save as).
2. Extract it (Right-click -> Extract all on Windows; double-click on Mac).
3. Keep the two Titanic files somewhere you can find them during the S3 episode, e.g.

- `~/Downloads/data/titanic/titanic_train.csv`
- `~/Downloads/data/titanic/titanic_test.csv`


### 4. Review the Workshop Glossary Page  
Skim the [Glossary](https://carpentries-incubator.github.io/ML_with_AWS_SageMaker/reference.html) before the workshop (it is also in the top menu of every lesson page); the terms come up from the first episode.

### 5. Episodes 1 to 4 come first

The first four episodes are the setup for everything after them, and the instructor walks through them with the class rather than leaving them as reading. Episode 2 creates the S3 bucket that every later episode writes to, and episode 3 creates the notebook instance and clones this lesson's repository onto it, which is where the prefilled notebooks for every episode come from: each later episode opens with "Open the notebook from `/ML_with_AWS_SageMaker/notebooks/...`", and that folder exists only once episode 3 has been done. Episode 4 is where the notebook first reads your bucket, with the helper that every later data step uses.

If you want to jump ahead to a later episode, do episodes 1 to 4 first on the instance you will use, and then start wherever you like. After that, the training episodes 6, 7 and 9 are a sequence (the tuning episode, 9, uses the files 7 saves), the fine-tuning episode 8 stands on its own, and the RAG episodes (10 to 13) are a sequence, since each one reuses what the one before it cached in your bucket.

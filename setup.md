---
title: Setup
---

## Setup (Complete Before The Workshop)
Before attending this workshop, you'll need to complete a few setup steps to ensure you can follow along smoothly. The main requirements are:

1. **GitHub Account (optional)** – Only needed for the optional episode on pushing your own code to GitHub.
2. **AWS Access** – Use a **shared AWS account** (UW-Madison) or sign up for an AWS Free Tier account.
3. **Data** – Download one small zip.
4. **Visit Glossary** — Find and briefly review the workshop glossary

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
* This setup ensures that all participants have the same environment and eliminates concerns about unexpected costs for attendees.
* These shared AWS credits should not be wasted, as we repurpose them for additional training events each year. Three rules follow from that:
    * Stick to the lesson materials, so that expensive pipelines (e.g., training/tuning LLMs) do not lead to high costs and wasted credits.
    * Do not use any tools we do not explicitly cover without discussing with the instructors first.
    * Stop (and at the end of the workshop, delete) every notebook instance you create. A GPU notebook left running costs about \$60 a day.

### Your own AWS account

If you are not attending through the shared account, follow these steps:

1. Go to the [AWS Free Tier page](https://aws.amazon.com/free/) and click **Create a Free Account**.
2. Complete the sign-up process. AWS's free tier covers very little of this workshop: SageMaker GPU instances, Bedrock calls and Processing jobs are all billed. Following the materials end to end costs about **\$30** (the Titanic episodes a few dollars, the Bedrock episode cents, the Processing Jobs episode about \$2, the fine-tuning episode about \$1.55 in class plus about \$1.75 for the optional homework). The one way to spend much more is to leave a GPU notebook instance running: about \$60 a day. Set a [Budgets alert](https://docs.aws.amazon.com/cost-management/latest/userguide/budgets-managing-costs.html) at \$50 before you start.
3. **Request GPU quotas three to four weeks before the workshop.** New accounts have a quota of 0 for GPU instances, and increases can take weeks. In the console, go to **Service Quotas → Amazon SageMaker** and request: `ml.g5.xlarge for training job usage` (1), `ml.g4dn.xlarge for training job usage` (1), `ml.g5.xlarge for processing job usage` (2), `ml.g6e.xlarge for processing job usage` (1), and `ml.g6e.xlarge for notebook instance usage` (1, optional GPU-notebook episode only). A justification like "self-paced machine learning workshop" is enough.
4. Enable the two Bedrock models the RAG episode uses, under **Amazon Bedrock → Model access**: DeepSeek V3 and Titan Text Embeddings V2.
5. Stage the shared datasets yourself, since there is no instructor bucket: the commands are in the lesson's instructor page *Shared data bucket: staging before each term*.

:::::::::::::::::::::::::::::::::::::

Once your AWS account is set up, log in to the **AWS Management Console**. For the shared account that means the sign-in link from your instructor's email, of the form `https://<org>.awsapps.com/start/#/console?account_id=...&role_name=...`; for your own account, the regular console sign-in at `https://console.aws.amazon.com/`.

### 3. Download the Data

1. Download [data.zip](https://raw.githubusercontent.com/carpentries-incubator/ML_with_AWS_SageMaker/main/data/data.zip) (Right-click -> Save as).
2. Extract it (Right-click -> Extract all on Windows; double-click on Mac).
3. Keep the two Titanic files somewhere you can find them during the S3 episode, e.g.

- `~/Downloads/data/titanic/titanic_train.csv`
- `~/Downloads/data/titanic/titanic_test.csv`


### 4. Review the Workshop Glossary Page  
When learning cloud tools for the first time, understanding new terminology is half the battle. We encourage learners to *briefly review* the [Glossary page](https://carpentries-incubator.github.io/ML_with_AWS_SageMaker/reference.html#glossary) (also accessible from the top menu of each lesson page) before the workshop.  *You don't need to memorize the terms*—just a quick read-through will help familiarize you with key concepts. Once we start running our own AWS SageMaker experiments, these terms will start to make more sense in context. If you feel lost at any point during the workshop, please ask the instructor for assistance or refer back to the glossary.

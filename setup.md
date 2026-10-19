---
title: Setup
---

## Setup (Complete Before The Workshop)
Before attending this workshop, you'll need to complete a few setup steps to ensure you can follow along smoothly. The main requirements are:

1. **GitHub Account (optional)** – Only needed for the optional episode on pushing your own code to GitHub.
2. **AWS Access** – Use a **shared AWS account** (if attending Machine Learning Marathon or Research Bazaar) or sign up for an AWS Free Tier account.
3. **Data** – Download one small zip (Titanic CSVs); the two larger datasets are staged for you.
4. **Visit Glossary** — Find and briefly review the workshop glossary
5. **(Optional) AWS Skill Builder** — For a broader overview of AWS, visit the [Getting Started with the AWS Cloud Essentials](https://skillbuilder.aws/search?searchText=Getting+Started+with+the+AWS+Cloud+Essentials) course

Details on each step are outlined below.

### 1. GitHub Account (optional)
All of the workshop's code lives in the lesson repository itself, which you clone (read-only, no account needed) onto your notebook instance in the *Accessing S3* episode with one command that every later episode repeats. A GitHub account is only needed for the optional self-study episode on pushing your own notebooks to GitHub from SageMaker. If you plan to do that episode and don't have an account, [sign up for GitHub](https://github.com/) (free); the episode walks through the rest.

### 2. AWS Account 
There are two ways to get access to AWS for this lesson. Please wait for a pre-workshop email from the instructor to confirm which option to choose.

#### Option 1) Shared Account
If you are attending this lesson as part of the Machine Learning Marathon or Research Bazaar, the instructors will provide a shared AWS account for all attendees. You do not need to set up your own AWS account. What to expect:

* Before the workshop, the instructor adds you to the shared account and emails you a **sign-in link**. It opens the AWS console directly in the workshop account and role after you sign in with your university login (UW–Madison NetID for the ML Marathon). Bookmark it; it is the link you will use every session. The link is sent by email rather than posted here on purpose: treat it like any other sign-in page and only use the one your instructor sent.
* **Verify your access before the workshop** (two minutes, and it is much easier to fix beforehand than on the day):
  1. Open the link and sign in. If you see a "no access" or "not assigned" message, email the instructor with your username; you have not been added yet.
  2. Check the top-right of the console: the account name and the region should be the ones your instructor named in the email. If the region shows anything else, switch it; everything in the workshop lives in one region.
  3. Type **SageMaker** in the search bar and open **Amazon SageMaker AI**. If the page loads without an error banner, you are set.
* This setup ensures that all participants have the same environment and eliminates concerns about unexpected costs for attendees.
* These shared AWS credits should not be wasted, as we repurpose them for additional training events each year.
  * Attendees are expected to **stick to the lesson materials** to ensure expensive pipelines (e.g., training/tuning LLMs) do not lead to high costs and wasted credits.
  * Do not use any tools we do not explicitly cover without discussing with the instructors first.
  * Stop (and at the end of the workshop, delete) every notebook instance you create. A GPU notebook left running costs about $60 a day.

#### Option 2) AWS Free Tier — Skip If Using Shared Account
**If you are attending this lesson as part of the Machine Learning Marathon or Research Bazaar, you can skip this step**. We will provide all attendees with a shared account. Otherwise, please follow these steps:

1. Go to the [AWS Free Tier page](https://aws.amazon.com/free/) and click **Create a Free Account**.
2. Complete the sign-up process. AWS's free tier covers very little of this workshop: SageMaker GPU instances, Bedrock calls and Processing jobs are all billed. Following the materials end to end costs about **$30** (the Titanic episodes a few dollars, the Bedrock episode cents, the Processing Jobs episode about $2, the fine-tuning episode about $1.55 in class plus about $1.75 for the optional homework). The one way to spend much more is to leave a GPU notebook instance running: about $60 a day. Set a [Budgets alert](https://docs.aws.amazon.com/cost-management/latest/userguide/budgets-managing-costs.html) at $50 before you start.
3. **Request GPU quotas three to four weeks before the workshop.** New accounts have a quota of 0 for GPU instances, and increases can take weeks. In the console, go to **Service Quotas → Amazon SageMaker** and request: `ml.g5.xlarge for training job usage` (1), `ml.g4dn.xlarge for training job usage` (1), `ml.g5.xlarge for processing job usage` (2), `ml.g6e.xlarge for processing job usage` (1), and `ml.g6e.xlarge for notebook instance usage` (1, optional GPU-notebook episode only). A justification like "self-paced machine learning workshop" is enough.
4. Enable the two Bedrock models the RAG episode uses, under **Amazon Bedrock → Model access**: DeepSeek V3 and Titan Text Embeddings V2.
5. Stage the shared datasets yourself, since there is no instructor bucket: the commands are in the lesson's instructor page *Shared data bucket: staging before each term*.

Once your AWS account is set up, log in to the **AWS Management Console** to get started with SageMaker.

### 3. Download the Data

The workshop uses three datasets. You download only the first; the other two are already in AWS when you arrive.

| Dataset | Used in | Size | What you do |
|---|---|---|---|
| **Titanic** passenger CSVs (predict survival) | S3, training and tuning episodes | 90 KB | Download now, upload to your own bucket in the S3 episode |
| **WattBot** energy-literature Q&A (paper index, labelled questions, PDF corpus) | RAG episodes | ~200 MB with PDFs | Nothing. Read from the shared bucket |
| **Badger Scribe** German Kurrent letters with transcriptions (a [release on the lesson repository](https://github.com/carpentries-incubator/ML_with_AWS_SageMaker/releases)) | Fine-tuning episode | ~130 MB | Nothing. Read from the shared bucket |

1. Download this zip (Right-click -> Save as): [data.zip](https://raw.githubusercontent.com/carpentries-incubator/ML_with_AWS_SageMaker/main/data/data.zip)

2. Extract it (Right-click -> Extract all on Windows; double-click on Mac). It contains a `titanic/` folder and a `wattbot/` folder.

3. Keep the two Titanic files somewhere you can find them during the S3 episode, e.g.

- `~/Downloads/data/titanic/titanic_train.csv`
- `~/Downloads/data/titanic/titanic_test.csv`

The `wattbot/` folder holds the two WattBot CSVs (`metadata.csv`, `train_QA.csv`) for anyone staging their own bucket; in the workshop you will not use them from your laptop, because the shared bucket already has them alongside the PDF corpus, which is not in the zip.

In the S3 episode (*Data Storage: Setting up S3*), you will create one S3 bucket of your own, upload the Titanic data to it, and use that bucket for your outputs for the whole workshop. The WattBot and Badger Scribe data live in a shared, read-only bucket that the instructors stage before class; the instructor tells you its name at the start of the RAG and fine-tuning episodes.

### 4. Review the Workshop Glossary Page  
When learning cloud tools for the first time, understanding new terminology is half the battle. We encourage learners to *briefly review* the [Glossary page](https://carpentries-incubator.github.io/ML_with_AWS_SageMaker/reference.html#glossary) (also accessible from the top menu of each lesson page) before the workshop.  *You don't need to memorize the terms*—just a quick read-through will help familiarize you with key concepts. Once we start running our own AWS SageMaker experiments, these terms will start to make more sense in context. If you feel lost at any point during the workshop, please ask the instructor for assistance or refer back to the glossary.

### 5. (Optional) AWS Skill Builder — Getting Started with the AWS Cloud Essentials
Attendees who want a stronger foundational understanding of AWS before diving into SageMaker and other ML services are encouraged to complete this self-paced course, [Getting Started with the AWS Cloud Essentials](https://skillbuilder.aws/search?searchText=Getting+Started+with+the+AWS+Cloud+Essentials), as optional pre-work for this workshop. It's designed for beginners and provides a broad overview of the AWS Cloud, including core services, global infrastructure, pricing basics, and security concepts. 


---
title: Setup
---

## Setup (Complete Before The Workshop)
Before attending this workshop, you'll need to complete a few setup steps to ensure you can follow along smoothly. The main requirements are:

1. **GitHub Account** – Create an account and be ready to fork a repository.
2. **AWS Access** – Use a **shared AWS account** (if attending Machine Learning Marathon or Research Bazaar) or sign up for an AWS Free Tier account.
3. **Titanic Dataset** – Download the required CSV files in advance.
4. **Workshop Repository** – Fork the provided GitHub repository for use in AWS.
5. **Visit Glossary** — Find and briefly review the workshop glossary
6. **(Optional) AWS Skill Builder** — For a broader overview of AWS, visit the [Getting Started with the AWS Cloud Essentials](https://skillbuilder.aws/search?searchText=Getting+Started+with+the+AWS+Cloud+Essentials) course

Details on each step are outlined below.

### 1. GitHub Account
You will need a GitHub account to access the code provided during this lesson. If you don't already have a GitHub account, please [sign up for GitHub](https://github.com/) to create a free account. Don't worry if you're a little rusty on using GitHub/git; we will only use a couple of git commands during the lesson, and the instructor will provide guidance on these steps.

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
2. Complete the sign-up process. AWS's free tier covers very little of this workshop: SageMaker GPU instances, Bedrock calls and Processing jobs are all billed. Following the materials end to end costs about **$30** (the Titanic episodes a few dollars, the Bedrock episode cents, the Processing Jobs episode about $2, the fine-tuning episode about $3). The one way to spend much more is to leave a GPU notebook instance running: about $60 a day. Set a [Budgets alert](https://docs.aws.amazon.com/cost-management/latest/userguide/budgets-managing-costs.html) at $50 before you start.
3. **Request GPU quotas at least two weeks before the workshop.** New accounts have a quota of 0 for GPU instances, and increases take days. In the console, go to **Service Quotas → Amazon SageMaker** and request: `ml.g5.xlarge for training job usage` (1), `ml.g4dn.xlarge for training job usage` (1), `ml.g5.xlarge for processing job usage` (2), and `ml.g6e.xlarge for processing job usage` (1). A justification like "self-paced machine learning workshop" is enough.
4. Enable the two Bedrock models the RAG episode uses, under **Amazon Bedrock → Model access**: DeepSeek V3 and Titan Text Embeddings V2.
5. Stage the shared datasets yourself, since there is no instructor bucket: the commands are in the lesson's instructor page *Shared data bucket: staging before each term*.

Once your AWS account is set up, log in to the **AWS Management Console** to get started with SageMaker.

### 3. Download the Data

For this workshop, you will need the Titanic dataset, which can be used to train a classifier predicting survival. 

1. Please download the following zip file (Right-click -> Save as): [data.zip](https://raw.githubusercontent.com/carpentries-incubator/ML_with_AWS_SageMaker/main/data/data.zip)

2. Extract the zip folder contents (Right-click -> Extract all on Windows; Double-click on mac)

3. Save the two data files (train and test) to a location where they can easily be accessed. E.g., ... 

- `~/Downloads/data/titanic_train.csv`
- `~/Downloads/data/titanic_test.csv`

In the first episode, you will create an S3 bucket and upload this data to use with SageMaker.

The later episodes use two more datasets, and you do **not** need to download them: the WattBot papers (RAG episodes) and the Badger Scribe handwriting data (fine-tuning episode, published as a [release on the lesson repository](https://github.com/carpentries-incubator/ML_with_AWS_SageMaker/releases)) are staged by the instructors in a shared, read-only bucket. You create one bucket of your own in the S3 episode and use it for the whole workshop. No Kaggle account is required for any part of the workshop.

### 4. Get Access To Workshop Code (Fork GitHub Repo)

You will need a copy of our AWS_helpers repo on GitHub to explore how to manage your repo in AWS. This setup will allow you to follow along with the workshop and test out the Interacting with Repositories episode.

To do this:

1. Go to the [AWS_helpers GitHub repository](https://github.com/UW-Madison-DataScience/AWS_helpers).
2. Click **Fork** (top right) to create your own copy of the repository under your GitHub account. You will only need the main branch. You can leave "Copy the main branch only" selected. 
3. Once forked, you don't need to do anything else. We'll clone this fork once we start working in the AWS Jupyter environment using...

```python
!git clone https://github.com/YOUR_GITHUB_USERNAME/AWS_helpers.git
```

### 5. Review the Workshop Glossary Page  
When learning cloud tools for the first time, understanding new terminology is half the battle. We encourage learners to *briefly review* the [Glossary page](https://carpentries-incubator.github.io/ML_with_AWS_SageMaker/reference.html#glossary) (also accessible from the top menu of each lesson page) before the workshop.  *You don't need to memorize the terms*—just a quick read-through will help familiarize you with key concepts. Once we start running our own AWS SageMaker experiments, these terms will start to make more sense in context. If you feel lost at any point during the workshop, please ask the instructor for assistance or refer back to the glossary.

### 6. (Optional) AWS Skill Builder — Getting Started with the AWS Cloud Essentials
Attendees who want a stronger foundational understanding of AWS before diving into SageMaker and other ML services are encouraged to complete this self-paced course, [Getting Started with the AWS Cloud Essentials](https://skillbuilder.aws/search?searchText=Getting+Started+with+the+AWS+Cloud+Essentials), as optional pre-work for this workshop. It's designed for beginners and provides a broad overview of the AWS Cloud, including core services, global infrastructure, pricing basics, and security concepts. 


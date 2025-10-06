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

Details on each step are outlined below.

### 1. GitHub Account
You will need a GitHub account to access the code provided during this lesson. If you don't already have a GitHub account, please [sign up for GitHub](https://github.com/) to create a free account. Don't worry if you're a little rusty on using GitHub/git; we will only use a couple of git commands during the lesson, and the instructor will provide guidance on these steps.

### 2. AWS Account 
There are two ways to get access to AWs for this lesson. Please wait for a pre-workshop email from the instructor to confirm which option to choose.

#### Option 1) Shared Account
If you are attending this lesson as part of the Machine Learning Marathon or Research Bazaar, the instructors will provide a shared AWS account for all attendees. You do not need to set up your own AWS account. What to expect:

* Before the workshop, you will receive an email invitation from the instructor with access details for the shared AWS account.
* During the lesson, you will log in using the credentials provided in the email.
* This setup ensures that all participants have the same environment and eliminates concerns about unexpected costs for attendees.
* These shared AWS credits should not be wasted, as we repurpose them for additional training events eac year.
  * Attendees are expected to **stick to the lesson materials** to ensure expensive pipelines (e.g., training/tuning LLMs) do not lead to high costs and wasted credits.
  * Do not use any tools we do not explictly cover without discussing with the instructors first.

#### Option 2) AWS Free Tier — Skip If Using Shared Account
**If you are attending this lesson as part of the Machine Learning Marathon or Research Bazaar, you can skip this step**. We will provide all attendees with a shared account. Otherwise, please follow these steps:

1. Go to the [AWS Free Tier page](https://aws.amazon.com/free/) and click **Create a Free Account**.
2. Complete the sign-up process. AWS offers a free tier with limited monthly usage. Some services, including SageMaker, may incur charges beyond free-tier limits, so be mindful of usage during the workshop. If you follow along with the materials, you can expect to incur around $10 in compute fees (e.g., from training and tuning several different models with GPU enabled at times).

Once your AWS account is set up, log in to the **AWS Management Console** to get started with SageMaker.

### 3. Download the Data

For this workshop, you will need the Titanic dataset, which can be used to train a classifier predicting survival. 

1. Please download the following zip file (Right-click -> Save as): [data.zip](https://raw.githubusercontent.com/UW-Madison-DataScience/ml-with-aws-sagemaker/main/data/data.zip)

2. Extract the zip folder contents (Right-click -> Extract all on Windows; Double-click on mac)

3. Save the two data files (train and test) to a location where they can easily be accessed. E.g., ... 

- `~/Desktop/data/titanic_train.csv`
- `~/Desktop/data/titanic_test.csv`

In the first episode, you will create an S3 bucket and upload this data to use with SageMaker.

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
When learning cloud tools for the first time, understanding new terminology is half the battle. We encourage learners to *briefly review* the [Glossary page](https://carpentries-incubator.github.io/ML_with_AWS_SageMaker/reference.html#glossary) (also accessible from the top menu of each lesson page) before the workshop.  **You don't need to memorize the terms**—just a quick read-through will help familiarize you with key concepts. Once we start running our own AWS SageMaker experiments, these terms will start to make more sense in context. If you feel lost at any point during the workshop, please ask the instructor/helpers for assistance and/or refer back to the glossary.


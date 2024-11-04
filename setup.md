---
title: Setup
---

## Overview

This workshop introduces you to foundational workflows in **AWS SageMaker**, covering data setup, model training, hyperparameter tuning, and model deployment within AWS's managed environment. You’ll learn how to use SageMaker notebooks to control data pipelines, manage training jobs, and evaluate model performance effectively. We’ll also cover strategies to help you scale training and tuning efficiently, with guidance on choosing between CPUs and GPUs, as well as when to consider parallelized training.

To keep costs manageable, this workshop provides tips for tracking and monitoring AWS expenses, so your experiments remain affordable. While AWS isn’t entirely free, it’s very cost-effective for typical ML workflows—training roughly 100 models on a small dataset (under 10GB) can cost under $20, making it accessible for many research projects. 

### What This Workshop Does Not Cover

Currently, this workshop does not include:
- **AWS Lambda** for serverless function deployment,
- **MLFlow** or other MLOps tools for experiment tracking,
- Additional AWS services beyond the core SageMaker ML workflows.

If there’s a specific ML workflow or AWS service you’d like to see included in this curriculum, we’re open to developing more content to meet the needs of researchers and ML practitioners at UW–Madison. Please contact [endemann@wisc.edu](mailto:endemann@wisc.edu) with suggestions or requests.

## Accounts and Initial Setup

### GitHub Account

If you don't already have a GitHub account, [sign up for GitHub](https://github.com/) to create a free account. A GitHub account will be required to fork and interact with the lesson repository.

### AWS Account

If you don't have an AWS account, please follow these steps:

> **Note**: Hackathon attendees can skip this step since we are providing you with the account.

1. Go to the [AWS Free Tier page](https://aws.amazon.com/free/) and click **Create a Free Account**.
2. Complete the sign-up process. AWS offers a free tier with limited monthly usage. Some services, including SageMaker, may incur charges beyond free-tier limits, so be mindful of usage during the workshop. If you follow along with the materials, you can expect to incur around $10 in compute fees (e.g., from training and tuning several different models with GPU enabled at times).

Once your AWS account is set up, log in to the **AWS Management Console** to get started with SageMaker.

## Data Sets

For this workshop, you will need the **Titanic dataset**. Please download the following files:

- [titanic_train.csv](https://raw.githubusercontent.com/UW-Madison-DataScience/ml-with-aws-sagemaker/main/data/titanic_train.csv)
- [titanic_test.csv](https://raw.githubusercontent.com/UW-Madison-DataScience/ml-with-aws-sagemaker/main/data/titanic_test.csv)

Save these files to a location where they can easily be accessed. In the first episode, you will create an S3 bucket and upload this data to use with SageMaker.

## Workshop Repository Setup

You will need a copy of the lesson repository on GitHub to explore how to manage your repo in AWS. This setup will allow you to follow along with the workshop and test out the Interacting with Repositories episode.

To do this:

1. Go to the workshop's [GitHub repository page](https://github.com/UW-Madison-DataScience/ml-with-aws-sagemaker).
2. Click **Fork** (top right) to create your own copy of the repository under your GitHub account.
3. Once forked, you don't need to do anything else. We'll clone this fork once we start working in the AWS Jupyter environment using...

       ```bash
       !git clone https://github.com/YOUR_USERNAME/ml-with-aws-sagemaker.git
       ```  

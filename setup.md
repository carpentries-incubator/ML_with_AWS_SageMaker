---
title: Setup
---

## Setup (Complete Before The Workshop)
Before attending this workshop, you'll need to complete a few setup steps to ensure you can follow along smoothly. The main requirements are:

1. **GitHub Account** – Create an account and be ready to fork a repository.
2. **AWS Access** – Use a **shared AWS account** (if attending the 2025 Research Bazaar) or sign up for an AWS Free Tier account.
3. **Titanic Dataset** – Download the required CSV files in advance.
4. **Workshop Repository** – Fork the provided GitHub repository for use in AWS.

Details on each step are outlined below.

### 1. GitHub Account
You will need a GitHub account to access the code provided during this lesson. If you don't already have a GitHub account, please [sign up for GitHub](https://github.com/) to create a free account. Don't worry if you're a little rusty on using GitHub/git; we will only use a couple of git commands during the lesson, and the instructor will provide guidance on these steps.

### 2. AWS Account 
There are two ways to get access to AWs for this lesson. Please wait for a pre-workshop email from the instructor to confirm which option to choose.

#### Option 1) Shared Account
If you are attending this lesson as part of the 2025 Research Bazaar, we will provide a shared AWS account for all attendees. You do not need to set up your own AWS account. What to expect:

* Before the workshop, you will receive an email invitation from the instructor with access details for the shared AWS account.
* During the lesson, you will log in using the credentials provided in the email.
* This setup ensures that all participants have the same environment and eliminates concerns about unexpected costs.

#### Option 2) AWS Free Tier — Skip If Using Shared Account
**If you are attending this lesson as part of the 2025 Research Bazaar, you can skip this step**. We will provide all attendees with a shared account. Otherwise, please follow these steps:

1. Go to the [AWS Free Tier page](https://aws.amazon.com/free/) and click **Create a Free Account**.
2. Complete the sign-up process. AWS offers a free tier with limited monthly usage. Some services, including SageMaker, may incur charges beyond free-tier limits, so be mindful of usage during the workshop. If you follow along with the materials, you can expect to incur around $10 in compute fees (e.g., from training and tuning several different models with GPU enabled at times).

Once your AWS account is set up, log in to the **AWS Management Console** to get started with SageMaker.

### 3. Download the Data

For this workshop, you will need the **Titanic dataset**. Please download the following files by right clicking each and selecting `Save as`. Make sure to save them out as .csvs:

- [titanic_train.csv](https://raw.githubusercontent.com/UW-Madison-DataScience/ml-with-aws-sagemaker/main/data/titanic_train.csv)
- [titanic_test.csv](https://raw.githubusercontent.com/UW-Madison-DataScience/ml-with-aws-sagemaker/main/data/titanic_test.csv)

Save these files to a location where they can easily be accessed. In the first episode, you will create an S3 bucket and upload this data to use with SageMaker.

### 4. Get Access To Workshop Code (Fork GitHub Repo)

You will need a copy of our AWS_helpers repo on GitHub to explore how to manage your repo in AWS. This setup will allow you to follow along with the workshop and test out the Interacting with Repositories episode.

To do this:

1. Go to the [AWS_helpers GitHub repository](https://github.com/UW-Madison-DataScience/AWS_helpers).
2. Click **Fork** (top right) to create your own copy of the repository under your GitHub account. 
3. Once forked, you don't need to do anything else. We'll clone this fork once we start working in the AWS Jupyter environment using...

```python
!git clone https://github.com/YOUR_GITHUB_USERNAME/AWS_helpers.git
```  

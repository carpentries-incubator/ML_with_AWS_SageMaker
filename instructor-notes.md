---
title: 'AWS Nuke Script Tutorial'
name: Zain Waseem & Chris Endemann
---

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

**aws sts get-caller-identity**

### 2\. Create a config file

**In order to run the script, you'll need to make a config file. This file MUST be in .yml or .yaml format or else your script will FAIL.**

1. Select Windows + E to open file-explorer.
2. Click on the documents tab on the left. Then click on the "New" button in the
top left corner.
3. Select "Text Document," please use notepad for this as it will be easy to edit.
4. Once you've opened notepad. This is the format our config file will
follow:

regions:
  - global
  - us-east-2 # do not change

blocklist:
  - "551379999285" # production, do not modify

accounts:
"183295408236": # aws-nuke-account
  resource-types:
    includes: #Specifies which types of resources you want to clean out
     - S3Bucket
     - SageMakerNotebookInstance

**MAKE SURE TO USE SPACE INDENTING AND NOT TAB INDENTING AS IT IS SENSITIVE TO USE IN CLOUDSHELL AND MAY CAUSE THE SCRIPT TO FAIL**

5. Once you've gotten your config file set up, click on **"FILE" > "SAVE AS"**
6. Label your doc as "nuke-config.yml" under "File Name:" \& Ensure "All files (\*)"
is selected under "Save as type:"

### 3\. Upload your config file to CloudShell

In CloudShell, choose **"Actions" > "Upload file"**, then upload "nuke-config.yml"

To confirm it's inside your Shell:

**pwd
ls -l nuke-config.yml
cat nuke-config.yml**

### 4\. Download and install the linux version

AWS CloudShell uses Linux as their base distribution. So you will need to download
the linux version of AWS Nuke from Github. You should download only from the
official ekristen/aws-nuke releases-not use any local verision.

Link: https://github.com/ekristen/aws-nuke/releases

Run the following script:

mkdir -p "$HOME/bin"
cd "$HOME"

curl -fL   
-o aws-nuke-v3.67.0-linux-amd64.tar.gz   
https://github.com/ekristen/aws-nuke/releases/download/v3.67.0/aws-nuke-v3.67.0-linux-amd64.tar.gz

tar -xzf aws-nuke-v3.67.0-linux-amd64.tar.gz
mv aws-nuke "$HOME/bin/aws-nuke"
chmod 700 "$HOME/bin/aws-nuke"

export PATH="$HOME/bin:$PATH"
aws-nuke --version

To make aws-nuke stay on your PATH in later CloudShell sessions:

**echo 'export PATH="$HOME/bin:$PATH"' >> \~/.bashrc
source \~/.bashrc**

Reminder: CloudShell is based on Amazon Linux, and software saved in $HOME
persists; software installed elsewhere may disappear after the session ends.

### 5\. DO a dry run first

It is important that you first run a dry-run so that you can verify which
files you want to remove before executing the actual script.

Here is the script to do so:

**aws-nuke run --config "$HOME/nuke-config.yml"**

This should list resources it would remove, without deleting them. Review the
output carefully—especially S3 buckets, IAM-related resources, and the displayed
account identity.


### 6\. Only after review: actual deletion

Only if the dry-run output is correct and this is truly the intended non-production account:

**aws-nuke run --config "$HOME/nuke-config.yml" --no-dry-run**

Do not add --no-prompt / --force on your first real run; leave the confirmation
prompt enabled as an additional safeguard.


Oftentimes when running **aws-nuke run --config "$HOME/nuke-config.yml"**, you
will run into an error stating:

WARN\[0000] 'targets' is deprecated. Please use 'includes' instead.
aws-nuke - v3.67.0 - 9cc98ccaed6303dac29001642879320ab584d41a
FATA\[0000] specified account doesn't have an alias. For safety reasons you need
to specify an account alias. Your production account should contain the term 'prod'


Do not panic, this isn't related to CloudShell or credentials. Rather, it connected to AWS successfully but stopped abruptly since the account you are trying to target has **no AWS account alias.** By default, aws-nuke requires
an alias so you must visually confirm a recognizable account name before proceeding. It also blocks aliases containing prod by default

Recommended: create a non-production account alias
First, verify the exact account:

**aws sts get-caller-identity**

Then add an alias in the AWS Console:

1. Go to IAM
2. Open Dashboard
3. Under AWS Account, choose Create Alias
4. Set a clear, unique non-production alias—e.g. mycompany-sandbox or training-cleanup
5. Do not use a name containing prod, production, or live.

Then run the dry-run again:
**aws-nuke run --config "$HOME/nuke-config.yml"**

Congratulations, you have reached the end of this tutorial! By now, all scripts
should work and the specified notebooks/buckets you wanted to clean should
now be all empty!

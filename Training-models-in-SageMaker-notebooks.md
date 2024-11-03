---
title: "Training models in SageMaker notebooks"
teaching: 20
exercises: 10
---

:::::::::::::::::::::::::::::::::::::: questions 

- How can I initialize the SageMaker environment and set up data in S3?
- What are the differences between local training and SageMaker-managed training?
- How do Estimator classes in SageMaker streamline the training process for various frameworks?

::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: objectives

- Set up and initialize the SageMaker environment, including roles, sessions, and S3 data.
- Understand the difference between training locally in a SageMaker notebook and using SageMaker's managed infrastructure.
- Learn to configure and use SageMaker's Estimator classes for different frameworks (e.g., XGBoost, PyTorch, SKLearn).
- Compare performance, cost, and setup between custom scripts and built-in images in SageMaker.
- Conduct training with data stored in S3 and monitor training job status using the SageMaker console.

::::::::::::::::::::::::::::::::::::::::::::::::


## Initialize SageMaker environment

This code initializes the AWS SageMaker environment by defining the SageMaker role, session, and S3 client. It also specifies the S3 bucket and key for accessing the Titanic training dataset stored in an S3 bucket.

#### Boto3 API
> Boto3 is the official AWS SDK for Python, allowing developers to interact programmatically with AWS services like S3, EC2, and Lambda. It provides both high-level and low-level APIs, making it easy to manage AWS resources and automate tasks. With built-in support for paginators, waiters, and session management, Boto3 simplifies working with AWS credentials, regions, and IAM permissions. It’s ideal for automating cloud operations and integrating AWS services into Python applications.


```python
import boto3
import pandas as pd
import sagemaker
from sagemaker import get_execution_role

# Initialize the SageMaker role (will reflect notebook instance's policy)
role = sagemaker.get_execution_role()
print(f'role = {role}')

# Create a SageMaker session to manage interactions with Amazon SageMaker, such as training jobs, model deployments, and data input/output.
session = sagemaker.Session()

# Initialize an S3 client to interact with Amazon S3, allowing operations like uploading, downloading, and managing objects and buckets.
s3 = boto3.client('s3')

# Define the S3 bucket that we will load from
bucket = 'titanic-dataset-test'  # replace with your S3 bucket name

# Define train/test filenames
train_filename = 'titanic_train.csv'
test_filename = 'titanic_test.csv'
```

    sagemaker.config INFO - Not applying SDK defaults from location: /etc/xdg/sagemaker/config.yaml
    sagemaker.config INFO - Not applying SDK defaults from location: /home/ec2-user/.config/sagemaker/config.yaml
    role = arn:aws:iam::183295408236:role/ml-sagemaker-use


### Download copy into notebook environment
If you have larger dataset (> 1GB), you may want to skip this step and always read directly into memory. However, for smaller datasets, it can be convenient to have a "local" copy (i.e., one that you store in your notebook's instance).

Download data from S3 to notebook environment. You may need to hit refresh on the file explorer panel to the left to see this file. If you get any permission issues...

* check that you have selected the appropriate policy for this notebook
* check that your bucket has the appropriate policy permissions


```python
# Define the S3 bucket and file location
file_key = f"data/{train_filename}"  # Path to your file in the S3 bucket
local_file_path = f"./{train_filename}"  # Local path to save the file

# Download the file using the s3 client variable we initialized earlier
s3.download_file(bucket, file_key, local_file_path)
print("File downloaded:", local_file_path)
```

    File downloaded: ./titanic_train.csv


We can do the same for the test set.


```python
# Define the S3 bucket and file location
file_key = f"data/{test_filename}"  # Path to your file in the S3 bucket. W
local_file_path = f"./{test_filename}"  # Local path to save the file

# Initialize the S3 client and download the file
s3.download_file(bucket, file_key, local_file_path)
print("File downloaded:", local_file_path)

```

    File downloaded: ./titanic_test.csv




::::::::::::::::::::::::::::::::::::: keypoints

- **Environment Initialization**: Setting up a SageMaker session, defining roles, and specifying the S3 bucket are essential for managing data and running jobs in SageMaker.
- **Local vs. Managed Training**: Local training in SageMaker notebooks can be useful for quick tests but lacks the scalability and resource management available in SageMaker-managed training.
- **Estimator Classes**: SageMaker provides framework-specific Estimator classes (e.g., XGBoost, PyTorch, SKLearn) to streamline training setups, each suited to different model types and workflows.
- **Custom Scripts vs. Built-in Images**: Custom training scripts offer flexibility with preprocessing and custom logic, while built-in images are optimized for rapid deployment and simpler setups.
- **Training Data Channels**: Using `TrainingInput` ensures SageMaker manages data efficiently, especially for distributed setups where data needs to be synchronized across multiple instances.
- **Distributed Training Options**: Data parallelism (splitting data across instances) is common for many models, while model parallelism (splitting the model across instances) is useful for very large models that exceed instance memory.

::::::::::::::::::::::::::::::::::::::::::::::::

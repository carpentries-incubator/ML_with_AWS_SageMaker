---
title: "Accessing and Managing Data in S3 with SageMaker Notebooks"
teaching: 20
exercises: 10
---

:::::::::::::::::::::::::::::::::::::: questions 

- How can I load data from S3 into a SageMaker notebook?
- How do I monitor storage usage and costs for my S3 bucket?
- What steps are involved in pushing new data back to S3 from a notebook?

::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: objectives

- Read data directly from an S3 bucket into memory in a SageMaker notebook.
- Check storage usage and estimate costs for data in an S3 bucket.
- Upload new files from the SageMaker environment back to the S3 bucket.

::::::::::::::::::::::::::::::::::::::::::::::::

## 1A. Read Data from S3 into Memory

Our data is stored in an S3 bucket called `titanic-dataset-test`. This approach reads data directly from S3 into memory within the Jupyter notebook environment without creating a local copy of `train.csv`.

```python
import boto3
import pandas as pd
import sagemaker
from sagemaker import get_execution_role

# Define the SageMaker role and session
role = sagemaker.get_execution_role()
session = sagemaker.Session()
s3 = boto3.client('s3')

# Define the S3 bucket and object key
bucket = 'titanic-dataset-test'  # replace with your S3 bucket name
key = 'data/titanic_train.csv'   # replace with your object key
response = s3.get_object(Bucket=bucket, Key=key)

# Load the data into a pandas DataFrame
train_data = pd.read_csv(response['Body'])
print(train_data.shape)
train_data.head()

::::::::::::::::::::::::::::::::::::: callout 

### Why Direct Reading?

Directly reading from S3 into memory minimizes storage requirements on the notebook instance and can handle large datasets without local storage limitations.

::::::::::::::::::::::::::::::::::::::::::::::::

## 1B. Download Data as a Local Copy

For smaller datasets, it can be convenient to have a local copy within the notebook’s environment. However, if your dataset is large (>1GB), consider skipping this step.

### Steps to Download Data from S3

```python
# Define the S3 bucket and file location
file_key = "data/titanic_train.csv"  # Path to your file in the S3 bucket
local_file_path = "./titanic_train.csv"  # Local path to save the file

# Download the file from S3
s3.download_file(bucket, file_key, local_file_path)
print("File downloaded:", local_file_path)
```

:::::::::::::::::::::::::::::::::::::: callout 

## Resolving Permission Issues

If you encounter permission issues when downloading from S3:
- Ensure your IAM role has appropriate policies for S3 access.
- Verify the bucket policy allows access.

::::::::::::::::::::::::::::::::::::::::::::::::

## 2. Check Current Size and Storage Costs of the Bucket

It’s useful to periodically check the storage usage and associated costs of your S3 bucket. Using the **Boto3** library, you can calculate the total size of objects within a specified bucket.

```python
# Initialize the total size counter
total_size_bytes = 0

# List and sum the size of all objects in the bucket
paginator = s3.get_paginator('list_objects_v2')
for page in paginator.paginate(Bucket=bucket):
    for obj in page.get('Contents', []):
        total_size_bytes += obj['Size']

# Convert the total size to megabytes for readability
total_size_mb = total_size_bytes / (1024 ** 2)
print(f"Total size of bucket '{bucket}': {total_size_mb:.2f} MB")
```

::::::::::::::::::::::::::::::::::::: callout 

### Explanation

1. **Paginator**: Handles large listings in S3 buckets.
2. **Size Calculation**: Sums the `Size` attribute of each object.
3. **Unit Conversion**: Size is converted from bytes to megabytes for readability.

> **Tip**: For large buckets, consider filtering by folder or object prefix to calculate size for specific directories.

::::::::::::::::::::::::::::::::::::::::::::::::

## 3. Estimate Storage Costs

AWS S3 costs are based on data size, region, and storage class. The example below estimates costs for the **S3 Standard** storage class in **US East (N. Virginia)**.

```python
# Pricing per GB for different storage tiers
first_50_tb_price_per_gb = 0.023
next_450_tb_price_per_gb = 0.022
over_500_tb_price_per_gb = 0.021

# Calculate the cost based on the size
total_size_gb = total_size_bytes / (1024 ** 3)
if total_size_gb <= 50 * 1024:
    cost = total_size_gb * first_50_tb_price_per_gb
elif total_size_gb <= 500 * 1024:
    cost = (50 * 1024 * first_50_tb_price_per_gb) + \
           ((total_size_gb - 50 * 1024) * next_450_tb_price_per_gb)
else:
    cost = (50 * 1024 * first_50_tb_price_per_gb) + \
           (450 * 1024 * next_450_tb_price_per_gb) + \
           ((total_size_gb - 500 * 1024) * over_500_tb_price_per_gb)

print(f"Estimated monthly storage cost: ${cost:.4f}")
```

> For up-to-date pricing details, refer to the [AWS S3 Pricing page](https://aws.amazon.com/s3/pricing/).

## Important Considerations:

- **Pricing Tiers**: S3 has tiered pricing, so costs vary with data size.
- **Region and Storage Class**: Prices differ by AWS region and storage class.
- **Additional Costs**: Consider other costs for requests, retrievals, and data transfer.

## 4. Upload New Files from Notebook to S3

As your analysis generates new files, you may need to upload them to your S3 bucket. Here’s how to upload a file from the notebook environment to S3.

```python
# Define the S3 bucket name and file paths
train_file_path = "results.txt"  # File to upload
s3.upload_file(train_file_path, bucket, "results/results.txt")
print("Files uploaded successfully.")
```

:::::::::::::::::::::::::::::::::::::: keypoints 

- Load data from S3 into memory for efficient storage and processing.
- Periodically check storage usage and costs to manage S3 budgets.
- Use SageMaker to upload analysis results and maintain an organized workflow.

::::::::::::::::::::::::::::::::::::::::::::::::

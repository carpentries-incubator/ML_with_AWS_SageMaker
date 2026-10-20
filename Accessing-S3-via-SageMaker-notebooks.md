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

#### Set up AWS environment
To begin each notebook, it's important to set up an AWS environment that will allow seamless access to the necessary cloud resources. Here's what we'll do to get started:

1. **Define the Role**: We'll use `get_execution_role()` to retrieve the IAM role associated with the SageMaker instance. This role specifies the permissions needed for interacting with AWS services like S3, which allows SageMaker to securely read from and write to storage buckets.

2. **Initialize the SageMaker Session**: Next, we'll create a `sagemaker.Session()` object, which will help manage and track the resources and operations we use in SageMaker, such as training jobs and model artifacts. The session acts as a bridge between the SageMaker SDK commands in our notebook and AWS services.

3. **Set Up an S3 Client using boto3**: Using `boto3`, we'll initialize an S3 client for accessing S3 buckets directly. Boto3 is the official AWS SDK for Python, allowing developers to interact programmatically with AWS services like S3, EC2, and Lambda.

Starting with these initializations prepares our notebook environment to efficiently interact with AWS resources for model development, data management, and deployment.

```python
import boto3
import sagemaker
from sagemaker import get_execution_role

# Initialize the SageMaker role, session, and s3 client
role = sagemaker.get_execution_role() # specifies your permissions to use AWS tools
session = sagemaker.Session() 
s3 = boto3.client('s3')

```

Preview variable details.

```python
# Print relevant details 
print(f"Execution Role: {role}")  # Displays the IAM role being used
bucket_names = [bucket["Name"] for bucket in s3.list_buckets()["Buckets"]]
print(f"Available S3 Buckets: {bucket_names}")  # Every bucket in the account this role can list, including other learners' buckets in a shared account
print(f"AWS Region: {session.boto_region_name}")  # The region of the notebook instance; your bucket must be in this same region
```

## Reading data from S3

You can either (A) read data from S3 into memory or (B) download a copy of your S3 data into your notebook instance. Since we are using SageMaker notebooks as controllers—rather than performing training or tuning directly in the notebook—the best practice is to **read data directly from S3** whenever possible. However, there are cases where downloading a local copy may be useful. We'll show you both strategies.

### A) Reading data directly from S3 into memory  
This is the recommended approach for most workflows. By keeping data in S3 and reading it into memory when needed, we avoid local storage constraints and ensure that our data remains accessible for SageMaker training and tuning jobs.

**Pros**:

- **Scalability**: Data remains in S3, allowing multiple training/tuning jobs to access it without duplication.
- **Efficiency**: No need to manage local copies or manually clean up storage.
- **Cost-effective**: Avoids unnecessary instance storage usage.

**Cons**:

- **Network dependency**: Requires internet access to S3.
- **Potential latency**: Reading large datasets repeatedly from S3 may introduce small delays. This approach works best if you only need to load data once or infrequently.

#### Example: Reading data from S3 into memory
Our data is stored in the S3 bucket you created in the previous episode, named 'challenge-lastnamefirstname' (e.g., sinkorswim-doejohn). We can use the following code to read data directly from S3 into memory in the Jupyter notebook environment, without actually downloading a copy of train.csv as a local file.

```python
import pandas as pd
# Define the S3 bucket and object key
bucket_name = 'sinkorswim-doejohn'  # replace with your S3 bucket name

# Read the train data from S3
key = 'titanic_train.csv'  # replace with your object key
response = s3.get_object(Bucket=bucket_name, Key=key)
train_data = pd.read_csv(response['Body'])

# Read the test data from S3
key = 'titanic_test.csv'  # replace with your object key
response = s3.get_object(Bucket=bucket_name, Key=key)
test_data = pd.read_csv(response['Body'])

# check shape
print(train_data.shape)
print(test_data.shape)

# Inspect the first few rows of the DataFrame
train_data.head()
```

### B) Download copy into notebook environment
In some cases, downloading a local copy of the dataset may be useful, such as when performing repeated reads in an interactive notebook session.

**Pros**:

- **Faster access for repeated operations**: Avoids repeated S3 requests.
- **Works offline**: Useful if running in an environment with limited network access.

**Cons**:

- **Consumes instance storage**: Notebook instances have limited space.
- **Requires manual cleanup**: Downloaded files remain until deleted.

#### Example

```python
!pwd
```


```python
# Define the S3 bucket and file location
key = "titanic_train.csv"  # Path to your file in the S3 bucket
local_file_path = "/home/ec2-user/SageMaker/titanic_train.csv"  # Local path to save the file

# Initialize the S3 client and download the file
s3.download_file(bucket_name, key, local_file_path)
!ls
```

**Note**: You may need to hit refresh on the file explorer panel to the left to see this file. If you get any permission issues...

* check that you have selected the appropriate policy for this notebook
* check that your bucket has the appropriate policy permissions

#### Check the current size and storage costs of bucket
It's a good idea to periodically check how much storage you have used in your bucket. You can do this from a Jupyter notebook in SageMaker by using the **Boto3** library, which is the AWS SDK for Python. This will allow you to calculate the total size of objects within a specified bucket. 

The code below will calculate your bucket size for you. Here is a breakdown of the important pieces in the next code section:

1. **Paginator**: Since S3 buckets can contain many objects, we use a paginator to handle large listings.
2. **Size calculation**: We sum the `Size` attribute of each object in the bucket.
3. **Unit conversion**: The size is given in bytes, so dividing by `1024 ** 2` converts it to megabytes (MB).

> **Note**: If your bucket has very large objects or you want to check specific folders within a bucket, you may want to refine this code to only fetch certain objects or folders.

```python
# Initialize the total size counter (bytes)
total_size_bytes = 0

# Use a paginator to handle large bucket listings
# This ensures that even if the bucket contains many objects, we can retrieve all of them
paginator = s3.get_paginator("list_objects_v2")

# Iterate through all pages of object listings
for page in paginator.paginate(Bucket=bucket_name):
    # 'Contents' contains the list of objects in the current page, if available
    for obj in page.get("Contents", []):  
        total_size_bytes += obj["Size"]  # Add each object's size to the total

# Convert the total size to gigabytes for cost estimation
total_size_gb = total_size_bytes / (1024 ** 3)

# Convert the total size to megabytes for easier readability
total_size_mb = total_size_bytes / (1024 ** 2)

# Print the total size in MB
print(f"Total size of bucket '{bucket_name}': {total_size_mb:.2f} MB")

# Print the total size in GB
#print(f"Total size of bucket '{bucket_name}': {total_size_gb:.2f} GB")
```

### Using helper functions from GitHub
We have added code to calculate bucket size to a helper function called `get_s3_bucket_size(bucket_name)` for your convenience. There are also some other helper functions in `scripts/helpers.py` of the lesson repository to assist you with common AWS/SageMaker workflows. We'll show you how to clone this code into your notebook environment.

#### Directory setup
Let's make sure we're starting in the root directory of this instance, so that we all have our helpers.py file located in the same path (/home/ec2-user/SageMaker/ML_with_AWS_SageMaker/scripts/helpers.py). This directory survives stopping the instance, so a clone placed there is still present next session. The `test -d ... ||` guard skips the clone if the folder already exists; every later episode starts with the same cell.

```python
%cd /home/ec2-user/SageMaker/
!test -d ML_with_AWS_SageMaker || git clone https://github.com/carpentries-incubator/ML_with_AWS_SageMaker.git
import sys
sys.path.insert(0, "/home/ec2-user/SageMaker/ML_with_AWS_SageMaker/scripts")
import helpers
```

`sys.path.insert` tells Python where to find the lesson's `scripts/` folder; after it, `import helpers` works from any directory.

Our helpers.py file can be found in `ML_with_AWS_SageMaker/scripts/helpers.py`. With this file downloaded, you can call this function via...

```python
helpers.get_s3_bucket_size(bucket_name)
```


### Check storage costs of bucket
To estimate the storage cost of your Amazon S3 bucket directly from a Jupyter notebook in SageMaker, you can use the following approach. This method calculates the total size of the bucket and estimates the monthly storage cost based on AWS S3 pricing.

**Note**: AWS S3 pricing varies by region and storage class. The example below uses the S3 Standard storage class pricing for the US East (N. Virginia) region as of November 1, 2024. Please verify the current pricing for your specific region and storage class on the [AWS S3 Pricing page](https://aws.amazon.com/s3/pricing/).

```python
# AWS S3 Standard Storage pricing for US East (N. Virginia) region
# Pricing tiers as of November 1, 2024
first_50_tb_price_per_gb = 0.023  # per GB for the first 50 TB
next_450_tb_price_per_gb = 0.022  # per GB for the next 450 TB
over_500_tb_price_per_gb = 0.021  # per GB for storage over 500 TB

# Calculate the cost based on the size
if total_size_gb <= 50 * 1024:
    # Total size is within the first 50 TB
    cost = total_size_gb * first_50_tb_price_per_gb
elif total_size_gb <= 500 * 1024:
    # Total size is within the next 450 TB
    cost = (50 * 1024 * first_50_tb_price_per_gb) + \
           ((total_size_gb - 50 * 1024) * next_450_tb_price_per_gb)
else:
    # Total size is over 500 TB
    cost = (50 * 1024 * first_50_tb_price_per_gb) + \
           (450 * 1024 * next_450_tb_price_per_gb) + \
           ((total_size_gb - 500 * 1024) * over_500_tb_price_per_gb)

print(f"Estimated monthly storage cost: ${cost:.5f}")
print(f"Estimated annual storage cost: ${cost*12:.5f}")

```


For your convenience, we have also added this code to a helper function.


```python
monthly_cost, storage_size_gb = helpers.calculate_s3_storage_cost(bucket_name)
print(f"Estimated monthly cost ({storage_size_gb:.4f} GB): ${monthly_cost:.5f}")
print(f"Estimated annual cost ({storage_size_gb:.4f} GB): ${monthly_cost*12:.5f}")

```

**Important Considerations**:

- **Pricing Tiers**: AWS S3 pricing is tiered. The first 50 TB per month is priced at `$0.023 per GB`, the next 450 TB at `$0.022 per GB`, and storage over 500 TB at `$0.021 per GB`. Ensure you apply the correct pricing tier based on your total storage size.
- **Region and Storage Class**: Pricing varies by AWS region and storage class. The example above uses the S3 Standard storage class pricing for the US East (N. Virginia) region. Adjust the pricing variables if your bucket is in a different region or uses a different storage class.
- **Additional Costs**: This estimation covers storage costs only. AWS S3 may have additional charges for requests, data retrievals, and data transfers. For a comprehensive cost analysis, consider these factors as well.

For detailed and up-to-date information on AWS S3 pricing, please refer to the [AWS S3 Pricing page](https://aws.amazon.com/s3/pricing/).

## Writing output files to S3
As your analysis generates new files or demands additional documentation, you can upload files to your bucket as demonstrated below. For this demo, you can create a blank `Notes.txt` file to upload to your bucket. To do so, go to **File** -> **New** -> **Text file**, and save it out as `Notes.txt`. (The code below also writes the file, so the notebook runs end to end.)

```python
# Define the S3 bucket name and the file paths
notes_file_path = "Notes.txt" # assuming your file is in root directory of jupyter notebook (check file explorer tab)
with open(notes_file_path, "w") as f:
    f.write("Workshop notes: Titanic data uploaded from the S3 episode; outputs go under per-episode prefixes.\n")

# Upload the training file to a new folder called "docs". You can also just place it in the bucket's root directory if you prefer (remove docs/ in code below).
s3.upload_file(notes_file_path, bucket_name, "docs/Notes.txt")

print("Files uploaded successfully.")

```

After uploading, we can view the objects/files available on our bucket using...

```python
# List and print all objects in the bucket
response = s3.list_objects_v2(Bucket=bucket_name)

# Check if there are objects in the bucket
if 'Contents' in response:
    for obj in response['Contents']:
        print(obj['Key'])  # Print the object's key (its path in the bucket)
else:
    print("The bucket is empty or does not exist.")

```

Alternatively, we can substitute this for a helper function call as well.

```python
file_list = helpers.list_S3_objects(bucket_name)
file_list
```

::::::::::::::::::::::::::::::::::::: challenge

### Challenge: before and after

1. Using `helpers.get_s3_bucket_size(bucket_name)`, record the size of your bucket. Then create a second file (say `docs/Notes2.txt` with a few lines of text), upload it under the `docs/` prefix, and measure the bucket size again. By how much did it grow, and what does that cost per month at $0.023 per GB?
2. The next episodes train an XGBoost model with a SageMaker training job that writes its outputs to your bucket. Looking at how this episode organized `Notes.txt`, which prefix do you expect those outputs to land under, and why does the lesson use one bucket with prefixes instead of a new bucket per episode?

:::::::::::::::::::::::::::::::::::: solution

1. One way to do it:

```python
size_before = helpers.get_s3_bucket_size(bucket_name)

with open("Notes2.txt", "w") as f:
    f.write("A second note, to see the bucket grow.\n" * 20)
s3.upload_file("Notes2.txt", bucket_name, "docs/Notes2.txt")

size_after = helpers.get_s3_bucket_size(bucket_name)
print(size_before, size_after)
```

The bucket grows by the size of the file (a few hundred bytes here), and at $0.023 per GB-month that is well under a thousandth of a cent per month. The same two lines, run before and after a training job, tell you what the job wrote and whether something is accumulating that you should clean up.

2. The training episode writes under `xgboost/` (`s3://sinkorswim-doejohn/xgboost/`), the same way this episode wrote under `docs/`; the PyTorch episode uses `pytorch/`, the RAG episodes `wattbot/`, and the fine-tuning episode `badger-scribe/`. One bucket with prefixes keeps a single bucket policy, a single set of tags, and a single thing to delete at the end of the workshop; one bucket per episode would mean repeating the policy and tag setup five times and a longer cleanup list.

::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::::::::::::::

:::::::::::::::::::::::::::::::::::::: keypoints 

- Load data from S3 into memory for efficient storage and processing.
- Periodically check storage usage and costs to manage S3 budgets.
- Use SageMaker to upload analysis results and maintain an organized workflow.
- Clone the lesson repository once, into `/home/ec2-user/SageMaker/ML_with_AWS_SageMaker/`; every later episode starts with the same guarded clone-and-import cell.


::::::::::::::::::::::::::::::::::::::::::::::::

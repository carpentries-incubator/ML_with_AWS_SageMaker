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

Open the notebook from: `/ML_with_AWS_SageMaker/notebooks/04_Accessing-S3-via-SageMaker-notebooks.ipynb`. Select the **`conda_python3`** kernel (the same one you used in the last episode).

#### Set up AWS environment
Each notebook starts by setting up the AWS session it works in:

Start with the role and the session, and print both. The role is the identity every call from this notebook uses. Check the region line against the bucket you made in the S3 episode: the two must match, or the training jobs in later episodes will fail to find their data and anything you do read will be billed as cross-region transfer. If they differ, the fix is a new bucket in this region, not a new notebook instance.

Work from a project folder beside the lesson clone, so downloads and outputs stay out of the repository. The Titanic episodes all use this one; the fine-tuning and RAG episodes make their own.

```python
!mkdir -p /home/ec2-user/SageMaker/titanic   # a project folder beside the lesson clone, so downloads and outputs stay out of the repository
%cd /home/ec2-user/SageMaker/titanic
```

```python
import sagemaker                      # SageMaker Python SDK: higher-level API for training jobs, tuning, deployment
from sagemaker import get_execution_role

role = sagemaker.get_execution_role() # the IAM role attached to this notebook instance; every call below acts as this role
session = sagemaker.Session()         # tracks the region, default bucket and SageMaker client for this notebook
print(f"Execution Role: {role}")
print(f"AWS Region: {session.boto_region_name}")  # The region of the notebook instance; your bucket must be in this same region
```

::::::::::::::::::::::::::::::::::::: callout

### Messages this cell prints

The SageMaker SDK prints a few lines of its own when it is imported and when a session is created, and none of them is an error. `sagemaker.config INFO - Not applying SDK defaults from location: ...` means it looked for an optional defaults file in two standard places and found none. A line about telemetry says it reports usage to AWS. A warning that this version of the SDK is deprecated means a newer major version exists. The code in this lesson is written for the version the notebook instances ship. All three appear in later episodes too, and can be ignored.

::::::::::::::::::::::::::::::::::::::::::::::::

Next, an S3 client, and a listing of the buckets this role can see.

```python
import boto3                          # AWS SDK for Python: low-level access to every AWS service (S3, EC2, ...)

s3 = boto3.client('s3')               # client for S3 API calls (list, upload, download)
bucket_names = [bucket["Name"] for bucket in s3.list_buckets()["Buckets"]]
print(f"Available S3 Buckets: {bucket_names}")  # Every bucket in the account this role can list, including other learners' buckets in a shared account
```

## Reading data from S3

There are two ways to get a file from S3 into a notebook: read it straight into memory with `get_object`, or copy it onto the instance's disk with `download_file`. In memory is the usual choice here, since the notebook only inspects data and the jobs read S3 themselves; a copy on disk is for a file you will read many times in one session, or that another program needs as a file.

### A) Reading data directly from S3 into memory

#### Example: Reading data from S3 into memory
Our data is stored in the S3 bucket you created in the previous episode, named 'challenge-lastnamefirstname' (e.g., sinkorswim-doejohn). We can use the following code to read data directly from S3 into memory in the Jupyter notebook environment, without downloading a copy of `titanic_train.csv` as a local file.

```python
import pandas as pd
# Define the S3 bucket and object key
bucket_name = 'sinkorswim-doejohn'  # replace with your S3 bucket name

# Read the train data from S3
key = 'titanic_train.csv'  # the object key: the file's path inside the bucket
response = s3.get_object(Bucket=bucket_name, Key=key)  # fetch the object; its bytes arrive as a stream in response['Body']
train_data = pd.read_csv(response['Body'])

# Read the test data from S3
key = 'titanic_test.csv'
response = s3.get_object(Bucket=bucket_name, Key=key)
test_data = pd.read_csv(response['Body'])

# check shape
print(train_data.shape)
print(test_data.shape)

# Inspect the first few rows of the DataFrame
train_data.head()
```

### B) Download copy into notebook environment
`download_file` writes the object to the instance's disk, on the 20 GB `SageMaker/` volume. There is no transfer charge, because the instance and the bucket are in the same region, and in-region transfer between AWS services is free. The same call from a laptop, with the AWS CLI or with `boto3`, pulls the data out of AWS at `$0.09` per GB. The optional *Same Jobs from the Terminal* episode covers working from a laptop and when that charge matters. The file stays on the instance until you delete it.

#### Example


```python
# Define the S3 bucket and file location
key = "titanic_train.csv"  # Path to your file in the S3 bucket
local_file_path = "/home/ec2-user/SageMaker/titanic/titanic_train.csv"  # Local path to save the file

# Copy the object to the instance's disk (unlike get_object above, which only read it into memory)
s3.download_file(bucket_name, key, local_file_path)
!ls "/home/ec2-user/SageMaker/titanic/"
```

**Note**: You may need to hit refresh on the file explorer panel to the left to see this file. If you get any permission issues...

* the instance's IAM role is `ml-sagemaker-use`
* the bucket policy from the S3 episode names that role

#### Check the current size and storage costs of bucket
Bucket size is what storage is billed on, and the bill is a day or two behind, so read the size from the notebook.

The code below calculates your bucket size. The pieces that matter:

1. **Paginator**: Since S3 buckets can contain many objects, we use a paginator to handle large listings.
2. **Size calculation**: We sum the `Size` attribute of each object in the bucket.
3. **Unit conversion**: The size is given in bytes, so dividing by `1024 ** 2` converts it to megabytes (MB).


```python
# Initialize the total size counter (bytes)
total_size_bytes = 0

# Use a paginator to handle large bucket listings (one list call returns at most 1000 objects)
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

### Using the lesson's helper functions
The same computation is the helper function `get_s3_bucket_size(bucket_name)`. There are also some other helper functions in `scripts/helpers.py` of the lesson repository, which you cloned to `/home/ec2-user/SageMaker/ML_with_AWS_SageMaker/` in the previous episode.

Tell Python where the lesson's `scripts/` folder is:

```python
import sys
# Put the lesson's scripts/ folder first on Python's import path
sys.path.insert(0, "/home/ec2-user/SageMaker/ML_with_AWS_SageMaker/scripts")
import helpers  # the lesson's helper functions (bucket size, cost estimate, object listing)
```


Then:

```python
helpers.get_s3_bucket_size(bucket_name)
```

### The shared bucket

The S3 episode mentioned a second bucket: the one the instructors staged with the larger datasets, which the fine-tuning and RAG episodes read from. Your role can list it and read from it, and nothing else; an upload to it is refused. Look at what is in it now, so that the names in those episodes are not a surprise. `show_bucket` prints each top-level folder with how many objects it holds and how big they are:

```python
shared_bucket = "ml-marathon-shared"   # the instructor-staged, read-only bucket; the later episodes have the name filled in
helpers.show_bucket(shared_bucket)
```

Each folder is one episode's data: `badger-scribe/` holds the scanned pages and the model weights for the fine-tuning episode, `wattbot/` the papers and questions for the RAG episodes. The same call with a prefix, `helpers.show_bucket(shared_bucket, "wattbot/")`, goes one level down.


### Check storage costs of bucket
To estimate the storage cost of your Amazon S3 bucket directly from a Jupyter notebook in SageMaker, you can use the following approach. This method calculates the total size of the bucket and estimates the monthly storage cost based on AWS S3 pricing.

The rates are S3 Standard in us-east-1 from the AWS price list of 2026-10-16; other regions differ by a few percent and other storage classes by more ([AWS S3 Pricing](https://aws.amazon.com/s3/pricing/)).

```python
# AWS S3 Standard Storage pricing for US East (N. Virginia) region
# Pricing tiers from the AWS price list of 2026-10-16
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


The same code is a helper function:


```python
monthly_cost, storage_size_gb = helpers.calculate_s3_storage_cost(bucket_name)
print(f"Estimated monthly cost ({storage_size_gb:.4f} GB): ${monthly_cost:.5f}")
print(f"Estimated annual cost ({storage_size_gb:.4f} GB): ${monthly_cost*12:.5f}")

```

## Writing output files to S3
Uploading is the reverse call. For this demo, we write a small `Notes.txt` file on the instance and check that it exists:

```python
notes_file_path = "/home/ec2-user/SageMaker/titanic/Notes.txt"
with open(notes_file_path, "w") as f:
    f.write("Workshop notes.\n")

!ls -l "/home/ec2-user/SageMaker/titanic/Notes.txt"
```

Then upload it. Here it goes under a `docs/` prefix; drop the `docs/` to put it in the bucket's root instead.

```python
notes_s3_key = "docs/Notes.txt"   # the object key: a "docs/" prefix plus the file name
# local file -> bucket + key; S3 has no real folders, the "docs/" prefix exists as soon as one key uses it
s3.upload_file(notes_file_path, bucket_name, notes_s3_key)
print(f"Uploaded {notes_file_path} to s3://{bucket_name}/{notes_s3_key}")
```

After uploading, we can view the objects/files available on our bucket using...

```python
# List and print all objects in the bucket
response = s3.list_objects_v2(Bucket=bucket_name)  # one page (up to 1000 keys); the paginator above handles more

# Check if there are objects in the bucket
if 'Contents' in response:
    keys = [obj['Key'] for obj in response['Contents']]   # each object's key: its path in the bucket
    print(len(keys), "objects; the first 20:")           # a bucket that has seen a few training jobs holds hundreds of files, so do not print them all
    for key in keys[:20]:
        print(key)
else:
    print("The bucket is empty or does not exist.")

```

Alternatively, we can substitute this for a helper function call as well.

```python
file_list = helpers.list_S3_objects(bucket_name)   # every key in the bucket, through the paginator
file_list[:20]
```

::::::::::::::::::::::::::::::::::::: challenge

### Challenge: before and after

1. Using `helpers.get_s3_bucket_size(bucket_name)`, record the size of your bucket. Then create a second file (say `docs/Notes2.txt` with a few lines of text), upload it under the `docs/` prefix, and measure the bucket size again. By how much did it grow, and what does that cost per month at `$0.023` per GB?
2. The training episode launches a job that writes its outputs to your bucket. Why does the lesson keep everything in one bucket, organised with prefixes like `docs/`, instead of creating a bucket per task?

:::::::::::::::::::::::::::::::::::: solution

1. One way to do it:

```python
size_before = helpers.get_s3_bucket_size(bucket_name)

import time
with open("Notes2.txt", "w") as f:
    f.write("A second note, to see the bucket grow.\n" * 20)
notes2_key = f"docs/Notes2-{int(time.time())}.txt"   # a new key each run; uploading over an existing key of the same size would show no growth
s3.upload_file("Notes2.txt", bucket_name, notes2_key)

size_after = helpers.get_s3_bucket_size(bucket_name)
print(f"before {size_before['size_mb'] * 1024:,.1f} KB, after {size_after['size_mb'] * 1024:,.1f} KB: grew by {(size_after['size_mb'] - size_before['size_mb']) * 1024:.1f} KB")   # the helper returns a dict with size_mb and size_gb
```

The bucket grows by the size of the file (a few hundred bytes here), and at `$0.023` per GB-month that is well under a thousandth of a cent per month. The same two lines, run before and after a training job, tell you what the job wrote and whether something is accumulating that you should clean up.

2. A prefix is just part of the object key, so `docs/Notes.txt`, `xgboost/output/...` and `wattbot/chunks.jsonl` can all live in one bucket and still be listed, downloaded or deleted as groups. One bucket means one policy, one set of tags, one region to get right, and one thing to empty and delete at the end; a bucket per task would repeat the setup each time and lengthen the cleanup list.

::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::::::::::::::

:::::::::::::::::::::::::::::::::::::: keypoints 

- Read S3 objects into memory with `get_object`; copy one to the instance's disk with `download_file` only when a program needs it as a file.
- Periodically check storage usage and costs to manage S3 budgets.
- Write outputs back to your bucket under a prefix, and read the bucket's size from the notebook to see what accumulates.
- The lesson repository lives at `/home/ec2-user/SageMaker/ML_with_AWS_SageMaker/`; every later episode adds its `scripts/` folder to `sys.path` and imports `helpers`.


::::::::::::::::::::::::::::::::::::::::::::::::

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

## Initial setup 

#### Open .ipynb notebook
Once your newly created *instance* shows as `InService`, open the instance in Jupyter Lab. From there, we can create as many Jupyter notebooks as we would like within the instance environment. 

We will then select the standard python3 environment (conda_python3) to start our first .ipynb notebook (Jupyter notebook). We can use the standard conda_python3 environment since we aren't doing any training/tuning just yet.

After opening, you can right-click the Jupyter notebook name to "Rename" it to: `Interacting-with-S3.ipynb`, since interacting with S3 will be our focus for now.

#### Set up AWS environment
To begin each notebook, it's important to set up an AWS environment that will allow seamless access to the necessary cloud resources. Here's what we'll do to get started:

1. **Define the Role**: We'll use `get_execution_role()` to retrieve the IAM role associated with the SageMaker instance. This role specifies the permissions needed for interacting with AWS services like S3, which allows SageMaker to securely read from and write to storage buckets.

2. **Initialize the SageMaker Session**: Next, we'll create a `sagemaker.Session()` object, which will help manage and track the resources and operations we use in SageMaker, such as training jobs and model artifacts. The session acts as a bridge between the SageMaker SDK commands in our notebook and AWS services.

3. **Set Up an S3 Client**: Using `boto3`, we'll initialize an S3 client for accessing S3 buckets directly. This client enables us to handle data storage, retrieve datasets, and manage files in S3, which will be essential as we work through various machine learning tasks.

Starting with these initializations prepares our notebook environment to efficiently interact with AWS resources for model development, data management, and deployment.

```python
import boto3
import sagemaker
from sagemaker import get_execution_role

# Initialize the SageMaker role, session, and s3 client
role = sagemaker.get_execution_role() # specifies your permissions to use AWS tools
session = sagemaker.Session() 
s3 = boto3.client('s3')

# Print relevant details 
role_name = role.split("/")[-1]  # Extracts the last role (the actual name)
print(f"Execution Role: {role_name}")  # Displays the IAM role being used
bucket_names = [bucket["Name"] for bucket in s3.list_buckets()["Buckets"]]
print(f"Available S3 Buckets: {bucket_names}")  # Shows the default S3 bucket assigned to SageMaker
print(f"AWS Region: {session.boto_region_name}")  # Prints the region where the SageMaker session is running

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
Our data is stored on an S3 bucket called 'name-titanic-s3' (e.g., doejohn-titanic-s3). We can use the following code to read data directly from S3 into memory in the Jupyter notebook environment, without actually downloading a copy of train.csv as a local file.

```python
import pandas as pd
# Define the S3 bucket and object key
bucket_name = 'doejohn-titanic-s3'  # replace with your S3 bucket name

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

    sagemaker.config INFO - Not applying SDK defaults from location: /etc/xdg/sagemaker/config.yaml
    sagemaker.config INFO - Not applying SDK defaults from location: /home/ec2-user/.config/sagemaker/config.yaml
    (712, 12)
    (179, 12)


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
# Define the S3 bucket and file location
key = "titanic_train.csv"  # Path to your file in the S3 bucket
local_file_path = "./titanic_train.csv"  # Local path to save the file

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

    Total size of bucket 'doejohn-titanic-s3': 0.06 MB


### Using helper functions from lesson repo
We have added code to calculate bucket size to a helper function called `get_s3_bucket_size(bucket_name)` for your convenience. There are also some other helper functions in that repo to assist you with common AWS/SageMaker workflows. We'll show you how to clone this code into your notebook environment.

**Note**: Make sure you have already forked the lesson repo as described on the [setup page](https://uw-madison-datascience.github.io/ML_with_Amazon_SageMaker/#workshop-repository-setup). Replace "username" below with your GitHub username.

#### Directory setup
Let's make sure we're starting in the root directory of this instance, so that we all have our AWS_helpers.py file located in the same path (/test_AWS/scripts/AWS_helpers.py)

```python
%cd /home/ec2-user/SageMaker/
```

    /home/ec2-user/SageMaker

To clone the repo to our Jupyter notebook, use the following code, adjusting username to your GitHub username.
```python
!git clone https://github.com/username/AWS_helpers.git # downloads AWS_helpers folder/repo (refresh file explorer to see)
```

Our AWS_helpers.py file can be found in `AWS_helpers/helpers.py`. With this file downloaded, you can call this function via...

```python
import AWS_helpers.helpers as helpers
helpers.get_s3_bucket_size(bucket_name)
```

    {'size_mb': 0.060057640075683594, 'size_gb': 5.865003913640976e-05}

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

print(f"Estimated monthly storage cost: ${cost:.4f}")
print(f"Estimated annual storage cost: ${cost*12:.4f}")

```

    Estimated monthly storage cost: $0.0000


For your convenience, we have also added this code to a helper function.


```python
monthly_cost, storage_size_gb = helpers.calculate_s3_storage_cost(bucket_name)
print(f"Estimated monthly cost ({storage_size_gb:.4f} GB): ${monthly_cost:.5f}")
print(f"Estimated annual cost ({storage_size_gb:.4f} GB): ${monthly_cost*12:.5f}")

```

    Estimated monthly cost (0.0001 GB): $0.00000
    Estimated annual cost (0.0001 GB): $0.00002

**Important Considerations**:

- **Pricing Tiers**: AWS S3 pricing is tiered. The first 50 TB per month is priced at `$0.023 per GB`, the next 450 TB at `$0.022 per GB`, and storage over 500 TB at `$0.021 per GB`. Ensure you apply the correct pricing tier based on your total storage size.
- **Region and Storage Class**: Pricing varies by AWS region and storage class. The example above uses the S3 Standard storage class pricing for the US East (N. Virginia) region. Adjust the pricing variables if your bucket is in a different region or uses a different storage class.
- **Additional Costs**: This estimation covers storage costs only. AWS S3 may have additional charges for requests, data retrievals, and data transfers. For a comprehensive cost analysis, consider these factors as well.

For detailed and up-to-date information on AWS S3 pricing, please refer to the [AWS S3 Pricing page](https://aws.amazon.com/s3/pricing/).

## Writing output files to S3
As your analysis generates new files, you can upload files to your bucket as demonstrated below. For this demo, you can create a blank `results.txt` file to upload to your bucket. To do so, go to **File** -> **New** -> **Text file**, and save it out as `results.txt`.

```python
# Define the S3 bucket name and the file paths
train_file_path = "results.txt" # assuming your file is in root directory of jupyter notebook (check file explorer tab)

# Upload the training file to a new folder called "results". You can also just place it in the bucket's root directory if you prefer (remove results/ in code below).
s3.upload_file(train_file_path, bucket_name, "results/results.txt")

print("Files uploaded successfully.")

```

    Files uploaded successfully.

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

   ['results/results.txt', 'titanic_test.csv', 'titanic_train.csv']


:::::::::::::::::::::::::::::::::::::: keypoints 

- Load data from S3 into memory for efficient storage and processing.
- Periodically check storage usage and costs to manage S3 budgets.
- Use SageMaker to upload analysis results and maintain an organized workflow.

::::::::::::::::::::::::::::::::::::::::::::::::

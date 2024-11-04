---
title: "Data Storage: Setting up S3"
teaching: 15
exercises: 5
---

:::::::::::::::::::::::::::::::::::::: questions 

- How can I store and manage data effectively in AWS for SageMaker workflows?
- What are the best practices for using S3 versus EC2 storage for machine learning projects?

::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: objectives

- Explain data storage options in AWS for machine learning projects.
- Describe the advantages of S3 for large datasets and multi-user workflows.
- Outline steps to set up an S3 bucket and manage data within SageMaker.

::::::::::::::::::::::::::::::::::::::::::::::::

## Step 1: Data storage

> **Hackathon Attendees**: All data uploaded to AWS must relate to your specific Kaggle challenge, except for auxiliary datasets for transfer learning or pretraining. **DO NOT upload any restricted or sensitive data to AWS.**

## Options for storage: EC2 Instance or S3

### When to store data directly on EC2 (e.g., in Jupyter Notebook instance)

Using EC2 for data storage can be a quick solution for certain temporary needs. An EC2 instance provides a virtual server environment with its own local storage, which can be used to store and process data directly on the instance. This method is suitable for temporary or small datasets and for one-off experiments that don’t require long-term data storage or frequent access from multiple services. 

#### Limitations of EC2 storage:
- **Scalability**: EC2 storage is limited to the instance’s disk capacity, so it may not be ideal for very large datasets.
- **Cost**: EC2 storage can be more costly for long-term use compared to S3.
- **Data Persistence**: EC2 data may be lost if the instance is stopped or terminated, unless using Elastic Block Store (EBS) for persistent storage.


### What is an S3 bucket?
Storing data in an **S3 bucket** is generally preferred for machine learning workflows on AWS, especially when using SageMaker. An S3 bucket is a container in Amazon S3 (Simple Storage Service) where you can store, organize, and manage data files. Buckets act as the top-level directory within S3 and can hold a virtually unlimited number of files and folders, making them ideal for storing large datasets, backups, logs, or any files needed for your project. You access objects in a bucket via a unique **S3 URI** (e.g., `s3://your-bucket-name/your-file.csv`), which you can use to reference data across various AWS services like EC2 and SageMaker.

::::::::::::::::::::::::::::::::::::: callout 

### Benefits of using S3 (recommended for SageMaker and ML workflows)
The benefits will become more clear as you progress through these materials. However, to point out the most important benefits briefly...

- **Scalability**: S3 handles large datasets efficiently, enabling storage beyond the limits of an EC2 instance's disk space.
- **Cost efficiency**: S3 storage costs are generally lower than expanding EC2 disk volumes. You only pay for the storage you use.
- **Separation of storage and compute**: You can start and stop EC2 instances without losing access to data stored in S3.
- **Integration with AWS services**: SageMaker can read directly from and write back to S3, making it ideal for AWS-based workflows.
- **Easy data sharing**: Datasets in S3 are easier to share with team members or across projects compared to EC2 storage.
- **Cost-effective data transfer**: When S3 and EC2 are in the same region, data transfer between them is free.

::::::::::::::::::::::::::::::::::::::::::::::::


## Recommended approach: Use S3 for data storage

For flexibility, scalability, and cost efficiency, store data in S3 and load it into EC2 as needed. This setup allows:

- Starting and stopping EC2 instances as needed
- Scaling storage without reconfiguring the instance
- Seamless integration across AWS services

### Summary steps to access S3 and upload your dataset

1. Log in to AWS Console and navigate to S3.
2. Create a new bucket or use an existing one.
3. Upload your dataset files.
4. Use the object URL to reference your data in future experiments.

### Detailed procedure:

1. **Sign in to the AWS Management Console**:
   - Log in to AWS Console using your credentials.

2. **Navigate to S3**:
   - Type "S3" in the search bar
   - Protip: select the star icon to save S3 as a bookmark in your AWS toolbar 
   - Select **S3 - Scalable Storage in the Cloud**

4. **Create a new bucket**:
   - Click **Create Bucket** and enter a unique name. **Hackathon participants**: Use the following convention for your bucket name: `TeamName-DatasetName` (e.g., `EmissionImpossible-CO2data`).
   - **Region**: Leave as is (likely `us-east-1` (US East N. Virginia))
   - **Access Control**: Disable ACLs (recommended).
   - **Public Access**: Turn on "Block all public access".
   - **Versioning**: Disable unless you need multiple versions of objects.
   - **Tags**: Include suggested tags for easier cost tracking. Adding tags to your S3 buckets is a great way to track project-specific costs and usage over time, especially as data and resources scale up. While tags are required for hackathon participants, we suggest that all users apply tags to easily identify and analyze costs later. **Hackathon participants**: Use the following convention for your bucket name
      - Name: Your Name
      - ProjectName: Your team's name
      - Purpose: Dataset name (e.g., TitanicData if you're following along with this workshop)
      ![Example of Recommended Tags for an S3 Bucket](path/to/your-image.png){alt="Screenshot showing recommended tags for an S3 bucket, such as Team, Dataset, and Environment"}

   - Click **Create Bucket** at the bottom once everything above has been configured
     
4. **Upload Files to the Bucket**:
   - Click on your bucket’s name, then **Upload**.
   - **Add Files** (e.g., `train.csv`, `test.csv`) and click **Upload** to complete.

5. **Getting the S3 URI for Your Data**:
   - After uploading, click on a file to find its **Object URI** (e.g., `s3://titanic-dataset-test/test.csv`). Use this URI to load data into SageMaker or EC2.

## S3 Bucket Costs

S3 bucket storage incurs costs based on data storage, data transfer, and request counts.

### Storage costs:
- Storage is charged per GB per month.
- Example: Storing 10 GB costs approximately $0.23/month in S3 Standard.
- **Pricing Tiers**: S3 offers multiple storage classes (Standard, Intelligent-Tiering, Glacier, etc.), with different costs based on access frequency and retrieval times.
- To calculate specific costs based on your needs, refer to AWS's [S3 Pricing Information](https://aws.amazon.com/s3/pricing/).

### Data transfer costs:
- **Uploading** data to S3 is free.
- **Downloading** data (out of S3) incurs charges (~$0.09/GB).
- **In-region transfer** (e.g., S3 to EC2) is free, while cross-region data transfer is charged (~$0.02/GB).

> **[Data Transfer Pricing](https://aws.amazon.com/s3/pricing/)**

### Request costs:
- GET requests are $0.0004 per 1,000 requests.

> **[Request Pricing](https://aws.amazon.com/s3/pricing/)**

## Removing Unused Data

Choose one of these options:

### Option 1: Delete Data Only
- **When to Use**: You plan to reuse the bucket.
- **Steps**:
   - Go to S3, navigate to the bucket.
   - Select files to delete, then **Actions > Delete**.
   - **CLI** (optional): `!aws s3 rm s3://your-bucket-name --recursive`

### Option 2: Delete the S3 Bucket Entirely
- **When to Use**: You no longer need the bucket or data.
- **Steps**:
   - Select the bucket, click **Actions > Delete**.
   - Type the bucket name to confirm deletion.

Deleting the bucket stops all costs associated with storage, requests, and data transfer.

::::::::::::::::::::::::::::::::::::: keypoints 

- Use S3 for scalable, cost-effective, and flexible storage.
- EC2 storage is suitable for small, temporary datasets.
- Track your S3 storage costs, data transfer, and requests to manage expenses.
- Regularly delete unused data or buckets to avoid ongoing costs.

::::::::::::::::::::::::::::::::::::::::::::::::

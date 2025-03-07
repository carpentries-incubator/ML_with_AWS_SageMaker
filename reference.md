---
title: 'Glossary'
---

Understanding the terminology used in cloud computing and AWS is half the battle when working with SageMaker. Familiarity with key concepts will help you navigate AWS services, configure machine learning workflows, and troubleshoot issues more efficiently.  

We encourage you to briefly study this glossary before the workshop and refer back to it as needed. While we'll go over these terms throughout the workshop, early exposure will be helpful in building your understanding and making the hands-on exercises smoother.  

### Cloud Compute Essentials  

* **Virtual Machine (VM)**: A software-based computer that runs in a cloud or on-premises environment. In AWS, EC2 instances act as virtual machines with configurable CPU, memory, and storage resources.  
* **Instance**: A virtual machine that runs in the cloud. AWS provides different types of instances for various computing needs, including general-purpose, memory-optimized, and GPU-powered instances for machine learning.  
* **Container**: A lightweight virtualized environment that packages applications and their dependencies together. Containers allow for consistent execution across different computing environments and can be deployed on AWS services like ECS, EKS, and SageMaker.  
* **Docker**: A popular platform for developing, shipping, and running containerized applications. Docker containers encapsulate an application and its dependencies, making them portable across different cloud and local environments.  
* **Elasticity**: The ability to automatically scale computing resources up or down based on demand. Cloud platforms like AWS provide elasticity to help manage costs and performance.  

### AWS General  

* **EC2 (Elastic Compute Cloud)**: An AWS service that provides virtual machines (instances) on demand. These instances can be used to run applications, process data, or train machine learning models.  
* **Auto Scaling**: A feature that automatically adjusts the number of EC2 instances or other cloud resources based on demand to optimize cost and performance.  
* **Spot Instances**: Discounted EC2 instances that take advantage of unused AWS capacity. These instances are cheaper but can be interrupted if AWS reclaims resources.  

### Account Governance and Security  

* **IAM (Identity and Access Management)**: A service that manages user permissions and security policies for AWS resources. It controls who can access SageMaker, S3, and EC2, and what actions they can perform.  
* **IAM Role**: A security identity in AWS that grants specific permissions to users, applications, or services. Instead of assigning direct permissions to a user, an IAM role allows a service like SageMaker or EC2 to access other AWS resources (e.g., an EC2 instance reading data from an S3 bucket).  
  - **Relation to Bucket Policy**: IAM roles grant permissions at the AWS account level, while bucket policies control access at the bucket level. Both must be configured correctly for secure and functional data access.  
* **Bucket Policy**: A set of rules attached to an S3 bucket that define access permissions. Bucket policies allow administrators to specify who can read, write, or manage objects in a bucket. They work in conjunction with IAM roles—**a user or service must have both IAM role permissions and a bucket policy that grants access** to perform actions on an S3 bucket.  
* **Access Control List (ACLs)**: A legacy method for defining access to individual objects in an S3 bucket. AWS now recommends using **bucket policies and IAM roles** for managing access more securely.  
* **AWS Organizations**: A management service that allows multiple AWS accounts to be grouped together under a central organization, enabling consolidated billing and centralized policy enforcement.  
* **Service Quotas**: AWS places default limits on the usage of certain resources (e.g., maximum number of EC2 instances). Service quotas can be increased upon request, but understanding these limits helps prevent unexpected interruptions.  
* **Billing Alerts**: AWS allows users to set cost monitoring and budget alerts to track spending and avoid unexpected charges.  

### Data Storage and Management  

* **S3 (Simple Storage Service)**: A scalable storage service where you can store datasets, models, and other files. S3 is commonly used to store data that will be processed by SageMaker.  
* **S3 Bucket**: A container in S3 where data is stored. Think of it like a folder, but with more scalability and security options. Data is accessed via unique S3 URIs (e.g., `s3://your-bucket-name/your-file.csv`).  
* **S3 URI**: A unique identifier for an object in an S3 bucket, used for referencing data in AWS services like SageMaker.  
* **Elastic Block Store (EBS)**: Persistent storage volumes attached to EC2 instances. Unlike S3, which is an object store, EBS provides block-level storage and is commonly used for databases or applications requiring high-speed storage.  
 **Object URI**: The Uniform Resource Identifier (URI) is a unique address that specifies the location of the file within S3 (e.g., `s3://doejohn-titanic-s3/titanic_train.csv`). This URI is essential for referencing data in AWS services like SageMaker, where it will be used to load data for processing and model training.
  
### SageMaker and Machine Learning Workflows  

* **SageMaker**: A managed machine learning platform in AWS that provides tools for building, training, and deploying ML models. It simplifies the process of running ML workloads on the cloud.  
* **SageMaker Notebook Instance**: A Jupyter notebook environment hosted on AWS. It provides a pre-configured setup for writing and running Python code, accessing data, and training models.  
* **Controller**: In this workshop, we use the term "controller" to describe how a SageMaker Notebook Instance is used to launch and manage training jobs, inference endpoints, and other AWS services. Rather than performing all computations within the notebook itself, the notebook acts as a high-level interface to configure and execute cloud-based ML workflows.  
* **SageMaker Training Job**: A managed process in SageMaker where a model is trained on a specified dataset using EC2 instances. Training jobs can be configured to use GPUs or multiple instances for scalability.  
* **Hyperparameter Tuning Job (HPO)**: A SageMaker feature that automatically tests different hyperparameter values to find the best-performing model configuration.  

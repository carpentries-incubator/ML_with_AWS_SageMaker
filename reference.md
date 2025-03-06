---
title: 'Glossary'
---

## Cloud Computing and AWS Terminology  
Understanding the terminology used in cloud computing and AWS is half the battle when working with SageMaker. Familiarity with key concepts will help you navigate AWS services, configure machine learning workflows, and troubleshoot issues more efficiently.  

We encourage you to briefly study this glossary before the workshop and refer back to it as needed. While we'll go over these terms throughout the workshop, early exposure will be helpful in building your understanding and making the hands-on exercises smoother.  

* **Instance**: A virtual machine that runs in the cloud. AWS provides different types of instances for various computing needs, including general-purpose, memory-optimized, and GPU-powered instances for machine learning.  

* **EC2 (Elastic Compute Cloud)**: An AWS service that provides virtual machines (instances) on demand. These instances can be used to run applications, process data, or train machine learning models.  

* **S3 (Simple Storage Service)**: A scalable storage service where you can store datasets, models, and other files. S3 is commonly used to store data that will be processed by SageMaker.  

* **S3 Bucket**: A container in S3 where data is stored. Think of it like a folder, but with more scalability and security options. Data is accessed via unique S3 URIs (e.g., `s3://your-bucket-name/your-file.csv`).  

* **IAM (Identity and Access Management)**: A service that manages user permissions and security policies for AWS resources. It controls who can access SageMaker, S3, and EC2, and what actions they can perform.  

* **SageMaker**: A managed machine learning platform in AWS that provides tools for building, training, and deploying ML models. It simplifies the process of running ML workloads on the cloud.  

* **SageMaker Notebook Instance**: A Jupyter notebook environment hosted on AWS. It provides a pre-configured setup for writing and running Python code, accessing data, and training models.  

* **Controller**: In this workshop, we use the term "controller" to describe how a SageMaker Notebook Instance is used to launch and manage training jobs, inference endpoints, and other AWS services. Rather than performing all computations within the notebook itself, the notebook acts as a high-level interface to configure and execute cloud-based ML workflows.  

* **SageMaker Training Job**: A managed process in SageMaker where a model is trained on a specified dataset using EC2 instances. Training jobs can be configured to use GPUs or multiple instances for scalability.  

* **Hyperparameter Tuning Job (HPO)**: A SageMaker feature that automatically tests different hyperparameter values to find the best-performing model configuration.  

* **Docker Container**: A lightweight environment that packages ML models, dependencies, and code together, ensuring consistency when running models across different machines, including SageMaker.  


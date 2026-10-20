---
title: "Overview of Amazon SageMaker"
teaching: 15
exercises: 0
---

:::::::::::::::::::::::::::::::::::::::::::::: questions

- When is cloud compute the right tool for a researcher who already has a cluster or a workstation?
- What does SageMaker add on top of renting a machine?
  
::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: objectives

- Compare cloud compute with campus HPC and lab hardware on hardware generation, cost and control.
- Introduce SageMaker and the controller-plus-jobs pattern.
- Preview the three parts of the lesson and the cost practices used in all of them.
  
::::::::::::::::::::::::::::::::::::::::::::::


Amazon SageMaker is AWS's managed platform for training, tuning and running machine learning models. You bring your own scripts and data; AWS provides the machines, the containers with common ML frameworks installed, and the bookkeeping. This episode is about when that is worth it for a researcher who already has a campus cluster or a lab workstation.

### Cloud vs. the cluster down the hall

Most researchers already have somewhere to run ML: a campus HPC cluster, a departmental GPU server, or a workstation under the desk. The cloud is a different trade. Three things matter most:

- **Hardware generation.** A cluster or lab server is bought once and used for five or more years, so its GPUs are whatever generation was current at purchase. Cloud providers carry each new generation (H100, H200 and newer) from the month it ships, in single-GPU instances up to 8-GPU nodes. If a model or a method needs the current generation's memory or speed, the cloud is often the only place to get it without waiting for a procurement cycle.
- **Efficiency of spend.** Cloud instances bill per second and only while they run. A job that needs an H100 for forty minutes costs forty minutes of an H100; the rest of the day costs nothing. Owned hardware costs the same whether it is busy or idle. For bursty work, renting is cheaper; for GPUs that stay busy, owning is.
- **Control and data.** On-prem gives you full control: your own software stack, no quotas, no egress charges, and data that never leaves campus, which matters for restricted datasets. In the cloud you work within quotas and an account's permissions, and anything you move in or out crosses the network. An idle rented GPU costs the same as a busy one, so every job in this lesson has a runtime cap.

| | Campus cluster | Lab workstation | Cloud (SageMaker) |
|---|---|---|---|
| GPU generation | Fixed at purchase | Fixed at purchase | Current, by the hour |
| Wait to start | Queue | None | Minutes (within quota) |
| Cost to you | Allocation or free at use | Already paid | Per second while running |
| Idle cost | None to you | Power | Full price until stopped |
| Control | Shared, admin-managed | Full | Within account permissions |
| Data | Stays on campus | Stays local | Transfer in and out |

### Why use SageMaker for machine learning?

Within AWS, SageMaker is the service built for ML workloads:


- **High-performance compute only when needed**: SageMaker lets you develop interactively in lightweight, inexpensive notebook environments (or your own laptop) and then launch training, tuning, or inference jobs on more powerful [instance types](https://carpentries-incubator.github.io/ML_with_AWS_SageMaker/reference.html#cloud-compute-essentials) only when necessary. You pay for expensive compute only while a job runs.
  
- **Support for custom scripts**: Most training and inference scripts can be run using pre-configured *estimators* or *containers* that come with popular ML frameworks such as scikit-learn, PyTorch, TensorFlow, and Hugging Face already installed. In many cases, you can simply include a `requirements.txt` file to add any additional dependencies you need. When you need more control, SageMaker also supports fully custom Docker containers, so you can bring your own code, dependencies, and environments for training, tuning, and inference for training, tuning, and inference.

- **Flexible compute options**: SageMaker lets you easily select instance types tailored to your project needs. For exploratory analysis, use a lightweight CPU (e.g., ml.m5.large). For compute-intensive tasks, such as training deep learning models, you can switch to GPU instances for faster processing. We'll cover instances more in-depth throughout the lesson (and how to select them), but here's a preview of the the different types:

    - **CPU instances (e.g., ml.m5.large — $0.115/hour)**: Suitable for general ML workloads, feature engineering, and inference tasks. 
    - **Memory-optimized instances (e.g., ml.r5.2xlarge — approx. $0.65/hour)**: Best for handling large datasets in memory.
    - **GPU instances (e.g., ml.g5.xlarge — $1.41/hour)**: Optimized for compute-intensive tasks like deep learning training and fine-tuning, offering accelerated processing. 
    - For more details, check out the supplemental "[Instances for ML](https://carpentries-incubator.github.io/ML_with_AWS_SageMaker/instances-for-ML.html)" page. We'll discuss this topic more throughout the lesson. (Prices here and throughout the lesson are us-east-1 on-demand list prices from the AWS price list of 2026-10-16; other regions are the same or within a few percent.)

- **Parallel training and tuning**: training jobs can run across several instances for large models and datasets, and hyperparameter tuning jobs run many training jobs in parallel with a few lines of configuration.

- **Ease of orchestration / Simplified ML pipelines**: Traditional high-performance computing (HPC) or high-throughput computing (HTC) environments often require researchers to break ML workflows into separate batch jobs, manually orchestrating each step (e.g., submitting preprocessing, training, cross-validation, and evaluation as distinct tasks and stitching the results together later). In SageMaker, a training or tuning job is launched from a notebook with the script, the data location and the instance type, and the orchestration is handled for you.

- **Cost tracking**: every job reports its billable seconds, and every resource can be tagged so that the bill can be broken down by project. For many common research workloads SageMaker is cheap. For example, training roughly 100 small to medium-sized models (e.g., logistic regression, random forests, or lightweight deep learning models with a few million parameters) on a small dataset (under 10GB) can cost under $20, making it accessible for many research projects.

### What this lesson covers

The lesson has three parts. **Part one** covers the foundations: an S3 bucket for your data, a small notebook instance that acts as a *controller*, then SageMaker training jobs (XGBoost and PyTorch) and hyperparameter tuning, with guidance on CPU vs. GPU instances and when parallel instances help. **Part two** builds a retrieval-augmented generation (RAG) pipeline over a corpus of research papers two ways: on Amazon Bedrock (pay per token, nothing left running) and as SageMaker Processing jobs serving an open-weight model that Bedrock does not offer. **Part three** fine-tunes a 3B open-weight vision-language model with LoRA as a training job, for handwriting transcription, and scores it before and after. Every episode applies the same cost controls: tags on every resource, a runtime cap on every job, a cost estimate before each launch compared to the bill after, and cleanup at the end.

Most research groups use both. This lesson covers the cloud half.

::::::::::::::::::::::::::::::::::::: keypoints

- The cloud offers the current GPU generation on demand and bills per second while a job runs; on-prem hardware offers full control and a fixed cost but a fixed generation and a queue.
- Renting is cheaper for bursty research work; owning is cheaper for GPUs that stay busy.
- An idle rented GPU costs the same as a busy one, so every job in this lesson is launched with a runtime cap and a stopping rule.
- A cheap controller notebook launches jobs that run on separate instances and end on their own; each job carries tags and a cost estimate.
- SageMaker runs custom scripts in preconfigured framework containers, and parallelizes training and hyperparameter tuning with little extra code.

::::::::::::::::::::::::::::::::::::::::::::::::

---
title: "Overview: ML on AWS"
teaching: 15
exercises: 0
---

:::::::::::::::::::::::::::::::::::::::::::::: questions

- When is cloud compute the right tool for a researcher who already has a cluster or a workstation?
- What do SageMaker and Bedrock add on top of renting a machine?
  
::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: objectives

- Compare cloud compute with campus HPC and lab hardware on hardware generation, cost and control.
- Introduce SageMaker and Bedrock.
- Preview the three parts of the lesson.
  
::::::::::::::::::::::::::::::::::::::::::::::


This episode compares cloud compute with the compute researchers usually already have, then introduces the two AWS services the lesson uses, SageMaker and Bedrock.

### Cloud vs. the cluster down the hall

Most researchers already have somewhere to run ML, such as a campus HPC cluster, a departmental GPU server, or a lab workstation. How the cloud compares:

- **Buy once and maintain, or rent the latest.** A cluster or lab server is bought once, maintained by someone, and used for five or more years with whatever GPUs were current at purchase. Cloud providers offer each new generation soon after release (Blackwell B200 and GB200 today, Vera Rubin announced for cloud deployment in 2026), billed per second and only while a job runs. The largest systems (72-GPU GB200 racks) are liquid-cooled; most campus data centers are air-cooled and cannot host them without a facility upgrade. Owned hardware costs the same busy or idle, so for GPUs that stay busy owning is cheaper, and for bursty research work renting is.
- **Energy efficiency tracks facility scale.** Power usage effectiveness (PUE) is total facility power divided by IT power; 1.0 is perfect. Hyperscale and flagship HPC facilities are in the same range: AWS reports a global average of [1.14 for 2025](https://aws.amazon.com/sustainability/), and a [2025 survey of US research HPC](https://arxiv.org/html/2506.19019v1) puts TACC at about 1.15 and MGHPCC near 1.14. Smaller facilities do worse: the [Uptime Institute's 2025 survey](https://datacenter.uptimeinstitute.com/rs/711-RIA-145/images/2025.Annual.Survey.Report.pdf?version=0) puts enterprise and colocation data centers at 1.58-1.80, and a departmental server room is typically in that range or above. So the cloud is a large efficiency gain over a lab server, and roughly a wash against a campus HPC center. (Figures are self-reported, and PUE measures the building, not whether the GPUs in it are busy.)
- **Control and data.** On-prem gives full control over the software stack, no quotas, no egress charges, and data that never leaves campus. In the cloud you work within quotas and account permissions, and data moved in or out crosses the network. An idle rented GPU costs the same as a busy one, so every job in this lesson has a runtime cap.

### SageMaker and Bedrock

**Amazon SageMaker** is AWS's managed service for machine learning. It runs on the same instances as plain EC2, but adds a job model built for ML: you submit a script and an instance type, AWS provisions the machine with a framework container (PyTorch, scikit-learn, Hugging Face, ...) installed, runs the script, saves the outputs to S3, and terminates the instance. Renting a bare EC2 GPU gets you the hardware; with SageMaker you do not install the frameworks, move the data by hand, or remember to shut the machine down.

**Amazon Bedrock** is the other AWS service this lesson uses. Where SageMaker runs your script on a machine you rent by the second, Bedrock serves models that AWS hosts (DeepSeek, Claude, Amazon's own, and others) through an API billed per token. There is no instance to choose, start or stop; when you stop calling it, you stop paying. The RAG part of the lesson uses Bedrock first for that reason, and turns to SageMaker only for a model Bedrock does not offer.

### Why use SageMaker for machine learning?

- **High-performance compute only when needed**: SageMaker lets you develop interactively in lightweight, inexpensive notebook environments (or your own laptop) and then launch training, tuning, or inference jobs on more powerful [instance types](https://carpentries-incubator.github.io/ML_with_AWS_SageMaker/reference.html#cloud-compute-essentials) only when necessary. You pay for expensive compute only while a job runs.
  
- **Support for custom scripts**: Most training and inference scripts can be run using pre-configured *estimators* or *containers* that come with popular ML frameworks such as scikit-learn, PyTorch, TensorFlow, and Hugging Face already installed. In many cases, you can simply include a `requirements.txt` file to add any additional dependencies you need. When you need more control, SageMaker also supports fully custom Docker containers, so you can bring your own code, dependencies, and environments for training, tuning, and inference.

- **Flexible compute options**: SageMaker lets you easily select instance types tailored to your project needs. For exploratory analysis, use a lightweight CPU (e.g., ml.m5.large). For compute-intensive tasks, such as training deep learning models, you can switch to GPU instances for faster processing. We'll cover instances more in-depth throughout the lesson (and how to select them), but here's a preview of the different types:

    - **CPU instances (e.g., ml.m5.large — $0.115/hour)**: Suitable for general ML workloads, feature engineering, and inference tasks. 
    - **Memory-optimized instances (e.g., ml.r5.2xlarge — approx. $0.65/hour)**: Best for handling large datasets in memory.
    - **GPU instances (e.g., ml.g5.xlarge — $1.41/hour)**: Optimized for compute-intensive tasks like deep learning training and fine-tuning, offering accelerated processing. 
    - For more details, check out the supplemental "[Instances for ML](https://carpentries-incubator.github.io/ML_with_AWS_SageMaker/instances-for-ML.html)" page. We'll discuss this topic more throughout the lesson. (Prices here and throughout the lesson are us-east-1 on-demand list prices from the AWS price list of 2026-10-16; other regions are the same or within a few percent.)

- **Parallel training and tuning**: training jobs can run across several instances for large models and datasets, and hyperparameter tuning jobs run many training jobs in parallel with a few lines of configuration.

- **Ease of orchestration / Simplified ML pipelines**: Traditional high-performance computing (HPC) or high-throughput computing (HTC) environments often require researchers to break ML workflows into separate batch jobs, manually orchestrating each step (e.g., submitting preprocessing, training, cross-validation, and evaluation as distinct tasks and stitching the results together later). In SageMaker, a training or tuning job is launched from a notebook with the script, the data location and the instance type, and the orchestration is handled for you.

- **Cost tracking**: every job reports its billable seconds, and every resource can be tagged so that the bill can be broken down by project. For many common research workloads SageMaker is cheap. For example, training roughly 100 small to medium-sized models (e.g., logistic regression, random forests, or lightweight deep learning models with a few million parameters) on a small dataset (under 10GB) can cost under $20, making it accessible for many research projects.

### Lesson outline

The rest of the lesson is hands-on, in three parts:

1. **SageMaker foundations and classic ML.** Set up an S3 bucket and a small notebook instance, then launch training jobs (XGBoost and PyTorch) and a hyperparameter tuning job from it.
2. **Retrieval-augmented generation (RAG).** Build a RAG pipeline over research papers on Amazon Bedrock, then run the same pipeline as SageMaker Processing jobs with an open-weight model.
3. **Fine-tuning.** Fine-tune a vision-language model for handwriting transcription as a SageMaker training job.

Every job is tagged, capped, estimated before launch and checked against the bill after.

::::::::::::::::::::::::::::::::::::: keypoints

- The cloud offers the current GPU generation on demand and bills per second while a job runs; on-prem hardware offers full control and a fixed cost but a fixed generation and a queue.
- Renting is cheaper for bursty research work; owning is cheaper for GPUs that stay busy.
- An idle rented GPU costs the same as a busy one, so every job in this lesson is launched with a runtime cap.
- SageMaker runs your script in a preconfigured framework container on an instance that exists only for the job; Bedrock serves AWS-hosted models per token with nothing to start or stop.

::::::::::::::::::::::::::::::::::::::::::::::::

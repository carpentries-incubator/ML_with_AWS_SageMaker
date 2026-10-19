---
title: "Overview of Amazon SageMaker"
teaching: 15
exercises: 5
---

:::::::::::::::::::::::::::::::::::::::::::::: questions

- Why use SageMaker for machine learning?
  
::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: objectives

- Introduce SageMaker and the controller-plus-jobs pattern the whole lesson is built on.
- Preview the three parts of the lesson and the cost-control thread that runs through them.
  
::::::::::::::::::::::::::::::::::::::::::::::


Amazon SageMaker is a comprehensive machine learning (ML) platform that empowers users to build, train, tune, and deploy models at scale. Designed to streamline the ML workflow, SageMaker supports data scientists and researchers in tackling complex machine learning problems without needing to manage underlying infrastructure. This allows you to focus on developing and refining your models while leveraging AWS's robust computing resources for efficient training and deployment.

### Why use SageMaker for machine learning?

SageMaker provides several features that make it an ideal choice for researchers and ML practitioners:


- **High-performance compute only when needed**: SageMaker lets you develop interactively in lightweight, inexpensive notebook environments (or your own laptop) and then launch training, tuning, or inference jobs on more powerful [instance types](https://carpentries-incubator.github.io/ML_with_AWS_SageMaker/reference.html#cloud-compute-essentials) only when necessary. This approach keeps costs low during development and ensures you only pay for expensive compute when you're actively using it.
  
- **Support for custom scripts**: Most training and inference scripts can be run using pre-configured *estimators* or *containers* that come with popular ML frameworks such as scikit-learn, PyTorch, TensorFlow, and Hugging Face already installed. In many cases, you can simply include a `requirements.txt` file to add any additional dependencies you need. When you need more control, SageMaker also supports fully custom Docker containers, so you can bring your own code, dependencies, and environments for training, tuning, and inference — all deployed on scalable AWS infrastructure.

- **Flexible compute options**: SageMaker lets you easily select instance types tailored to your project needs. For exploratory analysis, use a lightweight CPU (e.g., ml.m5.large). For compute-intensive tasks, such as training deep learning models, you can switch to GPU instances for faster processing. We'll cover instances more in-depth throughout the lesson (and how to select them), but here's a preview of the the different types:

    - **CPU instances (e.g., ml.m5.large — $0.115/hour)**: Suitable for general ML workloads, feature engineering, and inference tasks. 
    - **Memory-optimized instances (e.g., ml.r5.2xlarge — approx. $0.65/hour)**: Best for handling large datasets in memory.
    - **GPU instances (e.g., ml.g5.xlarge — $1.41/hour)**: Optimized for compute-intensive tasks like deep learning training and fine-tuning, offering accelerated processing. 
    - For more details, check out the supplemental "[Instances for ML](https://carpentries-incubator.github.io/ML_with_AWS_SageMaker/instances-for-ML.html)" page. We'll discuss this topic more throughout the lesson. (Prices here and throughout the lesson are us-east-1 on-demand list prices from the AWS price list of 2026-10-16; other regions are the same or within a few percent.)

- **Parallelized training and tuning**: SageMaker enables parallelized training across multiple instances, reducing training time for large datasets and complex models. It also supports parallelized hyperparameter tuning, allowing efficient exploration of model configurations with minimal code while maintaining fine-grained control over the process. 

- **Ease of orchestration / Simplified ML pipelines**: Traditional high-performance computing (HPC) or high-throughput computing (HTC) environments often require researchers to break ML workflows into separate batch jobs, manually orchestrating each step (e.g., submitting preprocessing, training, cross-validation, and evaluation as distinct tasks and stitching the results together later). This can be time-consuming and cumbersome, as it requires converting standard ML code into complex Directed Acyclic Graphs (DAGs) and job dependencies. By eliminating the need to manually coordinate compute jobs, SageMaker dramatically reduces ML pipeline complexity, making it easier for researchers to quickly develop and iterate on models efficiently.

- **Cost management and monitoring**: SageMaker includes built-in monitoring tools to help you track and manage costs, ensuring you can scale up efficiently without unnecessary expenses. For many common use cases of ML/AI, SageMaker can be very affordable. For example, training roughly 100 small to medium-sized models (e.g., logistic regression, random forests, or lightweight deep learning models with a few million parameters) on a small dataset (under 10GB) can cost under $20, making it accessible for many research projects.

### What this lesson covers

The lesson has three parts, and one thread runs through all of them. In **part one** you build the foundations: an S3 bucket for your data, a small notebook instance that acts as a *controller*, and then SageMaker training jobs (XGBoost and PyTorch) and hyperparameter tuning on the Titanic dataset, with guidance on CPU vs. GPU instances and when parallel instances pay off. In **part two** you build retrieval-augmented generation (RAG) three ways for the WattBot energy-literature question-answering challenge: on Amazon Bedrock (pay per token, nothing left running), as SageMaker Processing jobs serving an open-weight model, and, as optional self-study, interactively on a GPU notebook instance. In **part three** you fine-tune a 3B open-weight vision-language model with LoRA as a training job, on the Badger Scribe handwriting-transcription challenge, and score it before and after. The thread is cost control, taught as practice rather than as a warning: tags on every resource, a runtime cap on every job, a cost estimate written down before each launch and compared to the bill afterwards, and explicit cleanup at the end.

::::::::::::::::::::::::::::::::::::: challenge

### Challenge: controller or job?

In this lesson, a cheap notebook instance (the *controller*) launches work that runs elsewhere, on a separate instance that starts for the job and shuts down when it ends. For each of the three tasks below, decide whether it belongs on the controller notebook or in a separate SageMaker job, and say why.

1. Reading the first 20 rows of `titanic_train.csv` from S3 to check the column names.
2. Training 50 XGBoost models with different hyperparameters on the Titanic data.
3. Fine-tuning a 3B-parameter vision-language model on 120 scanned pages.

:::::::::::::::::::::::::::::::::::: solution

1. **Controller.** It takes a second, needs no GPU, and the result is a quick look at a DataFrame. Launching a job would cost more in startup time than the work itself.
2. **Job** (a hyperparameter tuning job). Fifty training runs is real compute, and a tuning job runs them in parallel on instances that exist only for as long as the training lasts. The controller just submits the job and reads the results back from S3.
3. **Job** (a training job on a GPU instance such as `ml.g5.xlarge`). A `ml.t3.medium` controller has no GPU and 4 GB of RAM; the job needs an A10G for less than an hour, and you pay for that hour only. Attach tags, a `max_run` cap and a written cost estimate to the launch.

::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::::::::::::::

In summary, Amazon SageMaker is a fully managed machine learning platform that simplifies building, training, tuning, and deploying models at scale. Unlike traditional research computing environments, which often require manual job orchestration and complex dependency management, SageMaker provides an integrated and automated workflow, allowing users to focus on model development rather than infrastructure. With support for on-demand compute resources, parallelized training and hyperparameter tuning, and flexible model deployment options, SageMaker enables researchers to scale experiments efficiently. Built-in cost tracking and monitoring tools also help keep expenses manageable, making SageMaker a practical choice for both small-scale research projects and large-scale ML pipelines. By combining preconfigured machine learning algorithms, support for custom scripts, and robust computing power, SageMaker reduces the complexity of ML development, empowering researchers to iterate faster and bring models to production more seamlessly.

::::::::::::::::::::::::::::::::::::: keypoints

- SageMaker simplifies ML workflows by eliminating the need for manual job orchestration.
- Flexible compute options allow users to choose CPU, GPU, or memory-optimized instances based on workload needs.
- The lesson's pattern: a cheap controller notebook launches jobs that run elsewhere and end on their own, each launched with tags, a runtime cap, a written estimate, and a cleanup step.
- Parallelized training and hyperparameter tuning accelerate model development.
- SageMaker supports both built-in ML algorithms and custom scripts via Docker containers.
- Cost monitoring tools help track and optimize spending on AWS resources.
- SageMaker streamlines scaling from experimentation to deployment, making it suitable for both research and production.

:::::::::::::::::::::::::::::::::::::::::::::::: 

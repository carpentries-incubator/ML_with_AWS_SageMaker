---
title: "Overview of Amazon SageMaker"
teaching: 10
exercises: 0
---

:::::::::::::::::::::::::::::::::::::::::::::: questions

- Why use SageMaker for machine learning?
  
::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: objectives

- Introduce SageMaker
  
::::::::::::::::::::::::::::::::::::::::::::::


Amazon SageMaker is a comprehensive machine learning (ML) platform that empowers users to build, train, tune, and deploy models at scale. Designed to streamline the ML workflow, SageMaker supports data scientists and researchers in tackling complex machine learning problems without needing to manage underlying infrastructure. This allows you to focus on developing and refining your models while leveraging AWS's robust computing resources for efficient training and deployment.

### Why use SageMaker for machine learning?

SageMaker provides several features that make it an ideal choice for researchers and ML practitioners:


- **High-performance compute only when needed**: SageMaker lets you develop interactively in lightweight, inexpensive notebook environments (or your own laptop) and then launch training, tuning, or inference jobs on more powerful [instance types](https://carpentries-incubator.github.io/ML_with_AWS_SageMaker/reference.html#cloud-compute-essentials) only when necessary. This approach keeps costs low during development and ensures you only pay for expensive compute when you're actively using it.
  
- **Support for custom scripts**: Most training and inference scripts can be run using pre-configured *estimators* or *containers* that come with popular ML frameworks such as scikit-learn, PyTorch, TensorFlow, and Hugging Face already installed. In many cases, you can simply include a `requirements.txt` file to add any additional dependencies you need. When you need more control, SageMaker also supports fully custom Docker containers, so you can bring your own code, dependencies, and environments for training, tuning, and inference — all deployed on scalable AWS infrastructure.

- **Flexible compute options**: SageMaker lets you easily select instance types tailored to your project needs. For exploratory analysis, use a lightweight CPU (e.g., ml.m5.large). For compute-intensive tasks, such as training deep learning models, you can switch to GPU instances for faster processing. We'll cover instances more in-depth throughout the lesson (and how to select them), but here's a preview of the the different types:

    - **CPU instances (e.g., ml.m5.large — $0.12/hour)**: Suitable for general ML workloads, feature engineering, and inference tasks. 
    - **Memory-optimized instances (e.g., ml.r5.2xlarge — $0.65/hour)**: Best for handling large datasets in memory.
    - **GPU instances (e.g., ml.p3.2xlarge — $3.83/hour)**: Optimized for compute-intensive tasks like deep learning training, offering accelerated processing. 
    - For more details, check out the supplemental "[Instances for ML](https://carpentries-incubator.github.io/ML_with_AWS_SageMaker/instances-for-ML.html)" page. We'll discuss this topic more throughout the lesson.

- **Parallelized training and tuning**: SageMaker enables parallelized training across multiple instances, reducing training time for large datasets and complex models. It also supports parallelized hyperparameter tuning, allowing efficient exploration of model configurations with minimal code while maintaining fine-grained control over the process. 

- **Ease of orchestration / Simplified ML pipelines**: Traditional high-performance computing (HPC) or high-throughput computing (HTC) environments often require researchers to break ML workflows into separate batch jobs, manually orchestrating each step (e.g., submitting preprocessing, training, cross-validation, and evaluation as distinct tasks and stitching the results together later). This can be time-consuming and cumbersome, as it requires converting standard ML code into complex Directed Acyclic Graphs (DAGs) and job dependencies. By eliminating the need to manually coordinate compute jobs, SageMaker dramatically reduces ML pipeline complexity, making it easier for researchers to quickly develop and iterate on models efficiently.

- **Cost management and monitoring**: SageMaker includes built-in monitoring tools to help you track and manage costs, ensuring you can scale up efficiently without unnecessary expenses. For many common use cases of ML/AI, SageMaker can be very affordable. For example, training roughly 100 small to medium-sized models (e.g., logistic regression, random forests, or lightweight deep learning models with a few million parameters) on a small dataset (under 10GB) can cost under $20, making it accessible for many research projects.

In summary, Amazon SageMaker is a fully managed machine learning platform that simplifies building, training, tuning, and deploying models at scale. Unlike traditional research computing environments, which often require manual job orchestration and complex dependency management, SageMaker provides an integrated and automated workflow, allowing users to focus on model development rather than infrastructure. With support for on-demand compute resources, parallelized training and hyperparameter tuning, and flexible model deployment options, SageMaker enables researchers to scale experiments efficiently. Built-in cost tracking and monitoring tools also help keep expenses manageable, making SageMaker a practical choice for both small-scale research projects and large-scale ML pipelines. By combining preconfigured machine learning algorithms, support for custom scripts, and robust computing power, SageMaker reduces the complexity of ML development, empowering researchers to iterate faster and bring models to production more seamlessly.

::::::::::::::::::::::::::::::::::::: keypoints

- SageMaker simplifies ML workflows by eliminating the need for manual job orchestration.
- Flexible compute options allow users to choose CPU, GPU, or memory-optimized instances based on workload needs.
- Parallelized training and hyperparameter tuning accelerate model development.
- SageMaker supports both built-in ML algorithms and custom scripts via Docker containers.
- Cost monitoring tools help track and optimize spending on AWS resources.
- SageMaker streamlines scaling from experimentation to deployment, making it suitable for both research and production.

:::::::::::::::::::::::::::::::::::::::::::::::: 

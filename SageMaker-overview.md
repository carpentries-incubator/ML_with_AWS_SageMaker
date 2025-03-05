---
title: "Overview of Amazon SageMaker"
teaching: 10
exercises: 0
---

Amazon SageMaker is a comprehensive machine learning (ML) platform that empowers users to build, train, tune, and deploy models at scale. Designed to streamline the ML workflow, SageMaker supports data scientists and researchers in tackling complex machine learning problems without needing to manage underlying infrastructure. This allows you to focus on developing and refining your models while leveraging AWS's robust computing resources for efficient training and deployment.

### Why use SageMaker for machine learning?

SageMaker provides several features that make it an ideal choice for researchers and ML practitioners:

- **Simplified ML/AI Pipelines**: Traditional high-performance computing (HPC) or high-throughput computing (HTC) environments often require researchers to break ML workflows into separate batch jobs, manually orchestrating each step (e.g., submitting preprocessing, training, cross-validation, and evaluation as distinct tasks and stitching the results together later). This can be time-consuming and cumbersome, as it requires converting standard ML code into complex Directed Acyclic Graphs (DAGs) and job dependencies. By eliminating the need to manually coordinate compute jobs, SageMaker dramatically reduces ML pipeline complexity, making it easier for researchers to develop and iterate on models efficiently.

- **Flexible compute options**: SageMaker lets you easily select "instance types" tailored to your project needs. For compute-intensive tasks, such as training deep learning models, you can switch to GPU instances for faster processing. We'll cover instances more in-depth throughout the lesson, but here's a preview of the the different types:

    - **CPU instances (e.g., ml.m5.large)**: Suitable for general ML workloads, feature engineering, and inference tasks.
    - **GPU instances (e.g., ml.p3.2xlarge)**: Optimized for compute-intensive tasks like deep learning training, offering accelerated processing.
    - **Memory-optimized instances (e.g., ml.r5.xlarge)**: Best for handling large datasets in memory, such as training transformer models.
      
- **Parallelized training and tuning**: SageMaker enables parallelized training across multiple instances, reducing training time for large datasets and complex models. It also supports parallelized hyperparameter tuning, allowing efficient exploration of model configurations with minimal code while maintaining fine-grained control over the process.

- **Support for custom scripts**: While SageMaker offers built-in algorithms and models (e.g., XGBoost, KMeans, Resnet), it also allows users to bring their own customized scripts. This flexibility is crucial for researchers developing unique models or custom algorithms. SageMaker's support for Docker containers allows you to deploy fully customized code for training, tuning, and inference on scalable AWS infrastructure.

- **Cost management and monitoring**: SageMaker includes built-in monitoring tools to help you track and manage costs, ensuring you can scale up efficiently without unnecessary expenses. With thoughtful usage, SageMaker can be very affordable. For example, training roughly 100 small to medium-sized models (e.g., logistic regression, random forests, or lightweight deep learning models with a few million parameters) on a small dataset (under 10GB) can cost under $20, making it accessible for many research projects.

In summary, Amazon SageMaker is a fully managed machine learning platform that simplifies building, training, tuning, and deploying models at scale. Unlike traditional research computing environments, which often require manual job orchestration and complex dependency management, SageMaker provides an integrated and automated workflow, allowing users to focus on model development rather than infrastructure. With support for on-demand compute resources, parallelized training and hyperparameter tuning, and flexible model deployment options, SageMaker enables researchers to scale experiments efficiently. Built-in cost tracking and monitoring tools also help keep expenses manageable, making SageMaker a practical choice for both small-scale research projects and large-scale ML pipelines. By combining preconfigured machine learning algorithms, support for custom scripts, and robust computing power, SageMaker reduces the complexity of ML development, empowering researchers to iterate faster and bring models to production more seamlessly.

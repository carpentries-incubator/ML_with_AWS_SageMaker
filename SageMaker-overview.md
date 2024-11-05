---
title: "Overview of Amazon SageMaker"
teaching: 10
exercises: 0
---

Amazon SageMaker is a comprehensive machine learning platform that empowers users to build, train, tune, and deploy models at scale. Designed to streamline the ML workflow, SageMaker supports data scientists and researchers in tackling complex machine learning problems without needing to manage underlying infrastructure. This allows you to focus on developing and refining your models while leveraging AWS’s robust computing resources for efficient training and deployment.

### Why use SageMaker for machine learning?

SageMaker provides several features that make it an ideal choice for researchers and ML practitioners:

- **End-to-end workflow**: SageMaker covers the entire ML pipeline, from data preprocessing to model deployment. This unified environment reduces the need to switch between platforms or tools, enabling users to set up, train, tune, and deploy models seamlessly.

- **Flexible compute options**: SageMaker lets you easily select instance types tailored to your project needs. For compute-intensive tasks, such as training deep learning models, you can switch to GPU instances for faster processing. SageMaker’s scalability also supports parallelized training, enabling you to distribute large training jobs across multiple instances, which can significantly speed up training time for large datasets and complex models.

- **Efficient hyperparameter tuning**: SageMaker provides powerful tools for automated hyperparameter tuning, allowing users to perform complex cross-validation (CV) searches with a single chunk of code. This feature enables you to explore a wide range of parameters and configurations efficiently, helping you find optimal models without manually managing multiple training runs.

- **Support for Custom Scripts**: While SageMaker offers built-in algorithms, it also allows users to bring their own customized scripts. This flexibility is crucial for researchers developing unique models or custom algorithms. SageMaker’s support for Docker containers allows you to deploy fully customized code for training, tuning, and inference on scalable AWS infrastructure.

- **Cost management and monitoring**: SageMaker includes built-in monitoring tools to help you track and manage costs, ensuring you can scale up efficiently without unnecessary expenses. With thoughtful usage, SageMaker can be very affordable—for example, training roughly 100 models on a small dataset (under 1GB) can cost less than $20, making it accessible for many research projects.

SageMaker is designed to support machine learning at any scale, making it a strong choice for projects ranging from small experiments to large research deployments. With robust tools for every step of the ML process, it empowers researchers and practitioners to bring their models from development to production efficiently and effectively.

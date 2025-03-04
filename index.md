---
site: sandpaper::sandpaper_site
---

## Workshop Overview

This workshop introduces you to foundational workflows in **Amazon SageMaker**, covering data setup, code repo setup, model training, and hyperparameter tuning within AWS's managed environment. You’ll learn how to use SageMaker notebooks to control data pipelines, manage training and tuning jobs, and evaluate model performance effectively. We’ll also cover strategies to help you scale training and tuning efficiently, with guidance on choosing between CPUs and GPUs, as well as when to consider parallelized workflows (i.e., using multiple instances).

To keep costs manageable, this workshop provides tips for tracking and monitoring AWS expenses, so your experiments remain affordable. While AWS isn’t entirely free, it can be very cost-effective for common machine learning (ML) workflows in used in research. For example, training roughly 100 small to medium-sized models (e.g., logistic regression, random forests, or lightweight deep learning models with a few million parameters) on a small dataset (under 10GB) can cost under $20, making it accessible for many research projects. 

### What This Workshop Does Not Cover (Yet)

Currently, this workshop does not include:

- **AWS Lambda** for serverless function deployment,
- **MLFlow** or other MLOps tools for experiment tracking,
- Additional AWS services beyond the core SageMaker ML workflows.

If there's a specific ML or AI workflow or AWS service you'd like to see included in this curriculum, please let us know! We're happy to develop additional content to meet the needs of researchers, students, and ML/AI practitioners. Please [post an issue on the lesson GitHub](https://github.com/carpentries-incubator/ML_with_AWS_SageMaker/issues) or contact [endemann@wisc.edu](mailto:endemann@wisc.edu)) with suggestions or requests.

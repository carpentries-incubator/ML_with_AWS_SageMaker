---
site: sandpaper::sandpaper_site
---

## Workshop Overview

This workshop introduces you to foundational workflows in **Amazon SageMaker**, covering data setup, code repo setup, model training, and hyperparameter tuning within AWS's managed environment. You’ll learn how to use SageMaker notebooks to control data pipelines, manage training and tuning jobs, and evaluate model performance effectively. We’ll also cover strategies to help you scale training and tuning efficiently, with guidance on choosing between CPUs and GPUs, as well as when to consider parallelized workflows (i.e., using multiple instances).

To keep costs manageable, this workshop provides tips for tracking and monitoring AWS expenses, so your experiments remain affordable. While AWS isn’t entirely free, it can be very cost-effective for common machine learning (ML) workflows in used in research. For example, training roughly 100 small to medium-sized models (e.g., logistic regression, random forests, or lightweight deep learning models with a few million parameters) on a small dataset (under 10GB) can cost under $20, making it accessible for many research projects. 

### What This Workshop Does Not Cover (Yet)

Currently, this workshop does not include:

- **AWS Lambda**: Lambda lets you run small pieces of code without setting up or managing a server. You write a function, tell AWS when to run it (for example, when a file is uploaded or a request comes in), and it automatically runs and scales as needed. This is useful for simple tasks like cleaning data as it arrives, kicking off a training job, or running a quick analysis without having to keep a server running all the time.
- **Bedrock**: Amazon Bedrock can be used to build and scale generative AI applications by providing API access to a range of foundation models from AWS and third-party providers, without managing infrastructure. It's designed to simplify integrating text, image, and other generative capabilities into your workflows using familiar AWS tools. Much of what Bedrock enables can also be done in SageMaker, but Bedrock trades off flexibility for simplicity — you get faster access to models and lower setup overhead, but have less control over training, fine-tuning, and the underlying infrastructure, which can matter for research workflows that need custom architectures, reproducibility, or integration with existing pipelines
- **Additional AWS services** beyond the core SageMaker ML workflows.

If there's a specific ML or AI workflow or AWS service you'd like to see included in this curriculum, please let us know! We're happy to develop additional content to meet the needs of researchers, students, and ML/AI practitioners. Please [post an issue on the lesson GitHub](https://github.com/carpentries-incubator/ML_with_AWS_SageMaker/issues) or contact [endemann@wisc.edu](mailto:endemann@wisc.edu) with suggestions or requests.

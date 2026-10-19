---
site: sandpaper::sandpaper_site
---

## Workshop Overview

This workshop introduces machine learning workflows on AWS using **Amazon SageMaker**. It has three parts:

1. **SageMaker foundations and classic ML.** Storing data in S3, using a small notebook instance as a *controller*, running training jobs (XGBoost and PyTorch), and hyperparameter tuning, with guidance on choosing CPU vs. GPU instances and when parallel instances help.
2. **Retrieval-Augmented Generation (RAG).** Building a RAG pipeline over a corpus of research papers two ways: with **Amazon Bedrock** (pay per token, nothing left running) and with SageMaker **Processing jobs** serving an open-weight model that Bedrock does not offer.
3. **Fine-tuning.** Fine-tuning a 3B open-weight vision-language model with LoRA as a SageMaker training job, for handwriting transcription, with before/after evaluation on held-out pages.

Three optional self-study episodes can be skipped without affecting the rest: pushing code to GitHub from a notebook instance, running the RAG pipeline on a GPU notebook instance, and CLI equivalents of the lesson's commands.

### Cost controls

Every episode applies the same controls: cost-allocation tags on every resource, a runtime cap on every job, a cost estimate before each launch compared to the bill after, and cleanup at the end. The final episode covers monitoring and cleanup. Run as written, the full workshop costs a few dollars per learner.

### What This Workshop Does Not Cover (Yet)

Currently, this workshop does not include:

- **Inference endpoints**: serving a model behind an always-on API for an application with steady traffic. The lesson explains when an endpoint is the right tool and what it costs, but does not deploy one.
- **AWS Lambda**: Lambda lets you run small pieces of code without setting up or managing a server. You write a function, tell AWS when to run it (for example, when a file is uploaded or a request comes in), and it automatically runs and scales as needed. This is useful for simple tasks like cleaning data as it arrives, kicking off a training job, or running a quick analysis without having to keep a server running all the time.
- **Additional AWS services** beyond SageMaker, Bedrock and S3.

If there's a specific ML or AI workflow or AWS service you'd like to see included in this curriculum, please let us know! We're happy to develop additional content to meet the needs of researchers, students, and ML/AI practitioners. Please [post an issue on the lesson GitHub](https://github.com/carpentries-incubator/ML_with_AWS_SageMaker/issues) or contact [endemann@wisc.edu](mailto:endemann@wisc.edu) with suggestions or requests.

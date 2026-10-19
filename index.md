---
site: sandpaper::sandpaper_site
---

## Workshop Overview

This workshop introduces foundational machine learning workflows on AWS, with **Amazon SageMaker** as the backbone and cost control as the thread that runs through every episode. It is organized in three parts:

1. **SageMaker foundations and classic ML.** S3 for data, a small notebook instance as a *controller*, training jobs (XGBoost and PyTorch) and hyperparameter tuning on the Titanic dataset, with guidance on CPU vs. GPU instances and when parallel instances pay off. The episode on pushing to GitHub with a personal access token is optional self-study.
2. **Retrieval-Augmented Generation (RAG).** The [WattBot](https://www.kaggle.com/competitions/WattBot2025) energy-literature question-answering challenge built three ways: on **Amazon Bedrock** (pay per token, nothing left running), then as SageMaker **Processing jobs** serving an open-weight model that Bedrock does not offer, and, as optional self-study, interactively on a GPU notebook instance. The two optional episodes (GitHub PAT and GPU notebook) are not needed to follow the rest of the lesson.
3. **Fine-tuning.** A LoRA fine-tune of a 3B open-weight vision-language model as a SageMaker training job, on the [Badger Scribe](https://www.kaggle.com/competitions/badger-scribe) handwriting-transcription challenge (German Kurrent letters from UW–Madison's Max Kade Institute), with before/after scoring on held-out pages.

Cost control is taught as practice rather than as a warning: cost-allocation tags on every resource, a runtime cap on every job, a written cost estimate before each launch compared to the bill afterwards, back-of-the-envelope rules for what a fine-tune will need in memory and GPU-hours, and explicit cleanup. AWS is not free, but used this way it is affordable for research: training a hundred small models on a small dataset costs under $20, a full WattBot evaluation on Bedrock costs cents, and the fine-tuning episode costs about $1.55 in class (plus about $1.75 for the optional homework). The lesson is equally explicit about what goes wrong: a forgotten GPU notebook instance is about $60 a day, and a full fine-tune of a large model is where five-figure bills come from.

### What This Workshop Does Not Cover (Yet)

Currently, this workshop does not include:

- **Inference endpoints**: serving a model behind an always-on API for an application with steady traffic. The lesson explains when an endpoint is the right tool and what it costs, but does not deploy one.
- **AWS Lambda**: Lambda lets you run small pieces of code without setting up or managing a server. You write a function, tell AWS when to run it (for example, when a file is uploaded or a request comes in), and it automatically runs and scales as needed. This is useful for simple tasks like cleaning data as it arrives, kicking off a training job, or running a quick analysis without having to keep a server running all the time.
- **Additional AWS services** beyond SageMaker, Bedrock and S3.

If there's a specific ML or AI workflow or AWS service you'd like to see included in this curriculum, please let us know! We're happy to develop additional content to meet the needs of researchers, students, and ML/AI practitioners. Please [post an issue on the lesson GitHub](https://github.com/carpentries-incubator/ML_with_AWS_SageMaker/issues) or contact [endemann@wisc.edu](mailto:endemann@wisc.edu) with suggestions or requests.

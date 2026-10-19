---
site: sandpaper::sandpaper_site
---

## Workshop Overview

This workshop introduces machine learning workflows on AWS using **Amazon SageMaker**. It has three parts:

1. **SageMaker foundations and classic ML.** Storing data in S3, using a small notebook instance as a *controller*, running training jobs (XGBoost and PyTorch), and hyperparameter tuning, with guidance on choosing CPU vs. GPU instances and when parallel instances help.
2. **Retrieval-Augmented Generation (RAG).** Building a RAG pipeline over a corpus of research papers two ways: with **Amazon Bedrock** (pay per token, nothing left running) and with SageMaker **Processing jobs** serving an open-weight model that Bedrock does not offer.
3. **Fine-tuning.** Fine-tuning a 3B open-weight vision-language model with LoRA as a SageMaker training job, for handwriting transcription, with before/after evaluation on held-out pages.

### Who this workshop is for

Researchers and students who already train models and want to run that work on AWS. The lesson assumes a basic machine learning background: you have trained a model before, and you know what training, hyperparameter tuning, fine-tuning, embeddings and retrieval-augmented generation are. It does not teach these concepts; instructors may briefly review some of them depending on the audience, but the default schedule does not include that time. Python and Jupyter are used throughout; no prior AWS experience is needed.

### Cost controls

Every episode applies the same controls: cost-allocation tags on every resource, a runtime cap on every job, a cost estimate before each launch compared to the bill after, and cleanup at the end. The final episode covers monitoring and cleanup. Run as written, the full workshop costs a few dollars per learner.

### What This Workshop Does Not Cover

- **Deploying models as endpoints**: serving a trained or fine-tuned model behind an always-on API.
- **Agents and tool use**: LLM applications that call tools or chain multiple model calls (e.g., Bedrock Agents).
- **Evaluation beyond a single metric**: LLM-as-judge, human evaluation, and benchmark suites.
- **MLOps**: pipelines, experiment tracking, model registries, and scheduled retraining.

If there's a specific ML or AI workflow or AWS service you'd like to see included in this curriculum, please let us know! We're happy to develop additional content to meet the needs of researchers, students, and ML/AI practitioners. Please [post an issue on the lesson GitHub](https://github.com/carpentries-incubator/ML_with_AWS_SageMaker/issues) or contact [endemann@wisc.edu](mailto:endemann@wisc.edu) with suggestions or requests.

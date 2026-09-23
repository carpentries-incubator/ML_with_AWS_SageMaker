---
title: "Overview of RAG Workflows on AWS"
teaching: 10
exercises: 0
---

## Retrieval-Augmented Generation (RAG) on AWS

Retrieval-Augmented Generation (RAG) is a pattern where you **retrieve** relevant context from your data and then **generate** an answer using that context. Unlike model training, a standard RAG workflow does **not** fine‑tune or train a model — it combines retrieval + inference only.

This episode introduces the major ways to build RAG systems on AWS and prepares us for later episodes where we experiment with each approach.

:::::::::::::::::::::::::::::::::::::: questions

- What is Retrieval‑Augmented Generation (RAG)?
- What are the main architectural options for running RAG on AWS?
- When is each RAG workflow appropriate?

::::::::::::::::::::::::::::::::::::::::::::::::

:::::::::::::::::::::::::::::::::::::: objectives

- Understand that RAG does *not* require training or fine‑tuning a model.
- Recognize the three major architectural patterns for RAG systems on AWS.
- Understand the core trade‑offs that drive which approach to use.

::::::::::::::::::::::::::::::::::::::::::::::::

## What is RAG?

RAG combines two steps:

1. **Retrieve**: Search your document store (vector DB or FAISS index) to find relevant text.
2. **Generate**: Provide those retrieved chunks to a large language model (LLM) to answer a question.

No model weights are updated. No backprop. No training job.  
RAG is an inference‑only pattern that layers retrieval logic around an LLM.

## Approaches to Running RAG on AWS

There are several general approaches for setting up a Retrieval-Augmented Generation (RAG) workflow on AWS, each suited to different scales and constraints.

1. **Run everything inside a single GPU-backed notebook instance**
For small- to medium-sized models (< 70 B), it's often simplest to just pick a GPU instance (e.g., [p3.2xlarge](https://carpentries-incubator.github.io/ML_with_AWS_SageMaker/instances-for-ML.html)), load your embedding and generation models directly in the notebook, and run RAG end-to-end there. This keeps the architecture simple and avoids extra moving parts, as long as you're disciplined about shutting down the instance when you're done so you don't leak cost. It's also possible to do this with larger models, but the costs to use more powerful GPUs (e.g., $15/hour) may be a limiting factor.

2. **Use SageMaker Processing Jobs to run batch jobs for embeddings and/or generation.**
For large corpora or workflows where you want repeatable, offline computation, you can treat parts of the RAG pipeline—especially embedding and generation—like batch processing jobs rather than a live model. Instead of using a notebook GPU, you run a short-lived Hugging Face Processing job that spins up a GPU instance, loads your embedding or generation model, processes all the chunked text in one shot, and saves the results back to S3. This pattern is ideal for "compute once, use many times" workloads, such as generating embeddings for thousands of documents or producing long-form outputs that don't require low latency. Because the job only exists while the batch completes, you avoid the continuous cost of an always-on endpoint. However, this approach is not suited for per-query RAG retrieval—launching a job per user request would be far too slow, since starting a Processing job can take several minutes.

3. **Use Amazon Bedrock for managed embedding and generation APIs.** If you prefer fully managed foundation models and don't want to own model hosting at all, Bedrock lets you call embedding and generation models via API from your RAG pipeline. You still can still manage retrieval logic (e.g., add reranking), but you outsource the heavy model lifecycle work—at the trade-off of less control over architectures and sometimes higher per-token cost. Bedrock can also give RAG systems access to proprietary models which would need to be purhcased separately otherwise.

4. **Use long-lived inference endpoints for online RAG workloads.** For applications that need low-latency, interactive RAG (APIs, chatbots, dashboards), you can deploy your own embedding and generation models as SageMaker inference endpoints (or Bedrock-like managed endpoints) and call them from your retrieval service. This gives you control over the model, scaling policies, and autoscaling, but it's also the most expensive option if traffic is low or bursty, since you're keeping capacity online even when no one is querying the system.


## When Do You Use Which Approach?

1. **Notebook RAG**: Fastest to build. Great for learning, prototyping, and small‑scale research.

2. **Processing‑job RAG**: Ideal for embedding large corpora and running periodic batch generation. Clean, reproducible, cost‑efficient (especially if you spend a lot of time in your notebook viewing results, rather than generating them).

3. **Bedrock RAG**: Best for production or long‑term research tools that need scalability without hosting models. Bedrock can also give RAG systems access to proprietary models which would need to be purhcased separately otherwise.

4. TODO
   
:::::::::::::::::::::::::::::::::::::: keypoints

- RAG is an inference‑only workflow: no training or fine‑tuning required.
- AWS supports three broad approaches: notebook RAG, batch RAG, and Bedrock‑managed RAG.
- The right choice depends on latency needs, scale, cost sensitivity, and model‑management preferences.
- Later episodes will walk through each pattern in depth using hands‑on examples.

::::::::::::::::::::::::::::::::::::::::::::::::

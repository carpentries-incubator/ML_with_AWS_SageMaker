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

Below are the four common patterns used across research and industry.

### New Cloud Stuff

There are several general approaches for setting up a Retrieval-Augmented Generation (RAG) workflow on AWS, each suited to different scales and constraints.


1. **Run everything inside a single GPU-backed notebook instance**
For small- to medium-sized models (< 8 B), it's often simplest to just pick a GPU instance (e.g., [p3.2xlarge](https://carpentries-incubator.github.io/ML_with_AWS_SageMaker/instances-for-ML.html)), load your embedding and generation models directly in the notebook, and run RAG end-to-end there. This keeps the architecture simple and avoids extra moving parts, as long as you're disciplined about shutting down the instance when you’re done so you don’t leak cost. It's also possible to do this with larger models, but the costs to use more powerful GPUs (e.g., $15/hour) may be a limiting factor.

2. **Use SageMaker Estimators to run batch jobs for embeddings and/or generation.**
For large corpora or workflows where you want repeatable, offline computation, you can treat parts of the RAG pipeline—especially embedding—like a batch processing job rather than a live model. Instead of deploying an inference endpoint, you run a short-lived Hugging Face Estimator job that spins up a GPU instance, loads your embedding or generation model, processes all the chunked text in one shot, and saves the results back to S3. This pattern is ideal for “compute once, use many times” workloads, such as generating embeddings for thousands of documents or producing long-form outputs that don’t require low latency. Because the Estimator only exists while the batch completes, you avoid the continuous cost of an always-on endpoint. However, this approach is not suited for per-query RAG retrieval—launching an Estimator per user request would be far too slow, since starting a training job can take several minutes.

3. **Use Amazon Bedrock for managed embedding and generation APIs.** If you prefer fully managed foundation models and don’t want to own model hosting at all, Bedrock lets you call embedding and generation models via API from your RAG pipeline. You still manage your vector store and retrieval logic (for example in Lambda, ECS, or SageMaker), but you outsource the heavy model lifecycle work—at the trade-off of less control over architectures and sometimes higher per-token cost.

4. **Use long-lived inference endpoints for online RAG workloads.** For applications that need low-latency, interactive RAG (APIs, chatbots, dashboards), you can deploy your own embedding and generation models as SageMaker inference endpoints (or Bedrock-like managed endpoints) and call them from your retrieval service. This gives you control over the model, scaling policies, and autoscaling, but it’s also the most expensive option if traffic is low or bursty, since you’re keeping capacity online even when no one is querying the system.


### 1. **Notebook‑based RAG (single GPU instance)**
Load embedding + generation models inside a GPU‑backed SageMaker notebook instance.

**When this works well**
- Small/medium models (< 8–12B). Large models also work, but be mindful of instance costs!
- Workshops, demos, prototyping, exploratory RAG
- RAG pipelines that only need to be run once (not live in production mode where you can constantly query your bot)
- You want everything in one place with minimal architecture
- You are feeling lazy and don't want to convert your code to a processing job or API calls

**Trade‑offs**
- Must shut down instance to avoid cost leakage
- Limited by single‑GPU memory
- Not all of the time spent in your notebook requires a GPU (wasteful for chunking + indexing)
- If you want to use a better instance later, you have to stop your notebook, change the instance type, and restart it (annoying but doable)

### 2. **Batch RAG with SageMaker Processing Jobs** (effecient!)
Use short‑lived processing jobs for embedding corpora or offline batch generation.

**When this works well**
- Only pay for the GPU time that you actually need (embedding and generation steps)
- "Compute once, use many times" tasks
- Repeatable pipelines where you want clean logs + reproducibility

**Trade‑offs**
- Not suitable for *interactive* or per‑query RAG (startup time too slow)
- Requires setting up processing jobs in SageMaker, similar to how we've done training jobs before. This isn't terribly difficult once you learn the basics, but it can slow progress the first time through.

### 3. **Fully managed RAG using Amazon Bedrock APIs**
Use Bedrock models for embedding + generation via API calls.

**When this works well**
- No desire to host or manage models
- Need scalability without managing GPUs
- Want access to the largest and greatest foundation models with guaranteed availability

**Trade‑offs**
- Per‑token cost may be higher than renting an instance/GPU if you have a large corpus or high query volume
- Latency may be higher due to network calls

:::::::::::::::::::::::::::::::::::::: callout

### RAG Pipeline Responsibilities
Even with Bedrock, *you* still manage:
- Chunking
- Embedding storage
- Vector search
- Retrieval logic  

Bedrock only handles the embedding + generation models.

::::::::::::::::::::::::::::::::::::::::::::::::

## When Do You Use Which Approach?

**Notebook RAG**  
Fastest to build. Great for learning, prototyping, and small‑scale research.

**Processing‑job RAG**  
Ideal for embedding large corpora and running periodic batch generation. Clean, reproducible, cost‑efficient.

**Bedrock RAG**  
Best for production or long‑term research tools that need scalability without hosting models.

:::::::::::::::::::::::::::::::::::::: keypoints

- RAG is an inference‑only workflow: no training or fine‑tuning required.
- AWS supports three broad approaches: notebook RAG, batch RAG, and Bedrock‑managed RAG.
- The right choice depends on latency needs, scale, cost sensitivity, and model‑management preferences.
- Later episodes will walk through each pattern in depth using hands‑on examples.

::::::::::::::::::::::::::::::::::::::::::::::::

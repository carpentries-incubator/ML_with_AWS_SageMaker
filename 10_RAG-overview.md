---
title: "Overview of RAG Workflows on AWS"
teaching: 15
exercises: 0
---

:::::::::::::::::::::::::::::::::::::: questions

- What is Retrieval-Augmented Generation (RAG)?
- What are the four ways to run inference on AWS, and which does a RAG pipeline need?
- Why is Bedrock the default, and when is SageMaker the right choice instead?
- How do the billing models differ, and what does it cost to forget to shut something down?

::::::::::::::::::::::::::::::::::::::::::::::::

:::::::::::::::::::::::::::::::::::::: objectives

- Explain that RAG trains nothing, so running one does not need a GPU of your own.
- Name the four ways to run inference on AWS and what each one bills for.
- Say why pay-per-token Bedrock is the default for most research RAG work.
- Say when self-managed GPU compute (Processing jobs, notebook GPUs, endpoints) is worth it.
- Weigh the cost and control trade-offs that decide which to use.

::::::::::::::::::::::::::::::::::::::::::::::::

## Retrieval-Augmented Generation (RAG) on AWS

Retrieval-Augmented Generation (RAG) is a pattern where you **retrieve** relevant passages from your own documents and then **generate** an answer from them. The model is used, not changed. No weights are updated and no training job runs, so the documents can change every week and nothing has to be retrained. Using a trained model to get an output from it, a prediction, an embedding or an answer, is called **inference**, and everything in this part of the lesson is inference. The example throughout is [WattBot](https://www.kaggle.com/competitions/WattBot2026), a question-answering challenge over a corpus of research papers on AI and energy.

There are four ways to run inference on AWS. Call a model through Bedrock's managed API, run it in a Processing job that ends on its own, load it on a GPU notebook instance, or deploy it as an inference endpoint. This episode compares the four, says why Bedrock is the one to start with, and sets up the episodes that follow, which build the same RAG pipeline in three of these ways and weigh the trade-offs of each.

## What is RAG?

![RAG pipeline: PDFs are chunked and embedded once; per question, the question is embedded, the top-k chunks are retrieved by cosine similarity, and a generator answers from question plus chunks. The embedder and generator are the only parts that change between episodes.](fig/rag-pipeline.svg){alt="Diagram of the RAG pipeline with the embedder and generator highlighted as the two model calls"}

A language model cannot reliably answer questions about documents it has never seen. RAG connects the model to a collection of material you control. You break the documents into passages, turn each passage into a vector with an **embedding model**, and keep the vectors so you can search them later. When someone asks a question, you embed the question with the same model and find the passages whose vectors are most similar. Those passages go to a **generation model** together with the question, and it writes an answer grounded in what was retrieved, citing the passages that support it. When the evidence is not there, a well-built pipeline says so instead of guessing.

So there are two model calls, and neither is training. The embedding model is called once per chunk when the corpus is indexed and once per question. The generation model is called once per question. Both are inference with no state between calls, so neither needs a GPU of your own. The rest of the pipeline, preparing documents, searching vectors, assembling prompts and scoring answers, is ordinary Python. That code stays the same while the place the models run changes, which is what the next three episodes show.

## Bedrock is the default for inference

**For most RAG pipelines, and for inference in general, Amazon Bedrock is the fastest and most cost-effective starting point on AWS.** Bedrock serves embedding and generation models (Amazon Titan, Anthropic Claude, DeepSeek, Meta Llama and others) through pay-per-token APIs. Your retrieval code, chunking, indexing, similarity search, evaluation, stays yours and runs on a cheap CPU notebook or a laptop. Only the two model calls go to Bedrock.

The difference is in how each one bills.

| | Bedrock (managed API) | SageMaker GPU instance / endpoint |
|---|---|---|
| **Billing unit** | Per token processed | Per second while the instance runs, at an hourly rate |
| **Cost while idle** | `$0`: nothing runs between calls | The full hourly rate, around the clock, used or not |
| **What you have to remember** | The per-token price before a big batch; route calls through a tagged profile | Stop or delete the instance, every time |
| **Model updates** | AWS and the provider | You: containers, CUDA, GPU memory |
| **Model choice** | A curated catalog, proprietary models included | Any open-weight model you can load |

Self-hosting on SageMaker gives you control over the weights, the libraries and the hardware, which matters for a specific open-weight model or your own fine-tune. The trade is that a rented GPU bills while the instance runs, whether or not it is generating answers, and research work is mostly occasional experiments, not a steady stream of requests. A GPU instance is a rental whose meter runs until you turn it off, and forgetting to is the most common way a research team spends its cloud budget on nothing. Bedrock has no such failure. Stop calling it and you stop paying.

### What forgetting costs

A GPU notebook instance left running after you close your laptop (on-demand prices at the time of writing; the [instances page](https://carpentries-incubator.github.io/ML_with_AWS_SageMaker/instances-for-ML.html) has the current ones):

| Instance | GPU | `$`/hour | Forgotten overnight (12 h) | Forgotten over a weekend (64 h) | Forgotten for a month (730 h) |
|---|---|---|---|---|---|
| ml.g4dn.xlarge | 1 × T4 (16 GB) | `$0.74` | ~`$9` | ~`$47` | ~`$540` |
| ml.g5.xlarge | 1 × A10G (24 GB) | `$1.41` | ~`$17` | ~`$90` | ~`$1,030` |
| ml.g5.2xlarge | 1 × A10G (24 GB) | `$1.52` | ~`$18` | ~`$97` | ~`$1,110` |
| ml.g6e.xlarge | 1 × L40S (48 GB) | `$2.61` | ~`$31` | ~`$167` | ~`$1,906` |
| ml.p4d.24xlarge | 8 × A100 (40 GB) | `$25.25` | ~`$303` | ~`$1,616` | ~`$18,430` |

Against that, the whole WattBot pipeline of this lesson on Bedrock, with DeepSeek V3 for generation (the [Bedrock pricing page](https://aws.amazon.com/bedrock/pricing/) has current per-token prices) and Titan Text Embeddings V2 (`$0.02` per million tokens):

| Step | Rough volume | Cost |
|---|---|---|
| Embed ~9,000 chunks (~2.4M tokens) | 2.4M input tokens | ~`$0.05` |
| Answer and explain the 220 answerable training questions (two calls each, ~1.5K-token prompts) with DeepSeek V3 | 650K in / 40K out | ~`$0.50` |
| **Total** | | **about `$0.55`** |

One forgotten weekend on the cheapest GPU instance costs more than fifty full runs of the pipeline on Bedrock; a forgotten weekend on the `ml.g6e.xlarge` the optional GPU-notebook episode uses costs more than a hundred.

::::::::::::::::::::::::::::::::::::: callout

### Where Bedrock falls short

- **Model choice.** You can only call models in the catalog. If the work depends on a specific open-weight model, your own fine-tune, or changes to the model's internals, you host it yourself.
- **Very large batches.** Embedding tens of millions of chunks with a small open model can be cheaper per unit on a short-lived GPU job than per token. Do the arithmetic before assuming either way.
- **Access and regions.** Some models have to be enabled in the console first, and the catalog differs by region. Check before a workshop or a deadline.
- **Quotas.** On-demand throughput is rate-limited per model and per account. For a large batch, use the batch inference API or request a quota increase instead of retrying against the limit.
- **Data handling.** Bedrock does not train on your prompts and keeps data in the region, or within the US when a `us.` cross-region profile is used; your institution's policy on sending documents to a hosted API still applies.
- **Cost tracking takes one extra step.** On-demand calls carry no tags. To attribute spend to a project you route calls through a tagged *application inference profile* (the Tagging section below, and the Bedrock episode). Set up once; skip it and the usage is anonymous on the bill.

:::::::::::::::::::::::::::::::::::::::::::::::::

## Four ways to run inference on AWS

The four differ less in what the model computes than in how long the hardware stays rented. Here they are in the order to try them.

![Four columns, Bedrock API, Processing job, GPU notebook and Inference endpoint, compared on where the model runs, what runs between calls, billing unit, latency, model choice, what you must remember, a cost example and which episode covers it; Bedrock is marked as the default and Processing jobs as the safe way to use a GPU](fig/rag-approaches.svg){alt="Comparison table of four ways to run inference on AWS: Bedrock API (default, pay per token, nothing to shut down), Processing job (batch with open models, bills per second while the job runs, terminates itself), GPU notebook (learning and debugging only, bills per hour until stopped, about $90 if forgotten over a weekend) and Inference endpoint (steady traffic only, bills 24/7, about $1,000 if idle for a month)."}

### 1. Amazon Bedrock, the default

Call the embedding and generation models from your pipeline. You keep the retrieval logic; AWS keeps the models. You pay per token, nothing runs between calls, and there is nothing to shut down. Bedrock is also where proprietary models such as Claude are available without a separate contract.

**Use it for:** any RAG system where a catalog model will do, from prototypes and course projects to research tools and production chatbots.

### 2. SageMaker Processing jobs

For a large corpus, or for an open-weight model Bedrock does not carry, run the embedding and the batch generation as jobs instead of against a live model. A Hugging Face Processing job starts a GPU instance, loads the model, works through all the chunks or questions, writes the results to S3 and ends. The GPU exists only while the job runs, so you get a self-managed model without the risk of forgetting to stop it. This is the safest way to use a GPU on SageMaker.

The cost is latency. A job takes several minutes to start, so this is for work done once and used many times, not for answering one question at a time.

**Use it for:** embedding a large corpus with an open model, periodic batch regeneration, or research that needs a model Bedrock does not offer.

### 3. A GPU notebook instance

For models up to about 70B you can choose a GPU instance (an [ml.g5.xlarge](https://carpentries-incubator.github.io/ML_with_AWS_SageMaker/instances-for-ML.html), say), load the embedding and generation models in the notebook, and run RAG end to end there. It is the simplest setup to follow, and the best way to learn what happens inside a RAG pipeline, because every step is a line of Python you can inspect.

It is also the most expensive way to run RAG and the easiest to get wrong. The GPU bills by the hour from the moment the instance starts until you stop it, through every stretch spent downloading PDFs, chunking text, reading results or at lunch. It is the route most often forgotten, because a notebook instance looks like a workspace, not a running job. If you use it:

- Treat it as a learning exercise, not a way to deploy anything.
- Stop the instance the moment you are done, and check its status on the [Notebook instances page](https://us-east-2.console.aws.amazon.com/sagemaker/home?region=us-east-2#/notebook-instances).
- Set an [AWS Budget](https://docs.aws.amazon.com/cost-management/latest/userguide/budgets-managing-costs.html) alert, and consider a [lifecycle configuration](https://docs.aws.amazon.com/sagemaker/latest/dg/notebook-lifecycle-config.html) that stops idle notebooks.
- For larger models the hourly rate (`$25` an hour for a p4d.24xlarge; the [Instances for ML](https://carpentries-incubator.github.io/ML_with_AWS_SageMaker/instances-for-ML.html) page has the list) becomes the limit quickly.

**Use it for:** walking through RAG step by step, debugging a model before packaging it as a job, or short experiments where you stay at the keyboard.

### 4. SageMaker inference endpoints

For an application that needs interactive, low-latency answers (an API, a chatbot, a dashboard) from a model you host yourself, deploy the embedding and generation models as SageMaker inference endpoints and call them from your retrieval service. You control the model, the scaling and the hardware.

It is also the most expensive option when traffic is low or bursty, because you pay for capacity that stays up when nobody is asking, the same always-on problem as a notebook GPU, now by design and without end. Before choosing it, ask whether Bedrock already serves a model that will do; it gives the same interactive latency with no idle cost. Endpoints are the answer only when the model itself is why Bedrock will not work.

**Use it for:** production systems with steady traffic that need a custom or open-weight model. The lesson does not build one.

Bedrock charges per call, a Processing job charges for one finite run, and a notebook or an endpoint charges for as long as it is left running. Those lifecycles, not the RAG algorithm, decide which to use.

## Every billable resource needs its own tags

Cost tracking in this workshop relies on three tags, `Name`, `Project` and `Purpose`, so that Cost Explorer can split the shared bill by challenge. **Tags never propagate.** Tagging your notebook instance tags the notebook instance and nothing else. Every other resource you create is a separate line on the bill, anonymous unless you tag it when you create it.

| Resource | What it bills | How it gets tagged |
|---|---|---|
| Notebook instance | Hourly, while running | Tags you set in the console when creating it |
| S3 bucket | Storage + requests | Tags you set in the console when creating it |
| Training / tuning job | Per second while the job runs, at an hourly rate | `tags=` argument on the estimator or tuner, **every job** |
| Processing job | Per second while the job runs, at an hourly rate | `tags=` argument on the processor, **every job** |
| Inference endpoint | Hourly, while deployed | `tags=` argument on `deploy()` |
| Bedrock on-demand call | Per token | **Cannot be tagged directly.** Create an *application inference profile* with tags and pass its ARN as `modelId` instead of the model ID |

Three things follow:

1. **Each job gets its own tags.** A SageMaker job launched from a tagged notebook does not inherit the notebook's tags. Pass `tags=job_tags` on every estimator, tuner and processor, as the training, tuning and Processing-job episodes do. Vary `Purpose` per job so the bill says what the money went to.
2. **Bedrock is tagged through a profile, not a call.** `invoke_model` and `converse` have no tags parameter. An application inference profile is a free wrapper around one model that carries tags; usage routed through it is attributed to those tags. One profile per base model, so one for the embedding model and one for the generation model. Bedrock's tag format is lowercase `key`/`value` where SageMaker's is `Key`/`Value`; the tag names are the same.
3. **Only the three approved keys count.** `Name`, `Project` and `Purpose`, with that spelling and capitalisation; the bill ignores any other key, for the reason the S3 episode gives.

## Which to use

| Your situation | Route |
|---|---|
| You need to embed and query a document collection and a catalog model (Titan, Claude, Llama, ...) will do | **Bedrock** |
| You want an interactive chatbot or API over your documents | **Bedrock** |
| You need a specific open-weight model and the work is batch (embedding a corpus, bulk generation) | **Processing jobs** |
| You are learning how RAG works, or debugging a model step by step | **Notebook GPU**, stopped when you leave |
| You need interactive serving of a model Bedrock does not offer, with steady traffic | **Inference endpoint** |

If in doubt, Bedrock. Its worst case is a somewhat higher per-token bill; the worst case on the other routes is a four-figure invoice next month.

::::::::::::::::::::::::::::::::::::: callout

### The order of the hands-on episodes

The next two episodes build RAG on **Bedrock** first, then on **Processing jobs**. The Bedrock episode is the one to start from. It builds the whole WattBot pipeline, chunking the corpus included, from a cheap CPU notebook with nothing left running afterward. The Processing-jobs episode reruns the same pipeline on an open-weight model Bedrock does not carry, as jobs that end on their own, which shows what Bedrock does for you and what it costs to do yourself. An optional self-study episode runs the same self-hosted pipeline interactively on a GPU notebook instance, for the case where you need to work on a self-hosted model by hand, and only if you will remember to stop the instance. All three share one module of pipeline code, `scripts/wattbot_rag.py` in the lesson repository.

When you build your own RAG system afterward, start from the Bedrock episode.

:::::::::::::::::::::::::::::::::::::::::::::::::

:::::::::::::::::::::::::::::::::::::: keypoints

- RAG is inference only. Nothing is trained or fine-tuned, so running one does not need a GPU of your own.
- **Bedrock is the default for inference**: you pay per token, nothing runs between calls, and there is nothing to forget to shut down.
- Self-managed GPU compute (Processing jobs, notebook GPUs, inference endpoints) is for a model Bedrock does not offer, or for batch volumes so large that per-token pricing loses; it is not a starting point.
- Of the GPU options, Processing jobs are the safest, because the instance ends itself; notebook GPUs and endpoints bill by the hour until you stop them.
- A forgotten GPU instance costs tens of dollars overnight and hundreds to thousands over a month; a full WattBot run on Bedrock costs under a dollar.
- Tags never propagate. Tag every job at launch and route Bedrock calls through a tagged application inference profile, or the spend cannot be traced.
- The hands-on episodes start with Bedrock, then the same pipeline on Processing jobs; the notebook-GPU version is optional self-study.

::::::::::::::::::::::::::::::::::::::::::::::::

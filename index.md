---
site: sandpaper::sandpaper_site
---

## Workshop Overview

This workshop is about running machine learning and AI work on AWS without losing control of the bill: where the data goes, which machine does what, how a job is launched and paid for, and how to find out what it cost. It is in four parts.

**Part 1, orientation.** What the cloud offers a researcher compared with a laptop or the cluster down the hall, and what it costs. The two services the lesson uses: **SageMaker** rents machines by the second, for a notebook you work in and for jobs that run on their own and end; **Bedrock** serves hosted models by the token, with nothing to start or stop. The four things every episode uses, S3, a notebook instance, jobs and Bedrock, and the cost habits every later episode follows.

**Part 2, SageMaker: data, training, tuning and fine-tuning.**

- Put the data in an S3 bucket and start a small notebook instance that launches everything else. The notebook is the controller, not the workhorse: the expensive compute runs on instances SageMaker rents for each job and releases when it ends.
- Train XGBoost and PyTorch models on the Titanic data as training jobs. Estimate the cost before each launch and read the billable seconds after, compare a CPU instance with a GPU, and see what two instances buy and what they do not.
- Run a hyperparameter tuning job that trains ten models under a budget set up front.
- Fine-tune a 3B open-weight vision-language model with LoRA, as a training job with checkpoints and a runtime cap, for transcribing handwritten German letters (Badger Scribe), scored on held-out pages before and after.

**Part 3, Bedrock: inference and retrieval-augmented generation.**

- Build a RAG pipeline over a corpus of energy research papers (WattBot): embed the papers, retrieve, generate, score. The models are Bedrock's, paid per token through tagged inference profiles so the spend has an owner, and the whole run costs cents.
- Add multimodal embeddings for the questions whose answers live in tables and figures, and measure what they change.
- Run the same pipeline with an open-weight model Bedrock does not serve, as SageMaker Processing jobs that rent a GPU for the minutes they need; optionally, the same on a notebook GPU, as the comparison.

**Part 4, resource management and cleanup.** What to stop, what to delete, how to read the bill, and the rules the account enforces on every launch, with how to set the same rules on an account of your own. An optional episode shows the same jobs from a terminal.

### Who this workshop is for

Researchers and students who use ML or AI in their work and want or need to run it in the cloud. The lesson assumes a basic machine learning background: you know what training, hyperparameter tuning, fine-tuning, embeddings and retrieval-augmented generation are. It does not teach these concepts; instructors may briefly review some of them depending on the audience, but the default schedule does not include that time. Python and Jupyter are used throughout; no prior AWS experience is needed.

### Cost controls

The cloud is often very affordable for research workloads, and this lesson shows it: run as written, the whole workshop, training jobs, a tuning job, a LoRA fine-tune on a GPU and two RAG pipelines, costs each learner a few dollars. What keeps it there is habit rather than restraint. Every episode tags its resources, caps job runtime, estimates cost before launching and checks the bill after, and cleans up. The final episode covers monitoring and cleanup, and the rules a shared account can enforce.

### What This Workshop Does Not Cover

- **General ML/AI concepts**: the focus is on moving existing ML and AI work to the cloud, not on teaching training, fine-tuning or RAG themselves.
- **Deploying models as endpoints**: serving a trained or fine-tuned model behind an always-on API.
- **Agents and tool use**: LLM applications that call tools or chain multiple model calls (e.g., Bedrock Agents).

If there's a specific ML or AI workflow or AWS service you'd like to see included in this curriculum, please let us know! We're happy to develop additional content to meet the needs of researchers, students, and ML/AI practitioners. Please [post an issue on the lesson GitHub](https://github.com/carpentries-incubator/ML_with_AWS_SageMaker/issues) or contact [endemann@wisc.edu](mailto:endemann@wisc.edu) with suggestions or requests.

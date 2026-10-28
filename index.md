---
site: sandpaper::sandpaper_site
---

## Workshop Overview

This workshop teaches researchers how to run machine learning and AI work on AWS without losing track of what it costs. It works through the practical decisions, where to store the data, which compute to use, how to launch and monitor a job, and how to estimate and manage the cost.

We use Amazon SageMaker to train, fine-tune and run our own models, and Amazon Bedrock to call models AWS hosts through an API. Throughout, we work from a small SageMaker notebook instance that acts as the controller. The demanding work runs on separate instances that SageMaker starts for each job and releases when the job finishes.

The workshop is in four parts, from AWS basics to training, inference and resource management.

::::::::::::::::::::::::::::::::::::: tab

### 1 Orientation

What cloud computing offers a researcher, when AWS makes sense next to a laptop or a campus cluster, and how its pricing works. Then the setup every later episode uses.

- **AWS services and pricing.** What SageMaker, Bedrock and S3 each do, and the difference between paying for compute time and paying per token.
- **Working environment.** Create an S3 bucket for data and results, launch a small SageMaker notebook instance, and read and write files in S3 from Python. Optionally, put your notebooks under version control.
- **Cost management.** Choose compute to fit the job, estimate the cost before launching, and track spending across the resources you create.

### 2 SageMaker

Training, fine-tuning and hyperparameter tuning. SageMaker runs the training scripts on separate compute instances, starting with small models and ending with a vision-language model of billions of parameters. Along the way we compare hardware choices and see how training time, model size and the number of experiments set the cost.

- **Model training.** Train XGBoost and PyTorch models on the Titanic data. Launch the training jobs from the notebook, compare CPU and GPU, and see when more than one instance helps.
- **Fine-tuning.** Adapt CHURRO-3B, an open-weight vision-language model, to transcribe historical German handwriting from the Badger Scribe challenge. Use LoRA to cut the GPU memory needed, estimate the run's cost, and measure transcription accuracy before and after.
- **Hyperparameter tuning.** Set up a search over learning rates and epochs, let SageMaker run the training jobs, and compare the results under a cost ceiling.

### 3 Bedrock

Inference and retrieval-augmented generation (RAG). Training changes a model's parameters. Inference uses a model that is already trained. We build a RAG pipeline that retrieves passages from research papers and has a language model answer questions from that evidence, then build it again with open-weight models on SageMaker, on the same problem, to compare cost, performance and flexibility.

- **RAG with Bedrock.** Build the pipeline for WattBot, a challenge of questions about the environmental impact of AI. Split research papers into searchable passages, embed them, retrieve the relevant evidence, and produce answers with citations. The models are Bedrock's, with usage tracked through tagged inference profiles.
- **Multimodal retrieval.** Extend the pipeline to retrieve from figures and tables as well as text, and measure whether that improves the answers.
- **RAG with SageMaker.** Run the embedding and generation models yourself, as GPU Processing jobs that shut down when they finish, and compare the results and the costs with Bedrock. An optional episode runs the same pipeline interactively on a GPU notebook, to see model loading and inference step by step.

### 4 Cleanup

Resource management and cost tracking. Cloud resources keep billing after you stop using them, notebook instances and stored data above all. We finish by reviewing what was created, checking what it cost, and cleaning up the account.

- Find the running resources and stored data, stop the notebook instance, and remove what is no longer needed, including S3 buckets and Bedrock inference profiles.
- Use Cost Explorer and the resource tags to see where the workshop's spending went.
- Review the account rules that kept costs in check during the workshop, and how to set the same safeguards on an AWS account of your own.
- Optionally, launch and manage the same resources from a terminal with the AWS CLI.

:::::::::::::::::::::::::::::::::::::

The four pieces the lesson is built from, which the overview episode introduces.

![S3 holds data and outputs; a small notebook instance launches SageMaker jobs that read from and write to S3 and end on their own; Bedrock is called as an API per token](fig/aws-ml-ecosystem.svg){alt="Diagram of the lesson's AWS components: S3 buckets at the top, a notebook instance on the left launching SageMaker training, tuning and processing jobs in the center, and Bedrock on the right called as an API"}

### Who this workshop is for

Researchers and students who already use machine learning or AI and want to run that work in the cloud. You should be comfortable with Python and Jupyter and know the basics of model training, hyperparameter tuning, fine-tuning, embeddings and RAG. The emphasis is on using AWS, not on teaching those methods. Instructors may review some of them briefly depending on the audience, but the workshop assumes the background. No previous AWS experience is needed.

### Cost controls

One of the workshop's main points is that cloud computing is affordable for research when the resources are chosen and managed with care. Run as written, the exercises cost each learner a few dollars, CPU and GPU training jobs, hyperparameter tuning, a LoRA fine-tune and RAG inference included.

We estimate the cost before launching a job, set a limit on how long it can run, and check its billable time afterwards. Resources are tagged so that spending can be attributed to each learner and project. We also look at how fast the cost grows when a GPU instance is left running or an experiment is scaled up without checking what it needs.

The workshop uses a shared AWS account with restrictions that block the expensive mistakes. The final part looks at those safeguards and how to set the same ones on an account of your own.

### What this workshop does not cover

- **General ML/AI instruction.** The focus is on running existing ML and AI workflows on AWS, not on teaching the algorithms.
- **Model deployment.** We do not deploy trained or fine-tuned models as always-on inference endpoints. A model trained elsewhere, on a campus cluster say, can still be used here. Upload its weights to S3 and run inference over a dataset as a Processing job, which is what the RAG with Processing Jobs episode does with an open-weight model. Serving it to an application behind an endpoint is the part the lesson leaves out, and the RAG overview says when that is the right tool.
- **AI agents and tool use.** We do not cover agent frameworks or applications that let a language model call external tools.

If there is another ML or AI workflow you would like to see covered, we welcome suggestions. [Open an issue on the lesson GitHub](https://github.com/carpentries-incubator/ML_with_AWS_SageMaker/issues) or email [endemann@wisc.edu](mailto:endemann@wisc.edu).

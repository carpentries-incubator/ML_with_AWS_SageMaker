---
title: "Overview: ML on AWS"
teaching: 15
exercises: 0
---

:::::::::::::::::::::::::::::::::::::::::::::: questions

- When is cloud compute the right tool for a researcher who already has a cluster or a workstation?
- What do SageMaker and Bedrock add on top of renting a machine?

::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: objectives

- Compare cloud compute with campus HPC and lab hardware on hardware generation, cost, efficiency and control.
- Describe what S3, a notebook instance, SageMaker jobs and Bedrock each do in this lesson.
- Say when a SageMaker job is the tool and when a Bedrock call is.

::::::::::::::::::::::::::::::::::::::::::::::

Most researchers already have somewhere to run ML, a campus HPC cluster, a departmental GPU server, or their own workstation. The cloud is another option, with its own trade-offs in cost, performance, flexibility and control. This episode compares the options, then introduces the four AWS pieces every later episode uses, S3, a notebook instance, SageMaker jobs and Bedrock.

### Cloud vs. the cluster down the hall

Cloud computing means renting compute as needed instead of buying and maintaining it. Whether that is the better choice depends on the workload and on what you already have.

- **Hardware and cost.** A campus cluster or lab server is used for five or more years, so its GPUs are whatever was current when it was bought. Cloud providers offer each new generation soon after release, Blackwell B200 and GB200 today, with Vera Rubin announced for cloud deployment in 2026. The gains between generations are large. NVIDIA's B200 has more than twice the memory and about seven times the bf16 arithmetic of the A100 on about two and a half times the power, so far more work per watt on paper, with the real speedup depending on the workload (NVIDIA data sheets). The financial trade-off comes down to utilization. Owned hardware costs the same busy or idle, so it is the cheaper option when it stays busy. Rented hardware is billed per second and only while a job runs, so it is the cheaper option for occasional or bursty work, and renting GPUs around the clock soon costs more than owning them.

- **Control and data.** Local hardware gives you control over the software stack, the network and where the data sits, with no quotas and no transfer charges. In the cloud you work within account permissions and service quotas, data moved in or out crosses the network and may be charged, and sensitive data needs a governance check before it leaves campus. Cost needs attention too. An idle rented GPU costs the same as a busy one, so every job in this lesson has a runtime cap, and every resource is stopped when the work is done.

- **Energy efficiency.** The comparison depends on the hardware and on how well it is shared. Two things matter most.

    - **Data center overhead.** Not all of the electricity entering a data center reaches its computers. Some goes to cooling and power distribution. Large facilities run by [AWS](https://aws.amazon.com/sustainability/) and by [national HPC centers](https://arxiv.org/html/2506.19019v1) report about 15% on top of what the computing hardware draws, against 60 to 80% for typical enterprise and departmental server rooms ([Uptime Institute, 2025](https://datacenter.uptimeinstitute.com/rs/711-RIA-145/images/2025.Annual.Survey.Report.pdf?version=0)). Cooling matters more as GPUs draw more power. The newest systems, such as NVIDIA's 72-GPU GB200 racks, need liquid cooling, which removes heat with less energy than air but needs plumbing most campus data centers do not have, and the large cloud and HPC facilities are the ones that can afford to install it. These figures describe the building, not the GPUs in it.

    - **Hardware utilization.** A GPU does no useful work while idle. A lab's own GPUs sit unused between experiments. A campus cluster does better, because it pools demand across every group at the institution, and the national pools, such as [CHTC](https://chtc.cs.wisc.edu/)'s [OSPool](https://osg-htc.org/services/open_science_pool.html) and the [National Research Platform](https://nationalresearchplatform.org/), do better still, because they pool it across institutions, often at no direct cost to the researcher. Cloud providers pool demand across a larger customer base and across time zones, so a GPU one group releases is running someone else's job minutes later. Providers do not publish enough utilization data for a direct comparison. The research pools get much of the same sharing benefit. What they do not have is the providers' hardware budget, so they adopt each new GPU generation more slowly.

There is no option that is better in every case. Campus and national research computing is often the most economical place to start, above all where it is free to the researcher. The cloud becomes the right tool when a project needs newer hardware, extra capacity for a while, or managed services that would be hard to run locally.

### S3, the notebook instance, SageMaker and Bedrock

AWS offers many services. This workshop uses four pieces that work together.

![S3 holds data and outputs; a small notebook instance launches SageMaker jobs that read from and write to S3 and end on their own; Bedrock is called as an API per token](fig/aws-ml-ecosystem.svg){alt="Diagram of the lesson's AWS components: S3 buckets at the top, a notebook instance on the left launching SageMaker training, tuning and processing jobs in the center, and Bedrock on the right called as an API"}

The workflow is the diagram. Datasets and results live in S3. We work from a small JupyterLab notebook instance. SageMaker launches the compute-heavy jobs on separate machines. For tasks a model AWS already hosts can do, we call Bedrock instead.

::::::::::::::::::::::::::::::::::::: tab

### S3

**Amazon S3 stores the data and the results.** [S3](https://aws.amazon.com/s3/) is object storage, organised in *buckets*. Datasets stay in a bucket instead of on the notebook instance, every SageMaker job reads its inputs from a bucket and writes its outputs back to one, and storage costs about two cents per GB per month. You create a bucket of your own in the next episode and use it for the rest of the workshop. The larger datasets are already in a shared bucket the instructors staged.

### Notebook instance

**The SageMaker notebook instance is where you work.** It is a small machine running JupyterLab, started at the beginning of the workshop and stopped at the end, and every notebook in the lesson runs on it. It is where code is written, scripts are tested on small data, jobs are launched and results are read back. It needs little power, because the expensive work runs elsewhere, so it is an `ml.t3.medium` at about five cents an hour. The cheap notebook stays up while you work, and the larger instances are paid for only while a job needs them. The *Notebooks as Controllers* episode creates it, and the two after that connect it to S3 and to a copy of the lesson's code.

### SageMaker

**Amazon SageMaker runs the ML jobs.** [SageMaker](https://aws.amazon.com/sagemaker/) is AWS's managed machine learning service. Instead of setting up a server for each experiment, you give it a training or inference script, the input data and an instance type. SageMaker starts that instance, runs the script on it with the ML frameworks already installed, saves the results and shuts the instance down when the job finishes. What makes that useful for research:

- **Flexible compute.** Each job picks its own hardware. A small CPU instance such as `ml.m5.large` costs about `$0.115` an hour and suits traditional ML, preprocessing and light inference. A GPU instance such as `ml.g5.xlarge` costs about `$1.41` an hour and suits deep learning and fine-tuning. The supplemental [Instances for ML](https://carpentries-incubator.github.io/ML_with_AWS_SageMaker/instances-for-ML.html) page has the full list. Prices here and throughout the lesson are us-east-1 on-demand list prices from the AWS price list of 2026-10-16. Other regions are the same or within a few percent.

- **Your own code.** SageMaker provides preconfigured containers for scikit-learn, PyTorch, TensorFlow and Hugging Face. Usually a script and a `requirements.txt` for extra packages are all you supply. A custom Docker container is there for a project that needs its own environment.

- **Parallel training and hyperparameter tuning.** A large training job can run across several instances. A tuning job launches many training jobs with different hyperparameters, collects their results and picks the best configuration. Both come later in the workshop.

- **Simpler job management.** On an HPC or HTC system the researcher writes the batch scripts and coordinates the preprocessing, training and evaluation jobs. SageMaker handles the machine's setup and the job's lifecycle. You launch a job from Python, watch its progress and collect the results without configuring the machine that runs it.

- **Cost tracking.** Every job reports its billable runtime, and resource tags let the bill be split by project. Most research experiments are cheap at this scale. Training a hundred small or medium models, logistic regressions, random forests or small neural networks, on a dataset under 10 GB costs under `$20`.

### Bedrock

**Amazon Bedrock serves hosted models.** Sometimes there is no model to train or host. You want to send a model a prompt and get a response. [Bedrock](https://aws.amazon.com/bedrock/) gives API access to models AWS already hosts, with no GPU instance to provision. What that means in practice:

- **Nothing to manage.** You call the model through an API and pay for what you use, by the number of input and output tokens. There is no instance to size, start or shut down.

- **A catalog of models.** Bedrock hosts models from Anthropic, DeepSeek, Amazon and others, for text generation and for embeddings. You pick one by its ID, with nothing to download or deploy.

- **Cheap for small experiments.** Embedding a few thousand document chunks and answering a few dozen questions costs cents. The RAG episodes build the pipeline on Bedrock first.

- **Only the catalog.** Bedrock serves only the models in its catalog. For a model it does not carry, or for control of how a model is served, SageMaker is the alternative. The RAG episodes make that move, from Bedrock to a model you run yourself.

- **Cost attribution.** Bedrock calls carry no resource tags by default. The lesson routes them through tagged application inference profiles so they show up on the bill under the right project.

:::::::::::::::::::::::::::::::::::::

The difference between SageMaker and Bedrock is control against convenience. SageMaker runs your own code on compute you choose. Bedrock runs models AWS hosts, with no infrastructure of yours underneath.

### How the episodes use them

Through the workshop, code is written in JupyterLab, datasets and results go to S3, training and other heavy jobs go to SageMaker, and Bedrock is called when a hosted model is enough.

None of this needs the notebook instance. Everything here can be done from Python on your own computer or from the AWS command line, and the optional *The Same Jobs from the Terminal* episode shows how.

::::::::::::::::::::::::::::::::::::: keypoints

- The cloud offers the current GPU generation on demand, billed per second while a job runs. Campus and national research computing is often cheaper for sustained work, and free to many researchers.
- Cost and energy efficiency depend on the hardware generation, how busy it is kept, and the data center's overhead.
- S3 stores the datasets and results. A small SageMaker notebook instance is the workspace.
- SageMaker runs training, tuning, processing and inference jobs on instances that exist only for the job.
- Bedrock gives API access to hosted models, with no GPU instance to manage.

:::::::::::::::::::::::::::::::::::::::::::::::::

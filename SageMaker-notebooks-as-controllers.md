---
title: "Notebooks as Controllers"
teaching: 20
exercises: 10
---

:::::::::::::::::::::::::::::::::::::: questions 

- How do you set up and use SageMaker notebooks for machine learning tasks?
- How can you manage compute resources efficiently using SageMaker's controller notebook approach?

::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: objectives

- Describe how to use SageMaker notebooks for ML workflows.
- Set up a Jupyter notebook instance as a controller to manage compute tasks.
- Explain the controller pattern: a cheap notebook launches jobs that run on separate instances and end on their own.

::::::::::::::::::::::::::::::::::::::::::::::::

## What a SageMaker notebook instance is
A **SageMaker notebook instance** is a virtual machine that AWS runs for you with JupyterLab already installed, a set of Python kernels that come with the AWS SDKs (`boto3` and the SageMaker Python SDK), and an IAM role attached that decides what the code running on it may do in your account. You create it from the console, open JupyterLab in a browser tab, and from then on it behaves like the Jupyter you already know. It is billed per second while it is running and nothing while it is stopped, and the files you keep under `/home/ec2-user/SageMaker/` survive a stop.

### Why we work from a notebook instance in this lesson
Nothing in the lesson *requires* it: the SageMaker SDK runs anywhere Python does. We use a notebook instance anyway, for reasons that matter in a classroom and in a research group alike:

- **Identity comes from the instance, not from a file on your machine.** The IAM role attached to the notebook is what authorizes every call it makes. Nobody installs the AWS CLI, configures credentials, or has an access key that can end up in a notebook, a shell history or a GitHub commit. For a shared account with a hundred people in it, that is the single biggest security simplification available.
- **Everyone has the same environment.** The kernels, the SDK version and the paths are identical for every learner, so a cell that works for the instructor works for you. Package and path problems are the main thing that eats workshop time; this removes most of them.
- **It is inside AWS.** Reading your S3 bucket from the instance is in-region, fast and free of transfer charges; the same reads from a laptop would be slower and billed as data leaving AWS.
- **It is a resource like any other.** It has to be sized, tagged, started and, above all, stopped. Creating it is your first practice at the discipline that the whole lesson is about.

### The controller pattern
The instance you create here will be small (`ml.t3.medium`: 2 vCPUs, 4 GB of memory, $0.05 per hour) and that is deliberate. The notebook never does the heavy computation. Its job is to be a **controller**: it holds your code and your notes, and it uses the SageMaker SDK to launch separate, more powerful instances for training, hyperparameter tuning and batch processing. Those instances start, run your script, write their results to S3, and shut themselves down. You pay for a GPU for the minutes a job needs it, not for the hours you spend thinking in between, and a mistake in a notebook cell cannot leave an expensive machine running: the expensive machine is a job, and jobs end.

Every later episode uses the same instance. In this one we create it, open it, and learn to stop it.

We'll follow these steps to create our first "SageMaker notebook instance".

#### 1. Navigate to SageMaker
- In the AWS Console, search for **SageMaker**.
- Recommended: Select the star icon next to **Amazon SageMaker AI** to save SageMaker as a bookmark in your AWS toolbar 
- Select  **Amazon SageMaker AI**

#### 2. Create a new notebook instance
- In the SageMaker left-side menu, click on **Notebook instances**, then click **Create notebook instance**.
- **Notebook name**: Same convention as your bucket, with a purpose on the end: `challenge-lastnamefirstname-purpose`, all lowercase, e.g. `sinkorswim-doejohn-trainclassifier`. SageMaker allows letters, digits and hyphens (no spaces or underscores, up to 63 characters) and would accept capitals, but we keep everything lowercase so the bucket, the notebook and the tags all carry the same string and sort together in the console and on the bill. This name is also reused as the prefix of every training job you launch from the notebook, which is how you will find your jobs in a shared account.
- **Instance type**: SageMaker notebooks run on AWS EC2 instances. The instance type determines the compute resources allocated to the notebook. Since our notebook will act as a **low-resource "controller"**, we'll select a small instance such as `ml.t3.medium` (2 vCPU, 4 GB RAM, $0.05/hour)  
  - This keeps costs low while allowing us to launch separate training/tuning jobs on more powerful instances when needed.  
  - For guidance on common instances for ML procedures, refer to our supplemental [Instances for ML webpage](https://carpentries-incubator.github.io/ML_with_AWS_SageMaker/instances-for-ML.html).  
- **Platform identifier**: This is an internal AWS setting related to the environment version and underlying platform. You can leave this as the default.
- **Permissions and encryption**:
   - **IAM role**: For this workshop, we have pre-configured the "ml-sagemaker-use" role to enable access to AWS services like SageMaker, with some restrictions to prevent overuse/misuse of resources. Select the "ml-sagemaker-use" role. Outside of the workshop, you create/select a role that includes the `AmazonSageMakerFullAccess` policy.
   - **Root access**: Determines whether the user can run administrative commands within the notebook instance.  You should **Enable root access** to allow installing additional packages if/when needed.  
   - **Encryption key (skip)**: While we won't use this feature for the workshop, it is possible to specify a KMS key for encrypting data at rest if needed. 
- **Network (skip)**: Networking settings are optional. Configure them if you're working within a specific VPC or need network customization.
- **Git repositories configuration (skip)**: You don't need to complete this configuration. Instead, we'll run a clone command from our notebook later to get our repo setup. This approach is a common strategy (allowing some flexiblity in which repo you use for the notebook).
- **Tags (NOT OPTIONAL)**: Adding tags helps track and organize resources for billing and management. This is particularly useful when you need to break down expenses by project, task, or team. To help track costs on our shared account, add the same three keys you put on your bucket in the previous episode (type the keys exactly as shown; capitalization matters):

  | Key | Value |
  |---|---|
  | `Project` | your challenge name, e.g. `sinkorswim` |
  | `Name` | your name, e.g. `John Doe` |
  | `Purpose` | what this resource is for, e.g. `Workshop-controller` |

![Tag Setup Example](fig/notebook_tags.PNG){alt="Screenshot showing required tags for a SageMaker notebook instance"}

  (The screenshot shows a fourth key, `ModelSize`. Leave it out: it is not an approved key, so it would never appear on the bill.)

::::::::::::::::::::::::::::::::::::: callout

#### Why these three keys, and why they must be spelled exactly

Tags only show up in cost reports if the tag *key* has been activated as a cost allocation tag at the account level. On UW–Madison's AWS accounts that activation is done centrally by the Public Cloud team for a fixed list of approved keys, documented at [AWS - Tagging with an AWS Account (KB 72454)](https://kb.wisc.edu/72454). `Name`, `Project`, and `Purpose` are on that list; `ModelSize`, `Team`, `project` (lowercase) or any other spelling are not, and spend tagged with them is invisible in Cost Explorer. The KB page says it plainly: cut and paste, because any deviation stops the tag from reporting.

Two more things to know now, because they come up in every later episode:

- **Tags never propagate.** Tagging this notebook instance tags the notebook instance only. Every S3 bucket, training job, processing job and Bedrock inference profile you create later needs its own copy of the same three tags.
- **Outside UW**, you activate the keys yourself, once, under **Billing and Cost Management → Cost allocation tags**. Until you do, the tags exist on the resources but not on the bill.

::::::::::::::::::::::::::::::::::::::::::::::::

- Click **Create notebook instance**. It may take a few minutes for the instance to start. Once its status is **InService**, you can open the notebook instance and start coding.

### Load pre-filled Jupyter notebooks
Once your newly created *instance* shows as `InService`, open the instance in Jupyter Lab. From there, we can create as many Jupyter notebooks as we would like within the instance environment. 

We will then select the standard python3 environment (conda_python3) to start our first .ipynb notebook (Jupyter notebook). We can use the standard conda_python3 environment since we aren't doing any training/tuning just yet.

##### Load pre-filled Jupyter notebooks
Within the Jupyter notebook, run the following command to clone the lesson repo into our Jupyter environment:

```sh
!git clone https://github.com/carpentries-incubator/ML_with_AWS_SageMaker.git
```

Then, navigate to `/ML_with_AWS_SageMaker/notebooks/Accessing-S3-via-SageMaker-notebooks.ipynb` to begin the first notebook.

::::::::::::::::::::::::::::::::::::: challenge

### Challenge: pick an instance, and decide what to tag

Open the supplemental [Instances for ML](https://carpentries-incubator.github.io/ML_with_AWS_SageMaker/instances-for-ML.html) page and answer:

1. A colleague has a **20 GB tabular dataset** (a CSV of sensor readings) and wants to train gradient-boosted trees on it. Which instance would you pick for the *training job*, and why? What would you pick for the *controller notebook* that launches it?
2. List every resource this colleague will create during that project, and say which of them need the three tags.

:::::::::::::::::::::::::::::::::::: solution

1. The table's rule of thumb is RAM of at least 1.5× the dataset, so 20 GB of data wants about 32 GB of memory: `ml.m5.2xlarge` (8 vCPU, 32 GiB, about $0.46/h) is the natural fit for a tree-ensemble training job, with `ml.m5.4xlarge` (64 GiB) as the safe choice if the feature engineering doubles the frame. No GPU: gradient-boosted trees on tabular data are CPU work, so a `g4dn` or `g5` would cost more and run no faster. The controller stays `ml.t3.medium` ($0.05/h): it only submits the job and reads the results back from S3, so it never needs to hold the dataset in memory.
2. At least four: the S3 bucket, the notebook instance, each training job, and (if they tune) each hyperparameter tuning job. Every one of them needs its own copy of `Name`, `Project` and `Purpose`, because tags never propagate from one resource to another. The training-job instance itself is not a separate resource to tag: it is created and destroyed by the job, and the job's tags cover it.

::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::::::::::::::

### Before you leave: stop the instance

A notebook instance bills for every second it is **InService**, whether or not you are typing in it. When you step away for the day, go to **Notebook instances**, select yours and choose **Stop**. Your files under `/home/ec2-user/SageMaker` (including the repos you clone) survive a Stop and are there when you start it again; packages you `pip install` are not, so expect to reinstall them. When the workshop is over, **Delete** the instance: a stopped instance still bills for its storage volume ($0.14 per GB-month), and the cleanup episode walks through this.

::::::::::::::::::::::::::::::::::::: callout

### Everything from here is one pattern

The rest of this lesson is one pattern applied again and again. A cheap controller notebook (the `ml.t3.medium` you just created) launches work that runs *elsewhere*, on an instance that exists only for the duration of the job and shuts itself down when the job ends: a training job on the Titanic data, a hyperparameter tuning job that runs many of them in parallel, a Processing job that serves a language model over a corpus of papers, and a fine-tuning job on a rented GPU. The controller never has to be big, and nothing is left running by accident. Four habits travel with every launch, and you will see them in every episode: the three **tags** on the job so its cost is traceable; a **runtime cap** (`max_run`) so a bug cannot run all night; a cost **estimate written down before** the launch and **compared to the bill after**, so your sense of what things cost gets calibrated; and **cleanup**, the stop-or-delete step that ends what the job did not end on its own.

::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: keypoints 

- Use a minimal SageMaker notebook instance as a controller to manage larger, resource-intensive tasks.
- The controller pattern: the notebook launches work (training, tuning, processing, fine-tuning) that runs on separate instances and ends on its own; the notebook itself never needs to be big.
- Tags can help track costs effectively, especially in multi-project or team settings; the same three keys (`Name`, `Project`, `Purpose`) go on every resource, and tags never propagate.
- Stop the notebook instance whenever you leave, and delete it when the workshop is over.
- Use the SageMaker SDK documentation to explore additional options for managing compute resources in AWS.

::::::::::::::::::::::::::::::::::::::::::::::::

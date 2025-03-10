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
- Use SageMaker SDK to launch training and tuning jobs on scalable instances.

::::::::::::::::::::::::::::::::::::::::::::::::

## Step 2: Running Python code with SageMaker notebooks
Amazon SageMaker provides a managed environment to simplify the process of building, training, and deploying machine learning models. By using SageMaker, you can focus on model development without needing to manually provision resources or set up environments. In this episode, we’ll guide you through setting up a **SageMaker notebook instance**—a Jupyter notebook hosted on AWS specifically for running SageMaker jobs. This setup allows you to efficiently manage and monitor machine learning workflows directly from a lightweight notebook controller. We’ll also cover loading data in preparation for model training and tuning in future episodes, using the Titanic dataset stored in S3.

### Using the notebook as a controller
In this setup, the notebook instance functions as a *controller* to manage more resource-intensive compute tasks. By selecting a minimal instance (e.g., `ml.t3.medium`) for the notebook, you can perform lightweight operations and leverage the **SageMaker Python SDK** to launch more powerful, scalable compute instances when needed for model training, batch processing, or hyperparameter tuning. This approach minimizes costs by keeping your controller instance lightweight while accessing the full power of SageMaker for demanding tasks.

### Setting up the notebook
We'll follow these steps to create our first "SageMaker notebook instance".

#### 1. Navigate to SageMaker
- In the AWS Console, search for **SageMaker**.
- Recommended: Select the star icon next to **Amazon SageMaker AI** to save SageMaker as a bookmark in your AWS toolbar 
- Select  **Amazon SageMaker AI**

#### 2. Create a new notebook instance
- In the SageMaker left-side menu, click on **Notebooks**, then click **Create notebook instance**.
- **Notebook name**: To easily track this resource in our shared account, please use the following naming convention: "YourName-ExploreSageMaker". For example, "DoeJohn-ExploreSageMaker". Can include hyphens, but not spaces. 
- **Instance type**: SageMaker notebooks run on AWS EC2 instances. The instance type determines the compute resources allocated to the notebook. Since our notebook will act as a **low-resource "controller"**, we'll select a small instance such as `ml.t3.medium`.  
  - This keeps costs low while allowing us to launch separate training/tuning jobs on more powerful instances when needed.  
  - For guidance on common instances for ML procedures, refer to our supplemental [Instances for ML webpage](https://carpentries-incubator.github.io/ML_with_AWS_SageMaker/instances-for-ML.html).  
- **Platform identifier**: This is an internal AWS setting related to the environment version and underlying platform. You can leave this as the default.
- **Permissions and encryption**:
   - **IAM role**: For this workshop, we have pre-configured the "ml-sagemmaker-use" role to enable access to AWS services like SageMaker, with some restrictions to prevent overuse/misuse of resources. Select the "ml-sagemmaker-use" role. Outside of the workshop, you create/select a role that includes the `AmazonSageMakerFullAccess` policy.
   - **Root access**: Determines whether the user can run administrative commands within the notebook instance.  You should **Enable root access** to allow installing additional packages if/when needed.  
   - **Encryption key (skip)**: While we won't use this feature for the workshop, it is possible to specify a KMS key for encrypting data at rest if needed. 
- **Network (skip)**: Networking settings are optional. Configure them if you're working within a specific VPC or need network customization.
- **Git repositories configuration (skip)**: You don't need to complete this configuration. Instead, we'll run a clone command from our notebook later to get our repo setup. This approach is a common strategy (allowing some flexiblity in which repo you use for the notebook).
- **Tags (NOT OPTIONAL)**: Adding tags helps track and organize resources for billing and management. This is particularly useful when you need to break down expenses by project, task, or team. To help track costs on our shared account, please use the tags found in the below image.

![Tag Setup Example](https://raw.githubusercontent.com/UW-Madison-DataScience/ml-with-aws-sagemaker/main/images/notebook_tags.PNG)

- Click **Create notebook instance**. It may take a few minutes for the instance to start. Once its status is **InService**, you can open the notebook instance and start coding.

### Managing training and tuning with the controller notebook

In the next couple expisodes, we'll use the **SageMaker Python SDK** within the notebook to launch compute-heavy tasks on more powerful instances as needed. Examples of tasks to launch include:

- **Training a model**: Use the SDK to submit a training job, specifying a higher-powered instance (e.g., `ml.p2.xlarge` or `ml.m5.4xlarge`) based on your model’s resource requirements.
- **Hyperparameter tuning**: Configure and launch tuning jobs, allowing SageMaker to automatically manage multiple powerful instances for optimal tuning.

This setup allows you to control costs by keeping the notebook instance minimal and only incurring costs for larger instances when they are actively training or tuning models. Detailed guidance on training, tuning, and batch processing will follow in later episodes.

For more details, refer to the [SageMaker Python SDK documentation](https://sagemaker.readthedocs.io/) for example code on launching and managing remote training jobs.

::::::::::::::::::::::::::::::::::::: keypoints 

- Use a minimal SageMaker notebook instance as a controller to manage larger, resource-intensive tasks.
- Launch training and tuning jobs on scalable instances using the SageMaker SDK.
- Tags can help track costs effectively, especially in multi-project or team settings.
- Use the SageMaker SDK documentation to explore additional options for managing compute resources in AWS.

::::::::::::::::::::::::::::::::::::::::::::::::

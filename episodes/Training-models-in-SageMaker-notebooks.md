---
title: "Training Models in SageMaker: Intro"
teaching: 20
exercises: 10
---

:::::::::::::::::::::::::::::::::::::: questions 

- What are the differences between local training and SageMaker-managed training?
- How do Estimator classes in SageMaker streamline the training process for various frameworks?
- How does SageMaker handle data and model parallelism, and when should each be considered?

::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: objectives

- Understand the difference between training locally in a SageMaker notebook and using SageMaker's managed infrastructure.
- Learn to configure and use SageMaker's Estimator classes for different frameworks (e.g., XGBoost, PyTorch, SKLearn).
- Understand data and model parallelism options in SageMaker, including when to use each for efficient training.
- Compare performance, cost, and setup between custom scripts and built-in images in SageMaker.
- Conduct training with data stored in S3 and monitor training job status using the SageMaker console.

::::::::::::::::::::::::::::::::::::::::::::::::

## Initial setup 

#### 1. Open prefilled .ipynb notebook
Open the notebook from: `/ML_with_AWS_SageMaker/notebooks/Training-models-in-SageMaker-notebooks.ipynb`

#### 2. CD to instance home directory
So we all can reference the helper functions using the same path, CD to...

```python
%cd /home/ec2-user/SageMaker/
```

#### 3. Initialize SageMaker environment
This code initializes the AWS SageMaker environment by defining the SageMaker role and S3 client. It also specifies the S3 bucket and key for accessing the Titanic training dataset stored in an S3 bucket.

#### Boto3 API
> Boto3 is the official AWS SDK for Python, allowing developers to interact programmatically with AWS services like S3, EC2, and Lambda. It provides both high-level and low-level APIs, making it easy to manage AWS resources and automate tasks. With built-in support for paginators, waiters, and session management, Boto3 simplifies working with AWS credentials, regions, and IAM permissions. It's ideal for automating cloud operations and integrating AWS services into Python applications.

```python
import boto3
import pandas as pd
import sagemaker
from sagemaker import get_execution_role

# Initialize the SageMaker role (will reflect notebook instance's policy)
role = sagemaker.get_execution_role()
print(f'role = {role}')

# Initialize an S3 client to interact with Amazon S3, allowing operations like uploading, downloading, and managing objects and buckets.
s3 = boto3.client('s3')

# Define the S3 bucket that we will load from
bucket_name = 'sinkorswim-doejohn-titanic'  # replace with your S3 bucket name

# Define train/test filenames
train_filename = 'titanic_train.csv'
test_filename = 'titanic_test.csv'
```

Create a SageMaker session to manage interactions with Amazon SageMaker, such as training jobs, model deployments, and data input/output.
```python
region = "us-east-2" # United States (Ohio). Make sure this matches what you see near top right of AWS Console menu
boto_session = boto3.Session(region_name=region) # Create a Boto3 session that ensures all AWS service calls (including SageMaker) use the specified region
session = sagemaker.Session(boto_session=boto_session)
```

#### 4. Get code from git repo (skip if completed already from earlier episodes)
If you didn't complete the earlier episodes, you'll need to clone our code repo before moving forward. Check to make sure we're in our EC2 root folder first (`/home/ec2-user/SageMaker`).

```python
%cd /home/ec2-user/SageMaker/
```

```python
# uncomment below line only if you still need to download the code repo (replace username with your GitHub usernanme)
#!git clone https://github.com/username/AWS_helpers.git 
```

## Testing train.py on this notebook's instance
In this next section, we will learn how to take a model training script that was written/designed to run locally, and deploy it to more powerful instances (or many instances) using SageMaker. This is helpful for machine learning jobs that require extra power, GPUs, or benefit from parallelization. However, before we try exploiting this extra power, it is essential that we test our code thoroughly! We don't want to waste unnecessary compute cycles and resources on jobs that produce bugs rather than insights. 

### General guidelines for testing ML pipelines before scaling
- **Run tests locally first** (if feasible) to avoid unnecessary AWS charges. Here, we assume that local tests are not feasible due to limited local resources. Instead, we use our SageMaker instance to test our script on a minimally sized EC2 instance.
- **Use a small dataset subset** (e.g., 1-5% of data) to catch issues early and speed up tests.
- **Start with a small/cheap instance** before committing to larger resources. Visit the [Instances for ML page](https://carpentries-incubator.github.io/ML_with_AWS_SageMaker/instances-for-ML.html) for guidance. 
- **Log everything** to track training times, errors, and key metrics.
- **Verify correctness first** before optimizing hyperparameters or scaling.

::::::::::::::::::::::::::::::::::::::: discussion

### What tests should we do before scaling?  

Before scaling to mutliple or more powerful instances (e.g., training on larger/multiple datsets in parallel or tuning hyperparameters in parallel), it's important to run a few quick sanity checks to catch potential issues early. **In your group, discuss:**  

- Which checks do you think are most critical before scaling up?  
- What potential issues might we miss if we skip this step?  

:::::::::::::::::::::::::::::::::::::::::::::::::::::


::::::::::::::::::::::::::::::::::::::: solution

### Solution

Which checks do you think are most critical before scaling up?  

- **Data loads correctly** – Ensure the dataset loads without errors, expected columns exist, and missing values are handled properly.  
- **Overfitting check** – Train on a small dataset (e.g., 100 rows). If it doesn't overfit, there may be a data or model setup issue.  
- **Loss behavior check** – Verify that training loss decreases over time and doesn't diverge.  
- **Training time estimate** – Run on a small subset to estimate how long full training will take.
- **Memory estimate** - Estimate the memory needs of the algorithm/model you're using, and understand how this scales with input size.
- **Save & reload test** – Ensure the trained model can be saved, reloaded, and used for inference without errors.

What potential issues might we miss if we skip the above checks?

- **Silent data issues** – Missing values, unexpected distributions, or incorrect labels could degrade model performance.  
- **Code bugs at scale** – Small logic errors might not break on small tests but could fail with larger datasets.  
- **Inefficient training runs** – Without estimating runtime, jobs may take far longer than expected, wasting AWS resources.  
- **Memory or compute failures** – Large datasets might exceed instance memory limits, causing crashes or slowdowns.  
- **Model performance issues** – If a model doesn't overfit a small dataset, there may be problems with features, training logic, or hyperparameters.  


:::::::::::::::::::::::::::::::::::::::::::::::::::::


::::::::::::::::::::::::::::::::::::::: callout  

### **Know Your Data Before Modeling**  
The sanity checks above focus on validating the code, but a model is only as good as the data it's trained on. A deeper look at feature distributions, correlations, and potential biases is critical before scaling up. We won't cover that here, but it's essential to keep in mind for any ML/AI practitioner.

:::::::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::::: challenge

### Understanding the XGBoost Training Script

Take a moment to review the `AWS_helpers/train_xgboost.py` script we just cloned into our notebook. This script handles preprocessing, training, and saving an XGBoost model, while also adapting to both local and SageMaker-managed environments.

Try answering the following questions:

1. **Data Preprocessing**: What transformations are applied to the dataset before training?

2. **Training Function**: What does the `train_model()` function do? Why do we print the training time?

3. **Command-Line Arguments**: What is the purpose of `argparse` in this script? How would you modify the script if you wanted to change the number of training rounds?

4. **Handling Local vs. SageMaker Runs**: How does the script determine whether it is running in a SageMaker training job or locally (within this notebook's instance)?

5. **Training and Saving the Model**: What format is the dataset converted to before training, and why? How is the trained model saved, and where will it be stored?

After reviewing, discuss any questions or observations with your group.

:::::::::::::::::::::::::::::::::::::::::::::::::::::


::::::::::::::::::::::::::::::::::::::: solution

### Solution

1. **Data Preprocessing**: The script fills missing values (`Age` with median, `Embarked` with mode), converts categorical variables (`Sex` and `Embarked`) to numerical values, and removes columns that don't contribute to prediction (`Name`, `Ticket`, `Cabin`).

2. **Training Function**: The `train_model()` function takes the training dataset (`dtrain`), applies XGBoost training with the specified hyperparameters, and prints the training time. Printing training time helps compare different runs and ensures that scaling decisions are based on performance metrics.

3. **Command-Line Arguments**: `argparse` allows passing parameters like `max_depth`, `eta`, `num_round`, etc., at runtime without modifying the script. To change the number of training rounds, you would update the `--num_round` argument when running the script:  `python train_xgboost.py --num_round 200`

4. **Handling Local vs. SageMaker Runs**: The script uses `os.environ.get("SM_CHANNEL_TRAIN", ".")` and `os.environ.get("SM_MODEL_DIR", ".")` to detect whether it’s running in SageMaker. `SM_CHANNEL_TRAIN` is the directory where SageMaker stores input training data, and `SM_MODEL_DIR` is the directory where trained models should be saved. If these environment variables are *not set* (e.g., running locally), the script defaults to `"."` (current directory).

5. **Training and Saving the Model**: The dataset is converted into **XGBoost's `DMatrix` format**, which is optimized for memory and computation efficiency. The trained model is saved using `joblib.dump()` to `xgboost-model`, stored either in the SageMaker `SM_MODEL_DIR` (if running in SageMaker) or in the local directory.

:::::::::::::::::::::::::::::::::::::::::::::::::::::

### Download data into notebook environment
It can be convenient to have a copy of the data (i.e., one that you store in your notebook's instance) to allow us to test our code before scaling things up. 

:::: callout
While we demonstrate how to download data into the notebook environment for testing our code (previously setup for local ML pipelines), keep in mind that S3 is the preferred location for dataset storage in a scalable ML pipeline. 
:::::

Run the next code chunk to download data from S3 to notebook environment. You may need to hit refresh on the file explorer panel to the left to see this file. If you get any permission issues...

* check that you have selected the appropriate policy for this notebook
* check that your bucket has the appropriate policy permissions

```python
# Define the S3 bucket and file location
file_key = f"{train_filename}"  # Path to your file in the S3 bucket
local_file_path = f"./{train_filename}"  # Local path to save the file

# Download the file using the s3 client variable we initialized earlier
s3.download_file(bucket_name, file_key, local_file_path)
print("File downloaded:", local_file_path)
```

We can do the same for the test set.


```python
# Define the S3 bucket and file location
file_key = f"{test_filename}"  # Path to your file in the S3 bucket. W
local_file_path = f"./{test_filename}"  # Local path to save the file

# Initialize the S3 client and download the file
s3.download_file(bucket_name, file_key, local_file_path)
print("File downloaded:", local_file_path)

```

#### Logging runtime & instance info
To compare our local runtime with future experiments, we'll need to know what instance was used, as this will greatly impact runtime in many cases. We can extract the instance name for this notebook using...

```python
# Replace with your notebook instance name.
# This does NOT refer to specific ipynb files, but to the SageMaker notebook instance.
notebook_instance_name = 'sinkorswim-DoeJohn-TrainClassifier'

# Make sure this matches what you see near top right of AWS Console menu
region = "us-east-2" # United States (Ohio)

# Initialize SageMaker client
sagemaker_client = boto3.client('sagemaker', region_name=region)

# Describe the notebook instance
response = sagemaker_client.describe_notebook_instance(NotebookInstanceName=notebook_instance_name)

# Display the status and instance type
print(f"Notebook Instance '{notebook_instance_name}' status: {response['NotebookInstanceStatus']}")
local_instance = response['InstanceType']
print(f"Instance Type: {local_instance}")

```

#### Helper:  `get_notebook_instance_info()` 
You can also use the `get_notebook_instance_info()` function found in `AWS_helpers.py` to retrieve this info for your own project.


```python
import AWS_helpers.helpers as helpers
helpers.get_notebook_instance_info(notebook_instance_name, region)
```


Test train.py on this notebook's instance (or when possible, on your own machine) before doing anything more complicated (e.g., hyperparameter tuning on multiple instances)


```python
!pip install xgboost # need to add this to environment to run train.py
```

### Local test
```python
import time as t # we'll use the time package to measure runtime

start_time = t.time()

# Define your parameters. These python vars wil be passed as input args to our train_xgboost.py script using %run

max_depth = 3 # Sets the maximum depth of each tree in the model to 3. Limiting tree depth helps control model complexity and can reduce overfitting, especially on small datasets.
eta = 0.1 #  Sets the learning rate to 0.1, which scales the contribution of each tree to the final model. A smaller learning rate often requires more rounds to converge but can lead to better performance.
subsample = 0.8 # Specifies that 80% of the training data will be randomly sampled to build each tree. Subsampling can help with model robustness by preventing overfitting and increasing variance.
colsample_bytree = 0.8 # Specifies that 80% of the features will be randomly sampled for each tree, enhancing the model's ability to generalize by reducing feature reliance.
num_round = 100 # Sets the number of boosting rounds (trees) to 100. More rounds typically allow for a more refined model, but too many rounds can lead to overfitting.
train_file = 'titanic_train.csv' #  Points to the location of the training data

# Use f-strings to format the command with your variables
%run AWS_helpers/train_xgboost.py --max_depth {max_depth} --eta {eta} --subsample {subsample} --colsample_bytree {colsample_bytree} --num_round {num_round} --train {train_file}

# Measure and print the time taken
print(f"Total local runtime: {t.time() - start_time:.2f} seconds, instance_type = {local_instance}")

```

Training on this relatively small dataset should take less than a minute, but as we scale up with larger datasets and more complex models in SageMaker, tracking both training time and total runtime becomes essential for efficient debugging and resource management.

**Note**: Our code above includes print statements to monitor dataset size, training time, and total runtime, which provides insights into resource usage for model development. We recommend incorporating similar logging to track not only training time but also total runtime, which includes additional steps like data loading, evaluation, and saving results. Tracking both can help you pinpoint bottlenecks and optimize your workflow as projects grow in size and complexity, especially when scaling with SageMaker's distributed resources.


### Sanity check: Quick evaluation on test set
This next section isn't SageMaker specific, but it does serve as a good sanity check to ensure our model is training properly. Here's how you would apply the outputted model to your test set using your local notebook instance.

```python
import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
import joblib
from AWS_helpers.train_xgboost import preprocess_data

# Load the test data
test_data = pd.read_csv('./titanic_test.csv')

# Preprocess the test data using the imported preprocess_data function
X_test, y_test = preprocess_data(test_data)

# Convert the test features to DMatrix for XGBoost
dtest = xgb.DMatrix(X_test)

# Load the trained model from the saved file
model = joblib.load('./xgboost-model')

# Make predictions on the test set
preds = model.predict(dtest)
predictions = np.round(preds)  # Round predictions to 0 or 1 for binary classification

# Calculate and print the accuracy of the model on the test data
accuracy = accuracy_score(y_test, predictions)
print(f"Test Set Accuracy: {accuracy:.4f}")

```

A reasonably high test set accuracy suggests our code/model is working correctly.

## Training via SageMaker (using notebook as controller) - custom train.py script
Unlike "local" training (using this notebook), this next approach leverages SageMaker's managed infrastructure to handle resources, parallelism, and scalability. By specifying instance parameters, such as instance_count and instance_type, you can control the resources allocated for training.

### Which instance to start with?
In this example, we start with one ml.m5.large instance, which is suitable for small- to medium-sized datasets and simpler models. Using a single instance is often cost-effective and sufficient for initial testing, allowing for straightforward scaling up to more powerful instance types or multiple instances if training takes too long. See here for further guidance on selecting an appropriate instance for your data/model: [EC2 Instances for ML](https://docs.google.com/spreadsheets/d/1uPT4ZAYl_onIl7zIjv5oEAdwy4Hdn6eiA9wVfOBbHmY/edit?usp=sharing)

### Overview of Estimator classes in SageMaker
To launch this training "job", we'll use the XGBoost "Estimator. In SageMaker, Estimator classes streamline the configuration and training of models on managed instances. Each Estimator can work with custom scripts and be enhanced with additional dependencies by specifying a `requirements.txt` file, which is automatically installed at the start of training. Here's a breakdown of some commonly used Estimator classes in SageMaker:

#### 1. **`Estimator` (Base Class)**
   - **Purpose**: General-purpose for custom Docker containers or defining an image URI directly.
   - **Configuration**: Requires specifying an `image_uri` and custom entry points.
   - **Dependencies**: You can use `requirements.txt` to install Python packages or configure a custom Docker container with pre-baked dependencies.
   - **Ideal Use Cases**: Custom algorithms or models that need tailored environments not covered by built-in containers.

#### 2. **`XGBoost` Estimator**
   - **Purpose**: Provides an optimized container specifically for XGBoost models.
   - **Configuration**:
      - `entry_point`: Path to a custom script, useful for additional preprocessing or unique training workflows.
      - `framework_version`: Select XGBoost version, e.g., `"1.5-1"`.
      - `dependencies`: Specify additional packages through `requirements.txt` to enhance preprocessing capabilities or incorporate auxiliary libraries.
   - **Ideal Use Cases**: Tabular data modeling using gradient-boosted trees; cases requiring custom preprocessing or tuning logic.

#### 3. **`PyTorch` Estimator**
   - **Purpose**: Configures training jobs with PyTorch for deep learning tasks.
   - **Configuration**:
      - `entry_point`: Training script with model architecture and training loop.
      - `instance_type`: e.g., `ml.p3.2xlarge` for GPU acceleration.
      - `framework_version` and `py_version`: Define specific versions.
      - `dependencies`: Install any required packages via `requirements.txt` to support advanced data processing, data augmentation, or custom layer implementations.
   - **Ideal Use Cases**: Deep learning models, particularly complex networks requiring GPUs and custom layers.

#### 4. **`SKLearn` Estimator**
   - **Purpose**: Supports scikit-learn workflows for data preprocessing and classical machine learning.
   - **Configuration**:
      - `entry_point`: Python script to handle feature engineering, preprocessing, or training.
      - `framework_version`: Version of scikit-learn, e.g., `"1.0-1"`.
      - `dependencies`: Use `requirements.txt` to install any additional Python packages required by the training script.
   - **Ideal Use Cases**: Classical ML workflows, extensive preprocessing, or cases where additional libraries (e.g., pandas, numpy) are essential.

#### 5. **`TensorFlow` Estimator**
   - **Purpose**: Designed for training and deploying TensorFlow models.
   - **Configuration**:
      - `entry_point`: Script for model definition and training process.
      - `instance_type`: Select based on dataset size and computational needs.
      - `dependencies`: Additional dependencies can be listed in `requirements.txt` to install TensorFlow add-ons, custom layers, or preprocessing libraries.
   - **Ideal Use Cases**: NLP, computer vision, and transfer learning applications in TensorFlow.

#### 6. **`HuggingFace` Estimator**
   - **Purpose**: Provides managed containers for running inference, fine-tuning, and Retrieval-Augmented Generation (RAG) workflows using the Hugging Face `transformers` library.  
   - **Configuration**:
      - `entry_point`: Custom script for training or inference (e.g., `train.py` or `rag_inference.py`).  
      - `transformers_version`, `pytorch_version`, `py_version`: Define framework versions.  
      - `dependencies`: Optional `requirements.txt` for extra libraries.  
   - **Ideal Use Cases**: RAG pipelines, LLM inference, NLP, vision, or multimodal tasks using pretrained Transformer models.  

::::::::::::::::::::::::::::::::::::: callout 
#### Configuring custom environments with `requirements.txt`

For all these Estimators, adding a `requirements.txt` file as a `dependencies` argument ensures that additional packages are installed before training begins. This approach allows the use of specific libraries that may be critical for custom preprocessing, feature engineering, or model modifications. Here's how to include it:

```python
# # Customizing estimator using requirements.txt
# from sagemaker.sklearn.estimator import SKLearn
# sklearn_estimator = SKLearn(
#     base_job_name=notebook_instance_name,
#     entry_point="train_script.py",
#     role=role,
#     instance_count=1,
#     instance_type="ml.m5.large",
#     output_path=f"s3://{bucket_name}/output",
#     framework_version="1.0-1",
#     dependencies=['requirements.txt'],  # Adding custom dependencies
#     hyperparameters={
#         "max_depth": 5,
#         "eta": 0.1,
#         "subsample": 0.8,
#         "num_round": 100
#     }
# )
```

This setup simplifies training, allowing you to maintain custom environments directly within SageMaker's managed containers, without needing to build and manage your own Docker images. The [AWS SageMaker Documentation](https://docs.aws.amazon.com/sagemaker/latest/dg/pre-built-containers-frameworks-deep-learning.html) provides lists of pre-built container images for each framework and their standard libraries, including details on pre-installed packages.
:::::::::::::::::::::::::::::::::::::::::::::

### Deploying to other instances
For this deployment, we configure the "XGBoost" estimator with a custom training script, train_xgboost.py, and define hyperparameters directly within the SageMaker setup. Here's the full code, with some additional explanation following the code.

#### Cost tracking
When you launch a SageMaker training job from a notebook, SageMaker creates new managed resources (EC2 instances, attached storage, logs) on your behalf. These resources do not automatically inherit the notebook instance's tags. 

To avoid this, we explicitly tag each training job at launch time. This ensures that compute usage is traceable to a project, a purpose, and a human-readable name, even after the job has completed.
```python
name = "John Doe" # replace with your name
project = "sinkorswim" # replace with your team name
purpose = "train_XGBoost"

job_tags = [
    {"Key": "Name", "Value": name},
    {"Key": "Project", "Value": project},
    {"Key": "Purpose", "Value": purpose},
]

```


```python
from sagemaker.inputs import TrainingInput
from sagemaker.xgboost.estimator import XGBoost

# Define instance type/count we'll use for training
instance_type="ml.m5.large"
instance_count=1 # always start with 1. Rarely is parallelized training justified with data < 50 GB. More on this later.

# Define max runtime in seconds to ensure you don't use more compute time than expected. Use a generous threshold (2x expected time but < 2 days) so that work isn't interrupted without any gains.
max_run = 2*60*60 # 2 hours

# Define S3 paths for input and output
train_s3_path = f's3://{bucket_name}/{train_filename}'

# we'll store all results in a subfolder called xgboost on our bucket. This folder will automatically be created if it doesn't exist already.
output_folder = 'xgboost'
output_path = f's3://{bucket_name}/{output_folder}/' 

# Set up the SageMaker XGBoost Estimator with custom script
xgboost_estimator = XGBoost(
    base_job_name=notebook_instance_name,
    max_run=max_run, # in seconds; always include (max 48 hours)
    entry_point='train_xgboost.py',      # Custom script path
    source_dir='AWS_helpers',               # Directory where your script is located
    role=role,
    tags=job_tags,
    instance_count=instance_count,
    instance_type=instance_type,
    output_path=output_path,
    sagemaker_session=session,
    framework_version="1.5-1",           # Use latest supported version for better compatibility
    hyperparameters={
        'train': train_file,
        'max_depth': max_depth,
        'eta': eta,
        'subsample': subsample,
        'colsample_bytree': colsample_bytree,
        'num_round': num_round
    }
)

# Define input data
train_input = TrainingInput(train_s3_path, content_type='csv')

# Measure and start training time
start = t.time()
xgboost_estimator.fit({'train': train_input})
end = t.time()

print(f"Runtime for training on SageMaker: {end - start:.2f} seconds, instance_type: {instance_type}, instance_count: {instance_count}")

```

When running longer training jobs, you can check on their status periodically from the AWS SageMaker Console (where we originally launched our Notebook instance) on left side panel under "Training".

#### Hyperparameters
The `hyperparameters` section in this code defines the input arguments of train_XGBoost.py. The first is the name of the training input file, and the others are hyperparameters for the XGBoost model, such as `max_depth`, `eta`, `subsample`, `colsample_bytree`, and `num_round`.

#### TrainingInput
Additionally, we define a TrainingInput object containing the training data's S3 path, to pass to `.fit({'train': train_input})`. SageMaker uses `TrainingInput` to download your dataset from S3 to a temporary location on the training instance. This location is mounted and managed by SageMaker and can be accessed by the training job if/when needed.

#### Model results
With this code, the training results and model artifacts are saved in a subfolder called `xgboost` in your specified S3 bucket. This folder (`s3://{bucket_name}/xgboost/`) will be automatically created if it doesn't already exist, and will contain:

1. **Model "artifacts"**: The trained model file (often a `.tar.gz` file) that SageMaker saves in the `output_path`.
2. **Logs and metrics**: Any metrics and logs related to the training job, stored in the same `xgboost` folder.
 
This setup allows for convenient access to both the trained model and related output for later evaluation or deployment.

### Extracting trained model from S3 for final evaluation
To evaluate the model on a test set after training, we'll go through these steps:

1. **Download the trained model from S3**.
2. **Load and preprocess** the test dataset. 
3. **Evaluate** the model on the test data.

Here's how you can implement this in your SageMaker notebook. The following code will:

- Download the `model.tar.gz` file containing the trained model from S3.
- Load the `test.csv` data from S3 and preprocess it as needed.
- Use the XGBoost model to make predictions on the test set and then compute accuracy or other metrics on the results. 

If additional metrics or custom evaluation steps are needed, you can add them in place of or alongside accuracy.


```python
# Model results are saved in auto-generated folders. Use xgboost_estimator.latest_training_job.name to get the folder name
model_s3_path = f'{output_folder}/{xgboost_estimator.latest_training_job.name}/output/model.tar.gz'
print(model_s3_path)
local_model_path = 'model.tar.gz'

# Download the trained model from S3
s3.download_file(bucket_name, model_s3_path, local_model_path)

# Extract the model file
import tarfile
with tarfile.open(local_model_path) as tar:
    tar.extractall()
```


```python
import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
import joblib
from AWS_helpers.train_xgboost import preprocess_data

# Load the test data
test_data = pd.read_csv('./titanic_test.csv')

# Preprocess the test data using the imported preprocess_data function
X_test, y_test = preprocess_data(test_data)

# Convert the test features to DMatrix for XGBoost
dtest = xgb.DMatrix(X_test)

# Load the trained model from the saved file
model = joblib.load('./xgboost-model')

# Make predictions on the test set
preds = model.predict(dtest)
predictions = np.round(preds)  # Round predictions to 0 or 1 for binary classification

# Calculate and print the accuracy of the model on the test data
accuracy = accuracy_score(y_test, predictions)
print(f"Test Set Accuracy: {accuracy:.4f}")

```

Now that we've covered training using a custom script with the `XGBoost` estimator, let's examine the built-in image-based approach. Using SageMaker's pre-configured XGBoost image streamlines the setup by eliminating the need to manage custom scripts for common workflows, and it can also provide optimization advantages. Below, we'll discuss both the code and pros and cons of the image-based setup compared to the custom script approach.

## Training with SageMaker's Built-in XGBoost Image

With the SageMaker-provided XGBoost container, you can bypass custom script configuration if your workflow aligns with standard XGBoost training. This setup is particularly useful when you need quick, default configurations without custom preprocessing or additional libraries.


### Comparison: Custom Script vs. Built-in Image

| Feature                | Custom Script (`XGBoost` with `entry_point`)      | Built-in XGBoost Image                       |
|------------------------|--------------------------------------------------|----------------------------------------------|
| **Flexibility**        | Allows for custom preprocessing, data transformation, or advanced parameterization. Requires a Python script and custom dependencies can be added through `requirements.txt`. | Limited to XGBoost's built-in functionality, no custom preprocessing steps without additional customization. |
| **Simplicity**         | Requires setting up a script with `entry_point` and managing dependencies. Ideal for specific needs but requires configuration. | Streamlined for fast deployment without custom code. Simple setup and no need for custom scripts.  |
| **Performance**        | Similar performance, though potential for overhead with additional preprocessing. | Optimized for typical XGBoost tasks with faster startup. May offer marginally faster time-to-first-train. |
| **Use Cases**          | Ideal for complex workflows requiring unique preprocessing steps or when adding specific libraries or functionalities. | Best for quick experiments, standard workflows, or initial testing on datasets without complex preprocessing. |

**When to use each approach**:

- **Custom script**: Recommended if you need to implement custom data preprocessing, advanced feature engineering, or specific workflow steps that require more control over training.
- **Built-in image**: Ideal when running standard XGBoost training, especially for quick experiments or production deployments where default configurations suffice.

Both methods offer powerful and flexible approaches to model training on SageMaker, allowing you to select the approach best suited to your needs. Below is an example of training using the built-in XGBoost Image.

#### Setting up the data path
In this approach, using `TrainingInput` directly with SageMaker's built-in XGBoost container contrasts with our previous method, where we specified a custom script with argument inputs (specified in hyperparameters) for data paths and settings. Here, we use hyperparameters only to specify the model's hyperparameters.

```python
from sagemaker.estimator import Estimator # when using images, we use the general Estimator class

# Define instance type/count we'll use for training
instance_type="ml.m5.large"
instance_count=1 # always start with 1. Rarely is parallelized training justified with data < 50 GB. More on this later.

# Use Estimator directly for built-in container without specifying entry_point
xgboost_estimator_builtin = Estimator(
    base_job_name=notebook_instance_name,
    max_run=max_run, # in seconds; always include (max 48 hours)
    image_uri=sagemaker.image_uris.retrieve("xgboost", session.boto_region_name, version="1.5-1"),
    role=role,
    tags=job_tags,
    instance_count=instance_count,
    instance_type=instance_type,
    output_path=output_path,
    sagemaker_session=session,
    hyperparameters={
        'max_depth': max_depth,
        'eta': eta,
        'subsample': subsample,
        'colsample_bytree': colsample_bytree,
        'num_round': num_round
    }
)

# Define input data
train_input = TrainingInput(train_s3_path, content_type="csv")

# Measure and start training time
start = t.time()
xgboost_estimator_builtin.fit({'train': train_input})
end = t.time()

print(f"Runtime for training on SageMaker: {end - start:.2f} seconds, instance_type: {instance_type}, instance_count: {instance_count}")

```


## Monitoring training

To view and monitor your SageMaker training job, follow these steps in the AWS Management Console. Since training jobs may be visible to multiple users in your account, it's essential to confirm that you're interacting with your own job before making any changes.

1. **Navigate to the SageMaker Console**  
   - Go to the AWS Management Console and open the **SageMaker** service (can search for it)

2. **View training jobs**  
   - In the left-hand navigation menu, select **Training jobs**. You'll see a list of recent training jobs, which may include jobs from other users in the account.

3. **Verify your training Job**  
   - Identify your job by looking for the specific name format (e.g., `sagemaker-xgboost-YYYY-MM-DD-HH-MM-SS-XXX`) generated when you launched the job.  Click on its name to access detailed information. Cross-check the job details, such as the **Instance Type** and **Input data configuration**, with the parameters you set in your script. 

4. **Monitor the job status**  
   - Once you've verified the correct job, click on its name to access detailed information:
     - **Status**: Confirms whether the job is `InProgress`, `Completed`, or `Failed`.
     - **Logs**: Review CloudWatch Logs and Job Metrics for real-time updates.
     - **Output Data**: Shows the S3 location with the trained model artifacts.

5. **Stopping a training job**  
   - Before stopping a job, ensure you've selected the correct one by verifying job details as outlined above.
   - If you're certain it's your job, go to **Training jobs** in the SageMaker Console, select the job, and choose **Stop** from the **Actions** menu. Confirm your selection, as this action will halt the job and release any associated resources.
   - **Important**: Avoid stopping jobs you don't own, as this could disrupt other users' work and may have unintended consequences.

Following these steps helps ensure you only interact with and modify jobs you own, reducing the risk of impacting other users' training processes.

## When training takes too long

When training time becomes excessive, two main options can improve efficiency in SageMaker:

- **Option 1: Upgrading to a more powerful instance**  
- **Option 2: Using multiple instances for distributed training**  

Generally, Option 1 is the preferred approach and should be explored first.

### Option 1: Upgrade to a more powerful instance (preferred starting point)

Upgrading to a more capable instance, particularly one with GPU capabilities, is often the simplest and most cost-effective way to speed up training. Check the [Instances for ML page](https://carpentries-incubator.github.io/ML_with_AWS_SageMaker/instances-for-ML.html) for guidance.

When to use a single instance upgrade:  
- Dataset size – The dataset is small to moderate (e.g., <10 GB), fitting comfortably within memory.  
- Model complexity – XGBoost models are typically small enough to fit in memory.  
- Training time – If training completes in a few hours but could be faster, upgrading may help.  

Upgrading a single instance is usually the most efficient option. It avoids the communication overhead of multi-instance setups and works well for small to medium datasets.

### Option 2: Use multiple instances for distributed training

If upgrading a single instance doesn’t sufficiently reduce training time, distributed training across multiple instances may be a viable alternative. For XGBoost, SageMaker applies only data parallelism (not model parallelism).

#### XGBoost uses data parallelism, not model parallelism

- Data parallelism – The dataset is split across multiple instances, with each instance training on a portion of the data. The gradient updates are then synchronized and aggregated.  
- Why not model parallelism? – Unlike deep learning models, XGBoost decision trees are small enough to fit in memory, so there’s no need to split the model itself across multiple instances.  

#### How SageMaker implements data parallelism for XGBoost

- When `instance_count > 1`, SageMaker automatically splits the dataset across instances.  
- Each instance trains on a subset of the data, computing gradient updates in parallel.  
- Gradient updates are synchronized across instances before the next iteration.  
- The final trained model is assembled as if it had been trained on the full dataset.  

### When to consider multiple instances

Using multiple instances makes sense when:  
- Dataset size – The dataset is large and doesn't fit comfortably in memory.  
- Expected training time – A single instance takes too long (e.g., >10 hours).  
- Need for faster training – Parallelization can speed up training but introduces communication overhead.  

If scaling to multiple instances, monitoring training time and efficiency is critical. In many cases, a single, more powerful instance may be more cost-effective than multiple smaller ones.  

### Implementing distributed training with XGBoost in SageMaker

In SageMaker, setting up distributed training for XGBoost can offer significant time savings as dataset sizes and computational requirements increase. Here's how you can configure it:

1. **Select multiple instances**: Specify `instance_count > 1` in the SageMaker `Estimator` to enable distributed training.
2. **Optimize instance type**: Choose an instance type suitable for your dataset size and XGBoost requirements 
3. **Monitor for speed improvements**: With larger datasets, distributed training can yield time savings by scaling horizontally. However, gains may vary depending on the dataset and computation per instance.


```python
# Define instance type/count we'll use for training
instance_type="ml.m5.large"
instance_count=1 # always start with 1. Rarely is parallelized training justified with data < 50 GB.

# Define the XGBoost estimator for distributed training
xgboost_estimator = Estimator(
    base_job_name=notebook_instance_name,
    max_run=max_run, # in seconds; always include (max 48 hours)
    image_uri=sagemaker.image_uris.retrieve("xgboost", session.boto_region_name, version="1.5-1"),
    role=role,
    tags=job_tags,
    instance_count=instance_count,  # Start with 1 instance for baseline
    instance_type=instance_type,
    output_path=output_path,
    sagemaker_session=session,
)

# Set hyperparameters
xgboost_estimator.set_hyperparameters(
    max_depth=5,
    eta=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    num_round=100,
)

# Specify input data from S3
train_input = TrainingInput(train_s3_path, content_type="csv")

# Run with 1 instance
start1 = t.time()
xgboost_estimator.fit({"train": train_input})
end1 = t.time()


# Now run with 2 instances to observe speedup
xgboost_estimator.instance_count = 2
start2 = t.time()
xgboost_estimator.fit({"train": train_input})
end2 = t.time()

print(f"Runtime for training on SageMaker: {end1 - start1:.2f} seconds, instance_type: {instance_type}, instance_count: {instance_count}")
print(f"Runtime for training on SageMaker: {end2 - start2:.2f} seconds, instance_type: {instance_type}, instance_count: {xgboost_estimator.instance_count}")

```

### Why scaling instances might not show speedup here

* Small dataset: With only 892 rows, the dataset might be too small to benefit from distributed training. Distributing small datasets often adds overhead (like network communication between instances), which outweighs the parallel processing benefits.

* Distributed overhead: Distributed training introduces coordination steps that can add latency. For very short training jobs, this overhead can become a larger portion of the total training time, reducing the benefit of additional instances.

* Tree-based models: Tree-based models, like those in XGBoost, don't benefit from distributed scaling as much as deep learning models when datasets are small. For large datasets, distributed XGBoost can still offer speedups, but this effect is generally less than with neural networks, where parallel gradient updates across multiple instances become efficient.

### When multi-instance training helps
* Larger datasets: Multi-instance training shines with larger datasets, where splitting the data across instances and processing it in parallel can significantly reduce the training time.

* Complex models: For highly complex models with many parameters (like deep learning models or large XGBoost ensembles) and long training times, distributing the training can help speed up the process as each instance contributes to the gradient calculation and optimization steps.

* Distributed algorithms: XGBoost has a built-in distributed training capability, but models that perform gradient descent, like deep neural networks, gain more obvious benefits because each instance can compute gradients for a batch of data simultaneously, allowing faster convergence.

### For cost optimization
* Single-instance training is typically more cost-effective for small or moderately sized datasets, while multi-instance setups can reduce wall-clock time for larger datasets and complex models, at a higher instance cost.
* Increase instance count only if training time becomes prohibitive even with more powerful single instances, while being mindful of communication overhead and scaling efficiency.


::::::::::::::::::::::::::::::::::::: keypoints

- **Environment initialization**: Setting up a SageMaker session, defining roles, and specifying the S3 bucket are essential for managing data and running jobs in SageMaker.
- **Local vs. managed training**: Always test your code locally (on a smaller scale) before scaling things up. This avoids wasting resources on buggy code that doesn't produce reliable results.
- **Estimator classes**: SageMaker provides framework-specific Estimator classes (e.g., XGBoost, PyTorch, SKLearn) to streamline training setups, each suited to different model types and workflows.
- **Custom scripts vs. built-in images**: Custom training scripts offer flexibility with preprocessing and custom logic, while built-in images are optimized for rapid deployment and simpler setups.
- **Training data channels**: Using `TrainingInput` ensures SageMaker manages data efficiently, especially for distributed setups where data needs to be synchronized across multiple instances.
- **Distributed training options**: Data parallelism (splitting data across instances) is common for many models, while model parallelism (splitting the model across instances) is useful for very large models that exceed instance memory.

::::::::::::::::::::::::::::::::::::::::::::::::

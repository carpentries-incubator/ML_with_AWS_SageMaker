---
title: "Training Models in SageMaker Notebooks"
teaching: 20
exercises: 10
---

:::::::::::::::::::::::::::::::::::::: questions 

- What are the differences between local training and SageMaker-managed training?
- How do Estimator classes in SageMaker streamline the training process for various frameworks?
- How does SageMaker handle data and model parallelism, and when should each be considered?
- When should you consider using a GPU instance for training neural networks in SageMaker, and what are the benefits and limitations?

::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: objectives

- Understand the difference between training locally in a SageMaker notebook and using SageMaker's managed infrastructure.
- Learn to configure and use SageMaker's Estimator classes for different frameworks (e.g., XGBoost, PyTorch, SKLearn).
- Understand data and model parallelism options in SageMaker, including when to use each for efficient training.
- Compare performance, cost, and setup between custom scripts and built-in images in SageMaker.
- Conduct training with data stored in S3 and monitor training job status using the SageMaker console.
- Determine when to use GPU instances for neural network training in SageMaker, and assess their impact on performance and cost.

::::::::::::::::::::::::::::::::::::::::::::::::

## Initial setup 

#### 1. Open a new .ipynb notebook
Open a fresh .ipynb notebook ("Jupyter notebook"), and select the conda_pytorch_p310 environment. This will save us the trouble of having to install pytorch in this notebook. You can name your Jupyter notebook something along the lines of, `Training-models.ipynb`.

#### 2. CD to instance home directory
So we all can reference the helper functions using the same path, CD to...

```python
%cd /home/ec2-user/SageMaker/
```

#### 3. Initialize SageMaker environment
This code initializes the AWS SageMaker environment by defining the SageMaker role, session, and S3 client. It also specifies the S3 bucket and key for accessing the Titanic training dataset stored in an S3 bucket.

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

# Create a SageMaker session to manage interactions with Amazon SageMaker, such as training jobs, model deployments, and data input/output.
session = sagemaker.Session()

# Initialize an S3 client to interact with Amazon S3, allowing operations like uploading, downloading, and managing objects and buckets.
s3 = boto3.client('s3')

# Define the S3 bucket that we will load from
bucket_name = 'myawesometeam-titanic'  # replace with your S3 bucket name

# Define train/test filenames
train_filename = 'titanic_train.csv'
test_filename = 'titanic_test.csv'
```

    sagemaker.config INFO - Not applying SDK defaults from location: /etc/xdg/sagemaker/config.yaml
    sagemaker.config INFO - Not applying SDK defaults from location: /home/ec2-user/.config/sagemaker/config.yaml
    role = arn:aws:iam::183295408236:role/ml-sagemaker-use


### 3. Download copy into notebook environment
It can be convenient to have a "local" copy (i.e., one that you store in your notebook's instance). Run the next code chunk to download data from S3 to notebook environment. You may need to hit refresh on the file explorer panel to the left to see this file. If you get any permission issues...

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

    File downloaded: ./titanic_train.csv


We can do the same for the test set.


```python
# Define the S3 bucket and file location
file_key = f"{test_filename}"  # Path to your file in the S3 bucket. W
local_file_path = f"./{test_filename}"  # Local path to save the file

# Initialize the S3 client and download the file
s3.download_file(bucket_name, file_key, local_file_path)
print("File downloaded:", local_file_path)

```

    File downloaded: ./titanic_test.csv

### 4. Get code from git repo (skip if completed already from earlier episodes)
If you didn't complete the earlier episodes, you'll need to clone our code repo before moving forward. Check to make sure we're in our EC2 root folder (`/home/ec2-user/SageMaker`).

```python
!pwd
```

    /home/ec2-user/SageMaker/test_AWS


If not, change directory using `%cd `.


```python
%cd /home/ec2-user/SageMaker/
```

    /home/ec2-user/SageMaker


```python
!git clone https://github.com/username/AWS_helpers.git
```

    fatal: destination path 'test_AWS' already exists and is not an empty directory.


## Testing train.py on this notebook's instance
In this next section, we will learn how to take a model training script, and deploy it to more powerful instances (or many instances). This is helpful for machine learning jobs that require extra power, GPUs, or benefit from parallelization. Before we try exploiting this extra power, it is essential that we test our code thoroughly. We don't want to waste unnecessary compute cycles and resources on jobs that produce bugs instead of insights. If you need to, you can use a subset of your data to run quicker tests. You can also select a slightly better instance resource if your current instance insn't meeting your needs. See the [Instances for ML spreadsheet](https://docs.google.com/spreadsheets/d/1uPT4ZAYl_onIl7zIjv5oEAdwy4Hdn6eiA9wVfOBbHmY/edit?usp=sharing) for guidance. 

#### Logging runtime & instance info
To compare our local runtime with future experiments, we'll need to know what instance was used, as this will greatly impact runtime in many cases. We can extract the instance name for this notebook using...

```python
# Replace with your notebook instance name.
# This does NOT refer to specific ipynb files, but to the SageMaker notebook instance.
notebook_instance_name = 'MyAwesomeTeam-ChrisEndemann-Titanic-Train-Tune-Xgboost-NN'

# Initialize SageMaker client
sagemaker_client = boto3.client('sagemaker')

# Describe the notebook instance
response = sagemaker_client.describe_notebook_instance(NotebookInstanceName=notebook_instance_name)

# Display the status and instance type
print(f"Notebook Instance '{notebook_instance_name}' status: {response['NotebookInstanceStatus']}")
local_instance = response['InstanceType']
print(f"Instance Type: {local_instance}")

```

    Notebook Instance 'MyAwesomeTeam-ChrisEndemann-Titanic-Train-Tune-Xgboost-NN' status: InService
    Instance Type: ml.t3.medium


#### Helper:  `get_notebook_instance_info()` 
You can also use the `get_notebook_instance_info()` function found in `AWS_helpers.py` to retrieve this info for your own project.


```python
import AWS_helpers.helpers as helpers
helpers.get_notebook_instance_info(notebook_instance_name)
```

    {'Status': 'InService', 'InstanceType': 'ml.t3.medium'}


Test train.py on this notebook's instance (or when possible, on your own machine) before doing anything more complicated (e.g., hyperparameter tuning on multiple instances)


```python
!pip install xgboost # need to add this to environment to run train.py
```
    Collecting xgboost
      Downloading xgboost-2.1.2-py3-none-manylinux2014_x86_64.whl.metadata (2.0 kB)
    Requirement already satisfied: numpy in /home/ec2-user/anaconda3/envs/pytorch_p310/lib/python3.10/site-packages (from xgboost) (1.26.4)
    Requirement already satisfied: scipy in /home/ec2-user/anaconda3/envs/pytorch_p310/lib/python3.10/site-packages (from xgboost) (1.14.1)
    Downloading xgboost-2.1.2-py3-none-manylinux2014_x86_64.whl (4.5 MB)
       ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 4.5/4.5 MB 82.5 MB/s eta 0:00:00
    Installing collected packages: xgboost
    Successfully installed xgboost-2.1.2

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

    Train size: (569, 8)
    Val size: (143, 8)
    Training time: 0.06 seconds
    Model saved to ./xgboost-model
    Total local runtime: 1.01 seconds, instance_type = ml.t3.medium


    /home/ec2-user/anaconda3/envs/pytorch_p310/lib/python3.10/site-packages/xgboost/core.py:265: FutureWarning: Your system has an old version of glibc (< 2.28). We will stop supporting Linux distros with glibc older than 2.28 after **May 31, 2025**. Please upgrade to a recent Linux distro (with glibc 2.28+) to use future versions of XGBoost.
    Note: You have installed the 'manylinux2014' variant of XGBoost. Certain features such as GPU algorithms or federated learning are not available. To use these features, please upgrade to a recent Linux distro with glibc 2.28+, and install the 'manylinux_2_28' variant.
      warnings.warn(


Training on this relatively small dataset should take less than a minute, but as we scale up with larger datasets and more complex models in SageMaker, tracking both training time and total runtime becomes essential for efficient debugging and resource management.

**Note**: Our code above includes print statements to monitor dataset size, training time, and total runtime, which provides insights into resource usage for model development. We recommend incorporating similar logging to track not only training time but also total runtime, which includes additional steps like data loading, evaluation, and saving results. Tracking both can help you pinpoint bottlenecks and optimize your workflow as projects grow in size and complexity, especially when scaling with SageMaker's distributed resources.


### Quick evaluation on test set
This next section isn't SageMaker specific, so we'll cover it quickly. Here's how you would apply the outputted model to your test set using your local notebook instance.

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

    Test Set Accuracy: 0.8156


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

::::::::::::::::::::::::::::::::::::: callout 
#### Configuring custom environments with `requirements.txt`

For all these Estimators, adding a `requirements.txt` file under `dependencies` ensures that additional packages are installed before training begins. This approach allows the use of specific libraries that may be critical for custom preprocessing, feature engineering, or model modifications. Here's how to include it:

```python
sklearn_estimator = SKLearn(
    entry_point="train_script.py",
    role=role,
    instance_count=1,
    instance_type="ml.m5.large",
    output_path="s3://your-bucket/output",
    framework_version="1.0-1",
    dependencies=['requirements.txt'],  # Adding custom dependencies
    hyperparameters={
        "max_depth": 5,
        "eta": 0.1,
        "subsample": 0.8,
        "num_round": 100
    }
)
```

This setup simplifies training, allowing you to maintain custom environments directly within SageMaker's managed containers, without needing to build and manage your own Docker images. The [AWS SageMaker Documentation](https://docs.aws.amazon.com/sagemaker/latest/dg/pre-built-containers-frameworks-deep-learning.html) provides lists of pre-built container images for each framework and their standard libraries, including details on pre-installed packages.
:::::::::::::::::::::::::::::::::::::::::::::

### Deploying to other instances
For this deployment, we configure the "XGBoost" estimator with a custom training script, train_xgboost.py, and define hyperparameters directly within the SageMaker setup. Here's the full code, with some additional explanation following the code.


```python
from sagemaker.inputs import TrainingInput
from sagemaker.xgboost.estimator import XGBoost

# Define instance type/count we'll use for training
instance_type="ml.m5.large"
instance_count=1 # always start with 1. Rarely is parallelized training justified with data < 50 GB. More on this later.

# Define S3 paths for input and output
train_s3_path = f's3://{bucket_name}/{train_filename}'

# we'll store all results in a subfolder called xgboost on our bucket. This folder will automatically be created if it doesn't exist already.
output_folder = 'xgboost'
output_path = f's3://{bucket_name}/{output_folder}/' 

# Set up the SageMaker XGBoost Estimator with custom script
xgboost_estimator = XGBoost(
    entry_point='train_xgboost.py',      # Custom script path
    source_dir='AWS_helpers',               # Directory where your script is located
    role=role,
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

    INFO:sagemaker:Creating training-job with name: sagemaker-xgboost-2024-11-03-21-10-03-577



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

    xgboost/sagemaker-xgboost-2024-11-03-21-10-03-577/output/model.tar.gz



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

    Test Set Accuracy: 0.8156


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
    image_uri=sagemaker.image_uris.retrieve("xgboost", session.boto_region_name, version="1.5-1"),
    role=role,
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
    
    2024-11-03 21:16:19 Uploading - Uploading generated training model
    2024-11-03 21:16:19 Completed - Training job completed
    Training seconds: 135
    Billable seconds: 135
    Runtime for training on SageMaker: 197.50 seconds, instance_type: ml.m5.large, instance_count: 1


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

When training time becomes excessive, two main options can improve efficiency in SageMaker.
* **Option 1: Upgrading to a more powerful instance** 
* **Option 2: Using multiple instances for distributed training**. 

Generally, **Option 1 is the preferred approach** and should be explored first.

### Option 1: Upgrade to a more powerful instance (preferred starting point)

Upgrading to a more capable instance, particularly one with GPU capabilities (e.g., for deep learning), is often the simplest and most cost-effective way to speed up training. Here's a breakdown of instances to consider. Check the [Instances for ML spreadsheet](https://docs.google.com/spreadsheets/d/1uPT4ZAYl_onIl7zIjv5oEAdwy4Hdn6eiA9wVfOBbHmY/edit?usp=sharing) for guidance on selecting a better instance.

**When to use a single instance upgrade**  
Upgrading a single instance works well if:
   - **Dataset size**: The dataset is small to moderate (e.g., <10 GB), fitting comfortably within the memory of a larger instance.
   - **Model complexity**: The model is not so large that it requires distribution across multiple devices.
   - **Training time**: Expected training time is within a few hours, but could benefit from additional power.

Upgrading a single instance is typically the most efficient option in terms of both cost and setup complexity. It avoids the communication overhead associated with multi-instance setups (discussed below) and is well-suited for most small to medium-sized datasets.

### Option 2: Use multiple instances for distributed training
If upgrading a single instance doesn't sufficiently reduce training time, distributed training across multiple instances may be a viable alternative, particularly for larger datasets and complex models. SageMaker supports two primary distributed training techniques: **data parallelism** and **model parallelism**.

#### Understanding data parallelism vs. model parallelism

- **Data parallelism**: This approach splits the dataset across multiple instances, allowing each instance to process a subset of the data independently. After each batch, gradients are synchronized across instances to ensure consistent updates to the model. Data parallelism is effective when the model itself fits within an instance's memory, but the data size or desired training speed requires faster processing through multiple instances.

- **Model parallelism**: Model parallelism divides the model itself across multiple instances, making it ideal for very large models (e.g., deep learning models in NLP or image processing) that cannot fit in memory on a single instance. Each instance processes a segment of the model, and results are combined during training. This approach is suitable for memory-intensive models that exceed the capacity of a single instance.

#### How SageMaker chooses between data and model parallelism

In SageMaker, the choice between data and model parallelism is not entirely automatic. Here's how it typically works:

- **Data parallelism (automatic)**: When you set `instance_count > 1`, SageMaker will automatically apply data parallelism. This splits the dataset across instances, allowing each instance to process a subset independently and synchronize gradients after each batch. Data parallelism works well when the model can fit in the memory of a single instance, but the data size or processing speed needs enhancement with multiple instances.

- **Model parallelism (manual setup)**: To enable model parallelism, you need to configure it explicitly using the **SageMaker Model Parallel Library**, suitable for deep learning models in frameworks like PyTorch or TensorFlow. Model parallelism splits the model itself across multiple instances, which is useful for memory-intensive models that exceed the capacity of a single instance. Configuring model parallelism requires setting up a distribution strategy in SageMaker's Python SDK.

- **Hybrid parallelism (manual setup)**: For extremely large datasets and models, SageMaker can support both data and model parallelism together, but this setup requires manual configuration. Hybrid parallelism is beneficial for workloads that are both data- and memory-intensive, where both the model and the data need distributed processing.

**When to use distributed training with multiple instances**  
Consider multiple instances if:
   - **Dataset size**: The dataset is large (>10 GB) and doesn't fit comfortably within a single instance's memory.
   - **Model complexity**: The model is complex, requiring extensive computation that a single instance cannot handle in a reasonable time.
   - **Expected training time**: Training on a single instance takes prohibitively long (e.g., >10 hours), and distributed computing overhead is manageable.

### Cost of distributed computing 
**tl;dr** Use 1 instance unless you are finding that you're waiting hours for the training/tuning to complete.

Let's break down some key points for deciding between **1 instance vs. multiple instances** from a cost perspective:

1. **Instance cost per hour**:
   - SageMaker charges per instance-hour. Running **multiple instances** in parallel can finish training faster, reducing wall-clock time, but the **cost per hour will increase** with each added instance.

2. **Single instance vs. multiple instance wall-clock time**:
   - When using a single instance, training will take significantly longer, especially if your data is large. However, the wall-clock time difference between 1 instance and 10 instances may not translate to a direct 10x speedup when using multiple instances due to **communication overheads**.
   - For example, with data-parallel training, instances need to synchronize gradients between batches, which introduces **communication costs** and may slow down training on larger clusters.

3. **Scaling efficiency**:
   - Parallelizing training does not scale perfectly due to those overheads. Adding instances generally provides **diminishing returns** on training time reduction.
   - For example, doubling instances from 1 to 2 may reduce training time by close to 50%, but going from 8 to 16 instances may only reduce training time by around 20-30%, depending on the model and batch sizes.

4. **Typical recommendation**:
   - For **small-to-moderate datasets** or cases where training time isn't a critical factor, a **single instance** may be more cost-effective, as it avoids parallel processing overheads.
   - For **large datasets** or where training speed is a high priority (e.g., tuning complex deep learning models), using **multiple instances** can be beneficial despite the cost increase due to time savings.

5. **Practical cost estimation**:
   - Suppose a single instance takes `T` hours to train and costs `$C` per hour. For a 10-instance setup, the cost would be approximately:
     - **Single instance:** `T * $C`
     - **10 instances (parallel):** `(T / k) * (10 * $C)`, where `k` is the speedup factor (<10 due to overhead).
   - If the speedup is only about 5x instead of 10x due to communication overhead, then the cost difference may be minimal, with a slight edge to a single instance on total cost but at a higher wall-clock time.


> In summary:
> - **Start by upgrading to a more powerful instance (Option 1)** for datasets up to 10 GB and moderately complex models. A single, more powerful, instance is usually more cost-effective for smaller workloads and where time isn't critical. Running initial tests with a single instance can also provide a benchmark. You can then experiment with small increases in instance count to find a balance between cost and time savings, particularly considering communication overheads that affect parallel efficiency.
> - **Consider distributed training across multiple instances (Option 2)** only when dataset size, model complexity, or training time demand it.


## XGBoost's distributed training mechanism
In the event that option 2 explained above really is better for your use-case (e.g., you have a very large dataset or model that takes a while to train even with high performance instances), the next example will demo setting this up. Before we do, though, we should ask what distributed computing really means for our specific model/setup. XGBoost's distributed training relies on a data-parallel approach that divides the dataset across multiple instances (or workers), enabling each instance to work on a portion of the data independently. This strategy enhances efficiency, especially for large datasets and computationally intensive tasks. 

> **What about a model parallelism approach?** Unlike deep learning models with vast neural network layers, XGBoost's decision trees are usually small enough to fit in memory on a single instance, even when the dataset is large. Thus, model parallelism is rarely necessary.
XGBoost does not inherently support model parallelism out of the box in SageMaker because the model architecture doesn't typically exceed memory limits, unlike massive language or image models. Although model parallelism can be theoretically applied (e.g., splitting large tree structures across instances), it's generally not supported natively in SageMaker for XGBoost, as it would require a custom distribution framework to split the model itself.

Here's how distributed training in XGBoost works, particularly in the SageMaker environment:

### Key steps in distributed training with XGBoost

#### 1. **Data partitioning**
   - The dataset is divided among multiple instances. For example, with two instances, each instance may receive half of the dataset.
   - In SageMaker, data partitioning across instances is handled automatically via the input channels you specify during training, reducing manual setup.

#### 2. **Parallel gradient boosting**
   - XGBoost performs gradient boosting by constructing trees iteratively based on calculated gradients.
   - Each instance calculates gradients (first-order derivatives) and Hessians (second-order derivatives of the loss function) independently on its subset of data.
   - This parallel processing allows each instance to determine which features to split and which trees to add to the model based on its data portion.

#### 3. **Communication between instances**
   - After computing gradients and Hessians locally, instances synchronize to share and combine these values.
   - Synchronization keeps the model parameters consistent across instances. Only computed gradients are communicated, not the raw dataset, minimizing data transfer overhead.
   - The combined gradients guide global model updates, ensuring that the ensemble of trees reflects the entire dataset, despite its division across multiple instances.

#### 4. **Final model aggregation**
   - Once training completes, XGBoost aggregates the trained trees from each instance into a single final model.
   - This aggregation enables the final model to perform as though it trained on the entire dataset, even if the dataset couldn't fit into a single instance's memory.

SageMaker simplifies these steps by automatically managing the partitioning, synchronization, and aggregation processes during distributed training with XGBoost.


## Implementing distributed training with XGBoost in SageMaker

In SageMaker, setting up distributed training for XGBoost can offer significant time savings as dataset sizes and computational requirements increase. Here's how you can configure it:

1. **Select Multiple Instances**: Specify `instance_count > 1` in the SageMaker `Estimator` to enable distributed training.
2. **Optimize Instance Type**: Choose an instance type suitable for your dataset size and XGBoost requirements 
3. **Monitor for Speed Improvements**: With larger datasets, distributed training can yield time savings by scaling horizontally. However, gains may vary depending on the dataset and computation per instance.


```python
# Define instance type/count we'll use for training
instance_type="ml.m5.large"
instance_count=1 # always start with 1. Rarely is parallelized training justified with data < 50 GB.

# Define the XGBoost estimator for distributed training
xgboost_estimator = Estimator(
    image_uri=sagemaker.image_uris.retrieve("xgboost", session.boto_region_name, version="1.5-1"),
    role=role,
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

    INFO:sagemaker.image_uris:Ignoring unnecessary instance type: None.
    INFO:sagemaker:Creating training-job with name: sagemaker-xgboost-2024-11-03-21-16-39-216


    2024-11-03 21:16:40 Starting - Starting the training job...
    2024-11-03 21:16:55 Starting - Preparing the instances for training...
    2024-11-03 21:17:22 Downloading - Downloading input data...
    2024-11-03 21:18:07 Downloading - Downloading the training image......
    2024-11-03 21:19:13 Training - Training image download completed. Training in progress.
    2024-11-03 21:19:13 Uploading - Uploading generated training model[34m/miniconda3/lib/python3.8/site-packages/xgboost/compat.py:36: FutureWarning:
    2024-11-03 21:19:32 Completed - Training job completed

    INFO:sagemaker:Creating training-job with name: sagemaker-xgboost-2024-11-03-21-19-57-254
    Training seconds: 130
    Billable seconds: 130
    
    2024-11-03 21:19:58 Starting - Starting the training job...
    2024-11-03 21:20:13 Starting - Preparing the instances for training...
    2024-11-03 21:20:46 Downloading - Downloading input data......
    2024-11-03 21:21:36 Downloading - Downloading the training image...
    2024-11-03 21:22:27 Training - Training image download completed. Training in progress..[35m/miniconda3/lib/python3.8/site-packages/xgboost/compat.py:36: 
    
    2024-11-03 21:23:01 Uploading - Uploading generated training model
    2024-11-03 21:23:01 Completed - Training job completed
    Training seconds: 270
    Billable seconds: 270
    Runtime for training on SageMaker: 198.04 seconds, instance_type: ml.m5.large, instance_count: 1
    Runtime for training on SageMaker: 197.66 seconds, instance_type: ml.m5.large, instance_count: 2


### Why scaling instances might not show speedup here

* Small dataset: With only 892 rows, the dataset might be too small to benefit from distributed training. Distributing small datasets often adds overhead (like network communication between instances), which outweighs the parallel processing benefits.

* Distributed overhead: Distributed training introduces coordination steps that can add latency. For very short training jobs, this overhead can become a larger portion of the total training time, reducing the benefit of additional instances.

* Tree-based models: Tree-based models, like those in XGBoost, don't benefit from distributed scaling as much as deep learning models when datasets are small. For large datasets, distributed XGBoost can still offer speedups, but this effect is generally less than with neural networks, where parallel gradient updates across multiple instances become efficient.

### When multi-instance training helps
* Larger datasets: Multi-instance training shines with larger datasets, where splitting the data across instances and processing it in parallel can significantly reduce the training time.

* Complex models: For highly complex models with many parameters (like deep learning models or large XGBoost ensembles) and long training times, distributing the training can help speed up the process as each instance contributes to the gradient calculation and optimization steps.

* Distributed algorithms: XGBoost has a built-in distributed training capability, but models that perform gradient descent, like deep neural networks, gain more obvious benefits because each instance can compute gradients for a batch of data simultaneously, allowing faster convergence.

## Training a neural network with SageMaker
Let's see how to do a similar experiment, but this time using PyTorch neural networks. We will again demonstrate how to test our custom model train script (train_nn.py) before deploying to SageMaker, and discuss some strategies (e.g., using a GPU) for improving train time when needed.

### Preparing the data (compressed npz files)
When deploying a PyTorch model on SageMaker, it's helpful to prepare the input data in a format that's directly accessible and compatible with PyTorch's data handling methods. The next code cell will prep our npz files from the existing csv versions. 

:::::::::::::::::::::::::::::::: callout
#### Why are we using this file format? 

1. **Optimized data loading**:  
   The `.npz` format stores arrays in a compressed, binary format, making it efficient for both storage and loading. PyTorch can easily handle `.npz` files, especially in batch processing, without requiring complex data transformations during training.

2. **Batch compatibility**:  
   When training neural networks in PyTorch, it's common to load data in batches. By storing data in an `.npz` file, we can quickly load the entire dataset or specific parts (e.g., `X_train`, `y_train`) into memory and feed it to the PyTorch `DataLoader`, enabling efficient batched data loading.

3. **Reduced I/O overhead in SageMaker**:  
   Storing data in `.npz` files minimizes the I/O operations during training, reducing time spent on data handling. This is especially beneficial in cloud environments like SageMaker, where efficient data handling directly impacts training costs and performance.

4. **Consistency and compatibility**:  
   Using `.npz` files allows us to ensure consistency between training and validation datasets. Each file (`train_data.npz` and `val_data.npz`) stores the arrays in a standardized way that can be easily accessed by keys (`X_train`, `y_train`, `X_val`, `y_val`). This structure is compatible with PyTorch's `Dataset` class, making it straightforward to design custom datasets for training.

5. **Support for multiple data types**:  
   `.npz` files support storage of multiple arrays within a single file. This is helpful for organizing features and labels without additional code. Here, the `train_data.npz` file contains both `X_train` and `y_train`, keeping everything related to training data in one place. Similarly, `val_data.npz` organizes validation features and labels, simplifying file management.

In summary, saving the data in `.npz` files ensures a smooth workflow from data loading to model training in PyTorch, leveraging SageMaker's infrastructure for a more efficient, structured training process.
:::::::::::::::::::::::::::::::::::::::

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import numpy as np

# Load and preprocess the Titanic dataset
df = pd.read_csv(train_filename)

# Encode categorical variables and normalize numerical ones
df['Sex'] = LabelEncoder().fit_transform(df['Sex'])
df['Embarked'] = df['Embarked'].fillna('S')  # Fill missing values in 'Embarked'
df['Embarked'] = LabelEncoder().fit_transform(df['Embarked'])

# Fill missing values for 'Age' and 'Fare' with median
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Fare'] = df['Fare'].fillna(df['Fare'].median())

# Select features and target
X = df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']].values
y = df['Survived'].values

# Normalize features (helps avoid exploding/vanishing gradients)
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split the data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Save the preprocessed data to our local jupyter environment
np.savez('train_data.npz', X_train=X_train, y_train=y_train)
np.savez('val_data.npz', X_val=X_val, y_val=y_val)

```

Next, we will upload our compressed files to our S3 bucket. Storage is farily cheap on AWS (around $0.023 per GB per month), but be mindful of uploading too much data. It may be convenient to store a preprocessed version of the data, just don't store too many versions that aren't being actively used.


```python
import boto3

train_file = "train_data.npz"  # Local file path in your notebook environment
val_file = "val_data.npz"  # Local file path in your notebook environment

# Initialize the S3 client
s3 = boto3.client('s3')

# Upload the training and validation files to S3
s3.upload_file(train_file, bucket_name, f"{train_file}")
s3.upload_file(val_file, bucket_name, f"{val_file}")

print("Files successfully uploaded to S3.")

```

    Files successfully uploaded to S3.


#### Testing our train script on notebook instance
You should always test code thoroughly before scaling up and using more resources. Here, we will test our script using a small number of epochs — just to verify our setup is correct.


```python
import torch

# Measure training time locally
start_time = t.time()
%run  test_AWS/scripts/train_nn.py --train train_data.npz --val val_data.npz --epochs 1000 --learning_rate 0.001
print(f"Local training time: {t.time() - start_time:.2f} seconds, instance_type = {local_instance}")

```


### Deploying PyTorch Neural Network via SageMaker
Now that we have tested things locally, we can try to train with a larger number of epochs and a better instance selected. We can do this easily by invoking the PyTorch estimator. Our notebook is currently configured to use ml.m5.large. We can upgrade this to `ml.m5.xlarge` with the below code (using our notebook as a controller). 

**Should we use a GPU?**: Since this dataset is farily small, we don't necessarily need a GPU for training. Considering costs, the m5.xlarge is `$0.17/hour`, while the cheapest GPU instance is `$0.75/hour`. However, for larger datasets (> 1 GB) and models, we may want to consider a GPU if training time becomes cumbersome (see [Instances for ML](https://docs.google.com/spreadsheets/d/1uPT4ZAYl_onIl7zIjv5oEAdwy4Hdn6eiA9wVfOBbHmY/edit?usp=sharing). If that doesn't work, we can try distributed computing (setting instance > 1). More on this in the next section.


```python
from sagemaker.pytorch import PyTorch
from sagemaker.inputs import TrainingInput

epochs = 10000
instance_count = 1
instance_type="ml.m5.large"
output_path = f's3://{bucket_name}/output_nn/' # this folder will auto-generate if it doesn't exist already

# Define the PyTorch estimator and pass hyperparameters as arguments
pytorch_estimator = PyTorch(
    entry_point="test_AWS/scripts/train_nn.py",
    role=role,
    instance_type=instance_type, # with this small dataset, we don't recessarily need a GPU for fast training. 
    instance_count=instance_count,  # Distributed training with two instances
    framework_version="1.9",
    py_version="py38",
    output_path=output_path,
    sagemaker_session=session,
    hyperparameters={
        "train": "/opt/ml/input/data/train/train_data.npz",  # SageMaker will mount this path
        "val": "/opt/ml/input/data/val/val_data.npz",        # SageMaker will mount this path
        "epochs": epochs,
        "learning_rate": 0.001
    }
)

# Define input paths
train_input = TrainingInput(f"s3://{bucket_name}/train_data.npz", content_type="application/x-npz")
val_input = TrainingInput(f"s3://{bucket_name}/val_data.npz", content_type="application/x-npz")

# Start the training job and time it
start = t.time()
pytorch_estimator.fit({"train": train_input, "val": val_input})
end = t.time()

print(f"Runtime for training on SageMaker: {end - start:.2f} seconds, instance_type: {instance_type}, instance_count: {instance_count}")

```
    
    2024-11-03 21:27:03 Uploading - Uploading generated training model
    2024-11-03 21:27:03 Completed - Training job completed
    Training seconds: 135
    Billable seconds: 135
    Runtime for training on SageMaker: 197.62 seconds, instance_type: ml.m5.large, instance_count: 1


### Deploying PyTorch Neural Network via SageMaker with a GPU Instance

In this section, we'll implement the same procedure as above, but using a GPU-enabled instance for potentially faster training. While GPU instances are more expensive, they can be cost-effective for larger datasets or more complex models that require significant computational power.

#### Selecting a GPU Instance
For a small dataset like ours, we don't strictly need a GPU, but for larger datasets or more complex models, a GPU can reduce training time. Here, we'll select an `ml.g4dn.xlarge` instance, which provides a single GPU and costs approximately `$0.75/hour` (check [Instances for ML](https://docs.google.com/spreadsheets/d/1uPT4ZAYl_onIl7zIjv5oEAdwy4Hdn6eiA9wVfOBbHmY/edit?usp=sharing) for detailed pricing).

#### Code Modifications for GPU Use
Using a GPU requires minor changes in your training script (`train_nn.py`). Specifically, you'll need to:
1. Check for GPU availability in PyTorch.
2. Move the model and tensors to the GPU device if available.

#### Enabling PyTorch to use GPU in `train_nn.py`  

The following code snippet to enables GPU support in `train_nn.py`:

```python
import torch

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```


```python
from sagemaker.pytorch import PyTorch
from sagemaker.inputs import TrainingInput
import time as t

epochs = 10000
instance_count = 1
instance_type="ml.g4dn.xlarge"
output_path = f's3://{bucket_name}/output_nn/'

# Define the PyTorch estimator and pass hyperparameters as arguments
pytorch_estimator_gpu = PyTorch(
    entry_point="test_AWS/scripts/train_nn.py",
    role=role,
    instance_type=instance_type,
    instance_count=instance_count,
    framework_version="1.9",
    py_version="py38",
    output_path=output_path,
    sagemaker_session=session,
    hyperparameters={
        "train": "/opt/ml/input/data/train/train_data.npz",
        "val": "/opt/ml/input/data/val/val_data.npz",
        "epochs": epochs,
        "learning_rate": 0.001
    }
)

# Define input paths
train_input = TrainingInput(f"s3://{bucket_name}/train_data.npz", content_type="application/x-npz")
val_input = TrainingInput(f"s3://{bucket_name}/val_data.npz", content_type="application/x-npz")

# Start the training job and time it
start = t.time()
pytorch_estimator_gpu.fit({"train": train_input, "val": val_input})
end = t.time()
print(f"Runtime for training on SageMaker: {end - start:.2f} seconds, instance_type: {instance_type}, instance_count: {instance_count}")

```
    
    2024-11-03 21:33:56 Uploading - Uploading generated training model
    2024-11-03 21:33:56 Completed - Training job completed
    Training seconds: 350
    Billable seconds: 350
    Runtime for training on SageMaker: 409.68 seconds, instance_type: ml.g4dn.xlarge, instance_count: 1


#### GPUs can be slow for small datasets/models
> This performance discrepancy might be due to the following factors:
> 
> 1. **Small Dataset/Model Size**: When datasets and models are small, the overhead of transferring data between the CPU and GPU, as well as managing the GPU, can actually slow things down. For very small models and datasets, CPUs are often faster since there's minimal data to process.
> 
> 2. **GPU Initialization Overhead**: Every time a training job starts on a GPU, there's a small overhead for initializing CUDA libraries. For short jobs, this setup time can make the GPU appear slower overall.
> 
> 3. **Batch Size**: GPUs perform best with larger batch sizes since they can process many data points in parallel. If the batch size is too small, the GPU is underutilized, leading to suboptimal performance. You may want to try increasing the batch size to see if this reduces training time.
> 
> 4. **Instance Type**: Some GPU instances, like the `ml.g4dn` series, have less computational power than the larger `p3` series. They're better suited for inference or lightweight tasks rather than intense training, so a more powerful instance (e.g., `ml.p3.2xlarge`) could help for larger tasks.
> 
> If training time continues to be critical, sticking with a CPU instance may be the best approach for smaller datasets. For larger, more complex models and datasets, the GPU's advantages should become more apparent.

### Distributed Training for Neural Networks in SageMaker
In the event that you do need distributed computing to achieve reasonable train times (remember to try an upgraded instance first!), simply adjust the instance count to a number between 2 and 5. Beyond 5 instances, you'll see diminishing returns and may be needlessly spending extra money/compute-energy.


```python
from sagemaker.pytorch import PyTorch
from sagemaker.inputs import TrainingInput
import time as t

epochs = 10000
instance_count = 2 # increasing to 2 to see if it has any benefit (likely won't see any with this small dataset)
instance_type="ml.m5.xlarge"
output_path = f's3://{bucket_name}/output_nn/'

# Define the PyTorch estimator and pass hyperparameters as arguments
pytorch_estimator = PyTorch(
    entry_point="test_AWS/scripts/train_nn.py",
    role=role,
    instance_type=instance_type, # with this small dataset, we don't recessarily need a GPU for fast training. 
    instance_count=instance_count,  # Distributed training with two instances
    framework_version="1.9",
    py_version="py38",
    output_path=output_path,
    sagemaker_session=session,
    hyperparameters={
        "train": "/opt/ml/input/data/train/train_data.npz",  # SageMaker will mount this path
        "val": "/opt/ml/input/data/val/val_data.npz",        # SageMaker will mount this path
        "epochs": epochs,
        "learning_rate": 0.001
    }
)

# Define input paths
train_input = TrainingInput(f"s3://{bucket_name}/train_data.npz", content_type="application/x-npz")
val_input = TrainingInput(f"s3://{bucket_name}/val_data.npz", content_type="application/x-npz")

# Start the training job and time it
start = t.time()
pytorch_estimator.fit({"train": train_input, "val": val_input})
end = t.time()

print(f"Runtime for training on SageMaker: {end - start:.2f} seconds, instance_type: {instance_type}, instance_count: {instance_count}")

```
    
    2024-11-03 21:36:35 Uploading - Uploading generated training model
    2024-11-03 21:36:47 Completed - Training job completed
    Training seconds: 228
    Billable seconds: 228
    Runtime for training on SageMaker: 198.36 seconds, instance_type: ml.m5.xlarge, instance_count: 2


### Distributed Training for Neural Networks in SageMaker: Understanding Training Strategies and How Epochs Are Managed
Amazon SageMaker provides two main strategies for distributed training: **data parallelism** and **model parallelism**. Understanding which strategy will be used depends on the model size and the configuration of your SageMaker training job, as well as the default settings of the specific SageMaker Estimator you are using.

#### 1. **Data Parallelism (Most Common for Mini-batch SGD)**
   - **How it Works**: In data parallelism, each instance in the cluster (e.g., multiple `ml.m5.xlarge` instances) maintains a **complete copy of the model**. The **training dataset is split across instances**, and each instance processes a different subset of data simultaneously. This enables multiple instances to complete forward and backward passes on different data batches independently.
   - **Epoch Distribution**: Even though each instance processes all the specified epochs, they only work on a portion of the dataset for each epoch. After each batch, instances synchronize their gradient updates across all instances using a method such as *all-reduce*. This ensures that while each instance is working with a unique data batch, the model weights remain consistent across instances.
   - **Key Insight**: Because all instances process the specified number of epochs and synchronize weight updates between batches, each instance's training contributes to a cohesive, shared model. The **effective epoch count across instances appears to be shared** because data parallelism allows each instance to handle a fraction of the data per epoch, not the epochs themselves. Data parallelism is well-suited for models that can fit into a single instance's memory and benefit from increased data throughput.

#### 2. **Model Parallelism (Best for Large Models)**
   - **How it Works**: Model parallelism divides the model itself across multiple instances, not the data. This approach is best suited for very large models that cannot fit into a single GPU or instance's memory (e.g., large language models).
   - **Epoch Distribution**: The model is partitioned so that each instance is responsible for specific layers or components. Data flows sequentially through these partitions, where each instance processes a part of each batch and passes it to the next instance.
   - **Key Insight**: This approach is more complex due to the dependency between model components, so **synchronization occurs across the model layers rather than across data batches**. Model parallelism generally suits scenarios with exceptionally large model architectures that exceed memory limits of typical instances.

### Determining Which Distributed Training Strategy is Used
SageMaker will select the distributed strategy based on:
   - **Framework and Estimator Configuration**: Most deep learning frameworks in SageMaker default to data parallelism, especially when using PyTorch or TensorFlow with standard configurations.
   - **Model and Data Size**: If you specify a model that exceeds a single instance's memory capacity, SageMaker may switch to model parallelism if configured for it.
   - **Instance Count**: When you specify `instance_count > 1` in your Estimator with a deep learning model, SageMaker will use data parallelism by default unless explicitly configured for model parallelism.

You observed that each instance ran all epochs with `instance_count=2` and 10,000 epochs, which aligns with data parallelism. Here, each instance processed the full set of epochs independently, but each batch of data was different, and the gradient updates were synchronized across instances.


### Summary of Key Points
- **Data Parallelism** is the default distributed training strategy and splits the dataset across instances, allowing each instance to work on different data batches.
   - Each instance runs all specified epochs, but the weight updates are synchronized, so **epoch workload is shared across the data** rather than by reducing epoch count per instance.
- **Model Parallelism** splits the model itself across instances, typically only needed for very large models that exceed the memory capacity of single instances.
- **Choosing Between Distributed Strategies**: Data parallelism is suitable for most neural network models, especially those that fit in memory, while model parallelism is intended for exceptionally large models with memory constraints.

For cost optimization:
- **Single-instance training** is typically more cost-effective for small or moderately sized datasets, while **multi-instance setups** can reduce wall-clock time for larger datasets and complex models, at a higher instance cost.
- For **initial testing**, start with data parallelism on a single instance, and increase instance count if training time becomes prohibitive, while being mindful of communication overhead and scaling efficiency.


::::::::::::::::::::::::::::::::::::: keypoints

- **Environment Initialization**: Setting up a SageMaker session, defining roles, and specifying the S3 bucket are essential for managing data and running jobs in SageMaker.
- **Local vs. Managed Training**: Local training in SageMaker notebooks can be useful for quick tests but lacks the scalability and resource management available in SageMaker-managed training.
- **Estimator Classes**: SageMaker provides framework-specific Estimator classes (e.g., XGBoost, PyTorch, SKLearn) to streamline training setups, each suited to different model types and workflows.
- **Custom Scripts vs. Built-in Images**: Custom training scripts offer flexibility with preprocessing and custom logic, while built-in images are optimized for rapid deployment and simpler setups.
- **Training Data Channels**: Using `TrainingInput` ensures SageMaker manages data efficiently, especially for distributed setups where data needs to be synchronized across multiple instances.
- **Distributed Training Options**: Data parallelism (splitting data across instances) is common for many models, while model parallelism (splitting the model across instances) is useful for very large models that exceed instance memory.

::::::::::::::::::::::::::::::::::::::::::::::::

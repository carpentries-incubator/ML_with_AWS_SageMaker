---
title: "Hyperparameter Tuning in SageMaker: Neural Network Example"
teaching: 60
exercises: 0
---

:::::::::::::::::::::::::::::::::::::: questions 

- How can we efficiently manage hyperparameter tuning in SageMaker?
- How can we parallelize tuning jobs to optimize time without increasing costs?

::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: objectives

- Set up and run a hyperparameter tuning job in SageMaker.
- Define `ContinuousParameter` and `CategoricalParameter` for targeted tuning.
- Log and capture objective metrics for evaluating tuning success.
- Optimize tuning setup to balance cost and efficiency, including parallelization.

::::::::::::::::::::::::::::::::::::::::::::::::


To conduct efficient hyperparameter tuning with neural networks (or any model) in SageMaker, we’ll leverage SageMaker’s **hyperparameter tuning jobs** while carefully managing parameter ranges and model count. Here’s an overview of the process, with a focus on both efficiency and cost-effectiveness.

### Key steps for hyperparameter tuning
The overall process involves these five below steps.

1. Setup estimator
2. Define parameter ranges
3. Set up HyperParamterTuner object
4. Prepare training script to log metrics
5. Set data paths and launch tuner.fit()
6. Monitor tuning job from SageMaker console
7. Extract best model for final evaluation

#### Code example for SageMaker hyperparameter tuning with neural networks
We'll walk through each step in detail by tuning a neural network. Specifcially, we will test out different values for our `epochs` and `learning_rate` parameters. We are sticking to just two hyperparameters for demonstration purposes, but you may wish to explore additional parameters in your own work. 
 
This setup provides:
- **Explicit control** over `epochs` using `CategoricalParameter`, allowing targeted testing of specific values.
- **Efficient sampling** for `learning_rate` using `ContinuousParameter`, covering a defined range for a balanced approach.
- **Cost control** by setting moderate `max_jobs` and `max_parallel_jobs`.

By managing these settings and configuring metrics properly, you can achieve a balanced and efficient approach to hyperparameter tuning for neural networks.

#### 0. Directory setup
Just to make we are all on the same directory starting point, let's cd to our instance's root directory


```python
%cd /home/ec2-user/SageMaker/
```

    /home/ec2-user/SageMaker


#### 1. Setup estimator
To kick off our hyperparameter tuning for a neural network model, we’ll start by defining the **SageMaker Estimator**. The estimator setup here is very similar to our previous episode, where we used it to configure and train a model directly. However, this time, rather than directly running a training job with the estimator, we’ll be using it as the foundation for **hyperparameter tuning**.

In SageMaker, the estimator serves as a blueprint for each tuning job, specifying the training script, instance type, and key parameters like data paths and metrics. Once defined, this estimator will be passed to a **Hyperparameter Tuner** that manages the creation of multiple training jobs with various hyperparameter combinations, helping us find an optimal configuration.

Here’s the setup for our PyTorch estimator, which includes specifying the entry script for training (`train_nn.py`) and defining hyperparameters that will remain fixed across tuning jobs. The hyperparameters we’re setting up to tune include `epochs` and `learning_rate`, with a few specific values or ranges defined:




```python
import sagemaker
from sagemaker.tuner import HyperparameterTuner, IntegerParameter, ContinuousParameter, CategoricalParameter
from sagemaker.pytorch import PyTorch
from sagemaker.inputs import TrainingInput
from sagemaker import get_execution_role

# Initialize SageMaker session and role
session = sagemaker.Session()
role = get_execution_role()
bucket_name = 'myawesometeam-titanic'  # replace with your S3 bucket name

# Define the PyTorch estimator with entry script and environment details
pytorch_estimator = PyTorch(
    entry_point="AWS_helpers/train_nn.py",  # Your script for training
    role=role,
    instance_count=1,
    instance_type="ml.m5.large",
    framework_version="1.9",
    py_version="py38",
    metric_definitions=[{"Name": "validation:accuracy", "Regex": "validation:accuracy = ([0-9\\.]+)"}],
    hyperparameters={
        "train": "/opt/ml/input/data/train/train_data.npz",  # SageMaker will mount this path
        "val": "/opt/ml/input/data/val/val_data.npz",        # SageMaker will mount this path
        "epochs": 100, # Placeholder initial value. Will be overridden by tuning by tuning values tested
        "learning_rate": 0.001 # Placeholder initial value. Will be overridden by tuning values tested
    },
    sagemaker_session=session,
)


```

#### 2. Define hyperparameter ranges

In SageMaker, you must explicitly define ranges for any hyperparameters you want to tune. SageMaker supports both `ContinuousParameter` and `CategoricalParameter` types:  

   - **`ContinuousParameter`** allows SageMaker to dynamically sample numeric values within a specified range, making it ideal for broad, exploratory tuning. The total number of values tested can be controlled through the upcoming `max_jobs` parameter, which defines how many different combinations SageMaker will evaluate.
   - **`CategoricalParameter`** specifies exact values for SageMaker to test, which is useful when you want the model to only try a specific set of options.

By default, SageMaker uses **Bayesian optimization**, adjusting future selections based on previous results to efficiently find optimal values. You can also set the `strategy` to **"Random"** for uniform sampling across the range, which is effective in larger or more exploratory search spaces. Random sampling may end up costing much more in time and resources, however. Generally, we recommend sticking with the default setting.



```python
# Hyperparameter tuning ranges
hyperparameter_ranges = {
    "epochs": CategoricalParameter([100, 1000, 10000]),       # Adjust as needed
    "learning_rate": ContinuousParameter(0.001, 0.1),  # Range for continuous values
}
```

#### Hyperparameter considerations in neural nets
When tuning hyperparameters in neural networks, it's essential to prioritize parameters that directly impact model performance while remaining mindful of diminishing returns and potential overfitting. Below are some core hyperparameters to consider and general strategies for tuning:

- **Learning Rate**: Often the most impactful parameter, learning rate controls the speed of convergence. A smaller learning rate can lead to more stable, though slower, convergence, while a larger rate may speed things up but risks overshooting optimal values. Testing ranges like `0.0001` to `0.1` with a `ContinuousParameter` is common practice, and `Bayesian` search can help home in on ideal values.

- **Batch Size**: Larger batch sizes often yield faster training times and may improve stability, but this can risk bypassing useful local minima, especially with small datasets. Smaller batch sizes can capture more nuanced updates but are computationally expensive. Ranges from `16` to `256` are worth exploring for most use cases, although, for large datasets or high-capacity models, even larger values may be beneficial.

- **Number of Epochs**: While larger epochs allow the model to learn from data longer, increasing epochs doesn't always equate to better performance and can lead to overfitting. Exploring `CategoricalParameter([50, 100, 500, 1000])` can help balance performance without excessive training costs. 

- **Layer Width and Depth**: Increasing the width or depth (i.e., number of neurons and layers) can improve model capacity but also risks overfitting if the dataset is small or lacks variability. Testing a range of layer sizes or depths (e.g., `32, 64, 128` neurons per layer) can reveal whether additional complexity yields benefits. Notably, understanding *double descent* is essential here, as larger networks may initially seem to overfit before unexpectedly improving in the *second descent*—a phenomenon worth investigating in high-capacity networks.

- **Regularization Parameters**: Regularization techniques, such as dropout rates or weight decay, can help control overfitting by limiting model capacity. For example, dropout rates from `0.1` to `0.5` or weight decay values of `0.0001` to `0.01` often balance underfitting and overfitting effectively. Higher regularization might inhibit learning, especially if the model is relatively small.

- **Early Stopping**: While not a traditional hyperparameter, setting up early stopping based on the validation performance can prevent overfitting without the need to exhaustively test for epoch limits. By allowing the model to halt once performance plateaus or worsens, early stopping can improve efficiency in hyperparameter tuning.

- **Special Phenomena - Grokking and Double Descent**: For certain complex datasets or when tuning particularly large models, keep an eye on phenomena like *grokking* (sudden shifts from poor to excellent generalization) and *double descent* (an unexpected second drop in error after initial overfitting). These behaviors are more likely to appear in models with high capacity and prolonged training periods, potentially requiring longer epochs or lower learning rates to observe. 

In summary, hyperparameter tuning is a balancing act between expanding model capacity and mitigating overfitting. Prioritize parameters that have shown past efficacy in similar problems, and limit the search to a manageable range—often 20–30 model configurations are sufficient to observe gains. This approach keeps resource consumption practical while achieving meaningful improvements in model performance.


#### 3. Set up HyperParamterTuner object
In step 3, we set up the `HyperparameterTuner`, which controls the tuning process by specifying the...

- `estimator`: Here, we connect the previously defined `pytorch_estimator` to `tuner`, ensuring that the tuning job will run with our PyTorch model configuration.
- `objectives`:
- The `metric_definitions` and `objective_metric_name` indicate which metric SageMaker should monitor to find the best-performing model; in this case, we’re tracking "validation:accuracy" and aiming to maximize it. We'll show you how to setup your training script to report this information in the next step.
- `hyperparameter ranges`: Defined above.
- `tuning strategy`: SageMaker uses a **Bayesian strategy** by default, which iteratively improves parameter selection based on past performance to find an optimal model more efficiently. Although it’s possible to adjust to a "Random" strategy, Bayesian optimization generally provides better results, so it’s recommended to keep this setting. 
- `max_jobs` and `max_parallel_jobs`: Finally, we set `max_jobs` to control the total number of configurations SageMaker will explore and `max_parallel_jobs` to limit the number of jobs that run simultaneously, balancing resource usage and tuning speed. Since SageMaker tests different hyperparameter values dynamically, it's important to limit total parallel instances to <= 4. 


> **Resource-Conscious Approach**: To control costs and energy-needs, choose efficient instance types and limit the search to impactful parameters at a sensible range, keeping resource consumption in check. Hyperparameter tuning often does yield better performing models, but these gains can be marginal after exhausting a reasonable search window of 20-30 model configurations. As a researcher, it's also imortant to do some digging on past work to see which parameters may be worthwhile to explore. Make sure you understand what each parameter is doing before you adjust them. 

**Always start with `max_jobs = 1` and `max_parallel_jobs=1`.**
Before running the full search, let's test our setup by setting max_jobs = 1. This will test just one possible hyperparameter configuration. This critical step helps ensure our code is functional before attempting to scale up. 


```python
# Tuner configuration
tuner = HyperparameterTuner(
    estimator=pytorch_estimator,
    metric_definitions=[{"Name": "validation:accuracy", "Regex": "validation:accuracy = ([0-9\\.]+)"}],
    objective_metric_name="validation:accuracy",  # Ensure this matches the metric name exactly
    objective_type="Maximize",                   # Specify if maximizing or minimizing the metric
    hyperparameter_ranges=hyperparameter_ranges,
    strategy="Bayesian",  # Default setting (recommend sticking with this!); can adjust to "Random" for uniform sampling across the range
    max_jobs=1,                # Always start with 1 instance for debugging purposes. Adjust based on exploration needs (keep below 30 to be kind to environment). 
    max_parallel_jobs=1         # Always start with 1 instance for debugging purposes. Adjust based on available resources and budget. Recommended to keep this value < 4 since SageMaker tests values dynamically.
)

```

#### 4. Prepare training script to log metrics
To prepare `train_nn.py` for hyperparameter tuning, we added code to log validation metrics in a format that SageMaker recognizes for tracking. In the training loop, we added a print statement for `Val Accuracy` in a specific format that SageMaker can capture. 

**Note**: It's best to use an if statement to only print out metrics periodically (e.g., every 100 epochs), so that you print time does not each up too much of your training time. It may be a little counter-intuitive that printing can slow things down so dramatically, but it truly does become a significant factor if you're doing it every epoch. On the flipside of this, you don't want to print metrics so infrequently that you lose resolution in the monitored validation accuracy. Choose a number between 100-1000 epochs or divide your total epoch count by ~25 to yield a reasonable range. 

```python
if (epoch + 1) % 100 == 0 or epoch == epochs - 1:
    print(f"validation:accuracy = {val_accuracy:.4f}", flush=True)  # Log for SageMaker metric tracking. Needed for hyperparameter tuning later.
```

Paired with this, our `metric_definitions` above uses a regular expression `"validation:accuracy = ([0-9\\.]+)"` to extract the val_accuracy value from each log line. This regex specifically looks for validation:accuracy =, followed by a floating-point number, which corresponds to the format of our log statement in train_nn.py.

#### 5. Set data paths and launch tuner.fit()
In step 4, we define the input data paths for the training job and launch the hyperparameter tuning process. Using `TrainingInput`, we specify the S3 paths to our `train_data.npz` and `val_data.npz` files. This setup ensures that SageMaker correctly accesses our training and validation data during each job in the tuning process. We then call `tuner.fit` and pass a dictionary mapping each data channel ("train" and "val") to its respective path. This command initiates the tuning job, triggering SageMaker to begin sampling hyperparameter combinations, training the model, and evaluating performance based on our defined objective metric. Once the job is launched, SageMaker handles the tuning process automatically, running the specified number of configurations and keeping track of the best model parameters based on validation accuracy.



```python
# Define the input paths
train_input = TrainingInput(f"s3://{bucket_name}/train_data.npz", content_type="application/x-npz")
val_input = TrainingInput(f"s3://{bucket_name}/val_data.npz", content_type="application/x-npz")

# Launch the hyperparameter tuning job
tuner.fit({"train": train_input, "val": val_input})
print("Hyperparameter tuning job launched.")

```

    No finished training job found associated with this estimator. Please make sure this estimator is only used for building workflow config
    No finished training job found associated with this estimator. Please make sure this estimator is only used for building workflow config


    .......................................!
    Hyperparameter tuning job launched.


#### 6. Monitor tuning job from SageMaker console
After running the above cell, we can check on the progress by visiting the SageMaker Console and finding the "Training" tab located on the left panel. Click "Hyperparmater tuning jobs" to view running jobs.


### Scaling up our approach
If all goes well, we can scale up the experiment with the below code. In this configuration, we’re scaling up the search by allowing SageMaker to test more hyperparameter configurations (`max_jobs=20`) while setting `max_parallel_jobs=2` to manage parallelization efficiently. With two jobs running at once, SageMaker will be able to explore potential improvements more quickly than in a fully sequential setup, while still dynamically selecting promising values as it learns from completed jobs. This balance leverages SageMaker’s Bayesian optimization, which uses completed trials to inform subsequent ones, helping to avoid redundant testing of less promising parameter combinations. **Setting `max_parallel_jobs` higher than 2-4 could increase costs and reduce tuning efficiency, as SageMaker’s ability to learn from completed jobs decreases when too many jobs run simultaneously.**

With this approach, SageMaker is better able to refine configurations without overloading resources or risking inefficient exploration, making `max_parallel_jobs=2` a solid default for most use cases.


```python
import time as t # always a good idea to keep a runtime of your experiments 

# Configuration variables
instance_type = "ml.m5.large"
max_jobs = 2
max_parallel_jobs = 2

# Define the Tuner configuration
tuner = HyperparameterTuner(
    estimator=pytorch_estimator,
    metric_definitions=[{"Name": "validation:accuracy", "Regex": "validation:accuracy = ([0-9\\.]+)"}],
    objective_metric_name="validation:accuracy",  # Ensure this matches the metric name exactly
    objective_type="Maximize",                   # Specify if maximizing or minimizing the metric
    hyperparameter_ranges=hyperparameter_ranges,
    max_jobs=max_jobs,
    max_parallel_jobs=max_parallel_jobs
)

# Define the input paths
train_input = TrainingInput(f"s3://{bucket_name}/train_data.npz", content_type="application/x-npz")
val_input = TrainingInput(f"s3://{bucket_name}/val_data.npz", content_type="application/x-npz")

# Track start time
start_time = t.time()

# Launch the hyperparameter tuning job
tuner.fit({"train": train_input, "val": val_input})

# Calculate runtime
runtime = t.time() - start_time

# Print confirmation with runtime and configuration details
print(f"Tuning runtime: {runtime:.2f} seconds, Instance Type: {instance_type}, Max Jobs: {max_jobs}, Max Parallel Jobs: {max_parallel_jobs}")

```

    No finished training job found associated with this estimator. Please make sure this estimator is only used for building workflow config
    No finished training job found associated with this estimator. Please make sure this estimator is only used for building workflow config


    .......................................!
    Tuning runtime: 205.53 seconds, Instance Type: ml.m5.large, Max Jobs: 2, Max Parallel Jobs: 2


### Monitoring tuning
After running the above cell, we can check on the progress by visiting the SageMaker Console and finding the "Training" tab located on the left panel. Click "Hyperparmater tuning jobs" to view running jobs.

* Initial Jobs: SageMaker starts by running only max_parallel_jobs (2 in this case) as the initial batch. As each job completes, new jobs from the remaining pool are triggered until max_jobs (20) is reached.
* Job Completion: Once the first few jobs complete, SageMaker will continue to launch the remaining jobs up to the maximum of 20, but no more than two at a time.

### Can/should we run more instances in parallel?
Setting max_parallel_jobs to 20 (equal to max_jobs) will indeed launch all 20 jobs in parallel. This approach won’t affect the total cost (since cost is based on the number of total jobs, not how many run concurrently), but it can impact the final results and resource usage pattern due to SageMaker's ability to dynamically select hyperparameter values to test to maximize efficiency and improve model performance. This adaptability is especially useful for neural networks, which often have a large hyperparameter space with complex interactions. Here’s how SageMaker’s approach impacts typical neural network training:

#### 1. Adaptive Search Strategies
   - SageMaker offers **Bayesian optimization** for hyperparameter tuning. Instead of purely random sampling, it learns from previous jobs to choose the next set of hyperparameters more likely to improve the objective metric.
   - For neural networks, this strategy can help converge on better-performing configurations faster by favoring promising areas of the hyperparameter space and discarding poor ones.

#### 2. Effect of `max_parallel_jobs` on adaptive tuning
   - When using Bayesian optimization, a lower `max_parallel_jobs` (e.g., 2–4) can allow SageMaker to iteratively adjust and improve its choices. Each batch of jobs informs the subsequent batch, which may yield better results over time.
   - Conversely, if all jobs are run in parallel (e.g., `max_parallel_jobs=20`), SageMaker can’t learn and adapt within a single batch, making this setup more like a traditional grid or random search. This approach is still valid, especially for small search spaces, but it doesn’t leverage the full potential of adaptive tuning.

#### 3. Practical impact on neural network training
   - **For simpler models** or smaller parameter ranges, running jobs in parallel with a higher `max_parallel_jobs` works well and quickly completes the search.
   - **For more complex neural networks** or large hyperparameter spaces, an adaptive strategy with a smaller `max_parallel_jobs` may yield a better model with fewer total jobs by fine-tuning hyperparameters over multiple iterations.

#### Summary
- **For fast, straightforward tuning**: Set `max_parallel_jobs` closer to `max_jobs` for simultaneous testing.
- **For adaptive, refined tuning**: Use a smaller `max_parallel_jobs` (like 2–4) to let SageMaker leverage adaptive tuning for optimal configurations. 

This balance between exploration and exploitation is particularly impactful in neural network tuning, where training costs can be high and parameters interact in complex ways.


### Extracting and evaluating the best model after tuning

Tuning should only take about 5 minutes to complete — not bad for 20 models! After SageMaker completes the hyperparameter tuning job, the results, including the trained models for each configuration, are stored in an S3 bucket. Here’s a breakdown of the steps to locate and evaluate the best model on test data.


1. **Understanding the folder structure**:
   - SageMaker saves each tuning job's results in the specified S3 bucket under a unique prefix.
   - For the best model, SageMaker stores the model artifact in the format `s3://{bucket}/{job_name}/output/model.tar.gz`. Each model is compressed as a `.tar.gz` file containing the saved model parameters.

2. **Retrieve and load the best model**:
   - Using the `tuner.best_training_job()` method, you can get the name of the best-performing job.
   - From there, retrieve the S3 URI of the best model artifact, download it locally, and extract the files for use.

3. **Prepare test data for final assessment of model generalizability**
    - If not done already.

4. **Evaluate the model on test data**:
   - Once extracted, load the saved model weights and evaluate the model on your test dataset to get the final performance metrics.


Here's the code to implement these steps:

#### View best model details and storage info
We can easily view the best hyperparameters from the tuning procedure...


```python
# 1. Get the best training job from the completed tuning job
best_job_name = tuner.best_training_job()
print("Best training job name:", best_job_name)

# 2. Use describe_training_job to retrieve full details, including hyperparameters...
best_job_desc = session.sagemaker_client.describe_training_job(TrainingJobName=best_job_name)
best_hyperparameters = best_job_desc["HyperParameters"]
print("Best hyperparameters:", best_hyperparameters)

# ...  and model URI (location on S3)
best_model_s3_uri = best_job_desc['ModelArtifacts']['S3ModelArtifacts']
print(f"Best model artifact S3 URI: {best_model_s3_uri}")

```

    Best training job name: pytorch-training-241107-0025-001-72851d7f
    Best hyperparameters: {'_tuning_objective_metric': 'validation:accuracy', 'epochs': '"100"', 'learning_rate': '0.005250489250786233', 'sagemaker_container_log_level': '20', 'sagemaker_estimator_class_name': '"PyTorch"', 'sagemaker_estimator_module': '"sagemaker.pytorch.estimator"', 'sagemaker_job_name': '"pytorch-training-2024-11-07-00-25-35-999"', 'sagemaker_program': '"train_nn.py"', 'sagemaker_region': '"us-east-1"', 'sagemaker_submit_directory': '"s3://sagemaker-us-east-1-183295408236/pytorch-training-2024-11-07-00-25-35-999/source/sourcedir.tar.gz"', 'train': '"/opt/ml/input/data/train/train_data.npz"', 'val': '"/opt/ml/input/data/val/val_data.npz"'}
    Best model artifact S3 URI: s3://sagemaker-us-east-1-183295408236/pytorch-training-241107-0025-001-72851d7f/output/model.tar.gz


#### Retrieve and load best model


```python
import boto3
import tarfile

# Initialize S3 client
s3 = boto3.client('s3')

# Download and extract the model artifact
local_model_path = "best_model.tar.gz"
bucket_name, model_key = best_model_s3_uri.split('/')[2], '/'.join(best_model_s3_uri.split('/')[3:])
s3.download_file(bucket_name, model_key, local_model_path)

# Extract the model files from the tar.gz archive
with tarfile.open(local_model_path, 'r:gz') as tar:
    tar.extractall()
print("Best model downloaded and extracted.")
```

    Best model downloaded and extracted.


#### Prepare test set as test_data.npz
In our previous episode, we converted our train dataset into train/validate subsets, and saved them out as .npz files for efficient processing. We'll need to preprocess our test data the same way to evaluate it on our model. 

**Note**: It's always a good idea to keep preprocessing code as a function so you can apply the same exact procedure across datasets with ease. We'll import our preprocessing function from `train_nn.py`.


```python
import pandas as pd
import numpy as np

# Now try importing the function again
from AWS_helpers.train_nn import preprocess_data

# Example usage for test data (using the same scaler from training)
test_df = pd.read_csv("titanic_test.csv")
X_test, y_test, _ = preprocess_data(test_df)

# Save processed data for testing
np.savez('test_data.npz', X_test=X_test, y_test=y_test)
```

#### Evaluate the model on test data


```python
from AWS_helpers.train_nn import TitanicNet
from AWS_helpers.train_nn import calculate_accuracy
import torch

# Load the model (assuming it's saved as 'nn_model.pth' after extraction)
model = TitanicNet()  # Ensure TitanicNet is defined as per your training script
model.load_state_dict(torch.load("nn_model.pth"))
model.eval()

# Load test data (assuming the test set is saved as "test_data.npz" in npz format)
test_data = np.load("test_data.npz")  # Replace "test_data.npz" with actual test data path if different
X_test = torch.tensor(test_data['X_test'], dtype=torch.float32)
y_test = torch.tensor(test_data['y_test'], dtype=torch.float32)

# Evaluate the model on the test set
with torch.no_grad():
    predictions = model(X_test)
    accuracy = calculate_accuracy(predictions, y_test)  # Assuming calculate_accuracy is defined as in your training script
    print(f"Test Accuracy: {accuracy:.4f}")

```

    Test Accuracy: 98.0894


### Conclusions
In just under 5 minutes, we produced a model that is almost 100% accurate on the test set. However, this performance does come at a cost (albeit manageable if you've stuck with our advise thus far). This next section will help you assess the total compute time that was used by your tuning job.

In SageMaker, training time and billing time (extracted in our code below) are often expected to differ slightly for training jobs, but not for tuning jobs. Here’s a breakdown of what’s happening:

* Training Time: This is the actual wall-clock time that each training job takes to run, from start to end. This metric represents the pure time spent on training without considering the compute resources.

* Billing Time: This includes the training time but is adjusted for the resources used. Billing time considers:

Instance Count: The number of instances used for training affects billing time.
Round-Up Policy: SageMaker rounds up the billing time to the nearest second for each job and multiplies it by the instance count used. This means that for short jobs, the difference between training and billing time can be more pronounced.


```python
import boto3
import math

# Initialize SageMaker client
sagemaker_client = boto3.client("sagemaker")

# Retrieve tuning job details
tuning_job_name = tuner.latest_tuning_job.name  # Replace with your tuning job name if needed
tuning_job_desc = sagemaker_client.describe_hyper_parameter_tuning_job(HyperParameterTuningJobName=tuning_job_name)

# Extract relevant settings
instance_type = tuning_job_desc['TrainingJobDefinition']['ResourceConfig']['InstanceType']
max_jobs = tuning_job_desc['HyperParameterTuningJobConfig']['ResourceLimits']['MaxNumberOfTrainingJobs']
max_parallel_jobs = tuning_job_desc['HyperParameterTuningJobConfig']['ResourceLimits']['MaxParallelTrainingJobs']

# Retrieve all training jobs for the tuning job
training_jobs = sagemaker_client.list_training_jobs_for_hyper_parameter_tuning_job(
    HyperParameterTuningJobName=tuning_job_name, StatusEquals='Completed'
)["TrainingJobSummaries"]

# Calculate total training and billing time
total_training_time = 0
total_billing_time = 0

for job in training_jobs:
    job_name = job["TrainingJobName"]
    job_desc = sagemaker_client.describe_training_job(TrainingJobName=job_name)
    
    # Calculate training time (in seconds)
    training_time = job_desc["TrainingEndTime"] - job_desc["TrainingStartTime"]
    total_training_time += training_time.total_seconds()
    
    # Calculate billed time with rounding up
    billed_time = math.ceil(training_time.total_seconds())
    total_billing_time += billed_time * job_desc["ResourceConfig"]["InstanceCount"]

# Print configuration details and total compute/billing time
print(f"Instance Type: {instance_type}")
print(f"Max Jobs: {max_jobs}")
print(f"Max Parallel Jobs: {max_parallel_jobs}")
print(f"Total training time across all jobs: {total_training_time / 3600:.2f} hours")
print(f"Estimated total billing time across all jobs: {total_billing_time / 3600:.2f} hours")

```

    Instance Type: ml.m5.large
    Max Jobs: 2
    Max Parallel Jobs: 2
    Total training time across all jobs: 0.07 hours
    Estimated total billing time across all jobs: 0.07 hours


For convenience, we have added this as a function in helpers.py


```python
import AWS_helpers.helpers as helpers
import importlib
importlib.reload(helpers)
helpers.calculate_tuning_job_time(tuner)

```

    Instance Type: ml.m5.large
    Max Jobs: 2
    Max Parallel Jobs: 2
    Total training time across all jobs: 0.07 hours
    Estimated total billing time across all jobs: 0.07 hours



```python
!jupyter nbconvert --to markdown Hyperparameter-tuning.ipynb

```

    [NbConvertApp] Converting notebook Hyperparameter-tuning.ipynb to markdown
    [NbConvertApp] Writing 31418 bytes to Hyperparameter-tuning.md



```python

```

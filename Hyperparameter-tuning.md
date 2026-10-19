---
title: "Hyperparameter Tuning in SageMaker: Neural Network Example"
teaching: 45
exercises: 15
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
- Read back what a whole tuning job cost from its child training jobs.

::::::::::::::::::::::::::::::::::::::::::::::::

## Initial setup: open prefilled .ipynb notebook
Open the notebook from: `/ML_with_AWS_SageMaker/notebooks/Hyperparameter-tuning.ipynb`. Select the **`conda_pytorch_p310`** kernel.

## Hyperparameter tuning in SageMaker
To conduct efficient hyperparameter tuning with neural networks (or any model) in SageMaker, we'll leverage SageMaker's **hyperparameter tuning jobs** while carefully managing parameter ranges and model count. Here's an overview of the process, with a focus on both efficiency and cost-effectiveness.

### Key steps 
The overall process involves these seven steps.

1. Setup estimator
2. Define parameter ranges
3. Set up HyperparameterTuner object
4. Prepare training script to log metrics
5. Set data paths and launch tuner.fit()
6. Monitor tuning job from SageMaker console
7. Extract best model for final evaluation

#### Code example for SageMaker hyperparameter tuning with neural networks
We'll walk through each step in detail by tuning a neural network. Specifically, we will test out different values for our `epochs` and `learning_rate` parameters. We are sticking to just two hyperparameters for demonstration purposes, but you may wish to explore additional parameters in your own work. 
 
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

#### 1. Setup estimator
To kick off our hyperparameter tuning for a neural network model, we'll start by defining the **SageMaker Estimator**. The estimator setup here is very similar to our previous episode, where we used it to configure and train a model directly. However, this time, rather than directly running a training job with the estimator, we'll be using it as the foundation for **hyperparameter tuning**.

In SageMaker, the estimator serves as a blueprint for each tuning job, specifying the training script, instance type, and key parameters like data paths and metrics. Once defined, this estimator will be passed to a **Hyperparameter Tuner** that manages the creation of multiple training jobs with various hyperparameter combinations, helping us find an optimal configuration.

Here's the setup for our PyTorch estimator, which includes specifying the entry script for training (`train_nn.py`) and defining hyperparameters that will remain fixed across tuning jobs. The hyperparameters we're setting up to tune include `epochs` and `learning_rate`, with a few specific values or ranges defined.

We'll use our notebook instance name again to label the training jobs launched in this episode

```python
import boto3

notebook_instance_name = 'sinkorswim-DoeJohn-TrainClassifier' # adjust to your notebook name. we'll use this variable to name the training jobs launched by the tuner.

# the following code just verifies you have the right name for your notebook instance
region = boto3.Session().region_name  # inherited from the notebook instance; do not hard-code a region
sagemaker_client = boto3.client('sagemaker', region_name=region) # Initialize SageMaker client
response = sagemaker_client.describe_notebook_instance(NotebookInstanceName=notebook_instance_name) # Describe the notebook instance

# Display the status and instance type
print(f"Notebook Instance '{notebook_instance_name}' status: {response['NotebookInstanceStatus']}")
local_instance = response['InstanceType']
print(f"Instance Type: {local_instance}")
```

**Cost tracking**: Just like in the previous episodes, we explicitly tag our work at launch time so that compute usage is traceable to a project, a purpose, and a human-readable name. Tuning is the one case where tags *do* propagate: tags on the `HyperparameterTuner` are copied to every child training job it launches, so we only need to set them once, on the tuner (we also set them on the estimator so a plain `.fit()` on it is tagged too).

```python
name = "John Doe" # replace with your name
project = "sinkorswim" # replace with your team name
purpose = "tune_PyTorch"

job_tags = [
    {"Key": "Name", "Value": name},
    {"Key": "Project", "Value": project},
    {"Key": "Purpose", "Value": purpose},
]
```

Next, we'll configure the baseline estimator that we plan to do hyperparameter search on.

```python
import sagemaker
from sagemaker.tuner import HyperparameterTuner, IntegerParameter, ContinuousParameter, CategoricalParameter
from sagemaker.pytorch import PyTorch
from sagemaker.inputs import TrainingInput
from sagemaker import get_execution_role

# Initialize role, bucket, and SageMaker session variables
role = get_execution_role()
bucket_name = 'sinkorswim-doejohn'  # replace with your S3 bucket name
session = sagemaker.Session()
s3 = boto3.client('s3')

# Define instance type/count we'll use for training
instance_type="ml.m5.large"
instance_count=1 # always start with 1; tuning parallelism comes from max_parallel_jobs, not instance_count
price_per_hour = 0.115 # ml.m5.large, us-east-1 on-demand (2026-10-16); check the instances page for your region

# Define max runtime in seconds to ensure you don't use more compute time than expected. Use a generous threshold (2x expected time but < 2 days) so that work isn't interrupted without any gains.
max_run = 2*60*60 # 2 hours

# Define the PyTorch estimator with entry script and environment details
pytorch_estimator = PyTorch(
    base_job_name=notebook_instance_name, # adjust to your notebook name,
    max_run=max_run, # in seconds; always include (max 48 hours)
    entry_point="AWS_helpers/train_nn.py",  # Your script for training
    role=role,
    tags=job_tags,
    instance_count=instance_count,
    instance_type=instance_type,
    framework_version="2.3",  # if a job fails with an image error, check with: sagemaker.image_uris.retrieve("pytorch", region, version="2.3", py_version="py311", image_scope="training", instance_type="ml.m5.large")
    py_version="py311",
    metric_definitions=[{"Name": "validation:accuracy", "Regex": "validation:accuracy = ([0-9\\.]+)"}],
    hyperparameters={
        "train": "/opt/ml/input/data/train/train_data.npz",  # SageMaker will mount this path
        "val": "/opt/ml/input/data/val/val_data.npz",        # SageMaker will mount this path
        "epochs": 100, # Placeholder initial value. Will be overridden by the tuning values tested
        "learning_rate": 0.001 # Placeholder initial value. Will be overridden by the tuning values tested
    },
    sagemaker_session=session,
)
```

#### 2. Define hyperparameter ranges

In SageMaker, you must explicitly define ranges for any hyperparameters you want to tune. SageMaker supports `ContinuousParameter`, `IntegerParameter` and `CategoricalParameter` types:  

   - **`ContinuousParameter`** allows SageMaker to dynamically sample numeric values within a specified range, making it ideal for broad, exploratory tuning. The total number of values tested can be controlled through the upcoming `max_jobs` parameter, which defines how many different combinations SageMaker will evaluate.
   - **`IntegerParameter`** does the same for whole-number settings such as a batch size or layer width.
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

- **Layer Width and Depth**: Increasing the width or depth (i.e., number of neurons and layers) can improve model capacity but also risks overfitting if the dataset is small or lacks variability. Testing a range of layer sizes or depths (e.g., `32, 64, 128` neurons per layer) can reveal whether additional complexity yields benefits. Notably, understanding *double descent* is essential here, as larger networks may initially seem to overfit before unexpectedly improving in the *second descent*, a phenomenon worth investigating in high-capacity networks.

- **Regularization Parameters**: Regularization techniques, such as dropout rates or weight decay, can help control overfitting by limiting model capacity. For example, dropout rates from `0.1` to `0.5` or weight decay values of `0.0001` to `0.01` often balance underfitting and overfitting effectively. Higher regularization might inhibit learning, especially if the model is relatively small.

- **Early Stopping**: While not a traditional hyperparameter, setting up early stopping based on the validation performance can prevent overfitting without the need to exhaustively test for epoch limits. By allowing the model to halt once performance plateaus or worsens, early stopping can improve efficiency in hyperparameter tuning.

- **Special Phenomena - Grokking and Double Descent**: For certain complex datasets or when tuning particularly large models, keep an eye on phenomena like *grokking* (sudden shifts from poor to excellent generalization) and *double descent* (an unexpected second drop in error after initial overfitting). These behaviors are more likely to appear in models with high capacity and prolonged training periods, potentially requiring longer epochs or lower learning rates to observe. 

In summary, hyperparameter tuning is a balancing act between expanding model capacity and mitigating overfitting. Prioritize parameters that have shown past efficacy in similar problems, and limit the search to a manageable range; often 20-30 model configurations are sufficient to observe gains. This approach keeps resource consumption practical while achieving meaningful improvements in model performance.

#### 3. Set up HyperparameterTuner object
In step 3, we set up the `HyperparameterTuner`, which controls the tuning process by specifying the...

- `estimator`: Here, we connect the previously defined `pytorch_estimator` to `tuner`, ensuring that the tuning job will run with our PyTorch model configuration.
- `objective`: The `metric_definitions` and `objective_metric_name` indicate which metric SageMaker should monitor to find the best-performing model; in this case, we're tracking "validation:accuracy" and aiming to maximize it. We'll show you how to setup your training script to report this information in the next step.
- `hyperparameter_ranges`: Defined above.
- `strategy`: SageMaker uses a **Bayesian strategy** by default, which iteratively improves parameter selection based on past performance to find an optimal model more efficiently. Although it's possible to adjust to a "Random" strategy, Bayesian optimization generally provides better results, so it's recommended to keep this setting. 
- `max_jobs` and `max_parallel_jobs`: Finally, we set `max_jobs` to control the total number of configurations SageMaker will explore and `max_parallel_jobs` to limit the number of jobs that run simultaneously, balancing resource usage and tuning speed. Since SageMaker tests different hyperparameter values dynamically, it's important to limit total parallel jobs to <= 4. 

> **Resource-Conscious Approach**: To control costs and energy-needs, choose efficient instance types and limit the search to impactful parameters at a sensible range, keeping resource consumption in check. Hyperparameter tuning often does yield better performing models, but these gains can be marginal after exhausting a reasonable search window of 20-30 model configurations. As a researcher, it's also important to do some digging on past work to see which parameters may be worthwhile to explore. Make sure you understand what each parameter is doing before you adjust them. 

**Always start with `max_jobs = 1` and `max_parallel_jobs=1`.**
Before running the full search, let's test our setup by setting max_jobs = 1. This will test just one possible hyperparameter configuration. This critical step helps ensure our code is functional before attempting to scale up. 

```python
# Tuner configuration
tuner = HyperparameterTuner(
    base_tuning_job_name=notebook_instance_name,
    estimator=pytorch_estimator,
    metric_definitions=[{"Name": "validation:accuracy", "Regex": "validation:accuracy = ([0-9\\.]+)"}],
    objective_metric_name="validation:accuracy",  # Ensure this matches the metric name exactly
    objective_type="Maximize",                   # Specify if maximizing or minimizing the metric
    hyperparameter_ranges=hyperparameter_ranges,
    strategy="Bayesian",  # Default setting (recommend sticking with this!); can adjust to "Random" for uniform sampling across the range
    max_jobs=1,                # Always start with 1 job for debugging purposes. Adjust based on exploration needs (keep below 30 to be kind to environment). 
    max_parallel_jobs=1,         # Always start with 1 for debugging purposes. Recommended to keep this value <= 4 since SageMaker tests values dynamically.
    tags=job_tags,               # propagated to every training job the tuner launches
)
```

#### 4. Prepare training script to log metrics
To prepare `train_nn.py` for hyperparameter tuning, we added code to log validation metrics in a format that SageMaker recognizes for tracking. In the training loop, we added a print statement for the validation accuracy in a specific format that SageMaker can capture. 

**Note**: It's best to use an if statement to only print out metrics periodically (e.g., every 100 epochs), so that your print time does not eat up too much of your training time. It may be a little counter-intuitive that printing can slow things down so dramatically, but it truly does become a significant factor if you're doing it every epoch. On the flipside of this, you don't want to print metrics so infrequently that you lose resolution in the monitored validation accuracy. Choose a number between 100-1000 epochs or divide your total epoch count by ~25 to yield a reasonable range. 

```text
if (epoch + 1) % 100 == 0 or epoch == epochs - 1:
    print(f"validation:accuracy = {val_accuracy:.4f}", flush=True)  # Log for SageMaker metric tracking
```

Paired with this, our `metric_definitions` above uses a regular expression `"validation:accuracy = ([0-9\\.]+)"` to extract the val_accuracy value from each log line. This regex specifically looks for `validation:accuracy =`, followed by a floating-point number, which corresponds to the format of our log statement in `train_nn.py`.

#### 5. Set data paths and launch tuner.fit()
In step 5, we define the input data paths for the training job and launch the hyperparameter tuning process. Using `TrainingInput`, we specify the S3 paths to our `train_data.npz` and `val_data.npz` files (uploaded in the previous episode). This setup ensures that SageMaker correctly accesses our training and validation data during each job in the tuning process. We then call `tuner.fit` and pass a dictionary mapping each data channel ("train" and "val") to its respective path. This command initiates the tuning job, triggering SageMaker to begin sampling hyperparameter combinations, training the model, and evaluating performance based on our defined objective metric. Once the job is launched, SageMaker handles the tuning process automatically, running the specified number of configurations and keeping track of the best model parameters based on validation accuracy.

Estimate first, as always: one child job is the same ~1 minute of training plus ~3 minutes of startup we saw in the previous episode, so about $0.01.

```python
# Define the input paths
train_input = TrainingInput(f"s3://{bucket_name}/train_data.npz", content_type="application/x-npz")
val_input = TrainingInput(f"s3://{bucket_name}/val_data.npz", content_type="application/x-npz")

# Launch the hyperparameter tuning job (blocks until the single job has finished)
tuner.fit({"train": train_input, "val": val_input})
print("Hyperparameter tuning job finished.")
```

#### 6. Monitor tuning job from SageMaker console
While the above cell runs, we can check on the progress by visiting the SageMaker Console and finding the "Training" tab located on the left panel. Click "Hyperparameter tuning jobs" to view running jobs, and click into one to see its child training jobs.

#### What did that cost?
A tuning job is just a manager; the money is spent by the training jobs it launches, and that is where SageMaker records billable time. We bring back `job_cost_report()` from the previous episodes and run it over every child job of the tuner. The list of child jobs comes from `tuner.analytics().dataframe()` (one row per training job, with its hyperparameters and final objective value).

```python
def job_cost_report(job_name: str, price_per_hour: float):
    desc = session.sagemaker_client.describe_training_job(TrainingJobName=job_name)
    billable = desc["BillableTimeInSeconds"] * desc["ResourceConfig"]["InstanceCount"]
    print(f"{job_name}: status {desc['TrainingJobStatus']}, training {desc.get('TrainingTimeInSeconds', 0)/60:.1f} min, billable {billable/60:.1f} min, cost ${billable/3600*price_per_hour:.2f} at ${price_per_hour}/h")
    return desc

def tuning_cost_report(tuner, price_per_hour):
    """Call job_cost_report on every training job the tuner launched; return the total cost in dollars."""
    jobs_df = tuner.analytics().dataframe()
    total_billable = 0
    for job_name in jobs_df["TrainingJobName"]:
        desc = job_cost_report(job_name, price_per_hour)
        total_billable += desc["BillableTimeInSeconds"] * desc["ResourceConfig"]["InstanceCount"]
    total_cost = total_billable / 3600 * price_per_hour
    print(f"Tuning job {tuner.latest_tuning_job.name}: {len(jobs_df)} training jobs, {total_billable/60:.1f} billable min, ${total_cost:.2f} total")
    return total_cost

tuning_cost_report(tuner, price_per_hour)
```

### Scaling up our approach
If all goes well, we can scale up the experiment with the below code. In this configuration, we're scaling up the search by allowing SageMaker to test more hyperparameter configurations (`max_jobs=10`) while setting `max_parallel_jobs=2` to manage parallelization efficiently. With two jobs running at once, SageMaker will be able to explore potential improvements more quickly than in a fully sequential setup, while still dynamically selecting promising values as it learns from completed jobs. This balance leverages SageMaker's Bayesian optimization, which uses completed trials to inform subsequent ones, helping to avoid redundant testing of less promising parameter combinations. **Setting `max_parallel_jobs` higher than 2-4 reduces tuning efficiency, as SageMaker's ability to learn from completed jobs decreases when too many jobs run simultaneously.**

With this approach, SageMaker is better able to refine configurations without overloading resources or risking inefficient exploration, making `max_parallel_jobs=2` a solid default for most use cases.

Estimate first: 10 child jobs at roughly 3-4 billable minutes each is 30-40 billable minutes on `ml.m5.large`, so about $0.06 in total (the 10,000-epoch configurations run a little longer). Wall-clock is five rounds of two jobs, so expect 15-20 minutes; a good moment for a break.

```python
import time as t # always a good idea to keep a runtime of your experiments 

# Configuration variables
max_jobs = 10
max_parallel_jobs = 2

# Define the Tuner configuration
tuner = HyperparameterTuner(
    base_tuning_job_name=notebook_instance_name,
    estimator=pytorch_estimator,
    metric_definitions=[{"Name": "validation:accuracy", "Regex": "validation:accuracy = ([0-9\\.]+)"}],
    objective_metric_name="validation:accuracy",  # Ensure this matches the metric name exactly
    objective_type="Maximize",                   # Specify if maximizing or minimizing the metric
    hyperparameter_ranges=hyperparameter_ranges,
    max_jobs=max_jobs,
    max_parallel_jobs=max_parallel_jobs,
    tags=job_tags,
)

# Track start time
start_time = t.time()

# Launch the hyperparameter tuning job
tuner.fit({"train": train_input, "val": val_input})

# Calculate runtime
runtime = t.time() - start_time

# Print confirmation with runtime and configuration details
print(f"Tuning runtime: {runtime:.2f} seconds, Instance Type: {instance_type}, Max Jobs: {max_jobs}, Max Parallel Jobs: {max_parallel_jobs}")
```

### Monitoring tuning
While the above cell runs, we can check on the progress by visiting the SageMaker Console and finding the "Training" tab located on the left panel. Click "Hyperparameter tuning jobs" to view running jobs.

* Initial Jobs: SageMaker starts by running only `max_parallel_jobs` (2 in this case) as the initial batch. As each job completes, new jobs from the remaining pool are triggered until `max_jobs` (10) is reached.
* Job Completion: Once the first few jobs complete, SageMaker will continue to launch the remaining jobs up to the maximum of 10, but no more than two at a time.

And once it has finished, the bill for all 10 child jobs:

```python
tuning_cost_report(tuner, price_per_hour)
```

### Can/should we run more instances in parallel?
Setting `max_parallel_jobs` to 10 (equal to `max_jobs`) will launch all 10 jobs at once. This barely changes the total cost (you pay per training job, and each job's billable time is the same whether or not another job is running), but it shortens wall-clock time to roughly one job's duration and it changes the *quality* of the search, because SageMaker's Bayesian strategy can only use results that have already finished. This adaptability is especially useful for neural networks, which often have a large hyperparameter space with complex interactions. Here's how SageMaker's approach impacts typical neural network training:

#### 1. Adaptive Search Strategies
   - SageMaker offers **Bayesian optimization** for hyperparameter tuning. Instead of purely random sampling, it learns from previous jobs to choose the next set of hyperparameters more likely to improve the objective metric.
   - For neural networks, this strategy can help converge on better-performing configurations faster by favoring promising areas of the hyperparameter space and discarding poor ones.

#### 2. Effect of `max_parallel_jobs` on adaptive tuning
   - When using Bayesian optimization, a lower `max_parallel_jobs` (e.g., 2-4) can allow SageMaker to iteratively adjust and improve its choices. Each batch of jobs informs the subsequent batch, which may yield better results over time.
   - Conversely, if all jobs are run in parallel (e.g., `max_parallel_jobs=10`), SageMaker can't learn and adapt within a single batch, making this setup more like a traditional random search. This approach is still valid, especially for small search spaces, but it doesn't leverage the full potential of adaptive tuning.

#### 3. Practical impact on neural network training
   - **For simpler models** or smaller parameter ranges, running jobs in parallel with a higher `max_parallel_jobs` works well and quickly completes the search.
   - **For more complex neural networks** or large hyperparameter spaces, an adaptive strategy with a smaller `max_parallel_jobs` may yield a better model with fewer total jobs by fine-tuning hyperparameters over multiple iterations.

#### Summary
- **For fast, straightforward tuning**: Set `max_parallel_jobs` closer to `max_jobs` for simultaneous testing.
- **For adaptive, refined tuning**: Use a smaller `max_parallel_jobs` (like 2-4) to let SageMaker leverage adaptive tuning for optimal configurations. 

This balance between exploration and exploitation is particularly impactful in neural network tuning, where training costs can be high and parameters interact in complex ways.

### Extracting and evaluating the best model after tuning

After SageMaker completes the hyperparameter tuning job, the results, including the trained models for each configuration, are stored in an S3 bucket. Here's a breakdown of the steps to locate and evaluate the best model on test data.

1. **Understanding the folder structure**:
   - SageMaker saves each tuning job's results in the specified S3 bucket under a unique prefix.
   - For the best model, SageMaker stores the model artifact in the format `s3://{bucket}/{job_name}/output/model.tar.gz`. Each model is compressed as a `.tar.gz` file containing the saved model parameters.

2. **Retrieve and load the best model**:
   - Using the `tuner.best_training_job()` method, you can get the name of the best-performing job.
   - From there, retrieve the S3 URI of the best model artifact, download it locally, and extract the files for use.

3. **Prepare test data for final assessment of model generalizability**
    - Using the scaler that was fit on the training data.

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

#### Retrieve and load best model

```python
import tarfile

# Download and extract the model artifact
local_model_path = "best_model.tar.gz"
model_bucket, model_key = best_model_s3_uri.split('/')[2], '/'.join(best_model_s3_uri.split('/')[3:])
s3.download_file(model_bucket, model_key, local_model_path)

# Extract the model files from the tar.gz archive
with tarfile.open(local_model_path, 'r:gz') as tar:
    tar.extractall()
print("Best model downloaded and extracted.")
```

#### Prepare test set as test_data.npz
In our previous episode, we converted our train dataset into train/validate subsets, and saved them out as .npz files for efficient processing. We'll need to preprocess our test data the same way to evaluate it on our model. 

**Note**: It's always a good idea to keep preprocessing code as a function so you can apply the same exact procedure across datasets with ease. We'll import our preprocessing function from `train_nn.py`, and pass it the **scaler that was fit on the training data** (saved as `scaler.joblib` in the previous episode). Calling `preprocess_data(test_df)` without the scaler would fit a new one on the test set, which leaks test-set statistics into the evaluation and transforms the features differently from what the model saw in training.

```python
import pandas as pd
import numpy as np
import joblib
from AWS_helpers.train_nn import preprocess_data

# Fetch the test CSV and the training-set scaler from S3 (both uploaded in earlier episodes)
s3.download_file(bucket_name, "titanic_test.csv", "titanic_test.csv")
s3.download_file(bucket_name, "scaler.joblib", "scaler.joblib")
scaler = joblib.load("scaler.joblib")

# Preprocess test data with the same scaler used for training
test_df = pd.read_csv("titanic_test.csv")
X_test, y_test, _ = preprocess_data(test_df, scaler=scaler)

# Save processed data for testing
np.savez('test_data.npz', X_test=X_test, y_test=y_test)
```

#### Evaluate the model on test data

```python
from AWS_helpers.train_nn import TitanicNet
from AWS_helpers.train_nn import calculate_accuracy
import torch

# Load the model (saved as 'nn_model.pth' by train_nn.py)
model = TitanicNet()
model.load_state_dict(torch.load("nn_model.pth", map_location="cpu"))
model.eval()

# Load test data
test_data = np.load("test_data.npz")
X_test = torch.tensor(test_data['X_test'], dtype=torch.float32)
y_test = torch.tensor(test_data['y_test'], dtype=torch.float32).unsqueeze(1)

# Evaluate the model on the test set
with torch.no_grad():
    predictions = model(X_test)
    accuracy = calculate_accuracy(predictions, y_test)
    print(f"Test Accuracy: {accuracy:.4f}")
```

### Conclusions
In about 20 minutes of wall-clock time and a few cents of compute, we searched 10 configurations and produced a model that scores about 0.8 on the test set, roughly on par with the XGBoost model from the first episode (the Titanic dataset does not have much more signal to extract). The point is the workflow: one tuner call, every child job tagged, and a bill you can read back.

#### Reading the bill
`job_cost_report()` printed two numbers per job. **Training time** (`TrainingTimeInSeconds`) is the time your script was running. **Billable time** (`BillableTimeInSeconds`) is the time the instance was reserved for you, which also includes the startup overhead of pulling the container and downloading the data. SageMaker bills per second, with no rounding to minutes or hours, and it bills training jobs and the child jobs of a tuning job in exactly the same way; the tuning job itself costs nothing. Total cost is therefore the sum over child jobs of billable seconds x instance count x hourly rate. For our tiny network the startup overhead is most of every job, which is why ten short jobs cost noticeably more than one job that trained ten times as long would.

For convenience, the same calculation is available as a function in `helpers.py`:

```python
import AWS_helpers.helpers as helpers
import importlib
importlib.reload(helpers)
helpers.calculate_tuning_job_time(tuner, region)
```

::::::::::::::::::::::::::::::::::::::: challenge

### All at once

Rerun the 10-job search with `max_parallel_jobs=10` (equal to `max_jobs`). Before you launch it, predict what changes in (a) total cost, (b) wall-clock time, and (c) the quality of the search. Then run it and check (a) and (b) with `tuning_cost_report()` and the runtime print.

```python
tuner_parallel = HyperparameterTuner(
    base_tuning_job_name=notebook_instance_name,
    estimator=pytorch_estimator,
    metric_definitions=[{"Name": "validation:accuracy", "Regex": "validation:accuracy = ([0-9\\.]+)"}],
    objective_metric_name="validation:accuracy",
    objective_type="Maximize",
    hyperparameter_ranges=hyperparameter_ranges,
    max_jobs=10,
    max_parallel_jobs=10,
    tags=job_tags,
)

start_time = t.time()
tuner_parallel.fit({"train": train_input, "val": val_input})
print(f"Tuning runtime: {t.time() - start_time:.2f} seconds with max_parallel_jobs=10")
tuning_cost_report(tuner_parallel, price_per_hour)
print("Best objective, sequential-ish run:", tuner.analytics().dataframe()["FinalObjectiveValue"].max())
print("Best objective, all-parallel run:  ", tuner_parallel.analytics().dataframe()["FinalObjectiveValue"].max())
```

::::::::::::::::::::::::::::::::::::::: solution

### Solution

- **Cost**: about the same (~$0.06). You pay per child training job, and each job's billable time does not depend on how many other jobs are running. (If your account has a quota on concurrent `ml.m5.large` training instances, the request may be rejected or throttled, which is the practical limit on this approach.)
- **Wall-clock**: roughly one job's duration (4-5 minutes) instead of five rounds of two, so about 4-5x faster.
- **Search quality**: with all 10 launched at once, the Bayesian strategy has no completed results to learn from, so the 10 configurations are effectively a random sample of the space. On a 2-parameter problem this small you will usually see a similar best objective either way; on a larger search space with more jobs, the sequential-ish run tends to find a better configuration for the same number of jobs. That is the trade: parallelism buys time, not a better model.

:::::::::::::::::::::::::::::::::::::::::::::::::::::

:::::::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::::: challenge

### Tune the batch size too

Suppose you want to add `batch_size` to the search as `IntegerParameter(16, 256)`. What has to change in `train_nn.py` for that to do anything, and what has to change in the notebook?

::::::::::::::::::::::::::::::::::::::: solution

### Solution

The tuner can only vary what the script reads. Right now `train_nn.py` has no `--batch_size` argument and trains on the full dataset every epoch (one `model(X_train)` call per epoch), so a `batch_size` hyperparameter would be passed on the command line, rejected by `argparse`, and the job would fail. Three changes are needed:

1. **In `train_nn.py`**: add `parser.add_argument('--batch_size', type=int, default=64)`, pass it to `train()`, and replace the full-batch step with a mini-batch loop, for example by wrapping the tensors in a `TensorDataset` and iterating over a `DataLoader(dataset, batch_size=batch_size, shuffle=True)` inside each epoch (with `optimizer.zero_grad()`, forward, loss, `backward()` and `step()` per batch).

```text
parser.add_argument('--batch_size', type=int, default=64)
...
loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
for epoch in range(epochs):
    model.train()
    for xb, yb in loader:
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        optimizer.step()
```

2. **In the estimator**: add a placeholder `"batch_size": 64` to `hyperparameters`, so the argument is always passed.

3. **In the ranges**: add `"batch_size": IntegerParameter(16, 256)` to `hyperparameter_ranges`.

Two side effects worth anticipating: an epoch now takes several optimizer steps instead of one, so the same `epochs` values train for longer (consider shrinking the `epochs` choices), and the search space has grown, so 10 jobs will cover it less well; 20-30 would be a reasonable budget, which at these prices is still well under a dollar.

:::::::::::::::::::::::::::::::::::::::::::::::::::::

:::::::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: keypoints

- **Tuner = estimator + ranges + objective**: A `HyperparameterTuner` wraps an ordinary estimator; the script only has to print the objective metric in a format the `metric_definitions` regex can capture.
- **Start with one job**: `max_jobs=1, max_parallel_jobs=1` first, to catch script and configuration errors before paying for a search.
- **Parallelism buys time, not quality**: Cost is per child training job; `max_parallel_jobs` changes wall-clock time and, with the Bayesian strategy, how much each job can learn from the ones before it. 2-4 parallel jobs is a good default.
- **Tags propagate from the tuner**: Tags on the tuner are copied to every child training job, the one case in SageMaker where tags are inherited.
- **Read the bill from the child jobs**: SageMaker bills per second of `BillableTimeInSeconds` x instance count on each training job; the gap to `TrainingTimeInSeconds` is startup overhead, and it dominates short jobs.
- **Transform the test set with the training scaler**: Load the saved scaler and pass it to `preprocess_data(test_df, scaler=scaler)`; never refit on test data.

::::::::::::::::::::::::::::::::::::::::::::::::

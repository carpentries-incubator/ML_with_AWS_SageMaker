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
Open the notebook from: `/ML_with_AWS_SageMaker/notebooks/08_Hyperparameter-tuning.ipynb`. Select the **`conda_pytorch`** kernel.

## Hyperparameter tuning in SageMaker
SageMaker's **hyperparameter tuning jobs** search a parameter space for any model; what keeps them cheap is the choice of ranges and the number of models trained. This episode walks through one tuning job with both in view.

### Key steps 
The overall process involves these seven below steps.

1. Setup estimator
2. Define parameter ranges
3. Set up HyperparameterTuner object
4. Prepare training script to log metrics
5. Set data paths and launch tuner.fit()
6. Monitor tuning job from SageMaker console
7. Extract best model for final evaluation

#### Code example for SageMaker hyperparameter tuning with neural networks
We'll walk through each step in detail by tuning a neural network. Specifically, we will test out different values for our `epochs` and `learning_rate` parameters. We are sticking to just two hyperparameters for demonstration purposes, but you may wish to explore additional parameters in your own work. 
 
`epochs` is a `CategoricalParameter`, a list of values to try; `learning_rate` a `ContinuousParameter`, a range to sample; `max_jobs` and `max_parallel_jobs` bound the cost.

#### 0. Directory setup
Start in the Titanic project folder, as in the previous episodes:

```python
# The Titanic project folder from the S3 episode, beside the lesson clone: downloads and outputs land here, not in the repository
!mkdir -p /home/ec2-user/SageMaker/titanic
%cd /home/ec2-user/SageMaker/titanic
import sys
sys.path.insert(0, "/home/ec2-user/SageMaker/ML_with_AWS_SageMaker/scripts")
import helpers
```

#### 1. Setup estimator
The tuner needs an estimator first. The estimator setup here is very similar to our previous episode, where we used it to configure and train a model directly. However, this time, rather than directly running a training job with the estimator, we'll be using it as the foundation for **hyperparameter tuning**.

In SageMaker, the estimator serves as a blueprint for each tuning job, specifying the training script, instance type, and key parameters like data paths and metrics. Once defined, this estimator will be passed to a **Hyperparameter Tuner** that manages the creation of multiple training jobs with various hyperparameter combinations, helping us find an optimal configuration.

The PyTorch estimator, which includes specifying the entry script for training (`train_nn.py`) and defining hyperparameters that will remain fixed across tuning jobs. The hyperparameters we're setting up to tune include `epochs` and `learning_rate`, with a few specific values or ranges defined.

We'll use our notebook instance name again to label the training jobs launched in this episode

```python
import boto3

notebook_instance_name = 'SinkOrSwim-DoeJohn' # adjust to your notebook name. we'll use this variable to name the training jobs launched by the tuner.

# the following code just verifies you have the right name for your notebook instance
region = boto3.Session().region_name  # inherited from the notebook instance; do not hard-code a region
sagemaker_client = boto3.client('sagemaker', region_name=region) # Initialize SageMaker client
response = sagemaker_client.describe_notebook_instance(NotebookInstanceName=notebook_instance_name) # Describe the notebook instance

# Display the status and instance type
print(f"Notebook Instance '{notebook_instance_name}' status: {response['NotebookInstanceStatus']}")
local_instance = response['InstanceType']
print(f"Instance Type: {local_instance}")
```

**Cost tracking**: Just like in the previous episode, we explicitly tag each training job at launch time. This ensures that compute usage is traceable to a project, a purpose, and a human-readable name, even after the job has completed. Tags on the `HyperparameterTuner` are copied to every child training job it launches, so we only need to set them once, on the tuner (we also set them on the estimator so a plain `.fit()` on it is tagged too).

```python
name = "DoeJohn" # replace with your name
project = "SinkOrSwim"   # your challenge: WattBot2026, BadgerScribe, BadgerCode, BrainDecoding, PoisonedWell or SnapshotWI; SinkOrSwim is the workshop itself
purpose = "Tuning"        # one of: Storage, Notebook, Training, Tuning, Inference

job_tags = [
    {"Key": "Name", "Value": name},
    {"Key": "Project", "Value": project},
    {"Key": "Purpose", "Value": purpose},
]
helpers.check_launch(job_tags)   # the account's rules, checked here first: AWS's own refusal does not say which rule
print(f"Name: {name}, Project: {project}, Purpose: {purpose}")
```

Next, we'll configure the baseline estimator that we plan to do hyperparameter search on.

Every price in this episode is from the [Instances for ML](https://carpentries-incubator.github.io/ML_with_AWS_SageMaker/instances-for-ML.html) page, us-east-1 on-demand; the tuning job's cost is that rate times the sum of the child jobs' billable minutes.

```python
import os
os.environ["SAGEMAKER_SUPPRESS_V2_WARNING"] = "1"   # the SDK otherwise prints a deprecation notice on import and on every estimator
import sagemaker
import logging
logging.getLogger("sagemaker.telemetry.telemetry_logging").setLevel(logging.WARNING)   # silence the telemetry notice; the "Creating training-job" line still prints
logging.getLogger("sagemaker.image_uris").setLevel(logging.WARNING)
logging.getLogger("sagemaker.config").setLevel(logging.WARNING)   # and the two "Not applying SDK defaults" lines
logging.getLogger("sagemaker.estimator").setLevel(logging.ERROR)   # tuner.fit reads estimator.model_data before any job exists and logs a harmless "No finished training job found" warning
# range types for the search: ContinuousParameter (float range), IntegerParameter (whole-number range), CategoricalParameter (explicit list)
from sagemaker.tuner import HyperparameterTuner, IntegerParameter, ContinuousParameter, CategoricalParameter
from sagemaker.pytorch import PyTorch
from sagemaker.inputs import TrainingInput
from sagemaker import get_execution_role

# Initialize role, bucket, and SageMaker session variables
role = get_execution_role()
bucket_name = 'sinkorswim-doejohn'  # replace with your S3 bucket name
session = sagemaker.Session()
s3 = boto3.client('s3')
print("Role:", role)
print("Region:", session.boto_region_name)

# Define instance type/count we'll use for training
instance_type="ml.m5.large"
instance_count=1 # always start with 1; tuning parallelism comes from max_parallel_jobs, not instance_count
price_per_hour = 0.115 # ml.m5.large, us-east-1 on-demand (2026-10-16), from the Instances for ML page; check it for your region

# The runtime cap, in seconds: SageMaker stops the job and releases the instance when it is reached. Set it a little above the estimate;
# the estimate here is three or four minutes, so 20 minutes leaves room for a slow container pull without letting a hung job bill for hours.
max_run = 20*60 # 20 minutes

# Define the PyTorch estimator with entry script and environment details
pytorch_estimator = PyTorch(
    base_job_name=f"{notebook_instance_name}-nn",   # who + which model; the tuner appends its own suffix to each child job
    max_run=max_run, # in seconds; always include (max 48 hours)
    entry_point="/home/ec2-user/SageMaker/ML_with_AWS_SageMaker/scripts/train_nn.py",  # Your script for training
    role=role,
    tags=job_tags,  # tags a plain .fit() on this estimator; the tuner below sets its own
    instance_count=instance_count,
    instance_type=instance_type,
    framework_version="2.3",  # if a job fails with an image error, check with: sagemaker.image_uris.retrieve("pytorch", region, version="2.3", py_version="py311", image_scope="training", instance_type="ml.m5.large")
    py_version="py311",
    # the Regex pulls the number out of each "validation:accuracy = 0.8123" line the script prints; SageMaker records it as the metric
    metric_definitions=[{"Name": "validation:accuracy", "Regex": "validation:accuracy = ([0-9\\.]+)"}],
    hyperparameters={
        "train": "/opt/ml/input/data/train/train_data.npz",  # SageMaker will mount this path
        "val": "/opt/ml/input/data/val/val_data.npz",        # SageMaker will mount this path
        "epochs": 100, # Placeholder initial value. Will be overridden by the tuning values tested
        "learning_rate": 0.001 # Placeholder initial value. Will be overridden by the tuning values tested
    },
    output_path=f"s3://{bucket_name}/tuning/",   # every child job's model.tar.gz lands here, in your bucket
    sagemaker_session=session,
)
```

#### 2. Define hyperparameter ranges

In SageMaker, you must explicitly define ranges for any hyperparameters you want to tune. SageMaker supports `ContinuousParameter`, `IntegerParameter` and `CategoricalParameter` types:  

   - **`ContinuousParameter`** allows SageMaker to dynamically sample numeric values within a specified range, making it ideal for broad, exploratory tuning. The total number of values tested can be controlled through the upcoming `max_jobs` parameter, which defines how many different combinations SageMaker will evaluate.
   - **`IntegerParameter`** does the same for whole-number settings such as a batch size or layer width.
   - **`CategoricalParameter`** specifies exact values for SageMaker to test, which is useful when you want the model to only try a specific set of options.

By default, SageMaker uses **Bayesian optimization**, adjusting future selections based on previous results to efficiently find optimal values. You can also set the `strategy` to **"Random"** for uniform sampling across the range, which is effective in larger or more exploratory search spaces. Random search usually needs more jobs to find as good a configuration, and the bill is the number of jobs. Generally, we recommend sticking with the default setting.

```python
# Hyperparameter tuning ranges
hyperparameter_ranges = {
    "epochs": CategoricalParameter([100, 1000, 10000]),       # exact values to try, nothing in between; adjust as needed
    "learning_rate": ContinuousParameter(0.001, 0.1),  # any float in this range; the Bayesian strategy picks where to sample
}
```

#### 3. Set up HyperparameterTuner object
In step 3, we set up the `HyperparameterTuner`, which controls the tuning process by specifying the...

- `estimator`: Here, we connect the previously defined `pytorch_estimator` to `tuner`, ensuring that the tuning job will run with our PyTorch model configuration.
- `objective`: The `metric_definitions` and `objective_metric_name` indicate which metric SageMaker should monitor to find the best-performing model; in this case, we're tracking "validation:accuracy" and aiming to maximize it. We'll show you how to setup your training script to report this information in the next step.
- `hyperparameter_ranges`: Defined above.
- `strategy`: SageMaker uses a **Bayesian strategy** by default, which iteratively improves parameter selection based on past performance to find an optimal model more efficiently. Although it's possible to adjust to a "Random" strategy, Bayesian optimization generally provides better results, so it's recommended to keep this setting. 
- `max_jobs` and `max_parallel_jobs`: Finally, we set `max_jobs` to control the total number of configurations SageMaker will explore and `max_parallel_jobs` to limit the number of jobs that run simultaneously, balancing resource usage and tuning speed. Since SageMaker tests different hyperparameter values dynamically, keep `max_parallel_jobs` at 4 or fewer. 

> Tuning gains are usually marginal past 20 to 30 configurations, and the search space is where the money goes: a published range for the parameters that matter beats a wide search over all of them. 

**Always start with `max_jobs = 1` and `max_parallel_jobs=1`.**
Before running the full search, let's test our setup by setting max_jobs = 1. This will test just one possible hyperparameter configuration. 

```python
# Tuner configuration
tuner = HyperparameterTuner(
    base_tuning_job_name=f"{name}-nn-tune"[:20],   # tuning job names are capped at 32 characters and the SDK appends a 12-character timestamp, so the base gets at most 20; child jobs are named <tuning job>-001-<hash>
    estimator=pytorch_estimator,  # the blueprint: every child training job is this estimator with different hyperparameters
    # same regex as the estimator; the value it captures is what the tuner compares across child jobs
    metric_definitions=[{"Name": "validation:accuracy", "Regex": "validation:accuracy = ([0-9\\.]+)"}],
    objective_metric_name="validation:accuracy",  # Ensure this matches the metric name exactly (the "Name" above)
    objective_type="Maximize",                   # Specify if maximizing or minimizing the metric
    hyperparameter_ranges=hyperparameter_ranges,  # the search space defined above
    strategy="Bayesian",  # Default setting (recommend sticking with this!); can adjust to "Random" for uniform sampling across the range
    max_jobs=1,                # Always start with 1 job for debugging purposes. Adjust based on exploration needs (keep below 30 to be kind to environment). 
    max_parallel_jobs=1,         # Always start with 1 for debugging purposes. Recommended to keep this value <= 4 since SageMaker tests values dynamically.
    tags=job_tags,               # propagated to every training job the tuner launches
)
```

#### 4. Prepare training script to log metrics
To prepare `train_nn.py` for hyperparameter tuning, we added code to log validation metrics in a format that SageMaker recognizes for tracking. In the training loop, we added a print statement for the validation accuracy in a specific format that SageMaker can capture. 

**Note**: It's best to use an if statement to only print out metrics periodically (e.g., every 100 epochs), so that your print time does not eat up too much of your training time. Printing every epoch is a measurable share of a step this small. On the flipside of this, you don't want to print metrics so infrequently that you lose resolution in the monitored validation accuracy. Choose a number between 100-1000 epochs or divide your total epoch count by ~25 to yield a reasonable range. 

```py
if (epoch + 1) % 100 == 0 or epoch == epochs - 1:
    print(f"validation:accuracy = {val_accuracy:.4f}", flush=True)  # Log for SageMaker metric tracking
```

Paired with this, our `metric_definitions` above uses a regular expression `"validation:accuracy = ([0-9\\.]+)"` to extract the val_accuracy value from each log line. This regex specifically looks for `validation:accuracy =`, followed by a floating-point number, which corresponds to the format of our log statement in `train_nn.py`.

#### 5. Set data paths and launch tuner.fit()
In step 5, we define the input data paths for the training job and launch the hyperparameter tuning process. Using `TrainingInput`, we specify the S3 paths to our `train_data.npz` and `val_data.npz` files (uploaded in the previous episode). We then call `tuner.fit` and pass a dictionary mapping each data channel ("train" and "val") to its respective path. This command initiates the tuning job, triggering SageMaker to begin sampling hyperparameter combinations, training the model, and evaluating performance based on our defined objective metric. Once the job is launched, SageMaker handles the tuning process automatically, running the specified number of configurations and keeping track of the best model parameters based on validation accuracy.

Estimate first: one child job is the same three billable minutes as the previous episode's job, almost all of it startup, so about `$0.01`.

If the launch is refused with an `AccessDeniedException` ending in `explicit deny`, the account's checks are the cause: a missing tag, an instance type outside the lesson's list, or a resource kind the lesson never creates; the first training episode's callout *If the launch is refused* says what to look at.

```python
# Define the input paths
train_input = TrainingInput(f"s3://{bucket_name}/train_data.npz", content_type="application/x-npz")
val_input = TrainingInput(f"s3://{bucket_name}/val_data.npz", content_type="application/x-npz")

import time as t

# Launch the hyperparameter tuning job (blocks until the single job has finished)
start_time = t.time()
tuner.fit({"train": train_input, "val": val_input})
print(f"Hyperparameter tuning job finished in {t.time() - start_time:.1f} seconds (wall-clock, including instance startup)")
```

#### 6. Monitor tuning job from SageMaker console
While the above cell runs, we can check on the progress by visiting the SageMaker Console and finding the "Training" tab located on the left panel. Click "Hyperparameter tuning jobs" to view running jobs.

#### What did that cost?
The money is spent by the training jobs the tuner launches, and that is where SageMaker records billable time. We bring back `job_cost_report()` from the previous episodes and run it over every child job of the tuner. The list of child jobs comes from `tuner.analytics().dataframe()` (one row per training job, with its hyperparameters and final objective value).

```python
def job_cost_report(job_name: str, price_per_hour: float):
    desc = session.sagemaker_client.describe_training_job(TrainingJobName=job_name)
    if "BillableTimeInSeconds" not in desc:   # the job has not finished: no bill yet
        print(f"{job_name}: status {desc['TrainingJobStatus']}; billable time and the model path appear once it is Completed")
        return desc
    n = desc["ResourceConfig"]["InstanceCount"]
    billable = desc["BillableTimeInSeconds"] * n
    wall = (desc["TrainingEndTime"] - desc["CreationTime"]).total_seconds()   # launch request to end, including unbilled provisioning
    print(f"{job_name}: status {desc['TrainingJobStatus']}, wall-clock {wall/60:.1f} min, billable {billable/60:.1f} min on {n} instance(s), cost ${billable/3600*price_per_hour:.3f} at ${price_per_hour}/h")
    print(f"    model: {desc['ModelArtifacts']['S3ModelArtifacts']}")   # the model.tar.gz this job uploaded to output_path
    return desc

def tuning_cost_report(tuner, price_per_hour):
    """Call job_cost_report on every training job the tuner launched and print the total."""
    jobs_df = tuner.analytics().dataframe()  # one row per child training job: its hyperparameters, FinalObjectiveValue and TrainingJobName
    total_billable = 0
    for job_name in jobs_df["TrainingJobName"]:
        desc = job_cost_report(job_name, price_per_hour)
        total_billable += desc["BillableTimeInSeconds"] * desc["ResourceConfig"]["InstanceCount"]
    total_cost = total_billable / 3600 * price_per_hour
    print(f"Tuning job {tuner.latest_tuning_job.name}: {len(jobs_df)} training jobs, {total_billable/60:.1f} billable min, ${total_cost:.2f} total")

tuning_cost_report(tuner, price_per_hour)
```

### Scaling up our approach
If all goes well, we can scale up the experiment with the below code. In this configuration, we're scaling up the search by allowing SageMaker to test more hyperparameter configurations (`max_jobs=10`) while setting `max_parallel_jobs=2` to manage parallelization efficiently. With two jobs running at once, SageMaker will be able to explore potential improvements more quickly than in a fully sequential setup, while still dynamically selecting promising values as it learns from completed jobs. SageMaker's Bayesian optimization uses the completed trials to choose the next ones, so it spends fewer jobs on unpromising combinations. **Setting `max_parallel_jobs` higher than 2-4 reduces tuning efficiency, as SageMaker's ability to learn from completed jobs decreases when too many jobs run simultaneously.**

With this approach, SageMaker is better able to refine configurations without overloading resources or risking inefficient exploration, making `max_parallel_jobs=2` a solid default for most use cases.

Estimate first: 10 child jobs at roughly 3 billable minutes each is about 30 billable minutes on `ml.m5.large`, so about `$0.06` in total. Wall-clock is five rounds of two jobs, so expect 15-20 minutes. Write both numbers down; the bill below comes in under them.

```python
import time as t

# Configuration variables
max_jobs = 10
max_parallel_jobs = 2

# Define the Tuner configuration
tuner = HyperparameterTuner(
    base_tuning_job_name=f"{name}-nn-tune"[:20],   # tuning job names are capped at 32 characters and the SDK appends a 12-character timestamp, so the base gets at most 20; child jobs are named <tuning job>-001-<hash>
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

Once it has finished, the cost report for all 10 child jobs:

```python
tuning_cost_report(tuner, price_per_hour)
```

Our run finished in 9.3 minutes of wall-clock and billed 14.1 minutes in total, `$0.03`: half the estimate. Read the per-job lines and the reason is visible. The first two child jobs billed about 3 minutes each, like every job so far. The other eight billed about 1 minute each. A tuning job keeps its instances between child jobs, so jobs 3 through 10 ran on the two instances that were already up, with the container pulled and the data downloaded. The startup overhead that dominates a single small job was paid twice, not ten times.

### More jobs in parallel

Bayesian optimization chooses each job's values from the results so far. With `max_parallel_jobs` well below `max_jobs`, later jobs learn from earlier ones; with all ten at once, nothing has finished when the values are chosen, and the search is a random one. The *All at once* challenge below measures what that does to the result, the wall-clock and the bill.

### Extracting and evaluating the best model after tuning

Tuning takes about 10 minutes. After SageMaker completes the hyperparameter tuning job, the results, including the trained models for each configuration, are stored in an S3 bucket. The steps to locate and evaluate the best model on test data:

1. **Understanding the folder structure**:
   - Each child job uploads its model to the estimator's `output_path`, under its own job name: `s3://{bucket}/tuning/{job_name}/output/model.tar.gz`. Each model is compressed as a `.tar.gz` file containing the saved model parameters.

2. **Retrieve and load the best model**:
   - Using the `tuner.best_training_job()` method, you can get the name of the best-performing job.
   - From there, retrieve the S3 URI of the best model artifact, download it locally, and extract the files for use.

3. **Prepare test data for final assessment of model generalizability**
    - Using the scaler that was fit on the training data.

4. **Evaluate the model on test data**:
   - Once extracted, load the saved model weights and evaluate the model on your test dataset to get the final performance metrics.

The code:

#### View best model details and storage info
We can easily view the best hyperparameters from the tuning procedure...

```python
# 1. Get the best training job from the completed tuning job
best_job_name = tuner.best_training_job()  # the child job with the highest validation:accuracy
print("Best training job name:", best_job_name)

# 2. Use describe_training_job to retrieve full details, including hyperparameters...
best_job_desc = session.sagemaker_client.describe_training_job(TrainingJobName=best_job_name)
best_hyperparameters = {k: v for k, v in best_job_desc["HyperParameters"].items() if k in hyperparameter_ranges}   # the record also carries the SDK's own sagemaker_* entries
print("Best hyperparameters:", best_hyperparameters)

# ...  and model URI (location on S3)
best_model_s3_uri = best_job_desc['ModelArtifacts']['S3ModelArtifacts']  # the model.tar.gz that job uploaded when it finished
print(f"Best model artifact S3 URI: {best_model_s3_uri}")
```

#### Retrieve and load best model

Download the winning job's `model.tar.gz` and unpack it; SageMaker packs whatever the script saved to `/opt/ml/model` into that archive.

```python
import tarfile

# Download and extract the model artifact: SageMaker packs whatever the script saved to /opt/ml/model into model.tar.gz
local_model_path = "best_model.tar.gz"
model_bucket, model_key = best_model_s3_uri.split('/')[2], '/'.join(best_model_s3_uri.split('/')[3:])  # s3://bucket/key -> bucket, key
s3.download_file(model_bucket, model_key, local_model_path)

# Extract the model files from the tar.gz archive
with tarfile.open(local_model_path, 'r:gz') as tar:
    tar.extractall()
print("Best model downloaded and extracted.")
```

#### Prepare test set as test_data.npz
In our previous episode, we converted our train dataset into train/validate subsets, and saved them out as .npz files for efficient processing. We'll need to preprocess our test data the same way to evaluate it on our model. 

Preprocessing lives in a function so that the test set is transformed exactly as the training set was. We'll import our preprocessing function from `train_nn.py`, and pass it the scaler that was fit on the training data (saved as `scaler.joblib` in the previous episode). Calling `preprocess_data(test_df)` without the scaler would fit a new one on the test set and transform the features differently from what the model saw in training.

```python
!pip install scikit-learn   # train_nn.py imports it; not in the notebook kernels
```

Fetch the test CSV and the training-set scaler, transform the test rows the same way the training rows were, and save the arrays.

```python
import pandas as pd
import numpy as np
import joblib
from train_nn import preprocess_data

# Fetch the test CSV and the training-set scaler from S3 (both uploaded in earlier episodes)
s3.download_file(bucket_name, "titanic_test.csv", "titanic_test.csv")
s3.download_file(bucket_name, "scaler.joblib", "scaler.joblib")
scaler = joblib.load("scaler.joblib")  # the StandardScaler fit on the training data in the previous episode

# Preprocess test data with the same scaler used for training
test_df = pd.read_csv("titanic_test.csv")
X_test, y_test, _ = preprocess_data(test_df, scaler=scaler)  # transform only; passing the scaler stops it refitting on test data

# Save processed data for testing
np.savez('test_data.npz', X_test=X_test, y_test=y_test)
```

#### Evaluate the model on test data

Rebuild the network from the script's class, load the saved weights, and measure accuracy on the test set.

```python
from train_nn import TitanicNet
from train_nn import calculate_accuracy
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

::::::::::::::::::::::::::::::::::::: challenge

### All at once

Rerun the 10-job search with `max_parallel_jobs=10` (equal to `max_jobs`). Before you launch it, predict what changes in (a) total cost, (b) wall-clock time, and (c) the quality of the search. Then run it and check (a) and (b) with `tuning_cost_report()` and the runtime print.

```python
tuner_parallel = HyperparameterTuner(
    base_tuning_job_name=f"{name}-nn-tune"[:20],   # tuning job names are capped at 32 characters and the SDK appends a 12-character timestamp, so the base gets at most 20; child jobs are named <tuning job>-001-<hash>
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

- **Cost**: higher, not the same. Our all-parallel run billed 27.5 minutes (`$0.05`) against 14.1 minutes (`$0.03`) for the two-at-a-time run. Ten instances each pull the container and download the data, so the startup overhead is paid ten times; with two at a time it was paid twice and the other eight child jobs reused warm instances. (If your account has a quota on concurrent `ml.m5.large` training instances, the request may be rejected or throttled, which is the practical limit on this approach.)
- **Wall-clock**: roughly one job's duration. Our run took 4.3 minutes against 9.3, so about twice as fast, not five times: the two-at-a-time run's later jobs were short because they skipped startup.
- **Search quality**: with all 10 launched at once, the Bayesian strategy has no completed results to learn from, so the 10 configurations are effectively a random sample of the space. On a 2-parameter problem this small you will usually see a similar best objective either way; on a larger search space with more jobs, the sequential-ish run tends to find a better configuration for the same number of jobs.

:::::::::::::::::::::::::::::::::::::::::::::::::::::

:::::::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::::: challenge

### Tune the batch size too

Suppose you want to add `batch_size` to the search as `IntegerParameter(16, 256)`. What has to change in `train_nn.py` for that to do anything, and what has to change in the notebook?

::::::::::::::::::::::::::::::::::::::: solution

### Solution

The tuner can only vary what the script reads. Right now `train_nn.py` has no `--batch_size` argument and trains on the full dataset every epoch (one `model(X_train)` call per epoch), so a `batch_size` hyperparameter would be passed on the command line, rejected by `argparse`, and the job would fail. Three changes are needed:

1. **In `train_nn.py`**: add `parser.add_argument('--batch_size', type=int, default=64)`, pass it to `train()`, and replace the full-batch step with a mini-batch loop, for example by wrapping the tensors in a `TensorDataset` and iterating over a `DataLoader(dataset, batch_size=batch_size, shuffle=True)` inside each epoch (with `optimizer.zero_grad()`, forward, loss, `backward()` and `step()` per batch).

```py
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

Two side effects: an epoch now takes several optimizer steps instead of one, so the same `epochs` values train for longer (consider shrinking the `epochs` choices), and the search space has grown, so 10 jobs will cover it less well; 20-30 would be a reasonable budget, which at these prices is still well under a dollar.

:::::::::::::::::::::::::::::::::::::::::::::::::::::

:::::::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: keypoints

- **Tuner = estimator + ranges + objective**: A `HyperparameterTuner` wraps an ordinary estimator; the script only has to print the objective metric in a format the `metric_definitions` regex can capture.
- **Start with one job**: `max_jobs=1, max_parallel_jobs=1` first, to catch script and configuration errors before paying for a search.
- **Parallelism buys time, not quality**: Cost is per child training job; `max_parallel_jobs` changes wall-clock time and, with the Bayesian strategy, how much each job can learn from the ones before it. 2-4 parallel jobs is a good default.
- **Tags propagate from the tuner**: tags on the tuner are copied to every child training job.
- **Read the bill from the child jobs**: SageMaker bills per second of `BillableTimeInSeconds` x instance count on each training job; container pull and data download are inside that number and dominate short jobs.
- **Transform the test set with the training scaler**: Load the saved scaler and pass it to `preprocess_data(test_df, scaler=scaler)`; never refit on test data.

::::::::::::::::::::::::::::::::::::::::::::::::

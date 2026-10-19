---
title: "Training Models in SageMaker: PyTorch Example"
teaching: 35
exercises: 15
---

:::::::::::::::::::::::::::::::::::::: questions 

- When should you consider using a GPU instance for training neural networks in SageMaker, and what are the benefits and limitations?
- What does `instance_count=2` actually do to a training job, and what does real distributed training require?

::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: objectives

- Preprocess the Titanic dataset for efficient training using PyTorch, and keep the fitted scaler so the test set can be transformed the same way later.
- Save and upload training and validation data in `.npz` format to S3.
- Understand the trade-offs between CPU and GPU training for smaller datasets.
- Train a PyTorch model via SageMaker and compare instance types on time and cost.
- Explain what `instance_count > 1` does without a distribution strategy, and what data parallelism and model parallelism need in order to help.

::::::::::::::::::::::::::::::::::::::::::::::::

## Initial setup: open prefilled .ipynb notebook
Open the notebook from: `/ML_with_AWS_SageMaker/notebooks/Training-models-in-SageMaker-notebooks-part2.ipynb`. Select the **`conda_pytorch_p310`** kernel.

## Setup notebook as controller
Once your notebook is open, we can setup our SageMaker controller as usual:

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
bucket_name = 'sinkorswim-doejohn'  # replace with your S3 bucket name

# Define train/test filenames
train_filename = 'titanic_train.csv'
test_filename = 'titanic_test.csv'
```

Create a SageMaker session to manage interactions with Amazon SageMaker, such as training jobs, model deployments, and data input/output. As before, the region comes from the notebook instance; never hard-code it.

```python
region = boto3.Session().region_name  # inherited from the notebook instance; do not hard-code a region
session = sagemaker.Session()
sagemaker_client = session.sagemaker_client
print(f"region = {region}")
```

We should also record our local instance information to report this information during testing. First, the usual cell: start from the instance root, clone the lesson repository if needed, and import the helpers.

```python
%cd /home/ec2-user/SageMaker/
!test -d ML_with_AWS_SageMaker || git clone https://github.com/carpentries-incubator/ML_with_AWS_SageMaker.git
import sys
sys.path.insert(0, "/home/ec2-user/SageMaker/ML_with_AWS_SageMaker/scripts")
import helpers
```

```python
notebook_instance_name = 'sinkorswim-DoeJohn-TrainClassifier' # replace with your notebook instance name

local_instance_info = helpers.get_notebook_instance_info(notebook_instance_name, region)
local_instance = local_instance_info['InstanceType']
local_instance
```

If you are starting from a fresh notebook instance, grab the Titanic CSVs from S3 again (harmless if they are already here).

```python
s3.download_file(bucket_name, train_filename, train_filename)
s3.download_file(bucket_name, test_filename, test_filename)
```

## Training a neural network with SageMaker
Let's see how to do a similar experiment, but this time using PyTorch neural networks. We will again demonstrate how to test our custom model train script (`train_nn.py`) before deploying to SageMaker, and discuss some strategies (e.g., using a GPU) for improving train time when needed.

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

We reuse the `preprocess_data()` function from the lesson's `scripts/train_nn.py` rather than re-typing the preprocessing here: the same function will later be applied to the test set, so train and test are guaranteed to be transformed identically. It returns the fitted `StandardScaler` for exactly that reason, and we save it with `joblib` so the next episode can load it. Fitting a new scaler on the test set would leak test-set statistics into the evaluation.

```python
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from train_nn import preprocess_data

# Load and preprocess the Titanic dataset: encode Sex/Embarked, fill missing Age/Fare, standardize features
df = pd.read_csv(train_filename)
X, y, scaler = preprocess_data(df)   # scaler is fit on the training data only

# Split the data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Save the preprocessed data (and the fitted scaler) to our local jupyter environment
np.savez('train_data.npz', X_train=X_train, y_train=y_train)
np.savez('val_data.npz', X_val=X_val, y_val=y_val)
joblib.dump(scaler, 'scaler.joblib')
print(X_train.shape, X_val.shape)
```

Next, we will upload our compressed files (and the scaler) to our S3 bucket. Storage is fairly cheap on AWS (around $0.023 per GB per month), but be mindful of uploading too much data. It may be convenient to store a preprocessed version of the data at times, but try not to store too many versions that aren't being actively used.

```python
train_file = "train_data.npz"  # Local file path in your notebook environment
val_file = "val_data.npz"  # Local file path in your notebook environment
scaler_file = "scaler.joblib"

# Upload the training and validation files (and the scaler) to the bucket root
for f in [train_file, val_file, scaler_file]:
    s3.upload_file(f, bucket_name, f)

print("Files successfully uploaded to S3.")
```

## Testing on notebook instance
You should always test code thoroughly before scaling up and using more resources. Here, we will test our script using a small number of epochs (100) just to verify our setup is correct. Note that this script trains on the full batch every epoch (one forward/backward pass over all 713 rows), so an epoch takes milliseconds.

```python
import torch
import time as t # Measure training time locally

epochs = 100          # small smoke test
learning_rate = 0.001

start_time = t.time()
%run ML_with_AWS_SageMaker/scripts/train_nn.py --train train_data.npz --val val_data.npz --epochs {epochs} --learning_rate {learning_rate}
print(f"Local training time: {t.time() - start_time:.2f} seconds, instance_type = {local_instance}")
```

## Deploying PyTorch neural network via SageMaker
Now that we have tested things locally, we can train with a larger number of epochs (1000) through the PyTorch estimator. Our notebook instance is an `ml.m5.large`, and we will use the same instance type for the training job so the comparison is apples to apples; the point of this first run is the managed workflow, not more compute.

**Should we use a GPU?** Since this dataset is fairly small, we don't necessarily need a GPU for training. Considering costs, the `ml.m5.large` is about $0.115/hour, while the cheapest GPU instance (`ml.g4dn.xlarge`) is about $0.74/hour. However, for larger datasets (> 1 GB) and models, we may want to consider a GPU if training time becomes cumbersome (see [Instances for ML](https://carpentries-incubator.github.io/ML_with_AWS_SageMaker/instances-for-ML.html)). We'll test that in the exercise below.

#### Cost tracking
As in the XGBoost episode, every training job needs its own tags: a job does not inherit the notebook instance's tags, and an untagged job is anonymous on the bill. Define them once here and pass `tags=job_tags` to every estimator below.

```python
name = "John Doe" # replace with your name
project = "sinkorswim" # replace with your team name
purpose = "train_PyTorch"

job_tags = [
    {"Key": "Name", "Value": name},
    {"Key": "Project", "Value": project},
    {"Key": "Purpose", "Value": purpose},
]
```

We'll also bring back the `job_cost_report()` helper from the previous episode, plus the hourly rates of the instances we're about to use. Estimate first: 1000 full-batch epochs took a few seconds locally, so expect roughly 1 minute of training plus ~3 minutes of startup, i.e. about $0.01 on `ml.m5.large`.

```python
def job_cost_report(job_name: str, price_per_hour: float):
    desc = session.sagemaker_client.describe_training_job(TrainingJobName=job_name)
    billable = desc["BillableTimeInSeconds"] * desc["ResourceConfig"]["InstanceCount"]
    print(f"{job_name}: status {desc['TrainingJobStatus']}, training {desc.get('TrainingTimeInSeconds', 0)/60:.1f} min, billable {billable/60:.1f} min, cost ${billable/3600*price_per_hour:.2f} at ${price_per_hour}/h")
    return desc

# us-east-1 on-demand rates (2026-10-16); check the instances page for your region
prices = {"ml.m5.large": 0.115, "ml.m5.xlarge": 0.23, "ml.g4dn.xlarge": 0.74}
```

```python
from sagemaker.pytorch import PyTorch
from sagemaker.inputs import TrainingInput

epochs = 1000          # more epochs now that the script is known to work
instance_count = 1
instance_type = "ml.m5.large"
output_path = f's3://{bucket_name}/output_nn/' # this folder will auto-generate if it doesn't exist already

# Define max runtime in seconds to ensure you don't use more compute time than expected. Use a generous threshold (2x expected time but < 2 days) so that work isn't interrupted without any gains.
max_run = 2*60*60 # 2 hours

# Define the PyTorch estimator and pass hyperparameters as arguments
pytorch_estimator = PyTorch(
    base_job_name=notebook_instance_name,
    max_run=max_run, # in seconds; always include (max 48 hours)
    entry_point="ML_with_AWS_SageMaker/scripts/train_nn.py",
    role=role,
    tags=job_tags,
    instance_type=instance_type, # with this small dataset, we don't necessarily need a GPU for fast training. 
    instance_count=instance_count,
    framework_version="2.3",  # if fit() fails with an image error, check available versions with: sagemaker.image_uris.retrieve("pytorch", region, version="2.3", py_version="py311", image_scope="training", instance_type="ml.m5.large")
    py_version="py311",
    output_path=output_path,
    sagemaker_session=session,
    hyperparameters={
        "train": "/opt/ml/input/data/train/train_data.npz",  # SageMaker will mount this path
        "val": "/opt/ml/input/data/val/val_data.npz",        # SageMaker will mount this path
        "epochs": epochs,
        "learning_rate": learning_rate
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

```python
desc_cpu = job_cost_report(pytorch_estimator.latest_training_job.name, prices[instance_type])
```

Notice how little of the wall-clock time was your script: for a job this small, almost all of the billable minutes are startup.

#### How `train_nn.py` uses a GPU when one is present
Using a GPU requires only minor changes in a training script, and `train_nn.py` already has them: it checks for GPU availability and moves the model and tensors to the GPU device if one is found.

```text
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TitanicNet().to(device)
X_train, y_train = X_train.to(device), y_train.to(device)
```

So the same script runs unchanged on a CPU or a GPU instance; all we change is `instance_type`.

::::::::::::::::::::::::::::::::::::::: challenge

### CPU or GPU: predict, then measure

We're about to run the identical job on an `ml.g4dn.xlarge` (one NVIDIA T4 GPU, about $0.74/hour) instead of the `ml.m5.large` ($0.115/hour). Before you run the cell:

1. Predict which instance finishes first (total wall-clock, as `fit()` reports it).
2. Predict which costs less for this job.

Then run the GPU job, call `job_cost_report()` for it, and compute cost = rate x billable minutes for both runs. Were you right? Explain the result.

```python
instance_type = "ml.g4dn.xlarge"
instance_count = 1

pytorch_estimator_gpu = PyTorch(
    base_job_name=notebook_instance_name,
    max_run=max_run, # in seconds; always include (max 48 hours)
    entry_point="ML_with_AWS_SageMaker/scripts/train_nn.py",
    role=role,
    tags=job_tags,
    instance_type=instance_type,
    instance_count=instance_count,
    framework_version="2.3",
    py_version="py311",
    output_path=output_path,
    sagemaker_session=session,
    hyperparameters={
        "train": "/opt/ml/input/data/train/train_data.npz",
        "val": "/opt/ml/input/data/val/val_data.npz",
        "epochs": epochs,
        "learning_rate": learning_rate
    }
)

start = t.time()
pytorch_estimator_gpu.fit({"train": train_input, "val": val_input})
end = t.time()
print(f"Runtime for training on SageMaker: {end - start:.2f} seconds, instance_type: {instance_type}, instance_count: {instance_count}")

desc_gpu = job_cost_report(pytorch_estimator_gpu.latest_training_job.name, prices[instance_type])

for label, d, rate in [("CPU ml.m5.large", desc_cpu, prices["ml.m5.large"]), ("GPU ml.g4dn.xlarge", desc_gpu, prices["ml.g4dn.xlarge"])]:
    billable_min = d["BillableTimeInSeconds"] / 60
    print(f"{label}: billable {billable_min:.1f} min x ${rate}/h = ${billable_min/60*rate:.3f}")
```

::::::::::::::::::::::::::::::::::::::: solution

### Solution

Typically the CPU run finishes about as fast or faster, and it is several times cheaper. Both jobs spend roughly the same few minutes on startup (provisioning, pulling the container, downloading data), and the GPU container is larger, so it often takes a little *longer* to start. The actual training is a few seconds on either instance: the network has about 2,600 parameters and the whole training set is 713 rows, so each full-batch step is tiny and the T4 sits almost idle, and the one-time CUDA initialization adds a few seconds of its own. Startup dominates both bills, and the GPU bill is about 6x higher per minute (0.74 / 0.115), so the GPU costs roughly 6x more for no speedup.

A GPU pays off when the per-step work is large enough to keep it busy: bigger models, bigger batches, or datasets of hundreds of MB and up. If you ever do need more GPU than a `g4dn.xlarge`, the next sensible step is an `ml.g5.xlarge` (~$1.41/hour), not a multi-GPU box.

:::::::::::::::::::::::::::::::::::::::::::::::::::::

:::::::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::: callout
#### GPUs can be slow for small datasets/models
This performance discrepancy might be due to the following factors:

1. **Small dataset/model size**: When datasets and models are small, the overhead of transferring data between the CPU and GPU, as well as managing the GPU, can actually slow things down. For very small models and datasets, CPUs are often faster since there's minimal data to process.

2. **GPU initialization overhead**: Every time a training job starts on a GPU, there's a small overhead for initializing CUDA libraries. For short jobs, this setup time can make the GPU appear slower overall.

3. **Batch size**: GPUs perform best with larger batch sizes since they can process many data points in parallel. If the batch size is too small, the GPU is underutilized, leading to suboptimal performance.

4. **Instance type**: The `ml.g4dn` series is the entry-level GPU family; an `ml.g5.xlarge` (A10G GPU) is the next step up when a model is genuinely GPU-bound. Neither helps a job that spends its time starting up.

If training time continues to be critical, sticking with a CPU instance may be the best approach for smaller datasets. For larger, more complex models and datasets, the GPU's advantages should become more apparent.
::::::::::::::::::::::::::::::::::::::::::::::::::::::

## When training takes too long: more instances?

The previous episode gave the rule of thumb: upgrade the single instance first, and consider multiple instances only when the dataset no longer fits in one instance's memory (tens of GB or more) or a single-instance run takes longer than about 10 hours. The GPU exercise was Option 1. Let's now look honestly at Option 2, because the obvious thing to try, `instance_count=2`, does something different from what most people expect.

### What `instance_count=2` does on its own

Setting `instance_count=2` on an estimator makes SageMaker start two instances and run **the full training script on each of them**. Nothing is split:

- `TrainingInput` defaults to `distribution="FullyReplicated"`, so every instance downloads the complete dataset.
- `train_nn.py` has no distributed code, so each instance trains its own independent copy of the model on all of the data for all 1000 epochs, and each saves a model.
- SageMaker bills `BillableTimeInSeconds x InstanceCount`.

So without a distribution strategy, two instances means the same job twice, in parallel, at double the cost. Let's demonstrate that rather than take it on faith. Estimate first: about the same 4 minutes of wall-clock as the single-instance run, but billed for two instances, so about $0.02 on `ml.m5.large`.

```python
instance_count = 2   # demonstration: NOT expected to be faster, and expected to cost double
instance_type = "ml.m5.large"

pytorch_estimator_2x = PyTorch(
    base_job_name=notebook_instance_name,
    max_run=max_run, # in seconds; always include (max 48 hours)
    entry_point="ML_with_AWS_SageMaker/scripts/train_nn.py",
    role=role,
    tags=job_tags,
    instance_type=instance_type,
    instance_count=instance_count,
    framework_version="2.3",
    py_version="py311",
    output_path=output_path,
    sagemaker_session=session,
    hyperparameters={
        "train": "/opt/ml/input/data/train/train_data.npz",
        "val": "/opt/ml/input/data/val/val_data.npz",
        "epochs": epochs,
        "learning_rate": learning_rate
    }
)

start = t.time()
pytorch_estimator_2x.fit({"train": train_input, "val": val_input})
end = t.time()
print(f"Runtime for training on SageMaker: {end - start:.2f} seconds, instance_type: {instance_type}, instance_count: {instance_count}")

desc_2x = job_cost_report(pytorch_estimator_2x.latest_training_job.name, prices[instance_type])
print(f"Single-instance job billed {desc_cpu['BillableTimeInSeconds']/60:.1f} min; two-instance job billed {desc_2x['BillableTimeInSeconds']*2/60:.1f} min")
```

Look at the CloudWatch log for this job (SageMaker console, Training jobs, your job, View logs): there are two log streams, `algo-1` and `algo-2`, and each one prints all 1000 epochs. The wall-clock time is about the same as the single-instance run (slightly longer, if anything, because two instances have to be provisioned), and the billable minutes are double.

::::::::::::::::::::::::::::::::::::::: callout
#### What real distributed training needs

To actually split the work across instances you need two things the demo above did not have:

- **Data parallelism** (the common case; each instance holds a full copy of the model and trains on a different slice of the data, exchanging gradients after every step). For PyTorch, pass `distribution={"torch_distributed": {"enabled": True}}` to the estimator *and* write the script for it: initialise a process group, wrap the model in `DistributedDataParallel`, and give each rank a different shard of the data (e.g. a `DistributedSampler`). SageMaker's own data-parallel library (`smdistributed.dataparallel`) is the same idea with a faster all-reduce on supported GPU instances.
- **Model parallelism** (for models too large for one device's memory; the layers themselves are spread across devices). This is for large language models and the like, needs `smdistributed.modelparallel` or an equivalent library, and is well beyond what a tabular network needs.

For the **built-in XGBoost** image from the previous episode the recipe is simpler, because the algorithm already knows how to train across instances: split the training CSV into several S3 objects under one prefix and pass `TrainingInput(prefix, content_type="text/csv", distribution="ShardedByS3Key")`, so each instance receives a different subset of the files. Even then, a dataset of 891 rows will only show overhead.

Either way, the rule of thumb stands: a bigger single instance first, and multiple instances only once data no longer fits in memory or a run exceeds ~10 hours.
:::::::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: keypoints

- **Efficient data handling**: The `.npz` format is optimized for efficient loading, reducing I/O overhead and enabling batch compatibility for PyTorch's `DataLoader`. Save the fitted scaler alongside it so the test set can be transformed identically later.
- **Startup dominates small jobs**: Every SageMaker training job pays a few minutes of provisioning, container pull and data download; `job_cost_report()` shows how much of the bill that is.
- **GPU training**: Beneficial for larger models or datasets, but for a tiny network the GPU sits idle while you pay ~6x the CPU rate; choose the instance to match the per-step work.
- **`instance_count` alone is not distributed training**: Without `distribution=` on the estimator and distributed code in the script, `instance_count=2` runs the full script on the full data on each instance and bills twice.
- **Data vs. model parallelism**: Data parallelism (DDP, each instance on a data shard, gradients all-reduced) is the usual tool when it is needed; model parallelism is for models that do not fit on one device.
- **Testing locally first**: Before deploying large-scale training in SageMaker, test locally with a smaller setup to ensure code correctness and efficient resource usage.

::::::::::::::::::::::::::::::::::::::::::::::::

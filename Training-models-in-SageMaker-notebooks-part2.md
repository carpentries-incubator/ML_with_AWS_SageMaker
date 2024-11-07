---
title: "Training Models in SageMaker: PyTorch Example"
teaching: 20
exercises: 10
---

:::::::::::::::::::::::::::::::::::::: questions 

- When should you consider using a GPU instance for training neural networks in SageMaker, and what are the benefits and limitations?
- How does SageMaker handle data parallelism and model parallelism, and which is suitable for typical neural network training?

::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: objectives

- Preprocess the Titanic dataset for efficient training using PyTorch.
- Save and upload training and validation data in `.npz` format to S3.
- Understand the trade-offs between CPU and GPU training for smaller datasets.
- Deploy a PyTorch model to SageMaker and evaluate instance types for training performance.
- Differentiate between data parallelism and model parallelism, and determine when each is appropriate in SageMaker.

::::::::::::::::::::::::::::::::::::::::::::::::

## Initial setup
To keep things organized, you may wish to open a fresh jupyter notebook (pytorch environment). Name your notebook something along the lines of, "Training-part2.ipynb". Once your notebook is open, we can setup our SageMaker controller as usual:

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

We should also record our local instance information to report this information during testing.

```python
import AWS_helpers.helpers as helpers
notebook_instance_name = 'MyAwesomeTeam-ChrisEndemann-Titanic-Train-Tune-Xgboost-NN'
local_instance_info = helpers.get_notebook_instance_info(notebook_instance_name)
local_instance = local_instance_info['InstanceType']
local_instance
````

    'ml.t3.medium'

  

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


## Testing on notebook instance
You should always test code thoroughly before scaling up and using more resources. Here, we will test our script using a small number of epochs — just to verify our setup is correct.


```python
import torch
import time as t # Measure training time locally

epochs = 1000
learning_rate = 0.001

start_time = t.time()
%run  AWS_helpers/train_nn.py --train train_data.npz --val val_data.npz --epochs {epochs} --learning_rate {learning_rate}
print(f"Local training time: {t.time() - start_time:.2f} seconds, instance_type = {local_instance}")

```


## Deploying PyTorch neural network via SageMaker
Now that we have tested things locally, we can try to train with a larger number of epochs and a better instance selected. We can do this easily by invoking the PyTorch estimator. Our notebook is currently configured to use ml.m5.large. We can upgrade this to `ml.m5.xlarge` with the below code (using our notebook as a controller). 

**Should we use a GPU?**: Since this dataset is farily small, we don't necessarily need a GPU for training. Considering costs, the m5.xlarge is `$0.17/hour`, while the cheapest GPU instance is `$0.75/hour`. However, for larger datasets (> 1 GB) and models, we may want to consider a GPU if training time becomes cumbersome (see [Instances for ML](https://docs.google.com/spreadsheets/d/1uPT4ZAYl_onIl7zIjv5oEAdwy4Hdn6eiA9wVfOBbHmY/edit?usp=sharing). If that doesn't work, we can try distributed computing (setting instance > 1). More on this in the next section.


```python
from sagemaker.pytorch import PyTorch
from sagemaker.inputs import TrainingInput

instance_count = 1
instance_type="ml.m5.large"
output_path = f's3://{bucket_name}/output_nn/' # this folder will auto-generate if it doesn't exist already

# Define the PyTorch estimator and pass hyperparameters as arguments
pytorch_estimator = PyTorch(
    entry_point="AWS_helpers/train_nn.py",
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
    
    2024-11-03 21:27:03 Uploading - Uploading generated training model
    2024-11-03 21:27:03 Completed - Training job completed
    Training seconds: 135
    Billable seconds: 135
    Runtime for training on SageMaker: 197.62 seconds, instance_type: ml.m5.large, instance_count: 1


## Deploying PyTorch neural network via SageMaker with a GPU instance

In this section, we'll implement the same procedure as above, but using a GPU-enabled instance for potentially faster training. While GPU instances are more expensive, they can be cost-effective for larger datasets or more complex models that require significant computational power.

#### Selecting a GPU Instance
For a small dataset like ours, we don't strictly need a GPU, but for larger datasets or more complex models, a GPU can reduce training time. Here, we'll select an `ml.g4dn.xlarge` instance, which provides a single GPU and costs approximately `$0.75/hour` (check [Instances for ML](https://docs.google.com/spreadsheets/d/1uPT4ZAYl_onIl7zIjv5oEAdwy4Hdn6eiA9wVfOBbHmY/edit?usp=sharing) for detailed pricing).

#### Code modifications for GPU use
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

instance_count = 1
instance_type="ml.g4dn.xlarge"
output_path = f's3://{bucket_name}/output_nn/'

# Define the PyTorch estimator and pass hyperparameters as arguments
pytorch_estimator_gpu = PyTorch(
    entry_point="AWS_helpers/train_nn.py",
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
        "learning_rate": learning_rate
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


::::::::::::::::::::::::::: callout
#### GPUs can be slow for small datasets/models
This performance discrepancy might be due to the following factors:

1. **Small Ddtaset/model size**: When datasets and models are small, the overhead of transferring data between the CPU and GPU, as well as managing the GPU, can actually slow things down. For very small models and datasets, CPUs are often faster since there's minimal data to process.

2. **GPU initialization overhead**: Every time a training job starts on a GPU, there's a small overhead for initializing CUDA libraries. For short jobs, this setup time can make the GPU appear slower overall.

3. **Batch size**: GPUs perform best with larger batch sizes since they can process many data points in parallel. If the batch size is too small, the GPU is underutilized, leading to suboptimal performance. You may want to try increasing the batch size to see if this reduces training time.

4. **Instance type**: Some GPU instances, like the `ml.g4dn` series, have less computational power than the larger `p3` series. They're better suited for inference or lightweight tasks rather than intense training, so a more powerful instance (e.g., `ml.p3.2xlarge`) could help for larger tasks.

If training time continues to be critical, sticking with a CPU instance may be the best approach for smaller datasets. For larger, more complex models and datasets, the GPU's advantages should become more apparent.

## Distributed Training for Neural Networks in SageMaker
In the event that you do need distributed computing to achieve reasonable train times (remember to try an upgraded instance first!), simply adjust the instance count to a number between 2 and 5. Beyond 5 instances, you'll see diminishing returns and may be needlessly spending extra money/compute-energy.
::::::::::::::::::::::::::::::::::::::::::::::::::::::


```python
from sagemaker.pytorch import PyTorch
from sagemaker.inputs import TrainingInput
import time as t

instance_count = 2 # increasing to 2 to see if it has any benefit (likely won't see any with this small dataset)
instance_type="ml.m5.xlarge"
output_path = f's3://{bucket_name}/output_nn/'

# Define the PyTorch estimator and pass hyperparameters as arguments
pytorch_estimator = PyTorch(
    entry_point="AWS_helpers/train_nn.py",
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
    
    2024-11-03 21:36:35 Uploading - Uploading generated training model
    2024-11-03 21:36:47 Completed - Training job completed
    Training seconds: 228
    Billable seconds: 228
    Runtime for training on SageMaker: 198.36 seconds, instance_type: ml.m5.xlarge, instance_count: 2


### Distributed training for neural networks in SageMaker: Understanding training strategies and how epochs are managed
Amazon SageMaker provides two main strategies for distributed training: **data parallelism** and **model parallelism**. Understanding which strategy will be used depends on the model size and the configuration of your SageMaker training job, as well as the default settings of the specific SageMaker Estimator you are using.

#### 1. **Data parallelism (most common for mini-batch SGD)**
   - **How it works**: In data parallelism, each instance in the cluster (e.g., multiple `ml.m5.xlarge` instances) maintains a **complete copy of the model**. The **training dataset is split across instances**, and each instance processes a different subset of data simultaneously. This enables multiple instances to complete forward and backward passes on different data batches independently.
   - **Epoch distribution**: Even though each instance processes all the specified epochs, they only work on a portion of the dataset for each epoch. After each batch, instances synchronize their gradient updates across all instances using a method such as *all-reduce*. This ensures that while each instance is working with a unique data batch, the model weights remain consistent across instances.
   - **Key insight**: Because all instances process the specified number of epochs and synchronize weight updates between batches, each instance's training contributes to a cohesive, shared model. The **effective epoch count across instances appears to be shared** because data parallelism allows each instance to handle a fraction of the data per epoch, not the epochs themselves. Data parallelism is well-suited for models that can fit into a single instance's memory and benefit from increased data throughput.

#### 2. **Model parallelism (best for large models)**
   - **How it works**: Model parallelism divides the model itself across multiple instances, not the data. This approach is best suited for very large models that cannot fit into a single GPU or instance's memory (e.g., large language models).
   - **Epoch distribution**: The model is partitioned so that each instance is responsible for specific layers or components. Data flows sequentially through these partitions, where each instance processes a part of each batch and passes it to the next instance.
   - **Key insight**: This approach is more complex due to the dependency between model components, so **synchronization occurs across the model layers rather than across data batches**. Model parallelism generally suits scenarios with exceptionally large model architectures that exceed memory limits of typical instances.

### Determining which distributed training strategy is used
SageMaker will select the distributed strategy based on:
   - **Framework and Estimator configuration**: Most deep learning frameworks in SageMaker default to data parallelism, especially when using PyTorch or TensorFlow with standard configurations.
   - **Model and data size**: If you specify a model that exceeds a single instance's memory capacity, SageMaker may switch to model parallelism if configured for it.
   - **Instance count**: When you specify `instance_count > 1` in your Estimator with a deep learning model, SageMaker will use data parallelism by default unless explicitly configured for model parallelism.

You observed that each instance ran all epochs with `instance_count=2` and 10,000 epochs, which aligns with data parallelism. Here, each instance processed the full set of epochs independently, but each batch of data was different, and the gradient updates were synchronized across instances.

::::::::::::::::::::::::::::::::::::: keypoints

- **Efficient data handling**: The `.npz` format is optimized for efficient loading, reducing I/O overhead and enabling batch compatibility for PyTorch’s `DataLoader`.
- **GPU training**: While beneficial for larger models or datasets, GPUs may introduce overhead for smaller tasks; selecting the right instance type is critical for cost-efficiency.
- **Data parallelism vs. model parallelism**: Data parallelism splits data across instances and synchronizes model weights, suitable for typical neural network tasks. Model parallelism, which divides model layers, is ideal for very large models that exceed memory capacity.
- **SageMaker configuration**: By adjusting instance counts and types, SageMaker supports scalable training setups. Starting with CPU training and scaling as needed with GPUs or distributed setups allows for performance optimization.
- **Testing locally first**: Before deploying large-scale training in SageMaker, test locally with a smaller setup to ensure code correctness and efficient resource usage.

::::::::::::::::::::::::::::::::::::::::::::::::

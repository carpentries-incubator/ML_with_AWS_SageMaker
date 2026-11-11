"""Helper functions used throughout the lesson.

Learners clone the lesson repository once, in the Accessing S3 episode, into
``/home/ec2-user/SageMaker/ML_with_AWS_SageMaker/`` and import this module with::

    import sys
    sys.path.insert(0, "/home/ec2-user/SageMaker/ML_with_AWS_SageMaker/scripts")
    import helpers
"""
import boto3
from datetime import datetime, timedelta
import os

def get_s3_bucket_size(bucket_name):
    """
    Calculate the total size of an S3 bucket in megabytes and gigabytes.

    Parameters:
        bucket_name (str): The name of the S3 bucket.

    Returns:
        dict: A dictionary containing the total size in megabytes and gigabytes.
    """
    s3 = boto3.client('s3')
    total_size_bytes = 0

    # List and sum the size of all objects in the bucket
    paginator = s3.get_paginator('list_objects_v2')
    for page in paginator.paginate(Bucket=bucket_name):
        for obj in page.get('Contents', []):
            total_size_bytes += obj['Size']

    # Convert the total size to megabytes and gigabytes for readability
    total_size_mb = total_size_bytes / (1024 ** 2)
    total_size_gb = total_size_bytes / (1024 ** 3)

    return {
        'size_mb': total_size_mb,
        'size_gb': total_size_gb
    }

def calculate_s3_storage_cost(bucket_name):
    """
    Calculates the estimated monthly storage cost of an S3 bucket based on its total size.

    Parameters:
        bucket_name (str): The name of the S3 bucket.

    Returns:
        tuple: A tuple containing the estimated monthly storage cost in USD and the total storage size in gigabytes.
    """
    # Retrieve bucket size using the get_s3_bucket_size function
    size_info = get_s3_bucket_size(bucket_name)
    total_size_gb = size_info['size_gb']

    # AWS S3 Standard Storage pricing for US East (N. Virginia) region
    # Pricing tiers as of November 1, 2024
    first_50_tb_price_per_gb = 0.023  # per GB for the first 50 TB
    next_450_tb_price_per_gb = 0.022  # per GB for the next 450 TB
    over_500_tb_price_per_gb = 0.021  # per GB for storage over 500 TB

    # Calculate the cost based on the size
    if total_size_gb <= 50 * 1024:
        cost = total_size_gb * first_50_tb_price_per_gb
    elif total_size_gb <= 500 * 1024:
        cost = (50 * 1024 * first_50_tb_price_per_gb) + \
               ((total_size_gb - 50 * 1024) * next_450_tb_price_per_gb)
    else:
        cost = (50 * 1024 * first_50_tb_price_per_gb) + \
               (450 * 1024 * next_450_tb_price_per_gb) + \
               ((total_size_gb - 500 * 1024) * over_500_tb_price_per_gb)

    return cost, total_size_gb

def list_S3_objects(bucket_name):
    """
    Lists all objects in a specified S3 bucket.

    Parameters:
        bucket_name (str): The name of the S3 bucket.

    Returns:
        list: A list of object keys (paths) in the bucket, or an empty list if the bucket is empty.
    """
    s3 = boto3.client('s3')
    file_list = []

    # Use paginator to list all objects in the bucket
    paginator = s3.get_paginator('list_objects_v2')
    for page in paginator.paginate(Bucket=bucket_name):
        for obj in page.get('Contents', []):
            file_list.append(obj['Key'])  # Add each object's key to the list

    # If the bucket is empty, return an empty list
    return file_list
    
def show_bucket(bucket_name, prefix="", max_files=15):
    """Print what is under s3://bucket_name/prefix: each sub-folder with its object count
    and size, then the files at this level (up to max_files). Read-only: one list call per folder."""
    s3 = boto3.client("s3")
    paginator = s3.get_paginator("list_objects_v2")
    folders, files = [], []
    for page in paginator.paginate(Bucket=bucket_name, Prefix=prefix, Delimiter="/"):   # Delimiter="/" groups keys into "folders"
        folders += [cp["Prefix"] for cp in page.get("CommonPrefixes", [])]
        files += [obj for obj in page.get("Contents", []) if obj["Key"] != prefix]
    print(f"s3://{bucket_name}/{prefix}")
    for folder in folders:
        n, size = 0, 0
        for page in paginator.paginate(Bucket=bucket_name, Prefix=folder):   # everything under the folder, recursively
            for obj in page.get("Contents", []):
                n += 1
                size += obj["Size"]
        print(f"  {folder.removeprefix(prefix):<40} {n:>6} objects  {size / 1024**2:>10,.1f} MB")
    for obj in files[:max_files]:
        print(f"  {obj['Key'].removeprefix(prefix):<40}         {obj['Size'] / 1024**2:>10,.1f} MB")
    if len(files) > max_files:
        print(f"  ... and {len(files) - max_files} more files at this level")
    if not folders and not files:
        print("  (nothing here, or no permission to list it)")


def get_instance_cost(instance_type, days=1):
    """
    Fetches the cost for a specific instance type over a specified number of days.
    """
    client = boto3.client('ce', region_name='us-east-1')
    end = datetime.utcnow().date()
    start = end - timedelta(days=days)
    response = client.get_cost_and_usage(
        TimePeriod={'Start': start.strftime('%Y-%m-%d'), 'End': end.strftime('%Y-%m-%d')},
        Granularity='DAILY',
        Metrics=['UnblendedCost'],
        Filter={
            'And': [
                {'Dimensions': {'Key': 'SERVICE', 'Values': ['Amazon Elastic Compute Cloud - Compute']}},
                {'Dimensions': {'Key': 'INSTANCE_TYPE', 'Values': [instance_type]}}
            ]
        }
    )
    total_cost = sum(float(day['Total']['UnblendedCost']['Amount']) for day in response['ResultsByTime'])
    return total_cost

def list_running_ec2_instances():
    """
    Lists all running EC2 instances in the account with instance ID and instance type.
    """
    ec2_client = boto3.client('ec2')
    response = ec2_client.describe_instances(Filters=[{'Name': 'instance-state-name', 'Values': ['running']}])
    instances = []
    for reservation in response['Reservations']:
        for instance in reservation['Instances']:
            instances.append({'InstanceId': instance['InstanceId'], 'InstanceType': instance['InstanceType']})
    return instances

def get_sagemaker_notebook_status(notebook_instance_name):
    """
    Fetches the status of a SageMaker notebook instance.
    """
    sagemaker_client = boto3.client('sagemaker')
    response = sagemaker_client.describe_notebook_instance(NotebookInstanceName=notebook_instance_name)
    return response['NotebookInstanceStatus']

def get_notebook_instance_info(notebook_instance_name, region=None):
    """
    Fetches the status and instance type of a specific SageMaker notebook instance.

    Parameters:
        notebook_instance_name (str): Name of the SageMaker notebook instance (not an .ipynb file).
        region (str or None): AWS region of the instance. Defaults to the session's region
            (e.g. ``boto3.Session().region_name``); never hard-code a region in lesson code.

    Returns:
        dict: {'Status': ..., 'InstanceType': ...}
    """
    if region is None:
        region = boto3.Session().region_name
    sagemaker_client = boto3.client('sagemaker', region_name=region)
    response = sagemaker_client.describe_notebook_instance(NotebookInstanceName=notebook_instance_name)
    return {'Status': response['NotebookInstanceStatus'], 'InstanceType': response['InstanceType']}

def calculate_tuning_job_time(tuner, region=None):
    """
    Sums training and billable time across every training job launched by a SageMaker
    hyperparameter tuning job. Matches the call in the Hyperparameter-tuning episode:
    ``helpers.calculate_tuning_job_time(tuner, region)``.

    SageMaker bills per second: each training job's ``BillableTimeInSeconds`` (instance
    reserved for you, including container pull and data download) multiplied by its
    instance count. The tuning job itself is not billed.

    Parameters:
        tuner (sagemaker.tuner.HyperparameterTuner): A tuner whose ``.fit()`` has been called.
        region (str or None): AWS region. Defaults to the session's region.

    Returns:
        dict: instance type, max_jobs, max_parallel_jobs, number of completed jobs, and total
              training / billable seconds (billable already multiplied by instance count).
    """
    if region is None:
        region = boto3.Session().region_name
    sagemaker_client = boto3.client('sagemaker', region_name=region)

    tuning_job_name = tuner.latest_tuning_job.name
    tuning_job_desc = sagemaker_client.describe_hyper_parameter_tuning_job(
        HyperParameterTuningJobName=tuning_job_name)
    instance_type = tuning_job_desc['TrainingJobDefinition']['ResourceConfig']['InstanceType']
    limits = tuning_job_desc['HyperParameterTuningJobConfig']['ResourceLimits']

    # Page through all child training jobs of the tuning job
    paginator = sagemaker_client.get_paginator('list_training_jobs_for_hyper_parameter_tuning_job')
    job_names = []
    for page in paginator.paginate(HyperParameterTuningJobName=tuning_job_name, StatusEquals='Completed'):
        job_names.extend(j['TrainingJobName'] for j in page['TrainingJobSummaries'])

    total_training_seconds = 0
    total_billable_seconds = 0
    for job_name in job_names:
        job_desc = sagemaker_client.describe_training_job(TrainingJobName=job_name)
        total_training_seconds += job_desc.get('TrainingTimeInSeconds', 0)
        total_billable_seconds += job_desc.get('BillableTimeInSeconds', 0) * job_desc['ResourceConfig']['InstanceCount']

    print(f"Tuning job: {tuning_job_name}")
    print(f"Instance Type: {instance_type}")
    print(f"Max Jobs: {limits['MaxNumberOfTrainingJobs']}")
    print(f"Max Parallel Jobs: {limits['MaxParallelTrainingJobs']}")
    print(f"Completed training jobs: {len(job_names)}")
    print(f"Total training time across all jobs: {total_training_seconds / 60:.1f} minutes")
    print(f"Total billable time across all jobs: {total_billable_seconds / 60:.1f} minutes (x instance count)")

    return {
        'TuningJobName': tuning_job_name,
        'InstanceType': instance_type,
        'MaxJobs': limits['MaxNumberOfTrainingJobs'],
        'MaxParallelJobs': limits['MaxParallelTrainingJobs'],
        'CompletedJobs': len(job_names),
        'TotalTrainingSeconds': total_training_seconds,
        'TotalBillableSeconds': total_billable_seconds,
    }

def get_current_costs(days=1):
    """
    Fetches the current total costs for the specified time period across all AWS services.
    """
    client = boto3.client('ce', region_name='us-east-1')
    end = datetime.utcnow().date()
    start = end - timedelta(days=days)
    response = client.get_cost_and_usage(
        TimePeriod={'Start': start.strftime('%Y-%m-%d'), 'End': end.strftime('%Y-%m-%d')},
        Granularity='DAILY',
        Metrics=['UnblendedCost']
    )
    total_cost = sum(float(day['Total']['UnblendedCost']['Amount']) for day in response['ResultsByTime'])
    return total_cost


def convert_files(direction="notebook_to_python", directory="."):
    """Convert every notebook in ``directory`` to a .py script, or every .py script
    to a notebook, with jupytext (``pip install jupytext``).

    direction: "notebook_to_python" (default) or "python_to_notebook".
    Converts *every* matching file in the directory, so run it inside a folder that
    holds only the files you mean to convert.
    """
    import subprocess
    if direction == "notebook_to_python":
        src_ext, dst_ext, fmt = ".ipynb", ".py", "py"
    elif direction == "python_to_notebook":
        src_ext, dst_ext, fmt = ".py", ".ipynb", "notebook"
    else:
        raise ValueError("direction must be 'notebook_to_python' or 'python_to_notebook'")
    for name in sorted(os.listdir(directory)):
        if not name.endswith(src_ext):
            continue
        src = os.path.join(directory, name)
        dst = os.path.join(directory, name[: -len(src_ext)] + dst_ext)
        subprocess.run(["jupytext", "--to", fmt, src, "--output", dst], check=True)
        print(f"Converted {name} to {os.path.basename(dst)}")


if __name__ == "__main__":
    # Example: Get the cost of a specific EC2 instance type over the last day
    instance_type = 'g4dn.xlarge'  # Replace with your desired instance type
    cost = get_instance_cost(instance_type)
    print(f"Cost for {instance_type} over the last day: ${cost:.2f}")

    # Example: List all running EC2 instances with their IDs and types
    running_instances = list_running_ec2_instances()
    print("Running EC2 Instances:")
    for instance in running_instances:
        print(f"Instance ID: {instance['InstanceId']}, Instance Type: {instance['InstanceType']}")

    # Example: Check the status of a specific SageMaker notebook instance
    notebook_instance_name = 'YourNotebookInstanceName'  # Replace with your notebook instance name
    status = get_sagemaker_notebook_status(notebook_instance_name)
    print(f"SageMaker Notebook '{notebook_instance_name}' Status: {status}")

    # Example: Fetch status and instance type of a specific notebook instance
    notebook_info = get_notebook_instance_info(notebook_instance_name, boto3.Session().region_name)
    print(f"Notebook Instance Info for '{notebook_instance_name}':")
    print(f"Status: {notebook_info['Status']}, Instance Type: {notebook_info['InstanceType']}")

    # Example: Get the total AWS cost across all services over the last 7 days
    total_cost = get_current_costs(days=7)
    print(f"Total AWS costs over the last 7 days: ${total_cost:.2f}")


# ---------------------------------------------------------------------------
# Fine-tuning cost estimator (fine-tuning episode, "Back of the envelope")
# ---------------------------------------------------------------------------

# SageMaker training instances the lesson refers to: GPUs per box, GPU memory per GPU (GB),
# bf16 dense peak per GPU (TFLOPS, from NVIDIA's data sheets; the sheets' headline figures for the L4 and
# L40S are with sparsity, twice these), the us-east-1 on-demand
# price per instance-hour from the AWS price list of 2026-10-16 (for the whole box), the
# instance's network bandwidth (Gbit/s, from the instance's spec sheet) and, on multi-GPU boxes,
# the GPU-to-GPU link per GPU (GB/s: NVLink on p4d/p5, PCIe on g5/g6e). Links are approximate.
GPU_INSTANCES = {
    "ml.g4dn.xlarge":   {"gpus": 1, "gpu": "T4",        "gpu_mem_gb": 16, "tflops_bf16": 65,  "price": 0.74, "net_gbps": 25,  "note": "no bf16 support; modern LLM fine-tuning is slow or fails"},
    "ml.g6.xlarge":     {"gpus": 1, "gpu": "L4",        "gpu_mem_gb": 24, "tflops_bf16": 61,  "price": 1.13, "net_gbps": 10},
    "ml.g5.xlarge":     {"gpus": 1, "gpu": "A10G",      "gpu_mem_gb": 24, "tflops_bf16": 70,  "price": 1.41, "net_gbps": 10},
    "ml.g5.2xlarge":    {"gpus": 1, "gpu": "A10G",      "gpu_mem_gb": 24, "tflops_bf16": 70,  "price": 1.52, "net_gbps": 10},
    "ml.g6e.xlarge":    {"gpus": 1, "gpu": "L40S",      "gpu_mem_gb": 48, "tflops_bf16": 181, "price": 2.61, "net_gbps": 20},
    "ml.g6e.2xlarge":   {"gpus": 1, "gpu": "L40S",      "gpu_mem_gb": 48, "tflops_bf16": 181, "price": 2.80, "net_gbps": 20},
    "ml.g5.12xlarge":   {"gpus": 4, "gpu": "A10G",      "gpu_mem_gb": 24, "tflops_bf16": 70,  "price": 7.09, "net_gbps": 40, "gpu_link_gbs": 32},
    "ml.g6e.12xlarge":  {"gpus": 4, "gpu": "L40S",      "gpu_mem_gb": 48, "tflops_bf16": 181, "price": 13.12, "net_gbps": 400, "gpu_link_gbs": 32},
    "ml.p4d.24xlarge":  {"gpus": 8, "gpu": "A100 40GB", "gpu_mem_gb": 40, "tflops_bf16": 312, "price": 25.25, "net_gbps": 400, "gpu_link_gbs": 600},
    "ml.p5.48xlarge":   {"gpus": 8, "gpu": "H100 80GB", "gpu_mem_gb": 80, "tflops_bf16": 989, "price": 63.30, "net_gbps": 3200, "gpu_link_gbs": 900},
}

# The CPU instances the lesson uses, with the same us-east-1 on-demand prices as the Instances for ML page
CPU_INSTANCES = {
    "ml.t3.medium": {"price": 0.05},
    "ml.m5.large":  {"price": 0.115},
    "ml.m5.xlarge": {"price": 0.23},
}


def instance_price(instance_type):
    """Hourly price of an instance the lesson uses, from the two tables above; None if unknown."""
    spec = GPU_INSTANCES.get(instance_type) or CPU_INSTANCES.get(instance_type)
    return spec["price"] if spec else None

# Bytes of GPU memory per model parameter, from the fine-tuning episode's table.
BYTES_PER_PARAM = {"lora": 2.5, "full": 16.0}

# Multiplier on parameters x tokens: 6 for a full fine-tune (forward + backward through
# every weight), 4 for LoRA (no weight gradients for the frozen base).
FLOPS_PER_PARAM_TOKEN = {"lora": 4.0, "full": 6.0}


def estimate_finetune_cost(params_billions, examples, tokens_per_example, epochs=2, method="lora",
                           instance_type=None, runs=1, utilization=0.25, gradient_checkpointing=True,
                           activations_gb=4.0, startup_minutes=10, verbose=True, from_scratch=False):
    """
    Back-of-the-envelope cost of a fine-tuning run, using the two rules from the
    fine-tuning episode. Good to within about a factor of two, which is enough to tell
    a $5 run from a $500 one before anything is launched.

    Rule 1, memory: bytes per parameter x parameters + activations must fit the GPU memory
    (summed across GPUs on a multi-GPU box, where the model is sharded).
    Rule 2, time: FLOPs = k x parameters x tokens (k = 4 for LoRA, 6 for full), divided by
    the GPU's peak bf16 rate x utilization; +50% when gradient checkpointing recomputes
    the forward pass. Cost = instance-hours x price per instance-hour + startup.

    Parameters
    ----------
    params_billions : float   total parameters of the model you train (3, 7, 27, 70 ...).
                              For a vision-language model include the vision tower (CHURRO-3B ~3.75).
    examples : int            training examples (pages, documents, images, recordings ...)
    tokens_per_example : int  tokens the model sees per example. Text: words x 1.3. An image on a
                              Qwen-VL style model: pixels / 784 (about 1,300 for one megapixel) plus
                              the text. An EEG window or audio clip: ask the model card what one
                              example becomes after tokenization.
    epochs : float            passes over the data
    method : "lora" | "full"
    instance_type : str       a key of helpers.GPU_INSTANCES; None picks the cheapest that fits
    runs : int                how many runs the project will need (a first run is never the last;
                              five to ten is realistic before any hyperparameter sweep)
    utilization : float       fraction of peak the GPU achieves; 0.25 unless you have measured
    gradient_checkpointing : bool   adds ~50% compute; needed on small GPUs
    activations_gb : float    headroom for activations; raise for long sequences or big batches
    startup_minutes : float   per-job fixed overhead (container pull, data download, model load)

    Returns a dict with the numbers and prints a report when verbose.
    """
    method = method.lower()
    if method not in BYTES_PER_PARAM:
        raise ValueError("method must be 'lora' or 'full'")
    params = params_billions * 1e9
    tokens = examples * tokens_per_example * epochs
    mem_gb = BYTES_PER_PARAM[method] * params / 1e9 + activations_gb
    flops = FLOPS_PER_PARAM_TOKEN[method] * params * tokens * (1.5 if gradient_checkpointing else 1.0)

    def fits(spec):
        return spec["gpus"] * spec["gpu_mem_gb"] >= mem_gb

    if instance_type is None:
        candidates = [k for k, v in GPU_INSTANCES.items() if fits(v) and "note" not in v]
        if not candidates:
            raise ValueError(f"needs ~{mem_gb:.0f} GB of GPU memory; nothing in GPU_INSTANCES holds it. "
                             "Use LoRA, a smaller model, or more than one box.")
        instance_type = min(candidates, key=lambda k: GPU_INSTANCES[k]["price"])
    spec = GPU_INSTANCES[instance_type]

    gpu_seconds = flops / (spec["tflops_bf16"] * 1e12 * utilization)
    gpu_hours = gpu_seconds / 3600
    instance_hours = gpu_hours / spec["gpus"] + startup_minutes / 60
    cost_per_run = instance_hours * spec["price"]
    max_run_minutes = max(15, round(2 * instance_hours * 60 / 5) * 5)
    result = {
        "method": method, "params_billions": params_billions, "tokens": tokens,
        "memory_gb_needed": round(mem_gb, 1), "instance_type": instance_type,
        "fits": fits(spec), "gpu_hours": round(gpu_hours, 2),
        "wall_clock_hours": round(instance_hours, 2), "cost_per_run": round(cost_per_run, 2),
        "runs": runs, "cost_total": round(cost_per_run * runs, 2),
        "max_run_seconds": int(max_run_minutes * 60),
    }
    if verbose:
        gpus = f'{spec["gpus"]} x {spec["gpu"]}'
        what = "Training from scratch" if from_scratch else f"{method.upper()} fine-tune"
        print(f"{what} of a {params_billions:g}B model: {examples:,} examples x "
              f"{tokens_per_example:,} tokens x {epochs:g} passes = {tokens/1e6:,.1f}M tokens")
        print(f"memory:     ~{mem_gb:.0f} GB needed; {instance_type} has {spec['gpus'] * spec['gpu_mem_gb']} GB ({gpus})")
        if not fits(spec):
            fitting = [k for k, v in GPU_INSTANCES.items() if fits(v) and "note" not in v]
            cheapest = min(fitting, key=lambda k: GPU_INSTANCES[k]["price"]) if fitting else None
            print(f"\n*** DOES NOT FIT: {instance_type} cannot hold this job. It would start, bill ~{startup_minutes:g} min of startup "
                  f"(~${startup_minutes / 60 * spec['price']:.2f}), then die with a CUDA out-of-memory error. ***")
            if cheapest:
                c = GPU_INSTANCES[cheapest]
                print(f"    smallest instance that fits: {cheapest} ({c['gpus']} x {c['gpu']}, {c['gpus'] * c['gpu_mem_gb']} GB, ${c['price']}/h); "
                      f"rerun with instance_type=\"{cheapest}\"")
            else:
                print("    nothing in GPU_INSTANCES holds it: use LoRA, a smaller model, or more than one box")
            if method == "full":
                print("    or switch to method=\"lora\": about %.0f GB instead of %.0f" % (BYTES_PER_PARAM["lora"] * params / 1e9 + activations_gb, mem_gb))
            print("    the numbers below are what the run WOULD cost on this instance if it fit")
        print(f"compute:    {flops:.2e} FLOPs at {utilization:.0%} of the {spec['tflops_bf16']} TFLOPS peak "
              f"= {gpu_hours:.2f} GPU-hours")
        print(f"wall-clock: ~{instance_hours * 60:.0f} min including {startup_minutes:g} min startup")
        print(f"cost:       ~${cost_per_run:.2f} per run at ${spec['price']}/h; ~${cost_per_run * runs:.2f} for {runs} run(s)")
        print(f"set max_run={result['max_run_seconds']}  ({max_run_minutes:.0f} min, about twice the estimate)")
        if "note" in spec:
            print(f"warning:    {spec['gpu']}: {spec['note']}")
        if method == "full" and not from_scratch:
            print("warning:    full fine-tuning updates every weight; do not run one on the class account")
        if method == "lora" and params_billions > 7:
            print("warning:    LoRA above 7B is outside what the class account allows")
        if spec["gpus"] > 1:
            print("warning:    multi-GPU instance; not allowed on the class account, and more GPUs do not make the run cheaper")
    return result


def estimate_training_cost(params_billions, examples, tokens_per_example=1, epochs=1, instance_type=None, runs=1, **kw):
    """estimate_finetune_cost for a model trained from scratch: every weight is trained, so it is
    the "full" case, 6 FLOPs per parameter per example-token. For a dense network or a transformer
    tokens_per_example is the sequence length (1 for a plain feature vector). The rule assumes each
    weight is used once per token; convolutional networks reuse weights across positions and need
    their own FLOPs-per-image figure instead.

    Only worth running when the FLOPs reach the 1e15 range: below that the bill is startup, not
    compute, and a timed short local run is the better estimate (Training part 2 episode).
    """
    return estimate_finetune_cost(params_billions, examples, tokens_per_example, epochs=epochs, method="full",
                                  instance_type=instance_type, runs=runs, from_scratch=True, **kw)


# ---------------------------------------------------------------------------
# The account's guard rails, checked in the notebook before a launch
# ---------------------------------------------------------------------------
# guardrails.json, next to this file, is the IAM policy the administrator attaches to the
# role (see the instructor page "AWS account preparation"). AWS refuses a launch that breaks
# one of its rules with a message that names no rule, so check_launch applies the same rules
# here first and says which one, and what to change. The lists are read from the policy file,
# so the two cannot drift apart.
import json as _json

GUARDRAILS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "guardrails.json")


def load_guard_rails(path=GUARDRAILS_FILE):
    """Return the lists the policy enforces: projects, purposes, instance_types, max_run_seconds."""
    with open(path) as f:
        statements = {s["Sid"]: s for s in _json.load(f)["Statement"]}
    return {
        "projects": statements["ProjectTagFromTheList"]["Condition"]["StringNotLike"]["aws:RequestTag/Project"],
        "purposes": statements["PurposeTagFromTheList"]["Condition"]["StringNotLike"]["aws:RequestTag/Purpose"],
        "instance_types": statements["OnlyLessonInstanceTypes"]["Condition"]["ForAnyValue:StringNotEquals"]["sagemaker:InstanceTypes"],
        "max_run_seconds": int(statements["RuntimeCapAtMost48Hours"]["Condition"]["NumericGreaterThan"]["sagemaker:MaxRuntimeInSeconds"]),
    }


def check_launch(tags, instance_type=None, max_run=None):
    """Check a job's tags (and optionally its instance type and runtime cap) against the account's rules.

    Raises ValueError with a plain message naming every problem and the fix; returns the tags
    unchanged when they pass, so it can wrap the list: job_tags = helpers.check_launch([...]).
    """
    rules = load_guard_rails()
    problems = []
    values = {t["Key"]: t["Value"] for t in tags}
    for key in ("Name", "Project", "Purpose"):
        if key not in values:
            problems.append(f"no {key} tag: every job needs the three tags Name, Project and Purpose")
    if "Name" in values and not values["Name"].strip():
        problems.append("Name is blank: set name to your last and first name run together, e.g. DoeJohn")
    if "Project" in values and values["Project"] not in rules["projects"]:
        problems.append(f"Project {values['Project']!r} is not on the account's list: use one of {', '.join(rules['projects'])}, spelled exactly")
    if "Purpose" in values and values["Purpose"] not in rules["purposes"]:
        problems.append(f"Purpose {values['Purpose']!r} is not on the list: use one of {', '.join(rules['purposes'])}")
    if instance_type is not None and instance_type not in rules["instance_types"]:
        problems.append(f"instance type {instance_type} is not one the lesson uses: choose from {', '.join(rules['instance_types'])}")
    if max_run is not None and max_run > rules["max_run_seconds"]:
        problems.append(f"max_run of {max_run} s is above the account's cap of {rules['max_run_seconds']} s ({rules['max_run_seconds'] // 3600} hours)")
    if problems:
        raise ValueError("this launch would be refused by the account:\n  - " + "\n  - ".join(problems))
    return tags


def check_finetune_launch(tags, instance_type, max_run, estimated_minutes, price_per_hour, params_billions,
                          method="lora", job_name=None, checkpoint_s3_uri=None, hyperparameters=None):
    """Check a fine-tuning launch before the estimator is built: the account's rules (tags, instance type,
    cap), then the lesson's own rules for a fine-tune. Raises ValueError naming every problem and the fix;
    prints a one-line summary of what the launch is allowed to cost when everything passes."""
    problems = []
    try:
        check_launch(tags, instance_type, max_run)
    except ValueError as e:
        problems += str(e).split("\n  - ")[1:]
    if method != "lora":
        problems.append(f'method="{method}": a full fine-tune updates every weight, needs 16 bytes per parameter, '
                        "and is not run on the class account; use LoRA")
    elif params_billions > 7:
        problems.append(f"{params_billions}B parameters: a LoRA run above 7B does not fit the 24 GB GPU the lesson uses")
    if max_run is None:
        problems.append("no max_run: the SDK would send 24 hours, so a hung job bills a day; set it to about twice the estimate")
    else:
        cap_minutes = max_run / 60
        if cap_minutes < estimated_minutes:
            problems.append(f"max_run is {cap_minutes:.0f} min but the estimate is {estimated_minutes} min: the job would be killed before it finishes")
        elif cap_minutes > 4 * estimated_minutes:
            problems.append(f"max_run is {cap_minutes:.0f} min, more than four times the {estimated_minutes} min estimate: "
                            f"a hung job would bill ${price_per_hour * cap_minutes / 60:.2f}; set it to about twice the estimate")
        if cap_minutes > 60 and not checkpoint_s3_uri:
            problems.append("no checkpoint_s3_uri on a run that may exceed an hour: a crash or the cap loses all of it")
    if not job_name:
        problems.append("no job_name: the SDK makes one up each time, so running the launch cell twice starts two jobs")
    hp = hyperparameters or {}
    if hp.get("epochs", 0) > 10:
        problems.append(f"epochs={hp['epochs']}: more than ten passes over the data; run a few first and read the curve")
    if hp.get("max_pixels", 0) > 4 * 1280 * 28 * 28:
        problems.append(f"max_pixels={hp['max_pixels']}: four times the lesson's page budget, so tokens and cost per page are four times too")
    if problems:
        raise ValueError("this launch should not go out:\n  - " + "\n  - ".join(problems))
    print(f"launch check passed: {method} on a {params_billions}B model, {instance_type} at ${price_per_hour}/h, "
          f"estimate {estimated_minutes} min (${price_per_hour * estimated_minutes / 60:.2f}), "
          f"cap {max_run / 60:.0f} min (worst case ${price_per_hour * max_run / 3600:.2f}), "
          f"checkpoints {'on' if checkpoint_s3_uri else 'off'}, job name {job_name}")
    return tags


def rules(path=GUARDRAILS_FILE):
    """Print the account's launch rules, read from the policy file."""
    r = load_guard_rails(path)
    print("A job on this account is refused unless:")
    print("  - it carries the three tags Name, Project and Purpose, and Name is not blank")
    print(f"  - Project is one of: {', '.join(r['projects'])}")
    print(f"  - Purpose is one of: {', '.join(r['purposes'])}")
    print(f"  - its instance type is one of: {', '.join(r['instance_types'])}")
    print(f"  - its runtime cap is at most {r['max_run_seconds']} s ({r['max_run_seconds'] // 3600} hours); Processing jobs must set one")
    print("  - it is a job, not an endpoint or an EC2 instance")
    print("AWS's refusal names none of these; helpers.check_launch(tags, instance_type, max_run) does, before the launch,")
    print("and helpers.explain_refusal() does afterwards, from the request CloudTrail recorded.")


def explain_refusal(minutes=60, region=None):
    """Find recent refused launches in CloudTrail and say which rule each one broke.

    CloudTrail records every API call on the account, including the ones IAM refused, with the
    request as sent: tags, instance type and runtime cap. This reads the refusals of the last
    `minutes` minutes (events appear in CloudTrail 5 to 15 minutes after the call) and applies
    the account's rules to each, which gives the message the refusal itself could not carry.
    The role needs cloudtrail:LookupEvents; without it this says so and points at check_launch.
    """
    ct = boto3.client("cloudtrail", region_name=region)
    start = datetime.utcnow() - timedelta(minutes=minutes)
    try:
        pages = ct.get_paginator("lookup_events").paginate(
            LookupAttributes=[{"AttributeKey": "EventSource", "AttributeValue": "sagemaker.amazonaws.com"}],
            StartTime=start)
        events = [e for page in pages for e in page["Events"]]
    except Exception as e:
        print(f"could not read CloudTrail ({type(e).__name__}): the role needs cloudtrail:LookupEvents.")
        print("Check the launch before sending it instead: helpers.check_launch(tags, instance_type, max_run)")
        return []
    found = []
    for ev in events:
        d = _json.loads(ev["CloudTrailEvent"])
        if "AccessDenied" not in d.get("errorCode", ""):
            continue
        params = d.get("requestParameters") or {}
        tags = [{"Key": t["key"], "Value": t["value"]} for t in params.get("tags", [])]
        res = params.get("resourceConfig") or (params.get("processingResources") or {}).get("clusterConfig") or {}
        instance_type = res.get("instanceType")
        cap = (params.get("stoppingCondition") or {}).get("maxRuntimeInSeconds")
        when = ev["EventTime"].strftime("%H:%M:%S")
        print(f"{when}  {d['eventName']}  name={params.get('trainingJobName') or params.get('processingJobName') or params.get('hyperParameterTuningJobName') or '?'}")
        try:
            check_launch(tags, instance_type, cap)
            if d["eventName"] == "CreateProcessingJob" and cap is None:
                print("    Processing job without a runtime cap: pass max_runtime_in_seconds=")
            elif d["eventName"] in ("CreateEndpoint", "CreateEndpointConfig", "CreateModel"):
                print("    endpoints cannot be created on this account: use a job, or Bedrock")
            else:
                print("    tags, instance type and cap all pass: check that tags= is on the estimator, and that this is your job")
        except ValueError as e:
            print("    " + str(e).replace("\n", "\n    "))
        found.append(d)
    if not found:
        print(f"no refused SageMaker launches in the last {minutes} minutes; CloudTrail can take 15 minutes to show one")
    return found


def estimate_tuning_cost(train_minutes_per_job, price_per_hour, max_jobs, max_parallel_jobs=2, startup_minutes=2):
    """Estimate a hyperparameter tuning job's bill before launching it, from the parameters the tuner takes.

    train_minutes_per_job: the billable minutes one child job spends after startup (training, evaluation, model
    upload). For a small model, time a short run and scale it, as the PyTorch episode did; for a large one,
    estimate_training_cost gives it. The tuner keeps each of its max_parallel_jobs instances between child jobs,
    so the startup (container pull and data download, startup_minutes) is paid once per parallel slot, not once
    per job, and every later child job on a slot bills only its own minutes. The bill is the sum over child jobs;
    the tuning job itself is free. Wall-clock is the startup plus the rounds of parallel jobs, before the unbilled
    minute or two of provisioning.
    """
    slots = min(max_parallel_jobs, max_jobs)
    rounds = -(-max_jobs // slots)                                   # ceiling division
    billable_minutes = slots * startup_minutes + max_jobs * train_minutes_per_job
    wall_minutes = startup_minutes + rounds * train_minutes_per_job
    cost = billable_minutes / 60 * price_per_hour
    max_run_seconds = int(2 * (startup_minutes + train_minutes_per_job) * 60)   # per child job, twice the estimate
    print(f"child jobs:   {max_jobs} jobs, {slots} at a time, {rounds} round(s)")
    print(f"billable:     {billable_minutes:.0f} min = {slots} x {startup_minutes} min startup + {max_jobs} x {train_minutes_per_job} min per job")
    print(f"cost:         ~${cost:.2f} at ${price_per_hour}/h")
    print(f"wall-clock:   ~{wall_minutes:.0f} min, plus a minute or two of provisioning")
    print(f"max_run:      {max_run_seconds} s per child job")
    return {"billable_minutes": billable_minutes, "cost": round(cost, 3), "wall_minutes": wall_minutes,
            "rounds": rounds, "max_run_seconds": max_run_seconds}


def estimate_parallel_speedup(params_billions, tokens_per_step, instance_type, instance_count=1, method="full",
                              trainable_fraction=None, utilization=0.25):
    """Estimate what data parallelism buys for one training step: compute time per device, gradient-exchange
    time, and the speedup over one device, on `instance_count` boxes of `instance_type` using every GPU in each.

    tokens_per_step: the global batch in tokens (examples per step x tokens per example), split across the devices.
    The exchange moves the gradient of every trainable parameter once per step (ring all-reduce: about 2 x its bytes
    per device, 4 bytes per value), over the slowest link in use: the GPU-to-GPU link inside one box, or the
    network when there is more than one box. Plus one millisecond of latency per step. The exchange is not
    overlapped with the computation here, so the speedup is a lower bound; real runs overlap part of it.
    """
    inst = GPU_INSTANCES[instance_type]
    n_devices = inst["gpus"] * instance_count
    if trainable_fraction is None:
        trainable_fraction = 1.0 if method == "full" else 0.01     # LoRA adapters are about a percent of the model
    flops_per_token = FLOPS_PER_PARAM_TOKEN[method] * params_billions * 1e9 * 1.5    # x1.5: gradient checkpointing, as the cost estimator assumes
    compute_one = flops_per_token * tokens_per_step / (inst["tflops_bf16"] * 1e12 * utilization)       # one device does the whole step
    compute_each = compute_one / n_devices                                                             # each device does its share
    grad_bytes = params_billions * 1e9 * trainable_fraction * 4
    if instance_count > 1:
        link_bytes_s, link = inst["net_gbps"] * 1e9 / 8, f"{inst['net_gbps']} Gb/s network between instances"
    elif inst["gpus"] > 1:
        link_bytes_s, link = inst["gpu_link_gbs"] * 1e9, f"{inst['gpu_link_gbs']} GB/s GPU link inside the box"
    else:
        link_bytes_s, link = None, "one device, nothing to exchange"
    exchange = 0.0 if link_bytes_s is None or n_devices == 1 else 2 * grad_bytes * (n_devices - 1) / n_devices / link_bytes_s + 1e-3
    step = compute_each + exchange
    speedup = compute_one / step
    print(f"{n_devices} device(s): {instance_count} x {instance_type} ({inst['gpus']} x {inst['gpu']} each); {link}")
    fmt = lambda s: (f"{s*1e6:10.1f} us" if s < 1e-3 else f"{s*1e3:10.3f} ms" if s < 0.01 else f"{s*1e3:10.1f} ms")
    print(f"one device alone:   {fmt(compute_one)} per step of {tokens_per_step:,} tokens")
    print(f"each of {n_devices}:          {fmt(compute_each)} of computation + {fmt(exchange)} exchanging {grad_bytes/1e9:.3f} GB of gradients")
    print(f"speedup:            {speedup:10.2f}x on {n_devices} devices (ideal {n_devices}x); exchange is {100*exchange/step:.0f}% of the step")
    cost_ratio = n_devices / speedup
    print(f"cost per run:       {cost_ratio:10.2f}x the one-device cost" + ("" if cost_ratio < 1.3 else "  (more devices, same work, a bigger bill)"))
    return {"devices": n_devices, "compute_ms": compute_each * 1e3, "exchange_ms": exchange * 1e3, "speedup": speedup, "cost_ratio": cost_ratio}

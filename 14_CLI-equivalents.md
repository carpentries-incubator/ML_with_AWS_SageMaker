---
title: "(Optional) The Same Jobs from the Terminal"
teaching: 20
exercises: 10
---

:::::::::::::::::::::::::::::::::::::: questions 

- What does the AWS command-line interface (CLI) let me do that the lesson's notebooks did, and what should I not do with it?
- How do I see, and stop, everything that is costing money from a terminal?
- What request does the SageMaker SDK send when I call `estimator.fit()`?

::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: objectives

- Authenticate the CLI with short-lived Identity Center credentials, never with access keys.
- Match each SDK call and helper the episodes used to the CLI command that does the same thing.
- List, inspect and stop training jobs, processing jobs, tuning jobs and notebook instances from the terminal, and read a job's bill the way `job_cost_report()` did.
- Copy data to and from S3 in bulk with `aws s3 sync`.
- Relaunch the training episode's XGBoost job from its JSON description, with the same script, hyperparameters, cap and tags.
- State which resources the lesson never creates from the CLI, and why the account refuses them.

::::::::::::::::::::::::::::::::::::::::::::::::

### Optional, self-study episode

Everything in this lesson went through the SageMaker Python SDK in a notebook. The same account can be used from a terminal with the AWS CLI, to check on a job from another machine, to stop something when no notebook is open, to copy results out of S3 in bulk, or to script a launch. This episode does five things from a laptop terminal. It signs in to the account with Identity Center, lists what is running and reads what a job cost, stops jobs and notebook instances, copies data to and from S3, and relaunches the training episode's XGBoost job from a JSON file, which is the request `estimator.fit()` sent.

The case for the terminal is the notebook instance. Every job in this lesson was launched from an `ml.t3.medium` that bills while it is on, five cents an hour, a dollar a day, and that someone has to remember to stop. A job launched from a laptop needs no instance at all. The request goes from your terminal to SageMaker, the job runs on a machine rented for it and ends, and the model lands in S3. For a project that launches a job a day and otherwise works locally, that is the notebook's whole cost gone and nothing left running overnight. The training episode's `wait=False` pattern gets most of the way there from a notebook. The CLI gets the rest.

What you give up is what the instance did for you. On the instance, its role signs every call, the lesson's helper checks a launch against the account's rules before it goes, and the console shows what you started next to everyone else's work. On a laptop you sign in yourself, the JSON carries no price and no check, and a mistake is not seen until the bill. The rules in the callout and the request check in section 5 cover that.

### What the notebooks did, and the command that does the same

The CLI, the SDK and the console are one API, so every call the episodes made has a CLI command with the same name and the same fields. This table is the map. The sections below work through the rows that matter most.

| Episode | In the notebook | From the terminal |
|---|---|---|
| Controllers, Training | `sagemaker.Session()`, `get_execution_role()` | `aws sts get-caller-identity`, and the role ARN is the `RoleArn` field of any job you launched |
| Training | `helpers.get_notebook_instance_info(name, region)` | `aws sagemaker describe-notebook-instance --notebook-instance-name <name>` |
| Accessing S3 | `s3.upload_file`, `s3.download_file`, `helpers.show_bucket(bucket)` | `aws s3 cp`, `aws s3 sync`, `aws s3 ls --recursive` |
| Training, PyTorch | `xgboost_estimator.fit({'train': train_input})` | `aws sagemaker create-training-job --cli-input-json file://train-xgboost.json` (section 5) |
| Training | `fit(wait=True)`, the default | `aws sagemaker wait training-job-completed-or-stopped --training-job-name <name>` |
| Training | `job_cost_report(job_name, price)` | `aws sagemaker describe-training-job --training-job-name <name>` (section 2) |
| Hyperparameter tuning | `tuner.fit(...)` | `aws sagemaker create-hyper-parameter-tuning-job --cli-input-json file://...` |
| Hyperparameter tuning | `tuning_cost_report(tuner, price)`, `tuner.best_training_job()` | `aws sagemaker list-training-jobs-for-hyper-parameter-tuning-job`, `describe-hyper-parameter-tuning-job --query BestTrainingJob.TrainingJobName` |
| Fine-tuning | `helpers.check_launch(...)`, `helpers.check_finetune_launch(...)` | `python scripts/check_request.py <request>.json` (section 5) |
| RAG with Processing jobs | `rag_processor.run(...)`, `describe_processing_job(...)` | `aws sagemaker create-processing-job --cli-input-json file://...`, `aws sagemaker describe-processing-job --processing-job-name <name>` |
| RAG with Bedrock | `bedrock.list_inference_profiles(typeEquals="APPLICATION")`, `bedrock_runtime.converse(...)` | `aws bedrock list-inference-profiles --type-equals APPLICATION`, `aws bedrock-runtime converse --model-id <profile ARN> ...` |
| Cleanup | `sm.list_notebook_instances()`, `bedrock.delete_inference_profile(...)` | `aws sagemaker list-notebook-instances`, `aws bedrock delete-inference-profile --inference-profile-identifier <ARN>` (section 3) |

The SDK's job classes (`XGBoost`, `PyTorch`, `HuggingFace`, `HuggingFaceProcessor`, `HyperparameterTuner`) are builders for the JSON these commands take. `describe-*` on any job you launched from a notebook prints that JSON back, which is how section 5 gets its file.

::::::::::::::::::::::::::::::::::::: callout

### Read this before the first command

Nothing here is more powerful than a notebook cell. The difference is that a terminal shows nothing unless you ask. A notebook instance left running is visible in the console the next day, and an endpoint created from a terminal on a Friday is not noticed until the bill arrives. A past participant of this workshop spent a large share of a year's credits that way. Three rules apply on the shared account and are worth keeping on your own.

1. **Authenticate with `aws sso login` only.** Never `aws configure` with an access key ID and secret. Those are long-lived, and they end up in shell histories and committed files.
2. **Launch only jobs** (training, processing, tuning), with the same three tags and the same `MaxRuntimeInSeconds` the notebooks always set. Jobs end on their own. **Endpoints, EC2 instances and GPU notebook instances do not**, and the lesson never creates them from the CLI, or at all.
3. **A refused `create` command is the account's rules, not a fault.** The workshop account carries a guard-rail policy that denies endpoint and EC2 creation, instance types outside the lesson's list, caps over 48 hours, and any job whose three tags are missing, blank or off the `Project` and `Purpose` lists. Do not look for a workaround. The policy is on the notebook role today, which covers every launch from an instance. A laptop CLI runs under the permission set, which gets the same policy when the Identity Center administrator attaches it, and until then these rules are yours to keep by hand from a laptop. (Instructors: the policy and its status are on the [Account preparation checklist](../instructors/aws-account-prep.md) page.)

::::::::::::::::::::::::::::::::::::::::::::::::

## 1. Sign in with Identity Center

This episode runs from your own laptop, a terminal that works when no notebook is open. The AWS CLI is already installed in CloudShell and on every SageMaker notebook instance, where it uses the instance's role and needs no sign-in. A laptop has neither, so install [AWS CLI v2](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html) there, then register the account's Identity Center once. It asks four things. First the start URL and the region of the Identity Center sign-in page, the same ones you used in the browser, both in the sign-in email from your instructor. It then opens a browser tab for you to approve the device, lets you pick the account and the permission set, and finally asks for a default region, `us-east-2` for the workshop account, and a name for the profile.

```bash
aws configure sso
```

The commands in this episode are the same in Bash (macOS, Linux, Git Bash on Windows) and in PowerShell. Where the shell's own syntax differs, variables, line continuation, writing a file, the two are shown as tabs.

From then on, every session starts with a login, and a check that you are who you think you are, in the account you think you are in:

::::::::::::::::::::::::::::::::::::: tab

### Bash

```bash
aws sso login --profile workshop
export AWS_PROFILE=workshop
aws sts get-caller-identity
```

### PowerShell

```powershell
aws sso login --profile workshop
$env:AWS_PROFILE = "workshop"
aws sts get-caller-identity
```

::::::::::::::::::::::::::::::::::::::::::::::::

The output names the account ID and an `AWSReservedSSO_...` role. The credentials expire after a few hours. Run `aws sso login` again when a command returns `ExpiredToken`. Nothing that would still work next month is written anywhere.

If you are on the notebook instance's own terminal (JupyterLab, **File → New → Terminal**), skip this step. The instance's role is already the identity, which is the reason the lesson works from one.

## 2. What is running, and what a job cost

First the lists. Each should be short, or empty. `--query` is a [JMESPath](https://jmespath.org/) filter that keeps the output to what you asked about, and the cleanup episode's first cell ran the same four `list_*` calls through `boto3`.

```bash
aws sagemaker list-training-jobs   --status-equals InProgress --query 'TrainingJobSummaries[].TrainingJobName'
aws sagemaker list-processing-jobs --status-equals InProgress --query 'ProcessingJobSummaries[].ProcessingJobName'
aws sagemaker list-hyper-parameter-tuning-jobs --status-equals InProgress --query 'HyperParameterTuningJobSummaries[].HyperParameterTuningJobName'
aws sagemaker list-notebook-instances --query 'NotebookInstances[].[NotebookInstanceName,NotebookInstanceStatus,InstanceType]' --output table
aws sagemaker list-endpoints       # must print an empty list on the workshop account
```

Then one job's bill. `job_cost_report()` in the training episodes called `describe_training_job` and read four fields from it, the status, the instance count, `BillableTimeInSeconds`, and the model's S3 path, and computed the wall-clock from `CreationTime` and `TrainingEndTime`. This is the same call with the same fields. Use a job name from the training episode's `Training jobs` page, or from `list-training-jobs` without the status filter.

::::::::::::::::::::::::::::::::::::: tab

### Bash

```bash
JOB=<a training job name>
aws sagemaker describe-training-job --training-job-name $JOB \
  --query '{status:TrainingJobStatus, instance:ResourceConfig.InstanceType, count:ResourceConfig.InstanceCount, billable_s:BillableTimeInSeconds, created:CreationTime, ended:TrainingEndTime, model:ModelArtifacts.S3ModelArtifacts}'
```

### PowerShell

```powershell
$JOB = "<a training job name>"
aws sagemaker describe-training-job --training-job-name $JOB `
  --query '{status:TrainingJobStatus, instance:ResourceConfig.InstanceType, count:ResourceConfig.InstanceCount, billable_s:BillableTimeInSeconds, created:CreationTime, ended:TrainingEndTime, model:ModelArtifacts.S3ModelArtifacts}'
```

::::::::::::::::::::::::::::::::::::::::::::::::

Cost is `billable_s` × `count` ÷ 3600 × the hourly rate from the [Instances for ML](https://carpentries-incubator.github.io/ML_with_AWS_SageMaker/instances-for-ML.html) page, the arithmetic `job_cost_report()` did. The helper printed it. Here you do the multiplication.

The other job kinds have the same shape. The Processing Jobs episode's `job_cost()` read `CreationTime`, `ProcessingStartTime` and `ProcessingEndTime` from `describe_processing_job`, and the tuning episode's `tuning_cost_report()` looped `describe_training_job` over the tuner's child jobs:

```bash
aws sagemaker describe-processing-job --processing-job-name <name> --query '{status:ProcessingJobStatus, created:CreationTime, started:ProcessingStartTime, ended:ProcessingEndTime, instance:ProcessingResources.ClusterConfig.InstanceType}'
aws sagemaker list-training-jobs-for-hyper-parameter-tuning-job --hyper-parameter-tuning-job-name <name> --query 'TrainingJobSummaries[].[TrainingJobName,TrainingJobStatus,FinalHyperParameterTuningJobObjectiveMetric.Value]' --output table
aws sagemaker describe-hyper-parameter-tuning-job --hyper-parameter-tuning-job-name <name> --query 'BestTrainingJob.TrainingJobName'   # tuner.best_training_job()
```

Tags are returned by a separate call, which is also how you find the untagged job that shows on the bill under "No tag key":

::::::::::::::::::::::::::::::::::::: tab

### Bash

```bash
ARN=$(aws sagemaker describe-training-job --training-job-name $JOB --query TrainingJobArn --output text)
aws sagemaker list-tags --resource-arn $ARN --query 'Tags[].[Key,Value]' --output table
```

### PowerShell

```powershell
$ARN = aws sagemaker describe-training-job --training-job-name $JOB --query TrainingJobArn --output text
aws sagemaker list-tags --resource-arn $ARN --query 'Tags[].[Key,Value]' --output table
```

::::::::::::::::::::::::::::::::::::::::::::::::

## 3. Stopping jobs and notebook instances

Each of these stops a charge. They are the calls behind the **Stop** buttons the cleanup episode walks through in the console.

```bash
aws sagemaker stop-training-job   --training-job-name   <name>
aws sagemaker stop-processing-job --processing-job-name <name>
aws sagemaker stop-hyper-parameter-tuning-job --hyper-parameter-tuning-job-name <name>   # stops every child job
aws sagemaker stop-notebook-instance --notebook-instance-name <name>
```

A stopped job keeps its logs and whatever it had already written to S3. A stopped notebook instance keeps `/home/ec2-user/SageMaker/` and keeps billing for that volume. Deleting the instance is the step that stops the storage charge, and the cleanup episode's last two items, the bucket and the Bedrock inference profiles, have their CLI forms too:

```bash
aws sagemaker delete-notebook-instance --notebook-instance-name <name>   # only after it reports Stopped
aws s3 rb s3://<your-bucket> --force                                     # empties the bucket, then deletes it; the cleanup episode's delete_objects loop plus delete_bucket
aws bedrock list-inference-profiles --type-equals APPLICATION --query 'inferenceProfileSummaries[].[inferenceProfileName,inferenceProfileArn]' --output table
aws bedrock delete-inference-profile --inference-profile-identifier <ARN>  # one per profile, as the cleanup episode's loop did
```

## 4. Copying to and from S3

The Accessing S3 episode moved single files with `s3.upload_file` and `s3.download_file` and listed the bucket with `helpers.show_bucket`. For many files, or for pulling a job's outputs down to a laptop, the CLI is the better tool, and these are the same keys those episodes used:

```bash
aws s3 ls s3://<your-bucket>/ --recursive --human-readable --summarize             # helpers.show_bucket, for the whole bucket
aws s3 cp titanic_train.csv s3://<your-bucket>/titanic_train.csv                     # s3.upload_file(train_filename, bucket_name, train_filename)
aws s3 cp s3://<your-bucket>/xgboost/<job-name>/output/model.tar.gz model.tar.gz    # the download in the training episode's evaluate step
aws s3 sync s3://<your-bucket>/xgboost/ ./xgboost-outputs/                           # every job's outputs under the prefix; only new or changed files move
aws s3 rm s3://<your-bucket>/scratch/ --recursive                                    # delete a prefix; this one asks no questions
```

Downloads to a laptop are a transfer out of AWS (`$0.09` per GB). For anything larger than a few hundred megabytes, keep the data in the region and let a job read it, which is the lesson's pattern.

## 5. Launching the training episode's job from a JSON file

`xgboost_estimator.fit({'train': train_input})` in the training episode assembled a JSON request and sent it to `CreateTrainingJob`. SageMaker kept that request, and `describe-training-job` on the job you ran prints it back. Three fields come from there, because they are specific to your account and region, and the rest of the file below is the estimator's arguments under their API names.

::::::::::::::::::::::::::::::::::::: tab

### Bash

```bash
JOB=<the training episode's xgboost job name>
aws sagemaker describe-training-job --training-job-name $JOB \
  --query '{image:AlgorithmSpecification.TrainingImage, role:RoleArn, code:HyperParameters.sagemaker_submit_directory}'
```

### PowerShell

```powershell
$JOB = "<the training episode's xgboost job name>"
aws sagemaker describe-training-job --training-job-name $JOB `
  --query '{image:AlgorithmSpecification.TrainingImage, role:RoleArn, code:HyperParameters.sagemaker_submit_directory}'
```

::::::::::::::::::::::::::::::::::::::::::::::::

`image` is the XGBoost 1.7-1 container that `framework_version="1.7-1"` picked. `role` is the `role` the notebook printed. `code` is where `fit()` uploaded `source_dir`, the `sourcedir.tar.gz` in the SDK's default bucket that the training episode pointed out under *What landed in S3*. Reusing it means this job runs the same `train_xgboost.py` the notebook's job ran. Put the three values into the file, with your bucket, your tag values and a job name that is unique in the account:

::::::::::::::::::::::::::::::::::::: tab

### Bash

```bash
cat > train-xgboost.json <<'JSON'
{
  "TrainingJobName": "<notebook-instance-name>-xgboost-cli-1",
  "AlgorithmSpecification": {
    "TrainingImage": "<image>",
    "TrainingInputMode": "File"
  },
  "RoleArn": "<role>",
  "HyperParameters": {
    "train": "\"titanic_train.csv\"",
    "max_depth": "3",
    "eta": "0.1",
    "subsample": "0.8",
    "colsample_bytree": "0.8",
    "num_round": "100",
    "sagemaker_program": "\"train_xgboost.py\"",
    "sagemaker_submit_directory": "\"<code>\"",
    "sagemaker_region": "\"us-east-2\"",
    "sagemaker_container_log_level": "20"
  },
  "InputDataConfig": [
    {
      "ChannelName": "train",
      "ContentType": "csv",
      "DataSource": { "S3DataSource": {
        "S3DataType": "S3Prefix",
        "S3Uri": "s3://<your-bucket>/titanic_train.csv",
        "S3DataDistributionType": "FullyReplicated" } }
    }
  ],
  "OutputDataConfig": { "S3OutputPath": "s3://<your-bucket>/xgboost/" },
  "ResourceConfig": { "InstanceType": "ml.m5.large", "InstanceCount": 1, "VolumeSizeInGB": 30 },
  "StoppingCondition": { "MaxRuntimeInSeconds": 1200 },
  "Tags": [
    { "Key": "Name",    "Value": "<LastnameFirstname>" },
    { "Key": "Project", "Value": "<ChallengeName>" },
    { "Key": "Purpose", "Value": "Training" }
  ]
}
JSON
```

### PowerShell

```powershell
@'
{
  "TrainingJobName": "<notebook-instance-name>-xgboost-cli-1",
  "AlgorithmSpecification": {
    "TrainingImage": "<image>",
    "TrainingInputMode": "File"
  },
  "RoleArn": "<role>",
  "HyperParameters": {
    "train": "\"titanic_train.csv\"",
    "max_depth": "3",
    "eta": "0.1",
    "subsample": "0.8",
    "colsample_bytree": "0.8",
    "num_round": "100",
    "sagemaker_program": "\"train_xgboost.py\"",
    "sagemaker_submit_directory": "\"<code>\"",
    "sagemaker_region": "\"us-east-2\"",
    "sagemaker_container_log_level": "20"
  },
  "InputDataConfig": [
    {
      "ChannelName": "train",
      "ContentType": "csv",
      "DataSource": { "S3DataSource": {
        "S3DataType": "S3Prefix",
        "S3Uri": "s3://<your-bucket>/titanic_train.csv",
        "S3DataDistributionType": "FullyReplicated" } }
    }
  ],
  "OutputDataConfig": { "S3OutputPath": "s3://<your-bucket>/xgboost/" },
  "ResourceConfig": { "InstanceType": "ml.m5.large", "InstanceCount": 1, "VolumeSizeInGB": 30 },
  "StoppingCondition": { "MaxRuntimeInSeconds": 1200 },
  "Tags": [
    { "Key": "Name",    "Value": "<LastnameFirstname>" },
    { "Key": "Project", "Value": "<ChallengeName>" },
    { "Key": "Purpose", "Value": "Training" }
  ]
}
'@ | Set-Content -Path train-xgboost.json
```

::::::::::::::::::::::::::::::::::::::::::::::::

Read it against the estimator cell in the training episode. Every field is one of its arguments:

| In the estimator | In the JSON |
|---|---|
| `base_job_name=f"{notebook_instance_name}-xgboost"` plus the SDK's timestamp | `TrainingJobName`, which you write out in full |
| `framework_version="1.7-1"` | `AlgorithmSpecification.TrainingImage` |
| `role=role` | `RoleArn` |
| `hyperparameters={'train': train_file, 'max_depth': max_depth, ...}` | `HyperParameters`, the same six keys and values, as strings |
| `entry_point='train_xgboost.py'`, `source_dir=.../scripts` | `sagemaker_program` and `sagemaker_submit_directory`, which the SDK adds to the hyperparameters so the container knows what to run |
| `TrainingInput(train_s3_path, content_type='csv')` and `fit({'train': ...})` | `InputDataConfig`, one channel named `train` |
| `output_path=f's3://{bucket_name}/xgboost/'` | `OutputDataConfig.S3OutputPath` |
| `instance_type="ml.m5.large"`, `instance_count=1` | `ResourceConfig`, with the SDK's default 30 GB volume |
| `max_run=20*60` | `StoppingCondition.MaxRuntimeInSeconds`, 1200 |
| `tags=job_tags` | `Tags`, the same three keys |

The string values with escaped quotes, `"\"titanic_train.csv\""`, are how the SDK encodes a text hyperparameter, and the container decodes them before they reach `argparse` in the script, so `--train titanic_train.csv` arrives as it did from the notebook. Numbers are plain strings.

Before sending it, check it the way the notebooks did. `helpers.check_launch` read the account's rules from `scripts/guardrails.json` before every launch, and `scripts/check_request.py` applies the same file to a request. A `Project` or `Purpose` off the lists, an instance type off the list or a cap over 48 hours is named here, where AWS would refuse it silently:

```bash
git clone -q https://github.com/carpentries-incubator/ML_with_AWS_SageMaker.git   # once, anywhere on the laptop
python ML_with_AWS_SageMaker/scripts/check_request.py train-xgboost.json
```

Then send it. This one line is `estimator.fit()`:

```bash
aws sagemaker create-training-job --cli-input-json file://train-xgboost.json
```

The CLI returns the moment the request is accepted. Nothing waits, which is `fit(wait=False)`. To follow the job the way the default `wait=True` did, then read its bill as in section 2:

```bash
aws sagemaker wait training-job-completed-or-stopped --training-job-name <notebook-instance-name>-xgboost-cli-1   # blocks until it ends
aws sagemaker describe-training-job --training-job-name <notebook-instance-name>-xgboost-cli-1 --query '{status:TrainingJobStatus, billable_s:BillableTimeInSeconds, failure:FailureReason, model:ModelArtifacts.S3ModelArtifacts}'
```

Expect the same two to three billable minutes on `ml.m5.large` as the notebook's job, under a cent, and the same `xgboost-model.json` inside the `model.tar.gz` it names. Nothing in a JSON file prints a price, so look the instance type up on the [Instances for ML](https://carpentries-incubator.github.io/ML_with_AWS_SageMaker/instances-for-ML.html) page before sending it, as the notebooks did with their estimate cell.

The PyTorch job, a Processing job and a tuning job follow the same route. `describe-training-job`, `describe-processing-job` and `describe-hyper-parameter-tuning-job` on the ones you ran print the request, and `create-training-job`, `create-processing-job` and `create-hyper-parameter-tuning-job` take it back with `--cli-input-json`. A PyTorch job's `HyperParameters` carries `epochs` and `learning_rate` and two input channels. A Processing job's `AppSpecification.ContainerArguments` is the `arguments=` list. A tuning job's file has the estimator's request nested under `TrainingJobDefinition` and the ranges under `HyperParameterTuningJobConfig.ParameterRanges`.

::::::::::::::::::::::::::::::::::::: callout

### What the lesson never does from the CLI

- `aws sagemaker create-endpoint` / `create-endpoint-config`: an endpoint is a server that runs until you delete it, with no runtime cap. The lesson serves models with processing jobs and Bedrock so that nothing of this kind exists in the account.
- `aws sagemaker create-notebook-instance` with a GPU type: a GPU notebook is `$60` a day while you are asleep. The optional GPU-notebook episode creates one through the console, where it stays visible, and deletes it the same session.
- `aws ec2 run-instances`: bare virtual machines have no tags required, no runtime cap and no SageMaker job list to appear in.

On the workshop account all three are denied on the notebook role, as is any instance type outside the lesson's list, any cap over 48 hours and any job whose tags are missing or off the lists. The permission set a laptop uses gets the same policy once the administrator attaches it. The instructor page [Account preparation checklist](../instructors/aws-account-prep.md) has the policy. On your own account, the same denies are an afternoon's work.

::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: challenge

### Challenge: what is still running on Friday evening

You are leaving for the weekend and have a terminal, not a browser. Write the commands you would run, in order, to make sure the account is costing nothing until Monday, and the one command to run if each check finds something. Then run them against your own resources.

:::::::::::::::::::::::::::::::::::: solution

::: tab

### Bash

```bash
aws sso login --profile workshop && export AWS_PROFILE=workshop
```

### PowerShell

```powershell
aws sso login --profile workshop; $env:AWS_PROFILE = "workshop"
```

:::

```bash
aws sagemaker list-training-jobs   --status-equals InProgress --query 'TrainingJobSummaries[].TrainingJobName'
aws sagemaker list-processing-jobs --status-equals InProgress --query 'ProcessingJobSummaries[].ProcessingJobName'
aws sagemaker list-hyper-parameter-tuning-jobs --status-equals InProgress --query 'HyperParameterTuningJobSummaries[].HyperParameterTuningJobName'
aws sagemaker list-notebook-instances --query 'NotebookInstances[?NotebookInstanceStatus==`InService`].NotebookInstanceName'
aws sagemaker list-endpoints --query 'Endpoints[].EndpointName'
```

For anything the first four print, the matching `stop-...` command from section 3. A job you recognise as yours and expect to finish within its `MaxRuntimeInSeconds` can be left. A job you do not recognise is a question for its owner, and on a shared account the tags (`list-tags`) tell you who that is. For an endpoint, `aws sagemaker delete-endpoint --endpoint-name <name>`, then find out how it was created, because the notebook role refuses it and the permission set should once the policy is attached. Buckets cost almost nothing over a weekend and need no action.

::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::::::::::::::

:::::::::::::::::::::::::::::::::::::: keypoints 

- The CLI, the SDK and the console are one API. Every call the notebooks made has a CLI command with the same name and fields, and `describe-*` on a job you ran prints the request the SDK built.
- The CLI shows you nothing you do not ask for, so its first use is to look and to stop, not to launch.
- Authenticate with `aws sso login` and short-lived credentials. Never store an access key.
- `describe-training-job` is `job_cost_report()`. Cost is billable seconds × instance count × the hourly rate, and you do the multiplication.
- A `create-training-job` JSON is what `estimator.fit()` sent, the same script, hyperparameters, channel, cap and tags, with no price printed next to it.
- Endpoints, EC2 instances and GPU notebooks are not jobs and do not end. The workshop account's notebook role denies creating them, and your own account should too.

::::::::::::::::::::::::::::::::::::::::::::::::

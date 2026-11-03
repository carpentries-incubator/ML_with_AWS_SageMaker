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
- List and inspect training jobs, processing jobs, tuning jobs and notebook instances from the terminal, stop your own by name, and read a job's bill the way `job_cost_report()` did.
- Copy data to and from S3 in bulk with `aws s3 sync`.
- Relaunch the training episode's XGBoost job from its JSON description, with the same script, hyperparameters, cap and tags.
- State which resources the lesson never creates from the CLI, and why the account refuses them.

::::::::::::::::::::::::::::::::::::::::::::::::

### Optional, self-study episode

Everything in this lesson went through the SageMaker Python SDK in a notebook on a SageMaker notebook instance. That instance is a cost and a chore. It bills while it is on, five cents an hour for the `ml.t3.medium`, a dollar a day, and someone has to remember to stop it. It is also a second place to work, a Jupyter session in a browser tab, when your code, your editor and your files are on your laptop.

None of that is required. The notebook instance was a convenience for teaching and for governance in a shared account. SageMaker is an API, and the AWS CLI calls the same API from a terminal. A job launched from your laptop needs no instance at all. The request goes from your terminal to SageMaker, the job runs on a machine rented for it and ends, and the model lands in S3. For a project that launches a job a day and otherwise works locally, that removes the notebook instance's whole cost and the habit of stopping it. The terminal is also the tool for checking on a job from another machine, stopping something when no notebook is open, and copying results out of S3 in bulk.

This episode does five things from a laptop terminal. It signs in to the account with Identity Center, lists what is running and reads what a job cost, stops your own jobs and your own notebook instance by name, copies data to and from S3, and relaunches the training episode's XGBoost job from a JSON file.

Two things differ from the notebook. The first is identity. On the instance, its role signed every call. On a laptop you sign in yourself, which is section 1. The second is checking. The lesson's notebooks printed a cost estimate and ran a launch check before every job. A JSON request does neither on its own, so section 5 runs the lesson's checker on the file before the create call. It applies the account's rules and prints the most the request can bill.

On a shared workshop account, such as the ML Marathon's, keep launching from the notebook instance. It is required there, because every learner's launches and the code behind them then sit on an instance the administrators can see, which is how they help when something goes wrong. Launching from the terminal is for your own account, and for practicing it in this episode. Checking on jobs and stopping them from the terminal is fine on any account.

### What the notebooks did, and the command that does the same

The CLI, the SageMaker SDK, `boto3` and the console all talk to the same AWS API. The SDK's estimators built requests for it, the `boto3` calls in the cleanup and RAG episodes sent requests to it directly, and the console shows its results. So every call the episodes made has a CLI command with the same name and the same fields, and a `boto3` method name such as `describe_training_job` is the CLI command `describe-training-job` with the underscores swapped for hyphens. This table is the map. The sections below work through the rows that matter most.

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
| PyTorch, Hyperparameter tuning, Fine-tuning | `helpers.estimate_training_cost(...)`, `helpers.estimate_tuning_cost(...)`, `helpers.estimate_finetune_cost(...)` | the same functions from a laptop (section 5) |
| Fine-tuning | `helpers.check_launch(...)`, `helpers.check_finetune_launch(...)` | `python scripts/check_request.py <request>.json --max-cost <dollars>` (section 5) |
| RAG with Processing jobs | `rag_processor.run(...)`, `describe_processing_job(...)` | `aws sagemaker create-processing-job --cli-input-json file://...`, `aws sagemaker describe-processing-job --processing-job-name <name>` |
| RAG with Bedrock | `bedrock.list_inference_profiles(typeEquals="APPLICATION")`, `bedrock_runtime.converse(...)` | `aws bedrock list-inference-profiles --type-equals APPLICATION`, `aws bedrock-runtime converse --model-id <profile ARN> ...` |
| Cleanup | `sm.list_notebook_instances()`, `bedrock.delete_inference_profile(...)` | `aws sagemaker list-notebook-instances`, `aws bedrock delete-inference-profile --inference-profile-identifier <ARN>` (section 3) |

The SDK's job classes (`XGBoost`, `PyTorch`, `HuggingFace`, `HuggingFaceProcessor`, `HyperparameterTuner`) are builders for the JSON these commands take. `describe-*` on any job you launched from a notebook prints that JSON back, which is how section 5 gets its file.

### What to replace in the commands

Anything in angle brackets, `<like this>`, is a value of yours. Replace it, brackets included. This table lists every placeholder in the episode and how to find its value.

| Placeholder | What it is | Where to find it |
|---|---|---|
| `<notebook-instance-name>`, then `$NB` | The name you gave your notebook instance in the controllers episode | The first column of `aws sagemaker list-notebook-instances` in section 2, where it is set once as `$NB` |
| `<your-bucket>`, then `$BUCKET` | The bucket you created in the S3 episode | `aws s3 ls` prints every bucket you can see; set once as `$BUCKET` in section 2 |
| `<a training job name>` | Any training job you ran | The list command in section 2, or the name the training notebook printed when `fit()` ran |
| `<your xgboost job name>` | The XGBoost job from the training episode, named `<notebook-instance-name>-xgboost-<timestamp>` | `aws sagemaker list-training-jobs --name-contains xgboost --query 'TrainingJobSummaries[].TrainingJobName'` |
| `<name>` in section 3 | A job or notebook instance to stop | The lists in section 2 |
| `<image>`, `<role>`, `<code>` | The container, the role and the uploaded code of the XGBoost job | Printed by the `describe-training-job` command at the start of section 5 |
| `<LastnameFirstname>`, `<ChallengeName>` | Your `Name` and `Project` tag values | The same values you put on your bucket and your notebook instance |
| `<ARN>` | A Bedrock inference profile's ARN | The second column of `aws bedrock list-inference-profiles` in section 3 |

::::::::::::::::::::::::::::::::::::: callout

### Read this before the first command

Nothing here is more powerful than a notebook cell. The difference is that a terminal shows nothing unless you ask. A notebook instance left running is visible in the console the next day, and an endpoint created from a terminal on a Friday is not noticed until the bill arrives. A past participant of this workshop spent a large share of a year's credits that way. Three rules apply on the shared account and are worth keeping on your own.

1. **Authenticate with `aws sso login` only.** Never `aws configure` with an access key ID and secret. Those are long-lived, and they end up in shell histories and committed files.
2. **Launch only jobs** (training, processing, tuning), with the same three tags and the same `MaxRuntimeInSeconds` the notebooks always set. Jobs end automatically. **Endpoints, EC2 instances and GPU notebook instances do not**, and the lesson never creates them from the CLI, or at all.
3. **A refused `create` command is the account's rules, not a fault.** The workshop account carries a guard-rail policy that denies endpoint and EC2 creation, instance types outside the lesson's list, caps over 48 hours, and any job whose three tags are missing, blank or off the `Project` and `Purpose` lists. The administrator attaches that policy, to the notebook role and to the permission set your laptop signs in with. You attach nothing. If a `create` command is refused, that policy is why, and AWS says only `AccessDenied`. The checker in section 5 applies the same rules to a request before it is sent, so you see the reason instead. (Instructors: the policy and where it is attached are on the [Account preparation checklist](../instructors/aws-account-prep.md) page.)

::::::::::::::::::::::::::::::::::::::::::::::::

## 1. Sign in with Identity Center

This episode runs from your own laptop, in a terminal, so it works when no notebook is open. CloudShell and the SageMaker notebook instance already have the AWS CLI installed, and on the instance it uses the instance's role, so there is no sign-in. On a laptop you install the CLI and sign in yourself.

1. **Install [AWS CLI v2](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html)** from the installer for your operating system. Then open a terminal, Terminal on macOS, Git Bash or PowerShell on Windows, and check that it is on your path:

```bash
aws --version
```

2. **Register the account's Identity Center**, once. Run the command below in the same terminal and answer its prompts in this order. A profile is a named set of these answers, saved in `~/.aws/config` on your laptop, so that the CLI knows which account, role and region a command is for. One laptop can hold several, one per account you use.

```bash
aws configure sso
```

| Prompt | Answer |
|---|---|
| SSO session name | `workshop` |
| SSO start URL | Your sign-in link from the instructor's email, up to and including `/start`, with nothing after it (for UW-Madison it begins with `https://uw-madison` and ends in `awsapps.com/start`) |
| SSO region | The region Identity Center runs in, from the same email (`us-east-2` for UW-Madison). It is not always the account's working region |
| SSO registration scopes | Enter, for the default |
| A browser tab opens | Sign in with your university login if asked. On the page headed *Allow botocore-client-workshop to access your data?*, choose **Allow access** |
| Account | The workshop account, whose ID is in the sign-in email. If your login has only one, the CLI picks it for you |
| Role | The permission set your login carries (for UW-Madison, the one with `sagemaker` in its name). If there is only one, the CLI picks it |
| Default client region | `us-east-2` for the workshop account |
| Default output format | Enter, which is `json` |
| Profile name | `workshop`, in place of the long default it offers, since the rest of the episode uses that name |

The wizard ends by printing `aws sts get-caller-identity --profile workshop`. That is the identity check from the next block.

The role here is the permission set, not the notebook role. The permission set is the identity you sign in with, the same one the console uses. The notebook role, `ml-sagemaker-use`, is what a job runs as, and it stays in the job's `RoleArn` in section 5.

The commands in this episode are the same in Bash (macOS, Linux, Git Bash on Windows) and in PowerShell. Where the shell's own syntax differs, variables, line continuation, writing a file, the two are shown as tabs.

From then on, every session starts with three commands. The login fetches credentials for the profile, which takes you through the browser once more. Setting `AWS_PROFILE` makes every later command in that terminal use the profile without a `--profile` flag on each one. The identity check confirms that you are who you think you are, in the account you think you are in:

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

The login ends with `Successfully logged into Start URL: ...`. The identity check prints the account ID and an `Arn` that names an `AWSReservedSSO_...` role, the permission set, and ends with your email. Check the region too, since SageMaker resources are regional and a job in another region does not appear in any list:

```bash
aws configure get region     # us-east-2 for the workshop account; if it is wrong, run aws configure sso again
```

The credentials expire after a few hours. Run `aws sso login` again when a command returns `ExpiredToken`. Nothing that would still work next month is written anywhere.

If you are in the notebook instance's own terminal instead (JupyterLab, **File → New → Terminal**), skip the install and the sign-in. The instance's role is already the identity, and every command below works there as written.

The sections below are the AWS commands, so that you see what the SDK was doing. Most of the lesson's helpers wrapped one of these calls and are not needed here. The ones that do arithmetic, the cost estimators and the launch check, come back in section 5, from the lesson's `scripts/helpers.py`, plain Python that runs on a laptop as it did on the notebook instance. Clone the lesson once and install `boto3`, which the module imports:

```bash
git clone -q https://github.com/carpentries-incubator/ML_with_AWS_SageMaker.git   # once, anywhere on the laptop
pip install boto3
```

## 2. What is running, and what a job cost

First the lists. Each should be short, or empty. `--query` is a [JMESPath](https://jmespath.org/) filter that keeps the output to what you asked about, and the cleanup episode's first cell ran the same four `list_*` calls through `boto3`.

```bash
aws sagemaker list-training-jobs   --status-equals InProgress --query 'TrainingJobSummaries[].TrainingJobName'
aws sagemaker list-processing-jobs --status-equals InProgress --query 'ProcessingJobSummaries[].ProcessingJobName'
aws sagemaker list-hyper-parameter-tuning-jobs --status-equals InProgress --query 'HyperParameterTuningJobSummaries[].HyperParameterTuningJobName'
aws sagemaker list-notebook-instances --query 'NotebookInstances[].[NotebookInstanceName,NotebookInstanceStatus,InstanceType]' --output table
aws sagemaker list-endpoints       # must print an empty list on the workshop account
```

Your notebook instance is in the fourth list, and `aws s3 ls` prints your bucket. Those two names come up in most commands below, so set them as shell variables once per terminal session, after the login, and the commands use `$NB` and `$BUCKET` from here on:

::::::::::::::::::::::::::::::::::::: tab

### Bash

```bash
aws s3 ls
NB='<notebook-instance-name>'
BUCKET='<your-bucket>'
```

### PowerShell

```powershell
aws s3 ls
$NB = "<notebook-instance-name>"
$BUCKET = "<your-bucket>"
```

::::::::::::::::::::::::::::::::::::::::::::::::

Then one job's bill. `job_cost_report()` in the training episodes called `describe_training_job` and read four fields from it, the status, the instance count, `BillableTimeInSeconds`, and the model's S3 path, and computed the wall-clock from `CreationTime` and `TrainingEndTime`. This is the same call with the same fields. Pick a job from the list below, newest first with its status, and put its full name in place of `<a training job name>`. On the shared account the list is everyone's jobs, so do not take the newest. Take the one named after your notebook instance:

```bash
aws sagemaker list-training-jobs --sort-by CreationTime --sort-order Descending --query 'TrainingJobSummaries[].[TrainingJobName,TrainingJobStatus]' --output table
# only your own, since the lesson names every job after its notebook instance: filter on that name
# aws sagemaker list-training-jobs --name-contains $NB --sort-by CreationTime --sort-order Descending --query 'TrainingJobSummaries[].[TrainingJobName,TrainingJobStatus]' --output table
# the same list in name order, which groups each learner's jobs together
# aws sagemaker list-training-jobs --sort-by Name --sort-order Ascending --query 'TrainingJobSummaries[].[TrainingJobName,TrainingJobStatus]' --output table
```

::::::::::::::::::::::::::::::::::::: tab

### Bash

```bash
JOB='<a training job name>'
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

Cost is `billable_s` × `count` ÷ 3600 × the hourly rate from the [Instances for ML](https://carpentries-incubator.github.io/ML_with_AWS_SageMaker/instances-for-ML.html) page, the arithmetic `job_cost_report()` did. The helper printed it. Here you do the multiplication. `billable_s` can be missing for a job that failed. Read the status and `FailureReason` before reading anything into its cost.

The other job kinds have the same shape. The Processing Jobs episode's `job_cost()` read `CreationTime`, `ProcessingStartTime` and `ProcessingEndTime` from `describe_processing_job`, and the tuning episode's `tuning_cost_report()` looped `describe_training_job` over the tuner's child jobs:

```bash
aws sagemaker describe-processing-job --processing-job-name <name> --query '{status:ProcessingJobStatus, created:CreationTime, started:ProcessingStartTime, ended:ProcessingEndTime, instance:ProcessingResources.ClusterConfig.InstanceType}'
aws sagemaker list-training-jobs-for-hyper-parameter-tuning-job --hyper-parameter-tuning-job-name <name> --query 'TrainingJobSummaries[].[TrainingJobName,TrainingJobStatus,FinalHyperParameterTuningJobObjectiveMetric.Value]' --output table
aws sagemaker describe-hyper-parameter-tuning-job --hyper-parameter-tuning-job-name <name> --query 'BestTrainingJob.TrainingJobName'   # tuner.best_training_job()
```

Tags are returned by a separate call. The `Name` tag is how you confirm a job is yours before you stop it in the next section, and a job with no tags is the one that shows on the bill under "No tag key":

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

Each of these stops a charge. They are the calls behind the **Stop** buttons the cleanup episode walks through in the console. Every one takes a name, and on a shared account the name must be yours. The lists in section 2 show everyone's jobs and instances. Before any stop command, confirm the `Name` tag with the `list-tags` call from section 2, and stop only what you started. Someone else's job is a question for them, or for the instructor.

```bash
aws sagemaker stop-training-job   --training-job-name   <name>
aws sagemaker stop-processing-job --processing-job-name <name>
aws sagemaker stop-hyper-parameter-tuning-job --hyper-parameter-tuning-job-name <name>   # stops every child job
aws sagemaker stop-notebook-instance --notebook-instance-name <name>
```

A stopped job keeps its logs and whatever it had already written to S3. A stopped notebook instance keeps `/home/ec2-user/SageMaker/` and keeps billing for that volume. Deleting the instance is the step that stops the storage charge, and the cleanup episode's last two items, the bucket and the Bedrock inference profiles, have their CLI forms too:

```bash
aws sagemaker delete-notebook-instance --notebook-instance-name <name>   # only after it reports Stopped
aws s3 rb s3://$BUCKET --force                                             # empties the bucket, then deletes it; the cleanup episode's delete_objects loop plus delete_bucket
aws bedrock list-inference-profiles --type-equals APPLICATION --query 'inferenceProfileSummaries[].[inferenceProfileName,inferenceProfileArn]' --output table
aws bedrock delete-inference-profile --inference-profile-identifier <ARN>  # one per profile, as the cleanup episode's loop did
```

## 4. Copying to and from S3

The Accessing S3 episode moved single files with `s3.upload_file` and `s3.download_file` and listed the bucket with `helpers.show_bucket`. For many files, or for pulling a job's outputs down to a laptop, the CLI is the better tool, and these are the same keys those episodes used:

```bash
aws s3 ls s3://$BUCKET/ --recursive --human-readable --summarize             # helpers.show_bucket, for the whole bucket
aws s3 cp titanic_train.csv s3://$BUCKET/titanic_train.csv                     # s3.upload_file(train_filename, bucket_name, train_filename)
aws s3 cp s3://$BUCKET/xgboost/<job-name>/output/model.tar.gz model.tar.gz    # the download in the training episode's evaluate step
aws s3 sync s3://$BUCKET/xgboost/ ./xgboost-outputs/                           # every job's outputs under the prefix; only new or changed files move
aws s3 rm s3://$BUCKET/scratch/ --recursive                                    # delete a prefix; this one asks no questions
```

The last line, `sync --delete` and the `rb --force` in section 3 remove data for good and ask nothing first. Check the bucket name twice before any of them, and on the shared account run them only on your own bucket.

Downloads to a laptop are a transfer out of AWS (`$0.09` per GB). For anything larger than a few hundred megabytes, keep the data in the region and let a job read it, which is the lesson's pattern.

## 5. Launching the training episode's job from a JSON file

`xgboost_estimator.fit({'train': train_input})` in the training episode assembled a JSON request and sent it to `CreateTrainingJob`. SageMaker kept that request, and `describe-training-job` on the job you ran prints it back. Three fields come from there, because they are specific to your account and region, and the rest of the file below is the estimator's arguments under their API names.

First find the job's name. The training episode named it after your notebook instance, so it is the one with `xgboost` in it:

```bash
aws sagemaker list-training-jobs --name-contains "$NB-xgboost" --query 'TrainingJobSummaries[].TrainingJobName'
```

Put that name in place of `<your xgboost job name>` below.

::::::::::::::::::::::::::::::::::::: tab

### Bash

```bash
JOB='<your xgboost job name>'
aws sagemaker describe-training-job --training-job-name $JOB \
  --query '{image:AlgorithmSpecification.TrainingImage, role:RoleArn, code:HyperParameters.sagemaker_submit_directory}'
```

### PowerShell

```powershell
$JOB = "<your xgboost job name>"
aws sagemaker describe-training-job --training-job-name $JOB `
  --query '{image:AlgorithmSpecification.TrainingImage, role:RoleArn, code:HyperParameters.sagemaker_submit_directory}'
```

::::::::::::::::::::::::::::::::::::::::::::::::

The describe output is a description, not a request. It carries fields AWS fills in after the fact, so it cannot be saved and sent back as it is, which is why the file below is written out from the estimator's arguments and takes only three values from the description.

`image` is the XGBoost 1.7-1 container that `framework_version="1.7-1"` picked. `role` is the `role` the notebook printed. `code` is where `fit()` uploaded `source_dir`, the `sourcedir.tar.gz` in the SDK's default bucket that the training episode pointed out under *What landed in S3*. Reusing it means this job runs the same `train_xgboost.py` the notebook's job ran. Save the file below as `train-xgboost.json`, in the folder your terminal is in, with any text editor. The file is the same whichever shell you use, and a file cannot read shell variables, so the two names are written out in it. Put the three values in place of `<image>`, `<role>` and `<code>`, your bucket in place of `<your-bucket>`, your notebook instance's name in `TrainingJobName`, and your tag values in place of `<LastnameFirstname>` and `<ChallengeName>`. The `code` value prints with quotation marks inside it; put only the `s3://...` path in place of `<code>`, since the file supplies the quotes. The `-cli-1` suffix makes the job name unique; use `-cli-2` for a second run, since SageMaker refuses a name it has seen.

```json
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
```

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

Before sending it, estimate it, as the notebooks did before every launch. Nothing in a JSON file prints a price, so the hand method is the instance's hourly rate from the [Instances for ML](https://carpentries-incubator.github.io/ML_with_AWS_SageMaker/instances-for-ML.html) page times the minutes you expect, and `MaxRuntimeInSeconds` at about twice that. This XGBoost job is startup and little else, two to three minutes on `ml.m5.large`, under a cent. For a job with real compute the lesson's estimators do the arithmetic, with the same arguments the notebooks gave them. They are called from `python -c`, with the lesson's scripts folder put on the path first. These are the PyTorch episode's estimate, the tuning episode's and the fine-tuning episode's, and each prints the time, the cost and the `max_run` to put in the request:

```bash
python -c "import sys; sys.path.insert(0, 'ML_with_AWS_SageMaker/scripts'); import helpers; helpers.estimate_training_cost(params_billions=0.125, examples=1_000_000, tokens_per_example=256, epochs=3, instance_type='ml.g6.xlarge')"
python -c "import sys; sys.path.insert(0, 'ML_with_AWS_SageMaker/scripts'); import helpers; helpers.estimate_tuning_cost(train_minutes_per_job=1, price_per_hour=0.115, max_jobs=8)"
python -c "import sys; sys.path.insert(0, 'ML_with_AWS_SageMaker/scripts'); import helpers; helpers.estimate_finetune_cost(params_billions=3.75, examples=251, tokens_per_example=1600, epochs=2, instance_type='ml.g5.xlarge')"
```

Then check the request. `scripts/check_request.py` reads the JSON and does two things. It applies the account's rules from `scripts/guardrails.json`, as `helpers.check_launch` did before every launch in the notebooks, so a `Project` or `Purpose` off the lists, an instance type off the list or a cap over 48 hours is named here, where AWS would refuse it silently. Then it prints the most the request can bill: every instance it asks for, running to its cap, at the instance's hourly price, and for a tuning job every child job it allows. `--max-cost` is a dollar amount you choose, the most you are willing to spend on this job. When the worst case is above it, the checker exits with an error, and the launch command below is written so that an error stops it. Here it is `1`, one dollar, because the job's worst case is four cents:

```bash
python ML_with_AWS_SageMaker/scripts/check_request.py train-xgboost.json --max-cost 1   # refuse if this job could bill more than $1
```

```output
train-xgboost.json: tags, instance type and cap pass the account's rules
worst case if every job runs to its cap: 1 x 0.33 h x $0.115/h = $0.04
```

The same command takes a Processing job's or a tuning job's request, and the tuning job's worst case multiplies by its `MaxNumberOfTrainingJobs`, which is where a tuning request surprises people.

The check catches configuration and cost mistakes before a request reaches AWS. It does not grant permission to launch the job or guarantee that AWS accepts it. The account's own policy decides that, and on the shared workshop account the rule is the one above, launch from the notebook instance.

Then send it. This one line is `estimator.fit()`, and the `&&` in front of it means the create call runs only when the check passed:

```bash
python ML_with_AWS_SageMaker/scripts/check_request.py train-xgboost.json --max-cost 1 && aws sagemaker create-training-job --cli-input-json file://train-xgboost.json
```

The CLI returns the moment the request is accepted. Nothing waits, which is `fit(wait=False)`. To follow the job the way the default `wait=True` did, then read its bill as in section 2:

```bash
aws sagemaker wait training-job-completed-or-stopped --training-job-name "$NB-xgboost-cli-1"   # blocks until it ends
aws sagemaker describe-training-job --training-job-name "$NB-xgboost-cli-1" --query '{status:TrainingJobStatus, billable_s:BillableTimeInSeconds, failure:FailureReason, model:ModelArtifacts.S3ModelArtifacts}'
```

Expect the same two to three billable minutes on `ml.m5.large` as the notebook's job, under a cent, and the same `xgboost-model.json` inside the `model.tar.gz` it names.

The PyTorch job, a Processing job and a tuning job follow the same route. `describe-training-job`, `describe-processing-job` and `describe-hyper-parameter-tuning-job` on the ones you ran print the request, and `create-training-job`, `create-processing-job` and `create-hyper-parameter-tuning-job` take it back with `--cli-input-json`. A PyTorch job's `HyperParameters` carries `epochs` and `learning_rate` and two input channels. A Processing job's `AppSpecification.ContainerArguments` is the `arguments=` list. A tuning job's file has the estimator's request nested under `TrainingJobDefinition` and the ranges under `HyperParameterTuningJobConfig.ParameterRanges`.

::::::::::::::::::::::::::::::::::::: callout

### What the lesson never does from the CLI

- `aws sagemaker create-endpoint` / `create-endpoint-config`: an endpoint is a server that runs until you delete it, with no runtime cap. The lesson serves models with processing jobs and Bedrock so that nothing of this kind exists in the account.
- `aws sagemaker create-notebook-instance` with a GPU type: a GPU notebook is `$60` a day while you are asleep. The optional GPU-notebook episode creates one through the console, where it stays visible, and deletes it the same session.
- `aws ec2 run-instances`: bare virtual machines have no tags required, no runtime cap and no SageMaker job list to appear in.

On the workshop account all three are denied by the guard-rail policy the administrator attached, as is any instance type outside the lesson's list, any cap over 48 hours and any job whose tags are missing or off the lists. On your own account, the policy is on the instructor page [Account preparation checklist](../instructors/aws-account-prep.md). Attaching it to your own user or role takes an afternoon and is worth it.

::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: challenge

### Challenge: what is still running on Friday evening

You are leaving for the weekend and have a terminal, not a browser. Write the commands you would run, in order, to make sure nothing of yours is costing money until Monday, and the one command to run if a check finds something. Then run them.

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

For anything of yours the first four print, confirm it with `list-tags` and run the matching `stop-...` command from section 3. A job you expect to finish within its `MaxRuntimeInSeconds` can be left. A job that is not yours is its owner's, and the tags say who that is. For an endpoint, `aws sagemaker delete-endpoint --endpoint-name <name>`, then find out how it was created, because on the workshop account the guard-rail policy refuses it. Buckets cost almost nothing over a weekend and need no action.

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

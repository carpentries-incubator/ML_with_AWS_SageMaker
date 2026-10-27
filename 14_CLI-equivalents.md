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
- List, inspect, and stop training jobs, processing jobs, tuning jobs and notebook instances from the terminal.
- Copy data to and from S3 in bulk with `aws s3 sync`.
- Launch one training job from its JSON description and recognise it as the request the SDK built in the training episodes.
- State which resources the lesson never creates from the CLI, and why the account refuses them.

::::::::::::::::::::::::::::::::::::::::::::::::

### Optional, self-study episode

Everything in this lesson went through the SageMaker Python SDK in a notebook. The same account can be used from a terminal with the AWS CLI: to check on a job from another machine, to stop something when no notebook is open, to copy results out of S3 in bulk, or to script a launch. This episode does five things from a laptop terminal: signs in to the account with Identity Center, lists what is running and reads what a job cost, stops jobs and notebook instances, copies data to and from S3, and launches one training job from a JSON file, the same request `estimator.fit()` sent in the training episodes.

The case for the terminal is the notebook instance. Every job in this lesson was launched from an `ml.t3.medium` that bills while it is on, five cents an hour, a dollar a day, and that someone has to remember to stop. A job launched from a laptop needs no instance at all: the request goes from your terminal to SageMaker, the job runs on a machine rented for it and ends, and the model lands in S3. For a project that launches a job a day and otherwise works locally, that is the notebook's whole cost gone and nothing left running overnight. The training episode's `wait=False` pattern gets most of the way there from a notebook; the CLI gets the rest.

What you give up is what the instance did for you. On the instance, its role signs every call, the lesson's helper checks a launch against the account's rules before it goes, and the console shows what you started next to everyone else's work. On a laptop you sign in yourself, the JSON carries no price and no check, and a mistake is not seen until the bill. The rules in the callout and the request check in section 5 cover that.

::::::::::::::::::::::::::::::::::::: callout

### Read this before the first command

The CLI, the SDK and the console are the same API. Nothing here is more powerful than a notebook cell. The difference is that a terminal shows nothing unless you ask: a notebook instance left running is visible in the console the next day, while an endpoint created from a terminal on a Friday is not noticed until the bill arrives. A past participant of this workshop spent a large share of a year's credits that way. Three rules apply on the shared account and are worth keeping on your own:

1. **Authenticate with `aws sso login` only.** Never `aws configure` with an access key ID and secret. Those are long-lived, and they end up in shell histories and committed files.
2. **Launch only jobs** (training, processing, tuning), with the same three tags and the same `MaxRuntimeInSeconds` the notebooks always set. Jobs end on their own. **Endpoints, EC2 instances and GPU notebook instances do not**, and the lesson never creates them from the CLI, or at all.
3. **A refused `create` command is the account's rules, not a fault.** The workshop account carries a guard-rail policy that denies endpoint and EC2 creation, instance types outside the lesson's list, caps over 48 hours, and any job whose three tags are missing, blank or off the `Project` and `Purpose` lists. Do not look for a workaround. The policy is on the notebook role today, which covers every launch from an instance; a laptop CLI runs under the permission set, which gets the same policy when the Identity Center administrator attaches it, and until then these rules are yours to keep by hand from a laptop. (Instructors: the policy and its status are on the *Account preparation checklist* page.)

::::::::::::::::::::::::::::::::::::::::::::::::

## 1. Sign in with Identity Center

This episode runs from your own laptop, a terminal that works when no notebook is open. The AWS CLI is already installed in CloudShell and on every SageMaker notebook instance, where it uses the instance's role and needs no sign-in; a laptop has neither, so install [AWS CLI v2](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html) there, then register the account's Identity Center once. It asks four things. First the start URL and the region of the Identity Center sign-in page, the same ones you used in the browser; for the workshop account both are in the sign-in email from your instructor. It then opens a browser tab for you to approve the device, lets you pick the account and the permission set, and finally asks for a default region, `us-east-2` for the workshop account, and a name for the profile.

```bash
aws configure sso
```

The commands in this episode are the same in Bash (macOS, Linux, Git Bash on Windows) and in PowerShell; where the shell's own syntax differs, variables, line continuation, writing a file, the two are shown as tabs.

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

The output names the account ID and an `AWSReservedSSO_...` role. The credentials expire after a few hours; `aws sso login` again when a command returns `ExpiredToken`. Nothing that would still work next month is written anywhere.

If you are on the notebook instance's own terminal (JupyterLab, **File → New → Terminal**), skip this step: the instance's role is already the identity, which is the reason the lesson works from one.

## 2. What is running, and what a job cost

The SageMaker SDK calls you used map one-to-one onto `aws sagemaker` subcommands. `--query` is a [JMESPath](https://jmespath.org/) filter that keeps the output to what you asked about.

```bash
# Everything that is running right now, by type. Each list should be short, or empty.
aws sagemaker list-training-jobs   --status-equals InProgress --query 'TrainingJobSummaries[].TrainingJobName'
aws sagemaker list-processing-jobs --status-equals InProgress --query 'ProcessingJobSummaries[].ProcessingJobName'
aws sagemaker list-hyper-parameter-tuning-jobs --status-equals InProgress --query 'HyperParameterTuningJobSummaries[].HyperParameterTuningJobName'
aws sagemaker list-notebook-instances --query 'NotebookInstances[].[NotebookInstanceName,NotebookInstanceStatus,InstanceType]' --output table
aws sagemaker list-endpoints       # must print an empty list on the workshop account
```

`describe-training-job` is the call behind `job_cost_report()` in the training episodes. The same number is there: billable time, which includes the container pull and data download you paid for.

::::::::::::::::::::::::::::::::::::: tab

### Bash

```bash
JOB=<a training job name from the list above>
aws sagemaker describe-training-job --training-job-name $JOB \
  --query '{status:TrainingJobStatus, instance:ResourceConfig.InstanceType, count:ResourceConfig.InstanceCount, billable_s:BillableTimeInSeconds}'
```

### PowerShell

```powershell
$JOB = "<a training job name from the list above>"
aws sagemaker describe-training-job --training-job-name $JOB `
  --query '{status:TrainingJobStatus, instance:ResourceConfig.InstanceType, count:ResourceConfig.InstanceCount, billable_s:BillableTimeInSeconds}'
```

::::::::::::::::::::::::::::::::::::::::::::::::

Cost is billable seconds × instance count × the hourly rate, exactly as in the notebooks. Tags are returned by a separate call, which is also how you find the untagged job that shows on the bill under "No tag key":

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

Each of these stops a charge.

```bash
aws sagemaker stop-training-job   --training-job-name   <name>
aws sagemaker stop-processing-job --processing-job-name <name>
aws sagemaker stop-hyper-parameter-tuning-job --hyper-parameter-tuning-job-name <name>   # stops every child job
aws sagemaker stop-notebook-instance --notebook-instance-name <name>
```

A stopped job keeps its logs and whatever it had already written to S3; a stopped notebook instance keeps `/home/ec2-user/SageMaker/`. Deleting an instance when you are done with it is the step that stops the storage charge too:

```bash
aws sagemaker delete-notebook-instance --notebook-instance-name <name>   # only after it reports Stopped
```

## 4. Copying to and from S3

The S3 episode uploaded through the console and the SDK. For many files, or for pulling a job's outputs down to a laptop, the CLI is the better tool:

```bash
aws s3 ls s3://<your-bucket>/ --recursive --human-readable --summarize     # every object, and the bucket's total size
aws s3 sync s3://<your-bucket>/xgboost/ ./xgboost-outputs/                 # download a prefix; only new or changed files move
aws s3 sync ./my-data/ s3://<your-bucket>/my-data/                         # upload a folder the same way
aws s3 rm s3://<your-bucket>/scratch/ --recursive                          # delete a prefix; this one asks no questions
```

Downloads to a laptop are a transfer out of AWS (`$0.09` per GB). For anything larger than a few hundred megabytes, keep the data in the region and let a job read it, which is the lesson's pattern.

## 5. Launching a training job from a JSON file

Every `estimator.fit()` in the training episodes assembled a JSON request and sent it to `CreateTrainingJob`. Here is that request. Fill in the placeholders from your notebook: the image URI (run `print(xgboost_estimator.training_image_uri())` in the training episode's notebook), your role ARN (`print(role)`), your bucket, your tag values, and a job name, which must be unique in the account.

The request is a JSON file. Save it as `train-xgboost.json` with any editor, or from the shell:

::::::::::::::::::::::::::::::::::::: tab

### Bash

```bash
cat > train-xgboost.json <<'JSON'
{
  "TrainingJobName": "<notebook-instance-name>-cli-xgboost-1",
  "AlgorithmSpecification": {
    "TrainingImage": "<xgboost image URI for your region>",
    "TrainingInputMode": "File"
  },
  "RoleArn": "<execution role ARN>",
  "HyperParameters": { "objective": "binary:logistic", "num_round": "100", "max_depth": "5", "eta": "0.1" },
  "InputDataConfig": [
    {
      "ChannelName": "train",
      "ContentType": "text/csv",
      "DataSource": { "S3DataSource": {
        "S3DataType": "S3Prefix",
        "S3Uri": "s3://<your-bucket>/xgboost/builtin/train_builtin.csv",
        "S3DataDistributionType": "FullyReplicated" } }
    }
  ],
  "OutputDataConfig": { "S3OutputPath": "s3://<your-bucket>/xgboost/cli-output/" },
  "ResourceConfig": { "InstanceType": "ml.m5.large", "InstanceCount": 1, "VolumeSizeInGB": 10 },
  "StoppingCondition": { "MaxRuntimeInSeconds": 1200 },
  "Tags": [
    { "Key": "Project", "Value": "<ChallengeName>" },
    { "Key": "Name",    "Value": "<LastnameFirstname>" },
    { "Key": "Purpose", "Value": "Training" }
  ]
}
JSON
```

### PowerShell

```powershell
@'
{
  "TrainingJobName": "<notebook-instance-name>-cli-xgboost-1",
  "AlgorithmSpecification": {
    "TrainingImage": "<xgboost image URI for your region>",
    "TrainingInputMode": "File"
  },
  "RoleArn": "<execution role ARN>",
  "HyperParameters": { "objective": "binary:logistic", "num_round": "100", "max_depth": "5", "eta": "0.1" },
  "InputDataConfig": [
    {
      "ChannelName": "train",
      "ContentType": "text/csv",
      "DataSource": { "S3DataSource": {
        "S3DataType": "S3Prefix",
        "S3Uri": "s3://<your-bucket>/xgboost/builtin/train_builtin.csv",
        "S3DataDistributionType": "FullyReplicated" } }
    }
  ],
  "OutputDataConfig": { "S3OutputPath": "s3://<your-bucket>/xgboost/cli-output/" },
  "ResourceConfig": { "InstanceType": "ml.m5.large", "InstanceCount": 1, "VolumeSizeInGB": 10 },
  "StoppingCondition": { "MaxRuntimeInSeconds": 1200 },
  "Tags": [
    { "Key": "Project", "Value": "<ChallengeName>" },
    { "Key": "Name",    "Value": "<LastnameFirstname>" },
    { "Key": "Purpose", "Value": "Training" }
  ]
}
'@ | Set-Content -Path train-xgboost.json
```

::::::::::::::::::::::::::::::::::::::::::::::::

Before sending it, check it the way the notebooks did. The lesson repository has the account's rules in `scripts/guardrails.json` and a script that applies them to a request file; a `Project` or `Purpose` off the lists, an instance type off the list or a cap over 48 hours is named here rather than refused silently by AWS:

```bash
git clone -q https://github.com/carpentries-incubator/ML_with_AWS_SageMaker.git   # once, anywhere on the laptop
python ML_with_AWS_SageMaker/scripts/check_request.py train-xgboost.json
```

Then send it. This one line is `estimator.fit()`:

```bash
aws sagemaker create-training-job --cli-input-json file://train-xgboost.json
```

Three fields to notice. `StoppingCondition.MaxRuntimeInSeconds` is the `max_run` you set on every estimator; the SDK would have sent 24 hours if you had not. `Tags` is the `tags=` argument; without it the request is refused when sent from a notebook instance, and nothing checks it from a laptop until the permission set carries the policy. `ResourceConfig` sets the instance type and count, which determine the cost; nothing in a JSON file prints a price, so look the type up on the [Instances for ML](https://carpentries-incubator.github.io/ML_with_AWS_SageMaker/instances-for-ML.html) page before sending it, as you would in a notebook.

The CLI returns the moment the request is accepted; nothing waits. To follow the job the way `wait=True` did:

```bash
aws sagemaker wait training-job-completed-or-stopped --training-job-name <the job name>   # blocks until it ends
aws sagemaker describe-training-job --training-job-name <the job name> --query '{status:TrainingJobStatus, billable_s:BillableTimeInSeconds, failure:FailureReason}'
```

Estimate first, compare after, as always: this is the same ~4 minutes billable on `ml.m5.large`, about a cent.

::::::::::::::::::::::::::::::::::::: callout

### What the lesson never does from the CLI

- `aws sagemaker create-endpoint` / `create-endpoint-config`: an endpoint is a server that runs until you delete it, with no runtime cap. The lesson serves models with processing jobs and Bedrock so that nothing of this kind exists in the account.
- `aws sagemaker create-notebook-instance` with a GPU type: a GPU notebook is `$60` a day while you are asleep. The optional GPU-notebook episode creates one through the console, where it stays visible, and deletes it the same session.
- `aws ec2 run-instances`: bare virtual machines have no tags required, no runtime cap and no SageMaker job list to appear in.

On the workshop account all three are denied on the notebook role, as is any instance type outside the lesson's list, any cap over 48 hours and any job whose tags are missing or off the lists; the permission set a laptop uses gets the same policy once the administrator attaches it. The instructor page *Account preparation checklist* has the policy. On your own account, the same denies are an afternoon's work.

::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: challenge

### Challenge: the Friday-evening check

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

For anything the first four print: the matching `stop-...` command from section 3. A job you recognise as yours and expect to finish within its `MaxRuntimeInSeconds` can be left; a job you do not recognise is a question for its owner, and on a shared account the tags (`list-tags`) tell you who that is. For an endpoint: `aws sagemaker delete-endpoint --endpoint-name <name>`, then find out how it was created, because the notebook role refuses it and the permission set should once the policy is attached. Buckets cost almost nothing over a weekend and need no action.

::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::::::::::::::

:::::::::::::::::::::::::::::::::::::: keypoints 

- The CLI, the SDK and the console are one API; the CLI shows you nothing you do not ask for, so its first use is to look and to stop, not to launch.
- Authenticate with `aws sso login` and short-lived credentials; never store an access key.
- `list-*`, `describe-*` and `stop-*` for training, processing and tuning jobs and notebook instances cover almost every reason to open a terminal.
- A `create-training-job` JSON is what `estimator.fit()` sent: the same tags, the same `MaxRuntimeInSeconds`, the same instance choice, with no price printed next to it.
- Endpoints, EC2 instances and GPU notebooks are not jobs and do not end; the workshop account's notebook role denies creating them, and your own account should too.

::::::::::::::::::::::::::::::::::::::::::::::::

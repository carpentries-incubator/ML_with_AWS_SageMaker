---
title: "(Optional) The Same Jobs from the Terminal"
teaching: 20
exercises: 10
---

:::::::::::::::::::::::::::::::::::::: questions 

- What does the AWS command-line interface (CLI) let me do that the lesson's notebooks did, and what should I not do with it?
- How do I see, and stop, everything that is costing money from a terminal?
- What was the SageMaker SDK actually sending when I called `estimator.fit()`?

::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: objectives

- Authenticate the CLI with short-lived Identity Center credentials, never with access keys.
- List, inspect, and stop training jobs, processing jobs, tuning jobs and notebook instances from the terminal.
- Launch one training job from its JSON description and recognise it as the request the SDK built in the training episodes.
- State which resources the lesson never creates from the CLI, and why the account refuses them.

::::::::::::::::::::::::::::::::::::::::::::::::

### Optional, self-study episode

Everything in this lesson went through the SageMaker Python SDK in a notebook. The same account can be driven from a terminal with the AWS CLI, and sooner or later you will want to: to check on a job from your phone's SSH client, to stop something from a machine where you have no notebook open, to copy results out of S3 in bulk, or to script a launch. This episode shows the CLI equivalents of what you already did, in that order of usefulness: **look, stop, copy, then launch**.

::::::::::::::::::::::::::::::::::::: callout

### Read this before the first command

The CLI, the SDK and the console are the same API. Nothing here is more powerful than a notebook cell; the difference is that a terminal shows you *nothing* unless you ask. A notebook instance left running sits in a console tab you will see tomorrow; an endpoint created from a terminal on a Friday is invisible until the bill arrives. One past participant of this workshop spent a large share of a year's credits that way. So three rules, in force for the shared account and worth keeping on your own:

1. **Authenticate with `aws sso login` only.** Never `aws configure` with an access key ID and secret. Those are long-lived, they end up in shell histories and committed files, and they are the usual way a research group's keys reach the internet.
2. **Launch only jobs** (training, processing, tuning), with the same three tags and the same `MaxRuntimeInSeconds` the notebooks always set. Jobs end on their own. **Endpoints, EC2 instances and GPU notebook instances do not**, and the lesson never creates them from the CLI, or at all.
3. **If the account refuses a command, that is the system working.** The workshop permission set denies endpoint and EC2 creation, instance types outside the lesson's list, and any job launched without the tags. An `AccessDeniedException` on one of those is not a permissions bug to work around; it is the guard rail.

::::::::::::::::::::::::::::::::::::::::::::::::

## 1. Sign in with Identity Center

Install [AWS CLI v2](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html) if you do not have it, then register the account's Identity Center once. It asks for the start URL and region of the sign-in page you used in the browser (for the workshop account they are in your sign-in email), opens a browser to approve the device, lets you pick the account and permission set, and asks for a default region (`us-east-2` for the workshop account) and a profile name.

```bash
aws configure sso
```

From then on, every session starts with a login, and a check that you are who you think you are, in the account you think you are in:

```bash
aws sso login --profile workshop
export AWS_PROFILE=workshop          # PowerShell: $env:AWS_PROFILE="workshop"
aws sts get-caller-identity
```

The output names the account ID and an `AWSReservedSSO_...` role. The credentials expire after a few hours; `aws sso login` again when a command returns `ExpiredToken`. Nothing that would still work next month is written anywhere.

If you are on the notebook instance's own terminal (JupyterLab, **File → New → Terminal**), skip this step: the instance's role is already the identity, which is the reason the lesson works from one.

## 2. Look: what is running and what did it cost

The SageMaker SDK calls you used map one-to-one onto `aws sagemaker` subcommands. `--query` is a [JMESPath](https://jmespath.org/) filter that keeps the output to what you asked about.

```bash
# Everything that is running right now, by type. Each list should be short, or empty.
aws sagemaker list-training-jobs   --status-equals InProgress --query 'TrainingJobSummaries[].TrainingJobName'
aws sagemaker list-processing-jobs --status-equals InProgress --query 'ProcessingJobSummaries[].ProcessingJobName'
aws sagemaker list-hyper-parameter-tuning-jobs --status-equals InProgress --query 'HyperParameterTuningJobSummaries[].HyperParameterTuningJobName'
aws sagemaker list-notebook-instances --query 'NotebookInstances[].[NotebookInstanceName,NotebookInstanceStatus,InstanceType]' --output table
aws sagemaker list-endpoints       # must print an empty list on the workshop account
```

`describe-training-job` is the call behind `job_cost_report()` in the training episodes. The same two numbers are there: training time, and billable time, which includes the startup you paid for.

```bash
JOB=<a training job name from the list above>
aws sagemaker describe-training-job --training-job-name $JOB \
  --query '{status:TrainingJobStatus, instance:ResourceConfig.InstanceType, count:ResourceConfig.InstanceCount, training_s:TrainingTimeInSeconds, billable_s:BillableTimeInSeconds, tags:Tags}'
```

Cost is billable seconds × instance count × the hourly rate, exactly as in the notebooks. Tags are returned by a separate call, which is also how you find the untagged job that will show up as "unallocated" on the bill:

```bash
ARN=$(aws sagemaker describe-training-job --training-job-name $JOB --query TrainingJobArn --output text)
aws sagemaker list-tags --resource-arn $ARN --query 'Tags[].[Key,Value]' --output table
```

## 3. Stop: the commands worth memorising

These are the reason to have the CLI installed at all. Each one makes money stop.

```bash
aws sagemaker stop-training-job   --training-job-name   <name>
aws sagemaker stop-processing-job --processing-job-name <name>
aws sagemaker stop-hyper-parameter-tuning-job --hyper-parameter-tuning-job-name <name>   # stops every child job
aws sagemaker stop-notebook-instance --notebook-instance-name <name>
```

A stopped job keeps its logs and whatever it had already written to S3; a stopped notebook instance keeps `/home/ec2-user/SageMaker/`. Stopping is never the expensive mistake. Deleting an instance when you are done with it is the step that stops the storage charge too:

```bash
aws sagemaker delete-notebook-instance --notebook-instance-name <name>   # only after it reports Stopped
```

## 4. Copy: S3 in bulk

The S3 episode uploaded through the console and the SDK. For many files, or for pulling a job's outputs down to a laptop, the CLI is the better tool:

```bash
aws s3 ls s3://<your-bucket>/ --recursive --human-readable --summarize     # every object, and the bucket's total size
aws s3 sync s3://<your-bucket>/xgboost/ ./xgboost-outputs/                 # download a prefix; only new or changed files move
aws s3 sync ./my-data/ s3://<your-bucket>/my-data/                         # upload a folder the same way
aws s3 rm s3://<your-bucket>/scratch/ --recursive                          # delete a prefix; this one asks no questions
```

Downloads to a laptop are a transfer out of AWS ($0.09 per GB). For anything larger than a few hundred megabytes, keep the data in the region and let a job read it, which is the lesson's pattern.

## 5. Launch: one training job, as JSON

Every `estimator.fit()` in the training episodes assembled a JSON request and sent it to `CreateTrainingJob`. Seeing that request once makes the SDK's arguments stop looking like magic. Fill in the five placeholders from your notebook: the image URI (run `print(xgboost_estimator.training_image_uri())` in the training episode's notebook), your role ARN (`print(role)`), your bucket, your tag values, and a job name, which must be unique in the account.

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
    { "Key": "Project", "Value": "<challenge name>" },
    { "Key": "Name",    "Value": "<your name>" },
    { "Key": "Purpose", "Value": "train_XGBoost_cli" }
  ]
}
JSON
aws sagemaker create-training-job --cli-input-json file://train-xgboost.json
```

Three things to notice. `StoppingCondition.MaxRuntimeInSeconds` is the `max_run` you set on every estimator; the SDK would have sent 48 hours if you had not. `Tags` is the `tags=` argument, and without it the request is refused on the workshop account. And `ResourceConfig` is where the money is: the instance type you would think twice about in a notebook deserves the same second thought in a JSON file, where nothing prints a price.

The CLI returns the moment the request is accepted; nothing waits. To follow the job the way `wait=True` did:

```bash
aws sagemaker wait training-job-completed-or-stopped --training-job-name <the job name>   # blocks until it ends
aws sagemaker describe-training-job --training-job-name <the job name> --query '{status:TrainingJobStatus, billable_s:BillableTimeInSeconds, failure:FailureReason}'
```

Estimate first, compare after, as always: this is the same ~4 minutes billable on `ml.m5.large`, about a cent.

::::::::::::::::::::::::::::::::::::: callout

### What the lesson never does from the CLI

- `aws sagemaker create-endpoint` / `create-endpoint-config`: an endpoint is a server that runs until you delete it, with no runtime cap. The lesson serves models with processing jobs and Bedrock precisely so that nothing of this kind exists in the account.
- `aws sagemaker create-notebook-instance` with a GPU type: a GPU notebook is $60 a day while you are asleep. The optional GPU-notebook episode creates one through the console, where it stays visible, and deletes it the same hour.
- `aws ec2 run-instances`: bare virtual machines have no tags required, no runtime cap and no SageMaker job list to appear in.

On the workshop account all three are denied by the permission set, as is any instance type outside the lesson's list and any job without the three tags; the instructor page *Account preparation checklist* has the policy. On your own account, the same denies are a fifteen-minute setup and the best money you will not spend.

::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: challenge

### Challenge: the Friday-evening check

You are leaving for the weekend and have a terminal, not a browser. Write the commands you would run, in order, to make sure the account is costing nothing until Monday, and the one command to run if each check finds something. Then run them for real against your own resources.

:::::::::::::::::::::::::::::::::::: solution

```bash
aws sso login --profile workshop && export AWS_PROFILE=workshop
aws sagemaker list-training-jobs   --status-equals InProgress --query 'TrainingJobSummaries[].TrainingJobName'
aws sagemaker list-processing-jobs --status-equals InProgress --query 'ProcessingJobSummaries[].ProcessingJobName'
aws sagemaker list-hyper-parameter-tuning-jobs --status-equals InProgress --query 'HyperParameterTuningJobSummaries[].HyperParameterTuningJobName'
aws sagemaker list-notebook-instances --query 'NotebookInstances[?NotebookInstanceStatus==`InService`].NotebookInstanceName'
aws sagemaker list-endpoints --query 'Endpoints[].EndpointName'
```

For anything the first four print: the matching `stop-...` command from section 3. A job you recognise as yours and expect to finish within its `MaxRuntimeInSeconds` can be left; a job you do not recognise is a question for its owner, and on a shared account the tags (`list-tags`) tell you who that is. For an endpoint: `aws sagemaker delete-endpoint --endpoint-name <name>`, then find out how it was created, because the permission set should have refused it. Buckets cost almost nothing over a weekend and need no action.

::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::::::::::::::

:::::::::::::::::::::::::::::::::::::: keypoints 

- The CLI, the SDK and the console are one API; the CLI shows you nothing you do not ask for, so its first use is to look and to stop, not to launch.
- Authenticate with `aws sso login` and short-lived credentials; never store an access key.
- `list-*`, `describe-*` and `stop-*` for training, processing and tuning jobs and notebook instances cover almost every reason to open a terminal.
- A `create-training-job` JSON is what `estimator.fit()` sent: the same tags, the same `MaxRuntimeInSeconds`, the same instance choice, with no price printed next to it.
- Endpoints, EC2 instances and GPU notebooks are not jobs and do not end; the workshop account denies creating them, and your own account should too.

::::::::::::::::::::::::::::::::::::::::::::::::

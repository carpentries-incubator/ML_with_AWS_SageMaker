---
title: 'Running this lesson from your own machine'
---

Every episode in this lesson runs on a SageMaker notebook instance, and the *Notebooks as Controllers* episode explains why: the instance's role is the identity, so no credential is installed on anyone's machine, and everyone has the same environment. If you later want to run the same cells from JupyterLab on your own computer, this page lists what changes. It is short, because the *jobs* do the work and the controller is cheap: a laptop is just another controller.

## 1. Authenticate with short-lived credentials, never with access keys

Install the [AWS CLI v2](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html), then register the account's IAM Identity Center (the sign-in page you used in the browser during the workshop) as an SSO profile:

```bash
aws configure sso
```

It asks for the start URL and region of the Identity Center instance (ask your account administrator; for the workshop account they are in your sign-in email), opens a browser to approve the device, then lets you pick the account and permission set and asks for a default region (`us-east-2` for the workshop account) and a profile name. Afterwards, every new day starts with

```bash
aws sso login --profile <profile-name>
export AWS_PROFILE=<profile-name>      # Windows Git Bash: the same; PowerShell: $env:AWS_PROFILE="<profile-name>"
aws sts get-caller-identity           # should print the account id and an AWSReservedSSO_... role
```

The credentials this issues expire after a few hours, and nothing is written to disk that would still work next month. **Do not run `aws configure` and paste an access key ID and secret**: those are long-lived, they end up in shell histories and notebooks, and committing one to GitHub is the standard way research groups lose money. If someone hands you a key pair, ask for SSO access instead.

When a session expires mid-episode, a running job keeps running on the SageMaker side; only your notebook's polling fails. Run `aws sso login` again and reattach with `estimator.attach(job_name)` or `sagemaker.estimator.Estimator.attach(...)`.

## 2. Build the environment once

Python 3.10 or newer, in a virtual environment (`python -m venv`, `uv venv` or conda), then:

```bash
pip install "sagemaker>=2.200" boto3 jupyterlab pandas numpy scikit-learn xgboost torch joblib
jupyter lab
```

`torch` and `xgboost` are only used for the local smoke tests in the training episodes; the jobs bring their own containers. The Bedrock episode needs nothing extra. The fine-tuning and Processing-jobs episodes install their dependencies inside the job.

## 3. The cells that change

| Where | On the notebook instance | On your laptop |
|---|---|---|
| Every episode's setup cell | `role = get_execution_role()` | Off a SageMaker instance this raises *"The current AWS identity is not a role"*. Set it explicitly: `role = "arn:aws:iam::<account-id>:role/<execution-role>"`. For the workshop account use the execution role your notebook instance had (copy its ARN from the instance's details page in the console, under **Permissions and encryption**). Your SSO permission set must allow `iam:PassRole` on that role; the workshop's does. |
| Every episode's setup cell | `region = boto3.Session().region_name` | Unchanged; it now reads the profile's default region. If it prints `None`, set `AWS_DEFAULT_REGION` or re-run `aws configure sso`. |
| Accessing S3, PyTorch episodes | `%cd /home/ec2-user/SageMaker/`, paths under it | Any folder you like; clone the lesson repository there and point the `sys.path.insert` line at its `scripts/` folder. |
| PyTorch episode | `helpers.get_notebook_instance_info(...)` to label the local run | There is no notebook instance to describe. Replace with `local_instance = "laptop"` (or your CPU model). The local timing is still the point of the cell. |
| Accessing S3 episode | Reads directly from S3 in the same region, free | A `read_csv` or `download_file` on the laptop pulls the data out of AWS at $0.09/GB. Jobs you launch from the laptop are unaffected: they read S3 in-region. The lesson's data is megabytes; for larger datasets, let jobs read it and keep only results on the laptop. |
| Optional GPU-notebook episode | Needs a `ml.g6e.xlarge` notebook instance | Does not apply. The Processing-jobs episode is the laptop-friendly version of the same work. |
| Resource management | Stop the notebook instance when you leave | Nothing to stop; your laptop is the controller. Everything else on the cleanup list (buckets, jobs, Bedrock profiles, endpoints) is the same. |

Everything else, every estimator, processor, tag block, `max_run` cap and cost report, is identical: the SageMaker SDK does not care where it is called from.

## 4. One thing you lose, one thing you gain

You lose the uniform environment: the version of the SageMaker SDK on your laptop is yours to keep current (`pip install -U sagemaker`), and a cell that works for you may fail for a colleague with a two-year-old install. You gain your own editor, your own files, and no instance to remember to stop.

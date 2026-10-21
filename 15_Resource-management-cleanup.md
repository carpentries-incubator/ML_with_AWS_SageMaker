---
title: "Resource Management and Monitoring"

teaching: 25
exercises: 10
---

:::::::::::::::::::::::::::::::::::::: questions 


- How can I monitor and manage AWS resources to avoid unnecessary costs?
- What steps are necessary to clean up SageMaker and S3 resources after the workshop?
- What best practices can help with efficient resource utilization?

::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: objectives

- Understand how to shut down SageMaker notebook instances to minimize costs.
- Clean up S3 storage and stop running jobs.
- Explore basic resource management strategies and tools for AWS.

::::::::::::::::::::::::::::::::::::::::::::::::




## Initial setup

Open the notebook from: `/ML_with_AWS_SageMaker/notebooks/15_Resource-management-cleanup.ipynb`. Any kernel works; **`conda_python3`** is fine. This episode is mostly console steps, but the cells below ask AWS what is still running or stored under your name, so you can check the list against reality before and after you clean up.

```python
import boto3
import sagemaker
import sys

sys.path.insert(0, "/home/ec2-user/SageMaker/ML_with_AWS_SageMaker/scripts")
import helpers

session = sagemaker.Session()
sm = session.sagemaker_client
region = session.boto_region_name

name = "DoeJohn"                              # replace with your name, as used in job names and tags
project = "SinkOrSwim"                        # replace with your challenge name, as used in the Bedrock profile names
bucket_name = "sinkorswim-doejohn"            # replace with your bucket
```

## What is still running under your name?

Everything this lesson launched carries your name in its job name or its tags, so one pass over the job lists shows what is still billing. A completed job holds nothing; an InProgress one bills until it ends or hits its cap.

```python
print("Notebook instances (bill while InService; volume bills while Stopped):")
for nb in sm.list_notebook_instances()["NotebookInstances"]:
    print(f"  {nb['NotebookInstanceName']:<40} {nb['NotebookInstanceStatus']:<12} {nb['InstanceType']}")

print("\nJobs still InProgress with your name in them:")
running = 0
for kind, lister, key, name_key in [
    ("training",   sm.list_training_jobs,              "TrainingJobSummaries",            "TrainingJobName"),
    ("tuning",     sm.list_hyper_parameter_tuning_jobs, "HyperParameterTuningJobSummaries", "HyperParameterTuningJobName"),
    ("processing", sm.list_processing_jobs,            "ProcessingJobSummaries",          "ProcessingJobName"),
]:
    for job in lister(StatusEquals="InProgress", NameContains=name, MaxResults=50)[key]:
        print(f"  {kind:<11} {job[name_key]}")
        running += 1
print("  none" if running == 0 else f"  {running} job(s) still billing: stop them from the console if you no longer need them")
```

Then storage, which bills whether or not anything runs:

```python
helpers.get_s3_bucket_size(bucket_name)
```

And the Bedrock inference profiles, which cost nothing to keep but are yours to remove (the instructors' cleanup does not touch them):

```python
bedrock = boto3.client("bedrock", region_name=region)
mine = f"{project}-{name}-"   # the Bedrock episode named every profile <project>-<name>-<label>, so this prefix is yours and nobody else's
profiles = [p for p in bedrock.list_inference_profiles(typeEquals="APPLICATION")["inferenceProfileSummaries"]
            if p["inferenceProfileName"].startswith(mine)]
for p in profiles:
    print(p["inferenceProfileName"], p["inferenceProfileArn"])
print("none" if not profiles else f"{len(profiles)} profile(s) with your prefix; the next cell deletes them")
```

Deleting them is one call per profile. The cell deletes only the profiles the cell above listed, the ones whose name starts with your project and your name, so nobody else's are touched even though the account's policy would allow it; read the list before you set the flag. The notebook instance needs the Bedrock role from the RAG episodes for this call; on the old role it fails with an access-denied error, and the Bedrock console does the same job by hand (Inference and Assessment → Cross-region inference → Application inference profiles → select → Delete).

```python
confirm_delete = False   # set to True after reading the list above

if confirm_delete:
    for p in profiles:
        bedrock.delete_inference_profile(inferenceProfileIdentifier=p["inferenceProfileArn"])
        print("deleted", p["inferenceProfileName"])
    profiles = []
else:
    print("nothing deleted; set confirm_delete = True and rerun to delete the profiles listed above")
```

Now the console steps. Rerun the cells above when you are done; every list should be empty or Stopped.

## Shutting down notebook instances

Notebook instances in SageMaker are billed per second, so it’s essential to stop or delete them when they are no longer needed. Earlier in the **Notebooks as controllers** episode, we discussed using lower-cost instance types like `ml.t3.medium` (approximately `$0.05`/hour) for controlling workflows. While this makes open notebooks less costly than larger instances, it’s still a good habit to stop or delete notebooks to avoid unnecessary spending, especially if left idle for long periods.

1. **Navigate to SageMaker in the AWS Console.**
2. In the left-hand menu, click **Notebook instances**.
3. Locate your notebook instance and select it.
4. Choose **Stop** to shut it down temporarily or **Delete** to permanently remove it.
   > **Tip:** If you plan to reuse the notebook later, stopping it is sufficient. Deleting is recommended if you are finished with the workshop.

A stopped instance stops the compute charge but keeps its storage volume, which bills at about `$0.14` per GB-month. The default 5 GB is cents; the 100 GB volume the optional GPU-notebook episode asks for is `$14` a month. If you enlarged the volume to hold model weights, delete the instance once the episode is done.

## Cleaning up S3 storage

While S3 storage is relatively inexpensive, cleaning up unused buckets and files helps keep costs minimal and your workspace organized.

1. **Navigate to the S3 Console.**
2. Locate the bucket(s) you created for this workshop.
3. Open the bucket and select any objects (files) you no longer need.
4. Click **Delete** to remove the selected objects.
5. To delete an entire bucket:
   - Empty the bucket by selecting **Empty bucket** under **Bucket actions**.
   - Delete the bucket by clicking **Delete bucket**.

> **Reminder:** Earlier in the workshop, we set up tags for S3 buckets. Use these tags to filter and identify workshop-related buckets, ensuring that only unnecessary resources are deleted.

## Monitoring and stopping active jobs

SageMaker charges for training, tuning and Processing jobs while they run, so make sure to terminate unused jobs. Every job in this lesson was launched with a runtime cap (`max_run` on training and tuning jobs, `max_runtime_in_seconds` on Processing jobs), so a forgotten job ends on its own.

1. In the SageMaker Console, go to **Training jobs**, **Hyperparameter tuning jobs**, or **Processing jobs**.
2. Identify any active jobs that you no longer need.
3. Select the jobs and click **Stop**.
   > **Tip:** Review the job logs to ensure you’ve saved the results before stopping a job.

## Billing and cost monitoring

Managing your AWS expenses is vital to staying within budget. Follow these steps to monitor and control costs:

1. **Set up billing alerts:**
   - Go to the AWS **Billing Dashboard**.
   - Navigate to **Budgets** and create a budget alert to track your spending.
2. **Review usage and costs:**
   - Use the AWS **Cost Explorer** in the Billing Dashboard to view detailed expenses by service, such as SageMaker and S3.
3. **Use tags for cost tracking:**
   - Refer to the tags you set up earlier in the workshop. Remember that tags never propagate between resources: your notebook instance, each S3 bucket, each training/tuning/Processing job, and each Bedrock application inference profile carries its own tags, and anything you launched without tags is untraceable after the fact.
   - Only the three approved keys count on the bill, `Name`, `Project` and `Purpose`, spelled exactly (the S3 episode says why); a tag you just added can take up to 24 hours to appear in Cost Explorer.
   - Bedrock application inference profiles cost nothing to keep, but delete the ones you created once a project is finished; the cells at the top of this episode do it for the ones carrying your name.

## End-of-workshop checklist

Work through this list before you sign off for the last time.

- **Notebook instances**: Stop, then **Delete** (SageMaker → Notebook instances). Includes the optional GPU notebook instance if you created one.
- **S3 bucket**: empty and delete your `challenge-lastnamefirstname` bucket (S3 → select bucket → Empty, then Delete). Download anything you want to keep first, once: egress is `$0.09`/GB. Leave the shared data bucket alone; it is the instructors'.
- **Default SDK bucket** `sagemaker-<region>-<account-id>`: created automatically the first time a job staged its code. In the shared account the instructors delete it; on your own account, empty and delete it with the rest.
- **Training, tuning and Processing jobs**: confirm none are still running (SageMaker → each job list, filter by status *InProgress*). Completed jobs hold no billable resource; their artifacts are in your bucket and go with it.
- **Bedrock application inference profiles**: delete the ones you created, with the two cells at the top of this episode (they list and delete only the profiles named with your project and your name) or from the Bedrock console.
- **GitHub personal access token**: revoke it on GitHub (Settings → Developer settings → Personal access tokens) if you made one for the optional PAT episode.

Every stop and delete in this list has a one-line CLI equivalent; the optional episode *The Same Jobs from the Terminal* collects them, for the day you need to stop something from a machine with no browser.

In the shared workshop account, the instructors sweep the account with [aws-nuke](https://github.com/ekristen/aws-nuke) after the workshop. That sweep deletes notebook instances and the learner buckets only, so if you forget those, they will still go. It does not touch Bedrock application inference profiles, so those are yours to remove.

## Best practices for resource management

Efficient resource management can save significant costs and improve your workflows. Below are some best practices:

- **Automate resource cleanup:**
   Use AWS SDK or CLI scripts to automatically shut down instances and clean up S3 buckets when not in use. [Learn more about automation with AWS CLI](https://docs.aws.amazon.com/cli/latest/userguide/cli-chap-welcome.html).
- **Schedule resource usage:**
   Schedule instance start and stop times using AWS Lambda. [Learn how to schedule tasks with AWS Lambda](https://docs.aws.amazon.com/lambda/latest/dg/tutorial-scheduled-events.html).
- **Test workflows locally first:**
   Before scaling up experiments in SageMaker, test them locally to minimize cloud usage and costs. [Learn about SageMaker local mode](https://sagemaker.readthedocs.io/en/stable/overview.html#local-mode).
- **Use cost tracking tools:**
   Explore AWS’s cost allocation tags and budget tracking features. [Learn more about cost management in AWS](https://aws.amazon.com/aws-cost-management/).

By following these practices and leveraging the additional resources provided, you can optimize your use of AWS while keeping costs under control.

::::::::::::::::::::::::::::::::::::: challenge

### Challenge: your own cleanup list

Without looking at the checklist above, list every billable resource *you* created in this workshop, and for each one write down where it is deleted (which console page, or which command). Then compare with the checklist. Which items would have kept billing if you had only stopped your notebook instance and walked away?

:::::::::::::::::::::::::::::::::::: solution

A complete list for someone who did every episode, including the optional ones:

| Resource | Created in | Bills while it exists? | Where it is deleted |
|---|---|---|---|
| S3 bucket `challenge-lastnamefirstname` and everything in it (the Titanic files and `.npz` arrays at the root, and the `docs/`, `xgboost/`, `output_nn/`, `wattbot/` and `badger-scribe/` prefixes) | S3 episode | Yes: storage per GB-month | S3 console → Empty bucket → Delete bucket |
| The SDK's default bucket `sagemaker-<region>-<account-id>` (code tarballs staged by every `fit()` and `run()`) | Training episode, automatically | Yes, a little: storage per GB-month, shared by the whole account | Instructors delete it at the end of term; on your own account, empty and delete it like any other bucket |
| Controller notebook instance (`ml.t3.medium`) | Notebooks as controllers | Yes while InService; its volume bills while Stopped | SageMaker → Notebook instances → Stop → Delete |
| Training jobs (XGBoost, PyTorch), tuning jobs, Processing jobs | Training, tuning, RAG and fine-tuning episodes | Only while running; all capped by `max_run` / `max_runtime_in_seconds` | SageMaker → job lists → Stop if InProgress; nothing to delete afterwards |
| Bedrock application inference profiles | RAG on Bedrock | No (the calls billed per token when made) | The delete cell at the top of this episode, or the Bedrock console |
| GPU notebook instance (`ml.g6e.xlarge`, 100 GB volume) | Optional GPU-notebook episode | Yes: `$2.61`/h InService, `$14`/month for the volume when Stopped | SageMaker → Notebook instances → Delete |
| GitHub personal access token | Optional PAT episode | No, but it is a credential | GitHub → Settings → Developer settings → revoke |

Stopping the controller and walking away leaves the bucket billing for storage, a stopped GPU instance's volume billing at `$0.14`/GB-month, and any Processing job that had not hit its `max_run` yet. Model artifacts in your bucket are the only thing worth keeping; download them once, then delete the bucket.

::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: keypoints


- Always stop or delete notebook instances when not in use to avoid charges.
- Stop any training, tuning or Processing job you no longer need.

- Bedrock application inference profiles are not swept by the instructors' aws-nuke run; remove them yourself.
- Regularly clean up unused S3 buckets and objects to save on storage costs.
- Monitor your expenses through the AWS Billing Dashboard and set up alerts.
- Use tags (set up earlier in the workshop) to track and monitor costs by resource.
- Following best practices for AWS resource management can significantly reduce costs and improve efficiency.

::::::::::::::::::::::::::::::::::::::::::::::::
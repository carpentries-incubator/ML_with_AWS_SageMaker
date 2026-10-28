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

Open the notebook from: `/ML_with_AWS_SageMaker/notebooks/15_Resource-management-cleanup.ipynb`. Any kernel works; **`conda_python3`** is fine. This episode is mostly console steps, but the cells below ask AWS what is still running or stored under your name, so you can check the list against reality before and after you clean up.




## Initial setup


```python
import boto3
import sagemaker
import sys

sys.path.insert(0, "/home/ec2-user/SageMaker/ML_with_AWS_SageMaker/scripts")   # the lesson's helper module, cloned in the controllers episode
import helpers

session = sagemaker.Session()
sm = session.sagemaker_client          # the low-level SageMaker client: list_*, describe_*, stop_* calls
region = session.boto_region_name      # the region the notebook runs in (us-east-2 on the workshop account)

name = "DoeJohn"                              # replace with your name, as used in job names and tags
project = "SinkOrSwim"   # the Project value used in your Bedrock profile names
bucket_name = "sinkorswim-doejohn"            # replace with your bucket
```

## What is still running under your name?

One pass over the three job lists shows every job still running in the account; yours are the ones with your notebook name in the training jobs and the `WattBot-` Processing jobs you launched. A completed job holds nothing; an InProgress one bills until it ends or hits its cap.

```python
# Every notebook instance in the account (the list is shared, so other learners' instances appear too), with its status.
# InService bills the instance's hourly rate; Stopped bills only the volume.
print("Notebook instances (bill while InService; volume bills while Stopped):")
for nb in sm.list_notebook_instances()["NotebookInstances"]:
    print(f"  {nb['NotebookInstanceName']:<40} {nb['NotebookInstanceStatus']:<12} {nb['InstanceType']}")

# The three kinds of job, each with its own list call. The call returns a dictionary; the list of jobs is under a key
# whose name differs per kind ("TrainingJobSummaries", ...), and so does the key holding the job's name, so the loop
# carries both. StatusEquals keeps only running jobs; NameContains keeps the ones with your name in the job name.
print("\nJobs still InProgress in the account:")
running = 0
for kind, lister, key, name_key in [
    ("training",   sm.list_training_jobs,              "TrainingJobSummaries",            "TrainingJobName"),
    ("tuning",     sm.list_hyper_parameter_tuning_jobs, "HyperParameterTuningJobSummaries", "HyperParameterTuningJobName"),
    ("processing", sm.list_processing_jobs,            "ProcessingJobSummaries",          "ProcessingJobName"),
]:
    for job in lister(StatusEquals="InProgress", MaxResults=50)[key]:   # every running job in the account; the Processing jobs are not named after you, so no name filter
        print(f"  {kind:<11} {job[name_key]}")
        running += 1
print("  none" if running == 0 else f"  {running} job(s) still billing: stop them from the console if you no longer need them")
```

Then storage, which bills whether or not anything runs:

```python
helpers.get_s3_bucket_size(bucket_name)   # sums the size of every object in the bucket; S3 bills per GB-month whether or not anything runs
```

And the Bedrock inference profiles, which cost nothing to keep but are yours to remove (the instructors' cleanup does not touch them):

```python
bedrock = boto3.client("bedrock", region_name=region)   # the Bedrock control-plane client: profiles are created and deleted here, not on the runtime client
mine = f"{project}-{name}-"   # the Bedrock episode named every profile <project>-<name>-<label>, so this prefix is yours and nobody else's
# list_inference_profiles returns every application profile in the account; keep the ones whose name starts with your prefix
profiles = [p for p in bedrock.list_inference_profiles(typeEquals="APPLICATION")["inferenceProfileSummaries"]
            if p["inferenceProfileName"].startswith(mine)]
for p in profiles:
    print(p["inferenceProfileName"], p["inferenceProfileArn"])   # the ARN is what the delete call below takes
print("none" if not profiles else f"{len(profiles)} profile(s) with your prefix; the next cell deletes them")
```

Deleting them is one call per profile. The cell deletes only the profiles the cell above listed, the ones whose name starts with your project and your name, so nobody else's are touched even though the account's policy would allow it; read the list before you set the flag.

```python
confirm_delete = False   # set to True after reading the list above

if confirm_delete:
    for p in profiles:   # only the profiles the cell above listed: yours, by prefix
        bedrock.delete_inference_profile(inferenceProfileIdentifier=p["inferenceProfileArn"])   # one call per profile; there is no undo
        print("deleted", p["inferenceProfileName"])
    profiles = []        # so a rerun of this cell does not try to delete them twice
else:
    print("nothing deleted; set confirm_delete = True and rerun to delete the profiles listed above")
```

Now the console steps. Rerun the cells above when you are done; every list should be empty or Stopped.

## Shutting down notebook instances

Notebook instances bill per second at an hourly rate while they run, so stop or delete them when they are no longer needed. Earlier in the **Notebooks as controllers** episode, we discussed using lower-cost instance types like `ml.t3.medium` (approximately `$0.05`/hour) for controlling workflows. A small instance left running still bills a dollar a day.

1. **Navigate to SageMaker in the AWS Console.**
2. In the left-hand menu, click **Notebook instances**.
3. Locate your notebook instance and select it.
4. Choose **Stop** to shut it down temporarily or **Delete** to permanently remove it.
   > **Tip:** If you plan to reuse the notebook later, stopping it is sufficient. Deleting is recommended if you are finished with the workshop.

A stopped instance stops the compute charge but keeps its storage volume, which bills at about `$0.14` per GB-month. The 20 GB the controllers episode set is `$2.80` a month; the 100 GB volume the optional GPU-notebook episode asks for is `$14` a month. If you enlarged the volume to hold model weights, delete the instance once the episode is done.

## Cleaning up S3 storage

S3 storage is cheap, and a bucket left behind bills every month until it is deleted.

1. **Navigate to the S3 Console.**
2. Locate the bucket(s) you created for this workshop.
3. Open the bucket and select any objects (files) you no longer need.
4. Click **Delete** to remove the selected objects.
5. To delete an entire bucket:
   - Empty the bucket by selecting **Empty bucket** under **Bucket actions**.
   - Delete the bucket by clicking **Delete bucket**.

> **Reminder:** Earlier in the workshop, we set up tags for S3 buckets. Use these tags to filter and identify workshop-related buckets, so that only workshop resources are deleted.

## Monitoring and stopping active jobs

SageMaker charges for training, tuning and Processing jobs while they run, so stop jobs you no longer need. Every job in this lesson was launched with a runtime cap (`max_run` on training and tuning jobs, `max_runtime_in_seconds` on Processing jobs), so a forgotten job ends on its own.

1. In the SageMaker Console, go to **Training jobs**, **Hyperparameter tuning jobs**, or **Processing jobs**.
2. Identify any active jobs that you no longer need.
3. Select the jobs and click **Stop**.
   > **Tip:** Review the job logs to ensure you’ve saved the results before stopping a job.

## Billing and cost monitoring

To see what you are spending:

1. **Set up billing alerts:**
   - Go to the AWS **Billing Dashboard**.
   - Navigate to **Budgets** and create a budget alert to track your spending.
2. **Review usage and costs:**
   - Use the AWS **Cost Explorer** in the Billing Dashboard to view detailed expenses by service, such as SageMaker and S3. To turn a line on the bill back into hours, the rates are on the [Instances for ML](https://carpentries-incubator.github.io/ML_with_AWS_SageMaker/instances-for-ML.html) page.
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

Every stop and delete in this list has a one-line CLI equivalent; the optional episode [The Same Jobs from the Terminal](14_CLI-equivalents.md) collects them, for stopping something from a machine with no browser.

In the shared workshop account, the instructors sweep the account with [aws-nuke](https://github.com/ekristen/aws-nuke) after the workshop. That sweep deletes notebook instances and the learner buckets only, so if you forget those, they will still go. It does not touch Bedrock application inference profiles, so those are yours to remove.

## Rules the account enforces, and how to set them on your own

Several things you could have done wrong in this workshop were refused before they cost anything: a job launched without the three tags, with a blank name, with a Project that is not on the account's list, with an instance type outside the lesson's list, a Processing job with no runtime cap, and any endpoint or EC2 instance at all. None of that is SageMaker's own behaviour. It is a set of rules the account administrator attached to the role your notebook runs as, written in the same IAM policy language that gave the role its permissions, and each one says: refuse this request if it looks like this. An AWS account with no such rules lets a tagged job and an untagged one through alike, and lets a notebook instance run until someone remembers it.

If you administer an account of your own, a lab's or a project's, the same rules are an afternoon's work, and they hold across the console, the SDK and the CLI, because all three are the same API. The lesson's *Account preparation checklist* instructor page has the policy as JSON and the one command that attaches it to a role; its *Overnight safeguards* section has the three measures that go with it. In order of what they prevent for what they cost:

| Rule | What it stops | How it is done |
|---|---|---|
| Every job carries the three cost tags, with real values | spend nobody can attribute | an IAM deny on `CreateTrainingJob` and the like when `aws:RequestTag/Name` is missing or blank, or `Project` and `Purpose` are missing or not on their lists |
| Only instance types you have priced | the `$63`-an-hour box chosen by accident | an IAM deny when `sagemaker:InstanceTypes` is not on your list |
| A runtime cap on every job, at most 48 hours | a hung job billing for a week | an IAM deny when `sagemaker:MaxRuntimeInSeconds` is missing or too large |
| No endpoints, no EC2 | the two resource kinds with no runtime cap at all | an IAM deny on `CreateEndpoint` and `ec2:RunInstances` |
| Notebook instances stop at 11 pm | the forgotten GPU notebook, `$63` a night | a ten-line scheduled Lambda function |
| A budget per person and a brake on the total | a quiet overspend | AWS Budgets on the `Name` tag, with an action that attaches a deny policy at the ceiling |

A deny applies to everyone who uses the role, including you, so your own launches need real tag values and listed instance types too. And the error a refused request gets, `AccessDeniedException ... with an explicit deny in an identity-based policy`, does not say which rule matched, so write the rules down where the people using the account will find them. A deny cannot carry a custom message either: AWS says `explicit deny in an identity-based policy` and nothing more, which is why this lesson keeps a copy of the policy next to its helper code and checks every launch against it in the notebook first, with `helpers.check_launch`, so the message you get names the rule. The [If a Launch Is Refused](../learners/refused-launch.md) page is the same list in one place, for the day a job of your own is refused.

What you can do without an administrator:

- **Automate the cleanup you do by hand.** The inventory cells at the top of this episode are the SDK calls; a script that runs them and stops what it finds is the Lambda in the table above.
- **Test on a slice first.** Every expensive run in this lesson was preceded by a small one on the same code: a few epochs, a subset of pages, a handful of questions.
- **Estimate, then compare.** Write down what a run should cost before launching it, read `BillableTimeInSeconds` afterwards, and find out why when the two disagree, as the fine-tuning episode does.
- **Tag everything, yourself, every time.** Cost allocation tags are the only way a shared bill becomes individual bills, and they never propagate.

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

Stopping the controller and walking away leaves the bucket billing for storage, a stopped GPU instance's volume billing at `$0.14`/GB-month, and any Processing job that had not hit its `max_runtime_in_seconds` yet. Model artifacts in your bucket are the only thing worth keeping; download them once, then delete the bucket.

::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: keypoints


- Always stop or delete notebook instances when not in use to avoid charges.
- Stop any training, tuning or Processing job you no longer need.

- Bedrock application inference profiles are not swept by the instructors' aws-nuke run; remove them yourself.
- The refusals you met in this workshop (untagged jobs, wrong instance types, endpoints) are IAM deny rules an administrator attached to the role, not SageMaker defaults; on an account you administer, the instructor pages have the policy to copy.
- Regularly clean up unused S3 buckets and objects to save on storage costs.
- Monitor your expenses through the AWS Billing Dashboard and set up alerts.
- Use tags (set up earlier in the workshop) to track and monitor costs by resource.

::::::::::::::::::::::::::::::::::::::::::::::::
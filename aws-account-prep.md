---
title: 'Account preparation checklist (quotas, Bedrock, tags)'
---

Everything that has to be requested or configured on the class AWS account before the workshop, sized for a class of up to **100 students** (100 for one-per-student resources, 200 for anything a student can have two of at once). GPU quota increases often take days and sometimes a support exchange, so file them three to four weeks ahead. Instance usage per episode is taken from the code in the episodes; if you change an instance type in an episode, change the quota here.

## Region

The shared workshop account runs in **US East (Ohio), `us-east-2`**. Quotas are per region, so file them there; the nuke config targets it; learners are told the region in the sign-in email and the setup page's access check. The lesson's price tables quote `us-east-1` list prices (the AWS price list of 2026-10-16); Ohio prices are the same or within a few percent, which does not change any decision in the lesson. All notebook code reads the region from the session rather than hard-coding it, so the lesson also works unchanged in any other region a self-paced learner picks.

## SageMaker service quotas

Console: **Service Quotas → AWS services → Amazon SageMaker**, in the region the class uses. Names below are the quota names as the console lists them.

| Quota | Request | Used by |
|---|---|---|
| `ml.t3.medium for notebook instance usage` | 100 | controller notebook, every episode |
| `ml.g6e.xlarge for notebook instance usage` | 0 (optional) | only the optional self-study GPU-notebook episode; not needed for class |
| `Total number of notebook instances` | 100 | one controller per student |
| `ml.m5.large for training job usage` | 200 | XGBoost episode; PyTorch part 2; tuning (2 parallel jobs each) |
| `ml.m5.xlarge for training job usage` | 200 | PyTorch part 2 runs `instance_count=2` |
| `ml.g4dn.xlarge for training job usage` | 100 | PyTorch part 2 GPU comparison |
| `ml.g5.xlarge for training job usage` | 100 | fine-tuning episode (one job per student at once) |
| `ml.g6e.xlarge for training job usage` | 100 | optional: fine-tuning Exercise 2 |
| `Number of instances across all training jobs` | 250 (already applied) | account-level ceiling; 250 covers 100 students |
| `ml.g5.xlarge for processing job usage` | 200 | Processing Jobs episode (both jobs) and fine-tuning homework (two jobs per student, possibly at once) |
| `ml.g6e.xlarge for processing job usage` | 100 (optional) | only the Processing Jobs episode's 27B variant. **Default is usually 0** and the increase takes days; the episode runs Qwen3-8B on `ml.g5.xlarge` without it |
| `Number of instances across all processing jobs` | 250 (already applied) | account-level ceiling |

Also check `ml.g5.xlarge for training warm pool usage` is left at its default of 0; the episodes do not use warm pools and a nonzero quota lets a student leave an idle GPU billing.

## Learner access

Learners sign in through IAM Identity Center with their university login. Before the workshop, add every registered user to the group that carries the workshop permission set, then email them the Identity Center console link for that account and permission set (the `.../start/#/console?account_id=...&role_name=...` deep link), together with the account name and region to expect. The link is deliberately not printed in the public lesson: it is not a credential, but a public start URL plus role name is the raw material for a convincing phishing page, and the lesson is reused outside UW. Ask learners to run the "verify your access" steps in the setup page; a missing assignment is the most common day-one problem and it cannot be fixed from the classroom in a hurry.

## Permission set guard rails

Learners reach the account through the console, the SageMaker SDK in notebooks, and, whether or not you teach it, the AWS CLI: the Identity Center portal offers "Command line or programmatic access" next to every account. All three are the same API, so the only control that holds across them is the permission set. The lesson tells learners that the account refuses endpoints, EC2, out-of-list instance types and untagged jobs; this section is what makes that true. Attach it as an inline policy on the workshop permission set alongside the SageMaker allow policy, replacing the instance-type list with the one your quota table allows.

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "NothingThatRunsUntilDeleted",
      "Effect": "Deny",
      "Action": [
        "sagemaker:CreateEndpoint",
        "sagemaker:CreateEndpointConfig",
        "sagemaker:CreateInferenceComponent",
        "ec2:RunInstances"
      ],
      "Resource": "*"
    },
    {
      "Sid": "OnlyLessonInstanceTypes",
      "Effect": "Deny",
      "Action": [
        "sagemaker:CreateTrainingJob",
        "sagemaker:CreateProcessingJob",
        "sagemaker:CreateHyperParameterTuningJob",
        "sagemaker:CreateTransformJob",
        "sagemaker:CreateNotebookInstance",
        "sagemaker:UpdateNotebookInstance"
      ],
      "Resource": "*",
      "Condition": {
        "ForAnyValue:StringNotEquals": {
          "sagemaker:InstanceTypes": [
            "ml.t3.medium",
            "ml.m5.large",
            "ml.m5.xlarge",
            "ml.g4dn.xlarge",
            "ml.g5.xlarge",
            "ml.g6.xlarge",
            "ml.g6e.xlarge",
            "ml.g6e.2xlarge"
          ]
        }
      }
    },
    {
      "Sid": "ProjectTagRequired",
      "Effect": "Deny",
      "Action": [
        "sagemaker:CreateTrainingJob",
        "sagemaker:CreateProcessingJob",
        "sagemaker:CreateHyperParameterTuningJob",
        "sagemaker:CreateNotebookInstance"
      ],
      "Resource": "*",
      "Condition": {
        "Null": {
          "aws:RequestTag/Project": "true"
        }
      }
    },
    {
      "Sid": "NameTagRequired",
      "Effect": "Deny",
      "Action": [
        "sagemaker:CreateTrainingJob",
        "sagemaker:CreateProcessingJob",
        "sagemaker:CreateHyperParameterTuningJob",
        "sagemaker:CreateNotebookInstance"
      ],
      "Resource": "*",
      "Condition": {
        "Null": {
          "aws:RequestTag/Name": "true"
        }
      }
    },
    {
      "Sid": "PurposeTagRequired",
      "Effect": "Deny",
      "Action": [
        "sagemaker:CreateTrainingJob",
        "sagemaker:CreateProcessingJob",
        "sagemaker:CreateHyperParameterTuningJob",
        "sagemaker:CreateNotebookInstance"
      ],
      "Resource": "*",
      "Condition": {
        "Null": {
          "aws:RequestTag/Purpose": "true"
        }
      }
    },
    {
      "Sid": "ProjectTagFromTheList",
      "Effect": "Deny",
      "Action": [
        "sagemaker:CreateTrainingJob",
        "sagemaker:CreateProcessingJob",
        "sagemaker:CreateHyperParameterTuningJob",
        "sagemaker:CreateNotebookInstance"
      ],
      "Resource": "*",
      "Condition": {
        "StringNotLike": {
          "aws:RequestTag/Project": [
            "SinkOrSwim",
            "WattBot2026",
            "BadgerScribe",
            "BadgerCode",
            "BrainDecoding",
            "PoisonedWell",
            "SnapshotWI"
          ]
        }
      }
    },
    {
      "Sid": "NameTagNotBlank",
      "Effect": "Deny",
      "Action": [
        "sagemaker:CreateTrainingJob",
        "sagemaker:CreateProcessingJob",
        "sagemaker:CreateHyperParameterTuningJob",
        "sagemaker:CreateNotebookInstance"
      ],
      "Resource": "*",
      "Condition": {
        "StringNotLike": {
          "aws:RequestTag/Name": "?*"
        }
      }
    },
    {
      "Sid": "PurposeTagFromTheList",
      "Effect": "Deny",
      "Action": [
        "sagemaker:CreateTrainingJob",
        "sagemaker:CreateProcessingJob",
        "sagemaker:CreateHyperParameterTuningJob",
        "sagemaker:CreateNotebookInstance"
      ],
      "Resource": "*",
      "Condition": {
        "StringNotLike": {
          "aws:RequestTag/Purpose": [
            "Storage",
            "Notebook",
            "Training",
            "Tuning",
            "Inference"
          ]
        }
      }
    },
    {
      "Sid": "RuntimeCapAtMost48Hours",
      "Effect": "Deny",
      "Action": [
        "sagemaker:CreateTrainingJob",
        "sagemaker:CreateProcessingJob"
      ],
      "Resource": "*",
      "Condition": {
        "NumericGreaterThan": {
          "sagemaker:MaxRuntimeInSeconds": "172800"
        }
      }
    },
    {
      "Sid": "ProcessingJobsNeedACap",
      "Effect": "Deny",
      "Action": [
        "sagemaker:CreateProcessingJob"
      ],
      "Resource": "*",
      "Condition": {
        "Null": {
          "sagemaker:MaxRuntimeInSeconds": "true"
        }
      }
    }
  ]
}
```

What each statement does, and what it costs you:

- **NothingThatRunsUntilDeleted** blocks the three resource kinds with no runtime cap. Nothing in the lesson creates any of them, so no episode changes. This is the statement that would have stopped the participant who spent a year's credits from a terminal.
- **OnlyLessonInstanceTypes** refuses any job or notebook whose instance type is not on the list. The list must match the quota table above exactly, including the capacity fallbacks; a type you forget here fails for every learner at once, from the console too. Keep `ml.g6e.xlarge` on the list only if you want the optional GPU-notebook episode to be possible.
- **The three tag statements** refuse a job or notebook created without each of the cost-allocation keys. **NameTagNotBlank** refuses a `Name` that is present but empty, which passes the first check and still bills as nobody's. **ProjectTagFromTheList and PurposeTagFromTheList** refuse any `Project` that is not on the list and any `Purpose` that is not one of the five words the lesson uses. The project list is the workshop's own project, `SinkOrSwim`, plus the year's challenge names; it is the one thing in the policy that changes every year, and it also decides your own test runs. The lesson's example values, `DoeJohn` and `SinkOrSwim`, are allowed on purpose, so that a learner who runs a notebook without replacing them is not stopped at the first launch; what they cost is a job on the bill under `DoeJohn`, which Cost Explorer shows and the first episodes tell learners to check.
- **Buckets are not covered.** A bucket can be created without tags, from the console or the SDK, and the policy does not try to stop it: the S3 console creates the bucket and attaches the tags in a second call, so a deny on an untagged `CreateBucket` would refuse every bucket. An untagged bucket costs cents a month and shows on the bill as unallocated, which is where you find it at cleanup. The S3 episode makes the tags mandatory in words instead.
- **RuntimeCapAtMost48Hours and ProcessingJobsNeedACap** bound what one job can bill. A training job always carries a `StoppingCondition` (the API requires one, and the SDK's default is 24 hours); a Processing job does not, and without one it runs until the script ends. With the instance list above, the most one job can now cost is 48 hours of the dearest type, `ml.g6e.2xlarge` at `$2.80`: about `$135`. Lower the number to tighten it; the credit-request form promises learners 48 hours, so change both together.

An IAM deny cannot carry a custom message; the learner sees `explicit deny in an identity-based policy` and no reason. The lesson covers that on its side: `helpers.check_launch` applies the same rules in the notebook before the launch and names the problem, and `helpers.explain_refusal` reads refused requests back from CloudTrail afterwards and does the same, which is what a learner (or you, a year on) can run instead of emailing. The second needs one more permission on the role, since SageMaker's managed policy does not read CloudTrail:

```bash
aws iam put-role-policy --role-name ml-sagemaker-use --policy-name ReadOwnCloudTrail \\
  --policy-document '{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Action":"cloudtrail:LookupEvents","Resource":"*"}]}'
```

`LookupEvents` reads the account's 90-day management-event history, which CloudTrail keeps without a trail being configured, and costs nothing. It shows every learner every other learner's refused launches too, with their tags; nothing in them is sensitive. Test every statement with the IAM policy simulator before trusting it, from CloudShell; it reports `explicitDeny` and the statement that matched, without launching anything. The blank-value and vocabulary statements went through two forms before the simulator was used, and the first form, `StringEquals` against an empty string, passed blanks in practice:

```bash
ROLE=arn:aws:iam::$(aws sts get-caller-identity --query Account --output text):role/ml-sagemaker-use
sim() {   # sim <Name> <Project> <Purpose> <instance type>
  aws iam simulate-principal-policy --policy-source-arn $ROLE --action-names sagemaker:CreateTrainingJob \
    --context-entries "ContextKeyName=aws:RequestTag/Name,ContextKeyValues=$1,ContextKeyType=string" \
                      "ContextKeyName=aws:RequestTag/Project,ContextKeyValues=$2,ContextKeyType=string" \
                      "ContextKeyName=aws:RequestTag/Purpose,ContextKeyValues=$3,ContextKeyType=string" \
                      "ContextKeyName=sagemaker:InstanceTypes,ContextKeyValues=$4,ContextKeyType=stringList" \
    --query 'EvaluationResults[0].[EvalDecision, MatchedStatements[].SourcePolicyId]' --output text
}
sim EndemannChris WattBot2026 Training ml.m5.large     # allowed
sim "" WattBot2026 Training ml.m5.large                # explicitDeny: NameTagNotBlank
sim EndemannChris MyProject Training ml.m5.large       # explicitDeny: ProjectTagFromTheList
sim EndemannChris WattBot2026 Testing ml.m5.large      # explicitDeny: PurposeTagFromTheList
sim DoeJohn SinkOrSwim Training ml.m5.large            # allowed: the lesson's example values are on the lists
sim EndemannChris WattBot2026 Training ml.m5.4xlarge   # explicitDeny: OnlyLessonInstanceTypes
``` Every estimator, processor and tuner in the lesson passes all three, and the console's notebook-instance form has the tags field the S3 and controllers episodes fill in, so nothing in the lesson is affected; a learner who deletes the `tags=` line gets an `AccessDeniedException` instead of an unattributed bill. Tuning-job child jobs are created by the SageMaker service with the tuner's tags propagated, so they pass.

Three cautions. Test the policy with one job of each kind (training, processing, tuning, notebook creation from the console) a week before class, from a learner-level sign-in, not an administrator one. Second, the role the notebook instance assumes (`ml-sagemaker-use`) is a separate identity from the permission set; what the SDK does from inside a notebook is governed by that role's policy, so attach the same denies there, or the guard rail only covers the laptop. On the UW account the policy above, all ten statements, is attached to that role as the inline policy `WorkshopGuardRails` (26 October 2026, with the 2026 challenge list), from a clone of the lesson repository in CloudShell, where the JSON above is kept as `scripts/guardrails.json`:

```bash
git clone -q https://github.com/qualiaMachine/ML_with_Amazon_SageMaker.git 2>/dev/null || git -C ML_with_Amazon_SageMaker pull -q
aws iam put-role-policy --role-name ml-sagemaker-use --policy-name WorkshopGuardRails \
  --policy-document file://ML_with_Amazon_SageMaker/scripts/guardrails.json
```

The same two commands replace it after any edit. Edit the file, not the page: `helpers.check_launch`, which every episode calls before a launch, reads its project, purpose and instance lists from that file, so the message learners see and the rule the account applies cannot disagree. It was tested from a notebook on that role: an untagged `create_training_job` is refused with `AccessDeniedException ... explicit deny`. The permission set, which governs the console and a laptop CLI, has not been given the policy; that needs the Identity Center administrator in the management account. Third, set the permission set's session duration to the length of a class day (8 hours) so a CLI login does not expire mid-episode; it is on the permission set's **General settings**.

## Overnight safeguards

The guard rails above decide what a launch may be. They do not stop a notebook instance that was left running, or a hundred allowed jobs launched in a loop. Three more measures cover those, in order of how much they prevent for how little they cost. None of them touches an ordinary run at the budgets the Marathon works with.

### 1. Stop every notebook instance nightly

A forgotten `ml.g6e.xlarge` notebook is `$63` a night; a forgotten `ml.t3.medium` is a dollar. A scheduled function that stops every running notebook instance at 11 pm removes the first case for good. Training and Processing jobs are untouched, since they are not notebook instances, and a learner who needs an instance up overnight tags it `KeepRunning=yes`. The whole thing is one Lambda function and one schedule, from an administrative CloudShell:

```bash
cat > stop_notebooks.py <<'PY'
import boto3

def handler(event, context):
    sm = boto3.client("sagemaker")
    stopped = []
    for page in sm.get_paginator("list_notebook_instances").paginate(StatusEquals="InService"):
        for nb in page["NotebookInstances"]:
            tags = {t["Key"]: t["Value"] for t in sm.list_tags(ResourceArn=nb["NotebookInstanceArn"])["Tags"]}
            if tags.get("KeepRunning", "").lower() == "yes":
                continue
            sm.stop_notebook_instance(NotebookInstanceName=nb["NotebookInstanceName"])
            stopped.append(nb["NotebookInstanceName"])
    print("stopped:", stopped)
    return stopped
PY
zip -q stop_notebooks.zip stop_notebooks.py
ACCOUNT=$(aws sts get-caller-identity --query Account --output text)

aws iam create-role --role-name nightly-notebook-stop \
  --assume-role-policy-document '{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"lambda.amazonaws.com"},"Action":"sts:AssumeRole"}]}'
aws iam put-role-policy --role-name nightly-notebook-stop --policy-name StopNotebooks \
  --policy-document '{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Action":["sagemaker:ListNotebookInstances","sagemaker:ListTags","sagemaker:StopNotebookInstance"],"Resource":"*"},{"Effect":"Allow","Action":["logs:CreateLogGroup","logs:CreateLogStream","logs:PutLogEvents"],"Resource":"*"}]}'
sleep 15   # a new role takes a few seconds to become usable
aws lambda create-function --function-name nightly-notebook-stop --runtime python3.12 --handler stop_notebooks.handler \
  --zip-file fileb://stop_notebooks.zip --role arn:aws:iam::$ACCOUNT:role/nightly-notebook-stop --timeout 120

# cron expressions here are UTC: 05:00 UTC is 11 pm Central in winter and midnight in summer
aws events put-rule --name nightly-notebook-stop --schedule-expression "cron(0 5 * * ? *)"
aws lambda add-permission --function-name nightly-notebook-stop --statement-id events --action lambda:InvokeFunction \
  --principal events.amazonaws.com --source-arn arn:aws:events:us-east-2:$ACCOUNT:rule/nightly-notebook-stop
aws events put-targets --rule nightly-notebook-stop --targets "Id=1,Arn=arn:aws:lambda:us-east-2:$ACCOUNT:function:nightly-notebook-stop"
```

To test it, run the function once by hand. It stops every running instance that is not tagged `KeepRunning=yes`, your own included, so do it when nothing is running:

```bash
aws lambda invoke --function-name nightly-notebook-stop out.json && cat out.json
```

Say in the credit-request form and in class that instances stop at 11 pm, that a running cell is interrupted when they do, and that jobs are not.

### 2. A spend alert per learner, and a brake on the class total

AWS Budgets can watch one cost-allocation tag. One budget per learner on the `Name` tag, at the per-person credit with an alert at 80%, tells you who is about to run over, a day or two late because of Cost Explorer's lag. From CloudShell, with a file `names.txt` holding one `Name` tag value per line:

```bash
LIMIT=50; EMAIL=endemann@wisc.edu
ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
while read -r NAME; do
  aws budgets create-budget --account-id $ACCOUNT \
    --budget "{\"BudgetName\":\"mlm-$NAME\",\"BudgetLimit\":{\"Amount\":\"$LIMIT\",\"Unit\":\"USD\"},\"TimeUnit\":\"MONTHLY\",\"BudgetType\":\"COST\",\"CostFilters\":{\"TagKeyValue\":[\"user:Name\$$NAME\"]}}" \
    --notifications-with-subscribers "[{\"Notification\":{\"NotificationType\":\"ACTUAL\",\"ComparisonOperator\":\"GREATER_THAN\",\"Threshold\":80},\"Subscribers\":[{\"SubscriptionType\":\"EMAIL\",\"Address\":\"$EMAIL\"}]}]"
done < names.txt
```

Budgets are free for the first two and about `$0.60` a month each after that. The alert is information, not a block: AWS cannot stop one learner's jobs on a shared role. What it can do is a **budget action** on the class total: in the Budgets console, a budget for the account's SageMaker and Bedrock spend at the class ceiling, with an action that attaches a deny-everything policy to `ml-sagemaker-use` when the ceiling is crossed. That stops every launch for every learner at once, which is the right outcome at that point. Set it up once in the console (Budgets, Create budget, add an action of type IAM policy, target the role) and detach the policy by hand when the term is sorted out.

### 3. Lower the GPU quotas after class

The quotas in the table at the top are sized for a hundred learners launching at once; a `ml.g5.xlarge` training quota of 100 lets one learner launch a hundred jobs at `$141` an hour. Once the workshop day is over, ask Service Quotas to lower the per-type job quotas to what the Marathon needs, 10 to 20 of each, and leave the account-wide instance ceilings. Decreases apply at once. The credit-request form's "at most three instances" is a rule; the quota is the enforcement, shared across the class.

## Bedrock

Console: **Service Quotas → Amazon Bedrock**, and **Bedrock → Model access**.

- Enable model access for the two models the Bedrock episode uses: `deepseek.v3-v1:0` (generation) and `amazon.titan-embed-text-v2:0` (embeddings).
- Check the **requests per minute** and **tokens per minute** quotas for both (Service Quotas → Amazon Bedrock, search the model name). The class load, if fifty learners run the episode in the same quarter hour: the generation step is about 120 DeepSeek calls per learner (the class sample of 60 questions, two calls each) at ~3,000 input tokens each, so roughly 6,000 requests and 18 million input tokens per minute at the peak; the embedding step is about 9,000 Titan calls and 3 million tokens per learner, ten minutes for one learner alone at the 300,000 tokens-per-minute quota, so in class everyone pulls the staged copy of the matrix instead (`use_instructor_embeddings = True`, the default; see the *Shared data bucket* page). Ask for what the console lets you (many Bedrock quotas adjust in place, some need a support case); the notebook's client retries with backoff, so a quota below the peak slows the run rather than failing it. The cross-region profile `us.deepseek.v3-v1:0` usually carries a higher quota than the in-region ID and is a one-line swap in the episode. Splitting the room into a RAG track and a fine-tuning track halves the peak as well.
- **The learner execution role needs Bedrock permissions**, and the default SageMaker role has none: the Bedrock episode's first Bedrock call fails with `AccessDeniedException ... bedrock:ListInferenceProfiles` until they are added. On the UW account the policy below is attached as the inline policy `BedrockAccess` to `ml-sagemaker-use`, the role the controller-notebook episode tells learners to pick, and to `ml-sagemaker-bedrock-use`, so either works; it went onto the default role after the console's role dropdown kept saving the default instead of the chosen one, which cost a class session. Elsewhere, attach the policy below as an inline policy on whatever role the notebooks run with, with `<account-id>` filled in; `aws iam put-role-policy --role-name <role> --policy-name BedrockAccess --policy-document file://bedrock-policy.json` does it, and a policy change reaches running instances within a minute. `InvokeModel` must be allowed on **both** the application inference profile ARN and the underlying foundation-model ARN, or calls through the profile are denied; `DeleteInferenceProfile` is for the cleanup episode.

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "BedrockCallModels",
      "Effect": "Allow",
      "Action": ["bedrock:InvokeModel", "bedrock:InvokeModelWithResponseStream", "bedrock:Converse", "bedrock:ConverseStream"],
      "Resource": [
        "arn:aws:bedrock:*::foundation-model/*",
        "arn:aws:bedrock:*:<account-id>:application-inference-profile/*",
        "arn:aws:bedrock:*:<account-id>:inference-profile/*"
      ]
    },
    {
      "Sid": "BedrockTaggedProfiles",
      "Effect": "Allow",
      "Action": ["bedrock:CreateInferenceProfile", "bedrock:GetInferenceProfile", "bedrock:ListInferenceProfiles",
                 "bedrock:DeleteInferenceProfile", "bedrock:TagResource", "bedrock:ListTagsForResource"],
      "Resource": [
        "arn:aws:bedrock:*:<account-id>:application-inference-profile/*",
        "arn:aws:bedrock:*:<account-id>:inference-profile/*"
      ]
    },
    {
      "Sid": "BedrockCatalog",
      "Effect": "Allow",
      "Action": ["bedrock:GetFoundationModel", "bedrock:ListFoundationModels", "bedrock:ListInferenceProfiles"],
      "Resource": "*"
    }
  ]
}
```

## Cost allocation tags

- Confirm with the Public Cloud team that `Name`, `Project`, `Purpose` are active cost allocation keys on the account ([KB 72454](https://kb.wisc.edu/72454)). The lesson uses exactly those three everywhere.
- Set a Budgets alert on the account at the planned class total, and a Cost Explorer saved report filtered on `Project`.

## Data and files

- One shared read-only bucket holds the WattBot inputs (metadata, questions, PDF corpus) and the Badger Scribe data plus your full fine-tune outputs. Staging commands and the bucket policy: *Shared data bucket: staging before each term*. Learners create one bucket each in the S3 episode and write all outputs there.

## Capacity fallbacks to have on a slide

- Processing Jobs episode: if `ml.g6e.xlarge` capacity is short, `ml.g6e.2xlarge` (`$2.80`/h, same GPU) is the only same-memory alternative; a 24 GB instance cannot hold the 27B model.
- Fine-tuning episode: `ml.g6.xlarge` (`$1.13`/h, L4, 24 GB) runs the mini job at about the same speed as `ml.g5.xlarge`.

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
            "ml.m5.large", "ml.m5.xlarge",
            "ml.g4dn.xlarge", "ml.g5.xlarge", "ml.g6.xlarge", "ml.g6e.xlarge", "ml.g6e.2xlarge"
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
      "Condition": { "Null": { "aws:RequestTag/Project": "true" } }
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
      "Condition": { "Null": { "aws:RequestTag/Name": "true" } }
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
      "Condition": { "Null": { "aws:RequestTag/Purpose": "true" } }
    }
  ]
}
```

What each statement does, and what it costs you:

- **NothingThatRunsUntilDeleted** blocks the three resource kinds with no runtime cap. Nothing in the lesson creates any of them, so no episode changes. This is the statement that would have stopped the participant who spent a year's credits from a terminal.
- **OnlyLessonInstanceTypes** refuses any job or notebook whose instance type is not on the list. The list must match the quota table above exactly, including the capacity fallbacks; a type you forget here fails for every learner at once, from the console too. Keep `ml.g6e.xlarge` on the list only if you want the optional GPU-notebook episode to be possible.
- **The three tag statements** refuse a job or notebook created without each of the cost-allocation keys. Every estimator, processor and tuner in the lesson passes all three, and the console's notebook-instance form has the tags field the S3 and controllers episodes fill in, so nothing in the lesson is affected; a learner who deletes the `tags=` line gets an `AccessDeniedException` instead of an unattributed bill. Tuning-job child jobs are created by the SageMaker service with the tuner's tags propagated, so they pass.

Three cautions. Test the policy with one job of each kind (training, processing, tuning, notebook creation from the console) a week before class, from a learner-level sign-in, not an administrator one. Second, the role the notebook instance assumes (`ml-sagemaker-bedrock-use`) is a separate identity from the permission set; what the SDK does from inside a notebook is governed by that role's policy, so attach the same denies there, or the guard rail only covers the laptop. Third, set the permission set's session duration to the length of a class day (8 hours) so a CLI login does not expire mid-episode; it is on the permission set's **General settings**.

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

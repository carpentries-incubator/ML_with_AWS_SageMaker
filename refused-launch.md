---
title: "If a Launch Is Refused"
---

Every launch from a notebook instance on the workshop account is checked against a short list of rules before it costs anything. (The console and a laptop run under your sign-in's permission set, which does not carry the rules yet.) When a job breaks one, AWS refuses it with an `AccessDeniedException` whose message ends in `with an explicit deny in an identity-based policy`, and says nothing else: an IAM refusal cannot carry a custom message, so the error names neither the rule nor the fix. This page is the fix. It applies to any job you launch from this account, during the lesson or in your own notebooks afterwards.

## The rules, and what to change

| The refusal means | What to change |
|---|---|
| A tag is missing | Every job needs the three tags `Name`, `Project` and `Purpose`, passed as `tags=` on the estimator, processor or tuner, with those exact keys. |
| `Name` is blank | Your last and first name run together, e.g. `DoeJohn`. |
| `Project` is not on the list | One of `SinkOrSwim`, `WattBot2026`, `BadgerScribe`, `BadgerCode`, `BrainDecoding`, `PoisonedWell`, `SnapshotWI`, spelled exactly, capitals included. |
| `Purpose` is not on the list | One of `Storage`, `Notebook`, `Training`, `Tuning`, `Inference`. |
| The instance type is not one the lesson uses | `ml.t3.medium`, `ml.m5.large`, `ml.m5.xlarge`, `ml.g4dn.xlarge`, `ml.g5.xlarge`, `ml.g6.xlarge`, `ml.g6e.xlarge`, `ml.g6e.2xlarge`. Ask the instructor if your work needs another. |
| The runtime cap is over 48 hours | `max_run` (training and tuning jobs) or `max_runtime_in_seconds` (Processing jobs) of at most `172800`. Set it a little above your estimate. |
| A Processing job has no cap | Processing jobs do not get a default cap; pass `max_runtime_in_seconds=`. |
| It is an endpoint or an EC2 instance | Nothing that runs until deleted can be created on this account. Use a job, or Bedrock. |

The message is the same for every row, so check them in order. A launch that passes every row and is still refused is usually the `tags=` argument left off the estimator.

## Checking from the notebook

The lesson's helper module applies the same rules before the request is sent, and its message names the problem. It works in any notebook on your instance, since the lesson repository is cloned there:

```python
import sys
sys.path.insert(0, "/home/ec2-user/SageMaker/ML_with_AWS_SageMaker/scripts")
import helpers

job_tags = [
    {"Key": "Name", "Value": "DoeJohn"},
    {"Key": "Project", "Value": "WattBot2026"},
    {"Key": "Purpose", "Value": "Training"},
]
helpers.check_launch(job_tags, instance_type="ml.g5.xlarge", max_run=2 * 60 * 60)   # raises ValueError naming each problem; returns the tags when all is well
```

The fine-tuning episode uses `helpers.check_finetune_launch`, which runs the same check and then this lesson's own rules for a fine-tune: LoRA rather than a full fine-tune, a model of at most 7B, a cap between the estimate and four times it, checkpoints on any run that may pass an hour, a fixed job name, and sane values for the passes and the pixel budget.

A failing check looks like this:

```output
ValueError: this launch would be refused by the account:
  - Project 'MLMarathon' is not on the account's list: use one of SinkOrSwim, WattBot2026, BadgerScribe, BadgerCode, BrainDecoding, PoisonedWell, SnapshotWI, spelled exactly
  - Purpose 'Testing' is not on the list: use one of Storage, Notebook, Training, Tuning, Inference
```

The helper reads its lists from `scripts/guardrails.json`, a copy of the policy on the account, so a `git pull` in the lesson folder keeps it current when the list changes. Run it in a cell before the launch cell. Two companions:

```python
helpers.rules()             # prints the current rules
helpers.explain_refusal()   # after a refusal: reads the refused requests of the last hour from CloudTrail and names the rule each one broke
```

`explain_refusal` works after the fact, on launches made from any code, because AWS logs every API call on the account in CloudTrail, including the refused ones, with the request exactly as it was sent: tags, instance type and cap. The log is 5 to 15 minutes behind, so a refusal made a moment ago may not be there yet. The output lists each refused launch with its job name, so pick out yours by the job name. The workshop role has the CloudTrail permission this needs; a role on another account may not, and the helper says so if not.

## Why the rules exist

They are the account administrator's rules, written as an IAM policy on the role every notebook runs as; the cleanup episode lists the rules, and the *Account preparation checklist* instructor page has the policy to copy to an account of your own. The three tags are how the bill is split by person and project; the instance list and the cap bound what any one job can cost; the endpoint rule keeps anything from running unnoticed for a month.

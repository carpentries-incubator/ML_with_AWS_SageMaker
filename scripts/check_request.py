"""Check a job request, written as JSON for the AWS CLI, before it is sent.

    python check_request.py <request>.json [--max-cost DOLLARS]

Takes the request of `aws sagemaker create-training-job`, `create-processing-job`
or `create-hyper-parameter-tuning-job --cli-input-json`. Applies the same
guardrails.json the account enforces (next to this file) and prints what would
be refused and why, as helpers.check_launch did in the notebooks. Then prints the
most the request can bill: every instance it asks for, running to its runtime cap,
at the instance's hourly price, and for a tuning job every child job it allows.
With --max-cost, exits 1 when that worst case is above the amount, so a shell
script can stop before the create call.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import helpers  # noqa: E402


def worst_case(req):
    """(instances, cap_seconds, instance_type, jobs) for the three request shapes."""
    if "HyperParameterTuningJobConfig" in req:
        defn = req["TrainingJobDefinition"]
        limits = req["HyperParameterTuningJobConfig"].get("ResourceLimits", {})
        res, cap = defn.get("ResourceConfig", {}), (defn.get("StoppingCondition") or {}).get("MaxRuntimeInSeconds")
        jobs = limits.get("MaxNumberOfTrainingJobs", 1)
    elif "ProcessingJobName" in req:
        res, cap, jobs = (req.get("ProcessingResources") or {}).get("ClusterConfig", {}), (req.get("StoppingCondition") or {}).get("MaxRuntimeInSeconds"), 1
    else:
        res, cap, jobs = req.get("ResourceConfig", {}), (req.get("StoppingCondition") or {}).get("MaxRuntimeInSeconds"), 1
    return res.get("InstanceCount", 1), cap, res.get("InstanceType"), jobs


def main(path, max_cost=None):
    with open(path) as f:
        req = json.load(f)
    name = os.path.basename(path)
    instances, cap, instance_type, jobs = worst_case(req)

    # --- the account's rules, as the notebooks checked them before every launch ---
    if cap is None:
        print(f"{name}: no runtime cap. Add StoppingCondition.MaxRuntimeInSeconds; without it a hung job bills until someone notices")
        return 1
    try:
        helpers.check_launch(req.get("Tags", []), instance_type, cap)
    except ValueError as e:
        print(e)
        return 1
    print(f"{name}: tags, instance type and cap pass the account's rules")

    # --- the most this request can bill ---
    price = helpers.instance_price(instance_type)
    if price is None:
        print(f"no price on file for {instance_type}; look it up on the Instances for ML page")
        return 0
    hours = cap / 3600
    worst = instances * hours * price * jobs
    per_job = f"{instances} x {hours:.2f} h x ${price}/h" + (f" x {jobs} jobs" if jobs > 1 else "")
    print(f"worst case if every job runs to its cap: {per_job} = ${worst:,.2f}")
    if max_cost is not None and worst > max_cost:
        print(f"that is above your limit of ${max_cost:,.2f}: lower MaxRuntimeInSeconds, the instance count or the job count before sending it")
        return 1
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("request", help="the JSON file for --cli-input-json")
    ap.add_argument("--max-cost", type=float, default=None, help="exit 1 if the worst case is above this many dollars")
    args = ap.parse_args()
    sys.exit(main(args.request, args.max_cost))

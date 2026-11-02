"""Read what a finished fine-tuning job sent back and print the comparison.

    python read_results.py adapter.tar.gz

Unpacks the archive the job returned, prints the summary, the per-page CER of
the base model and the fine-tuned model on the same held-out pages, the page
that improved most, and saves the loss and holdout-CER curves as curves.png.
Needs pandas and matplotlib: run it on your own machine, or on the access point
inside the container with
    apptainer exec /staging/<n>/<netid>/churro.sif python read_results.py adapter.tar.gz
"""
import json
import os
import sys
import tarfile

import pandas as pd

archive = sys.argv[1] if len(sys.argv) > 1 else "adapter.tar.gz"
with tarfile.open(archive) as tar:
    tar.extractall("results", filter="data")
out = os.path.join("results", "model")
print("files:", sorted(os.listdir(out)))

summary = json.load(open(os.path.join(out, "train_summary.json")))
print("\nsummary:")
print(json.dumps(summary, indent=2))

before = pd.read_csv(os.path.join(out, "eval_before.csv"), keep_default_na=False)
after = pd.read_csv(os.path.join(out, "eval_after.csv"), keep_default_na=False)
comparison = before[["page_id", "cer"]].merge(after[["page_id", "cer"]], on="page_id", suffixes=("_base", "_lora"))
comparison["gain"] = comparison.cer_base - comparison.cer_lora        # positive: the fine-tune helped on that page
print("\nper-page character error rate, lower is better:")
print(comparison.sort_values("gain", ascending=False).round(3).to_string(index=False))
print(f"\nmean CER  base: {comparison.cer_base.mean():.3f}   fine-tuned: {comparison.cer_lora.mean():.3f}")

best = comparison.sort_values("gain", ascending=False).iloc[0].page_id
b, a = before.set_index("page_id"), after.set_index("page_id")
print(f"\npage that improved most: {best}")
print("REFERENCE\n", b.loc[best, "text_ref"][:400])
print("\nBASE\n", b.loc[best, "text_pred"][:400])
print("\nFINE-TUNED\n", a.loc[best, "text_pred"][:400])

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    log = json.load(open(os.path.join(out, "train_log.json")))
    loss = pd.DataFrame(log["loss"])
    curve = pd.DataFrame(log["holdout_cer_by_epoch"])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.8))
    ax1.plot(loss.epoch, loss.loss, marker=".")
    ax1.set(xlabel="epoch", ylabel="training loss", title="Training loss")
    ax2.plot(curve.epoch, curve.macro_cer, marker="o")
    ax2.set(xlabel="epoch (0 = base model)", ylabel="holdout CER", title=f"Holdout CER on {log['curve_pages']} pages")
    ax2.set_xticks(curve.epoch)
    plt.tight_layout()
    plt.savefig("curves.png", dpi=120)
    print("\nholdout CER by epoch:")
    print(curve.to_string(index=False))
    print("curves saved to curves.png")
except ImportError:
    print("\nmatplotlib is not installed here; the curves are in results/model/train_log.json")

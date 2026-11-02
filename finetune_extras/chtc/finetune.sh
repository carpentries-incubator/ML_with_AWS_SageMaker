#!/bin/bash
# The job's executable. Runs inside the container on the GPU machine, in a scratch
# directory that holds only what transfer_input_files brought. It unpacks the model
# and the data, runs the same training script the lesson's SageMaker job ran, and
# packs the adapter so HTCondor can carry it back. Everything after the script
# name on the submit file's arguments line is passed through to the script.
#
# The training script runs under a two-hour timeout. When the timeout fires, this
# wrapper exits with code 85, which tells HTCondor to keep output/ckpt (the submit
# file's transfer_checkpoint_files) and put the job back in the queue. The next run
# starts here again, finds the last epoch's checkpoint in output/ckpt and resumes.
# That is CHTC's time-based checkpointing, and it is what lets the job run on
# backfill GPUs, where it may be interrupted.
set -euo pipefail

echo "host: $(hostname)  gpu: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "started: $(date)"

tar xf churro-3B.tar && rm churro-3B.tar            # the base weights, 7.5 GB, as a local folder the script loads from
tar xzf badger-scribe-data.tar.gz && rm badger-scribe-data.tar.gz   # train.csv, metadata.csv, images/
mkdir -p output/model output/data output/ckpt    # output/ckpt already holds a checkpoint on a resumed run
ls output/ckpt

set +e                                              # so a timeout does not end the wrapper before it can exit 85
timeout 2h python finetune_churro_lora.py \
    --model_id churro-3B \
    --train_dir data \
    --model_dir output/model \
    --output_data_dir output/data \
    --checkpoint_dir output/ckpt \
    "$@"
status=$?
set -e
if [ $status -eq 124 ]; then                        # timeout's own exit code: time ran out, the script did not fail
    echo "two hours up at $(date); exiting 85 so HTCondor keeps output/ckpt and requeues the job"
    exit 85
fi
[ $status -eq 0 ] || exit $status                   # the script failed: HTCondor reports it and keeps nothing

# The adapter and its processor files, holdout list, scores and curves, as one archive
tar czf adapter.tar.gz -C output model
ls -lh adapter.tar.gz
echo "finished: $(date)"

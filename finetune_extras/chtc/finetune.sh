#!/bin/bash
# The job's executable. Runs inside the container on the GPU machine, in a scratch
# directory that holds only what transfer_input_files brought. It unpacks the model
# and the data, runs the same training script the lesson's SageMaker job ran, and
# packs the adapter so HTCondor can carry it back. Everything after the script
# name on the submit file's arguments line is passed through to the script.
set -euo pipefail

echo "host: $(hostname)  gpu: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "started: $(date)"

tar xf churro-3B.tar && rm churro-3B.tar            # the base weights, 7.5 GB, as a local folder the script loads from
tar xzf badger-scribe-data.tar.gz && rm badger-scribe-data.tar.gz   # train.csv, metadata.csv, images/
mkdir -p output/model output/data output/ckpt

python finetune_churro_lora.py \
    --model_id churro-3B \
    --train_dir data \
    --model_dir output/model \
    --output_data_dir output/data \
    --checkpoint_dir output/ckpt \
    "$@"

# The adapter and its processor files, holdout list, scores and curves, as one archive
tar czf adapter.tar.gz -C output model
ls -lh adapter.tar.gz
echo "finished: $(date)"

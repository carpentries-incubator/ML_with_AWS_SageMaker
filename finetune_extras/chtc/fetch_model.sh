#!/bin/bash
# Runs inside the container on an execute node: download the model's files from
# Hugging Face into a folder, then tar the folder so HTCondor moves one file.
set -euo pipefail

export HF_HOME="$PWD/hf_cache"                      # keep the download cache in the job's own directory
huggingface-cli download stanford-oval/churro-3B --local-dir churro-3B
rm -rf hf_cache
du -sh churro-3B
tar cf churro-3B.tar churro-3B                      # no compression: safetensors do not compress, and the tar is faster to make and to unpack
ls -lh churro-3B.tar

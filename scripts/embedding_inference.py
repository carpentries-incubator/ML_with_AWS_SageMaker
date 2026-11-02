"""Embed a JSONL file of text chunks with a Hugging Face embedding model. Runs
as a SageMaker Processing job (the Processing Jobs episode's first job).

Input:  <input_dir>/<input_filename>, one JSON object per line with a text field
Output: <output_dir>/embeddings.npy, one row per chunk, in the same order,
        unit-normalized so a dot product is a cosine similarity

The job's ProcessingInput copies the chunk file to input_dir before this runs,
and its ProcessingOutput copies output_dir to S3 when it exits. Nothing in the
script is WattBot-specific: any JSONL with a text field works.
"""

import argparse
import json
import os
import numpy as np
from hf_embedder import HFEmbedder   # sits next to this script in the job's source_dir


def main():
    """Parse the flags, read the chunks, embed them on whatever device is available, save the matrix."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, required=True)
    parser.add_argument("--input_filename", type=str, default="wattbot_chunks.jsonl")
    parser.add_argument("--text_key", type=str, default="text")
    parser.add_argument("--input_dir", type=str, default="/opt/ml/processing/input")
    parser.add_argument("--output_dir", type=str, default="/opt/ml/processing/output")
    args = parser.parse_args()

    infile = os.path.join(args.input_dir, args.input_filename)
    if not os.path.exists(infile):
        raise FileNotFoundError(f"Expected input file at {infile}")

    texts = []
    with open(infile, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            texts.append(obj[args.text_key])

    print(f"Loaded {len(texts)} chunks")

    model = HFEmbedder(args.model_id)   # downloads from Hugging Face the first time; uses the GPU if the job has one
    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        show_progress_bar=True,
        normalize_embeddings=True,   # unit length, so retrieval can use a plain dot product as cosine similarity
    )

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "embeddings.npy")
    np.save(out_path, embeddings)
    print(f"Wrote embeddings to {out_path}")


if __name__ == "__main__":
    main()

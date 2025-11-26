import argparse
import json
import os
import numpy as np
from sentence_transformers import SentenceTransformer


def main():
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

    model = SentenceTransformer(args.model_id)
    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "embeddings.npy")
    np.save(out_path, embeddings)
    print(f"Wrote embeddings to {out_path}")


if __name__ == "__main__":
    main()

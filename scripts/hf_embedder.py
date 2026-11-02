"""A minimal sentence embedder on plain `transformers`, for the Processing jobs.

The SageMaker Hugging Face container ships `transformers` and PyTorch but not
`sentence-transformers`, and installing the latter drags in whichever
`transformers` release it currently wants, which has broken the container's
other packages before. For a BERT-style embedding model such as
`thenlper/gte-large` the sentence-transformers recipe is just mean pooling
over the last hidden state followed by L2 normalization, so we do that here
and depend on nothing the container does not already have.

`HFEmbedder.encode` keeps sentence-transformers' signature for the arguments
the lesson uses, so `wattbot_rag.embed_query_with` works unchanged.
"""

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer


class HFEmbedder:
    def __init__(self, model_id: str, max_length: int = 512):
        """Load the tokenizer and model once; use the GPU if the job has one."""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id).to(self.device).eval()
        self.max_length = max_length

    @torch.no_grad()
    def encode(self, texts, batch_size: int = 32, normalize_embeddings: bool = True,
               convert_to_numpy: bool = True, show_progress_bar: bool = False):
        """Embed a list of strings: mean-pool the last hidden state over real tokens, then unit-normalize."""
        out = []
        for i in range(0, len(texts), batch_size):
            batch = self.tokenizer(texts[i:i + batch_size], padding=True, truncation=True,
                                   max_length=self.max_length, return_tensors="pt").to(self.device)
            hidden = self.model(**batch).last_hidden_state                    # (batch, tokens, dim)
            mask = batch["attention_mask"].unsqueeze(-1).to(hidden.dtype)     # 1 for real tokens, 0 for padding
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)        # mean over real tokens
            if normalize_embeddings:
                pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)   # unit length: dot product == cosine
            out.append(pooled.float().cpu())
            if show_progress_bar:
                print(f"embedded {min(i + batch_size, len(texts))} / {len(texts)}", flush=True)
        result = torch.cat(out)
        return result.numpy() if convert_to_numpy else result

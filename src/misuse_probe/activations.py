"""Extract per-layer hidden states from a Hugging Face causal LM, with an on-disk cache."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


def cache_key(texts: list[str], model_name: str, pooling: str, max_length: int) -> str:
    h = hashlib.sha256()
    h.update(json.dumps([model_name, pooling, max_length]).encode())
    for t in texts:
        h.update(t.encode("utf-8", "replace"))
        h.update(b"\x00")
    return h.hexdigest()[:16]


def extract_activations(
    texts: list[str],
    model_name: str,
    pooling: str = "last",
    batch_size: int = 16,
    max_length: int = 256,
    cache_dir: str | Path | None = "cache",
) -> np.ndarray:
    """Return an array of shape (n_texts, n_layers + 1, hidden_dim).

    Index 0 is the embedding layer. With pooling="last", each prompt is wrapped in the
    model's chat template with the assistant turn opened, and we read the final token:
    the point where the model is about to decide how to respond.
    """
    if pooling not in {"last", "mean"}:
        raise ValueError("pooling must be 'last' or 'mean'")

    path = None
    if cache_dir is not None:
        path = Path(cache_dir) / f"{cache_key(texts, model_name, pooling, max_length)}.npy"
        if path.exists():
            return np.load(path)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(model_name)
    tok.padding_side = "right"
    tok.truncation_side = "left"  # keep the end of the prompt (and the assistant prefix)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    if tok.chat_template:
        formatted = [
            tok.apply_chat_template([{"role": "user", "content": t}], tokenize=False, add_generation_prompt=True)
            for t in texts
        ]
        add_special = False  # the template already adds them
    else:
        formatted, add_special = list(texts), True

    dtype = torch.float16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype).to(device).eval()

    chunks = []
    with torch.no_grad():
        for i in range(0, len(formatted), batch_size):
            enc = tok(
                formatted[i : i + batch_size],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
                add_special_tokens=add_special,
            ).to(device)
            out = model(**enc, output_hidden_states=True)
            hs = torch.stack(out.hidden_states, dim=1)  # (B, L+1, T, D)
            mask = enc["attention_mask"]
            if pooling == "last":
                last = mask.sum(dim=1) - 1
                pooled = hs[torch.arange(hs.shape[0], device=device), :, last, :]  # (B, L+1, D)
            else:
                m = mask[:, None, :, None].to(hs.dtype)
                pooled = (hs * m).sum(dim=2) / m.sum(dim=2)
            chunks.append(pooled.float().cpu().numpy())

    acts = np.concatenate(chunks, axis=0)
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, acts)
    return acts

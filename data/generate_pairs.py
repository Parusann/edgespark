"""Generate the distillation corpus: prompt -> response pairs from the verifier.

DeepSpec-style: the verifier (Qwen3-4B, non-thinking mode) writes the responses,
so the drafter learns to imitate the exact model it will later propose for. Small
by design, tens of thousands of short sequences, not a 38 TB reproduction (spec
sections 4, 10). Streamed to JSONL so the corpus never has to sit in 32 GB of RAM.

torch/transformers imported lazily; run on the target machine.

    python data/generate_pairs.py --n 20000 --out data/datasets/pairs.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def generate(model_name: str, prompts, out_path: Path, max_new_tokens: int = 256,
             temperature: float = 0.7) -> int:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="cuda"
    ).eval()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with out_path.open("w", encoding="utf-8") as fh:
        for prompt in prompts:
            messages = [{"role": "user", "content": prompt}]
            # Non-thinking mode: no chain-of-thought scaffolding in the target.
            text = tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=False,
            )
            ids = tok(text, return_tensors="pt").input_ids.cuda()
            with torch.no_grad():
                gen = model.generate(
                    ids, max_new_tokens=max_new_tokens, do_sample=temperature > 0,
                    temperature=max(temperature, 1e-5), top_p=0.9,
                )
            response = tok.decode(gen[0, ids.shape[1]:], skip_special_tokens=True)
            fh.write(json.dumps({"prompt": prompt, "response": response}) + "\n")
            written += 1
    return written


def _seed_prompts(n: int):
    """A cheap prompt generator: cycle the built-in sets with index suffixes.

    Replace with a real blended instruction + code-instruction source for a
    production corpus (spec section 10); this keeps the script self-contained.
    """
    from data.prompts import _CHAT, _CODE

    base = _CODE * 3 + _CHAT  # code-heavy
    i = 0
    while i < n:
        p = base[i % len(base)]
        yield f"{p} (variant {i // len(base)})" if i >= len(base) else p
        i += 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--out", default="data/datasets/pairs.jsonl")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    args = ap.parse_args()
    n = generate(args.model, _seed_prompts(args.n), Path(args.out), args.max_new_tokens)
    print(f"wrote {n} pairs to {args.out}")


if __name__ == "__main__":
    main()

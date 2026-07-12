"""Build a small, streamed selected-layer hidden-state cache (spec section 10).

This is the RAM/disk pressure point of the whole project. The rule (section 10,
section 17): **never load the whole cache into 32 GB of RAM**. So the cache is
written as fixed-size shards to SSD and read back with a streaming loader that
holds one shard at a time.

Only the layers in ``target_layer_ids`` are stored, the drafter's fusion
sources, which is what keeps the footprint in the 50-200 GB range instead of the
multi-TB full-hidden-state cache. If GPU headroom allows, skip this entirely and
compute hidden states on the fly during training (``train/distill.py --on-the-fly``).

    python data/build_cache.py --pairs data/datasets/pairs.jsonl --out data/cache
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def build(model_name, pairs_path, out_dir, target_layer_ids, shard_size=512, max_len=512) -> int:
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="cuda", output_hidden_states=True
    ).eval()

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    shard, shard_idx, n = [], 0, 0

    def flush():
        nonlocal shard, shard_idx
        if not shard:
            return
        path = out_dir / f"shard_{shard_idx:05d}.npz"
        # float16 hidden keeps the cache small (the drafter upcasts); the int32
        # token ids ride alongside so the collator can build prefix/target tokens.
        payload = {}
        for i, (h, ids_i) in enumerate(shard):
            payload[f"hidden_{i}"] = h
            payload[f"ids_{i}"] = ids_i
        np.savez_compressed(path, **payload)
        manifest.append({"shard": path.name, "count": len(shard)})
        shard_idx += 1
        shard = []

    with Path(pairs_path).open(encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line)
            text = rec["prompt"] + "\n" + rec["response"]
            ids = tok(text, return_tensors="pt", truncation=True, max_length=max_len).input_ids.cuda()
            with torch.no_grad():
                out = model(ids, output_hidden_states=True)
            # Stack only the selected layers: [num_layers_kept, seq, hidden].
            sel = np.stack([
                out.hidden_states[lid][0].to(torch.float16).cpu().numpy()
                for lid in target_layer_ids
            ])
            ids_np = ids[0].to(torch.int32).cpu().numpy()  # [seq] verifier tokens
            shard.append((sel, ids_np))
            n += 1
            if len(shard) >= shard_size:
                flush()
    flush()
    (out_dir / "manifest.json").write_text(
        json.dumps({"layers": list(target_layer_ids), "shards": manifest}, indent=2),
        encoding="utf-8",
    )
    return n


def stream_cache(cache_dir):
    """Yield ``(hidden, ids)`` per cached item, one shard at a time (never all in RAM).

    ``hidden`` is ``[num_layers_kept, seq, Hv]`` float16 selected-layer states and
    ``ids`` is the ``[seq]`` int token sequence written beside it.
    """
    import numpy as np

    cache_dir = Path(cache_dir)
    manifest = json.loads((cache_dir / "manifest.json").read_text(encoding="utf-8"))
    for entry in manifest["shards"]:
        with np.load(cache_dir / entry["shard"]) as data:
            i = 0
            while f"hidden_{i}" in data.files:
                yield data[f"hidden_{i}"], data[f"ids_{i}"]
                i += 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--pairs", default="data/datasets/pairs.jsonl")
    ap.add_argument("--out", default="data/cache")
    ap.add_argument("--layers", type=int, nargs="+", default=[1, 17, 33])
    ap.add_argument("--shard-size", type=int, default=512)
    args = ap.parse_args()
    n = build(args.model, args.pairs, args.out, args.layers, args.shard_size)
    print(f"cached {n} sequences to {args.out} (layers {args.layers}, streamed shards)")


if __name__ == "__main__":
    main()

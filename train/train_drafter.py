"""Path B: train the shrunk drafter from scratch on the streamed cache (spec 11).

Feasible on a single 24 GB GPU with gradient checkpointing and a small batch; the
drafter only needs to be "good enough to expose the quantization/calibration
effects," not state of the art (spec section 11). Reads the hidden-state cache one
shard at a time so it never blows the 32 GB RAM budget.

    python train/train_drafter.py --config configs/drafter_qwen3_4b.yaml \
        --cache data/cache --steps 20000 --out checkpoints/drafter_fp16.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

from edgespark.utils.config import DrafterConfig


def train(config: DrafterConfig, cache_dir, steps, out_path, batch_size=4, lr=3e-4,
          grad_checkpoint=True, on_the_fly=False):
    import torch
    from torch.optim import AdamW

    from data.build_cache import stream_cache
    from edgespark.drafter import EdgeSparkDrafter
    from train.losses import drafter_loss

    device = "cuda" if torch.cuda.is_available() else "cpu"
    drafter = EdgeSparkDrafter(config).to(device)
    if grad_checkpoint and hasattr(drafter.backbone, "decoder"):
        # Trade compute for memory on the 24 GB card.
        drafter.backbone.decoder.gradient_checkpointing = True

    opt = AdamW(drafter.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.01)
    # GradScaler is an fp16 tool; bf16 shares fp32's exponent range and needs no
    # loss scaling, so it is only enabled for an fp16 run.
    scaler = torch.cuda.amp.GradScaler(enabled=config.precision_train == "fp16")

    drafter.train()
    step = 0
    writer = _tensorboard(out_path)
    for batch in _batches(stream_cache(cache_dir), batch_size, device,
                          config.target_layer_ids, config.block_size):
        if step >= steps:
            break
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                            enabled=config.precision_train == "bf16"):
            logits, conf_logit, block_feature = drafter(batch["hidden_by_layer"], batch["prefix_last"])
            loss, parts = drafter_loss(
                logits, block_feature, batch["target_tokens"],
                batch["target_hidden"], conf_logit, config,
            )
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        if step % 50 == 0 and writer is not None:
            for k, v in parts.items():
                writer.add_scalar(f"loss/{k}", v, step)
        step += 1

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": drafter.state_dict(), "config": config.__dict__}, out_path)
    return out_path


def _batches(shard_stream, batch_size, device, layer_ids, block_size):
    """Assemble training batches from the streamed hidden-state cache.

    Each streamed item is ``(hidden, ids)`` where ``hidden`` is
    ``[num_layers, seq, Hv]`` (the selected verifier layers, float16) and ``ids``
    is the ``[seq]`` verifier token sequence. A block window slides over each
    sequence: conditioning on the hidden state at anchor position ``t`` (ctx=1,
    matching the inference loop, which drafts from the single last-context
    hidden), the drafter predicts the ``block_size`` tokens at ``t+1 .. t+block``.

    Yields dicts with:
      * ``hidden_by_layer``  {lid: [b, 1, Hv]}   fusion inputs at the anchor
      * ``prefix_last``      [b]                 anchor token id (long)
      * ``target_tokens``    [b, block]          next verifier tokens (CE / accept)
      * ``target_hidden``    [b, block, Hv]      verifier hidden at block positions
                                                 (L1 target; deepest cached layer)

    Streaming contract: the loader holds one shard/item at a time, so the full
    cache never enters RAM. ``block_hidden`` and the confidence logit are not
    batch fields -- they come from the drafter's own forward pass.
    """
    import numpy as np
    import torch

    layer_ids = list(layer_ids)
    deep = len(layer_ids) - 1  # deepest cached layer -> feature-regression target

    def _fresh():
        return {"hidden": {lid: [] for lid in layer_ids}, "prefix": [], "tok": [], "hid": []}

    def _emit(buf):
        hidden_by_layer = {
            lid: torch.from_numpy(np.stack(buf["hidden"][lid])).to(device=device, dtype=torch.float32)
            for lid in layer_ids
        }
        return {
            "hidden_by_layer": hidden_by_layer,  # each [b, 1, Hv]
            "prefix_last": torch.tensor(buf["prefix"], device=device, dtype=torch.long),
            "target_tokens": torch.from_numpy(np.stack(buf["tok"])).to(device=device, dtype=torch.long),
            "target_hidden": torch.from_numpy(np.stack(buf["hid"])).to(device=device, dtype=torch.float32),
        }

    buf, n = _fresh(), 0
    for hidden, ids in shard_stream:
        hidden = np.asarray(hidden)
        ids = np.asarray(ids).ravel()
        seq = int(hidden.shape[1])
        for t in range(0, seq - block_size):
            for k, lid in enumerate(layer_ids):
                buf["hidden"][lid].append(hidden[k, t:t + 1, :].astype(np.float32))  # [1, Hv]
            buf["prefix"].append(int(ids[t]))
            buf["tok"].append(ids[t + 1:t + 1 + block_size].astype(np.int64))  # [block]
            buf["hid"].append(hidden[deep, t + 1:t + 1 + block_size, :].astype(np.float32))  # [block, Hv]
            n += 1
            if n >= batch_size:
                yield _emit(buf)
                buf, n = _fresh(), 0
    if n > 0:
        yield _emit(buf)


def _tensorboard(out_path):
    try:
        from torch.utils.tensorboard import SummaryWriter

        return SummaryWriter(log_dir=str(Path(out_path).parent / "tb"))
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/drafter_qwen3_4b.yaml")
    ap.add_argument("--cache", default="data/cache")
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--out", default="checkpoints/drafter_fp16.pt")
    ap.add_argument("--on-the-fly", action="store_true", help="compute hidden states live, skip the cache")
    args = ap.parse_args()
    cfg = DrafterConfig.from_yaml(args.config)
    path = train(cfg, args.cache, args.steps, args.out, on_the_fly=args.on_the_fly)
    print(f"saved drafter checkpoint to {path}")


if __name__ == "__main__":
    main()

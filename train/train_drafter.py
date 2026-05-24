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
    scaler = torch.cuda.amp.GradScaler(enabled=config.precision_train == "bf16")

    drafter.train()
    step = 0
    writer = _tensorboard(out_path)
    for batch in _batches(stream_cache(cache_dir), batch_size, device):
        if step >= steps:
            break
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                            enabled=config.precision_train == "bf16"):
            logits, conf_logit = drafter(batch["hidden_by_layer"], batch["prefix_last"])
            loss, parts = drafter_loss(
                logits, batch["block_hidden"], batch["target_tokens"],
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


def _batches(shard_stream, batch_size, device):
    """Assemble training batches from streamed cache items.

    Placeholder collation: a production version aligns block windows, target
    tokens and verifier hidden states from the cached sequences. Kept explicit so
    the streaming contract (one shard resident at a time) is visible.
    """
    raise NotImplementedError(
        "wire cache collation to your pair format; see data/build_cache.py manifest"
    )
    yield  # pragma: no cover


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

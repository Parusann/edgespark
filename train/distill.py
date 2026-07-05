"""Path A: distill a released drafter into the shrunk EdgeSpark architecture.

The preferred route (spec section 11). DeepSpec publishes DSpark / DFlash /
EAGLE-3 checkpoints with Qwen3 configs. If one targets our verifier and loads on
ROCm, use it as teacher/initialiser and distill into the smaller architecture
(section 9.2), which is far faster than training from scratch and frees the
schedule for the calibration study.

Distillation adds a KL term against the teacher's block distribution on top of
the standard drafter loss; the shrunk student keeps the L1 hidden-regression and
confidence terms so its confidence head is trained from the start.

    python train/distill.py --teacher deepspec/dspark-qwen3-4b \
        --config configs/drafter_qwen3_4b.yaml --out checkpoints/drafter_fp16.pt
"""

from __future__ import annotations

import argparse

from edgespark.utils.config import DrafterConfig


def load_teacher(teacher_id: str, device: str):
    """Load a released drafter checkpoint. Returns the teacher module.

    Confirmed viable in Phase 0 (spec section 16). If no compatible checkpoint
    loads on ROCm, fall back to Path B (``train_drafter.py``).

    Accepts a local ``.pt`` saved in EdgeSpark's own layout
    (``{"state_dict": ..., "config": {...}}``) -- e.g. a DeepSpec / EAGLE-3
    drafter already converted to this architecture. A raw upstream release with a
    different module layout must be converted first; if none loads on ROCm, use
    Path B.
    """
    import torch
    from pathlib import Path

    from edgespark.drafter import EdgeSparkDrafter
    from edgespark.utils.config import DrafterConfig

    path = Path(teacher_id)
    if not path.exists():
        raise FileNotFoundError(
            f"teacher checkpoint {teacher_id!r} not found. Path A needs a released "
            "drafter saved in EdgeSpark layout ({'state_dict':..., 'config':...}); "
            "if none loads on ROCm, use train/train_drafter.py (Path B)."
        )
    ckpt = torch.load(path, map_location=device)
    cfgd = dict(ckpt.get("config") or {})
    if "target_layer_ids" in cfgd:
        cfgd["target_layer_ids"] = tuple(cfgd["target_layer_ids"])
    teacher = EdgeSparkDrafter(DrafterConfig(**cfgd) if cfgd else DrafterConfig()).to(device)
    teacher.load_state_dict(ckpt["state_dict"])
    teacher.eval()
    return teacher


def distill(config: DrafterConfig, teacher_id: str, cache_dir, steps, out_path,
            kl_alpha: float = 0.5):
    import torch
    import torch.nn.functional as F
    from torch.optim import AdamW

    from edgespark.drafter import EdgeSparkDrafter
    from train.losses import drafter_loss

    device = "cuda" if torch.cuda.is_available() else "cpu"
    teacher = load_teacher(teacher_id, device)
    student = EdgeSparkDrafter(config).to(device)
    opt = AdamW(student.parameters(), lr=3e-4, betas=(0.9, 0.95))

    student.train()
    for step, batch in enumerate(
        _distill_batches(cache_dir, device, config.target_layer_ids, config.block_size)
    ):
        if step >= steps:
            break
        with torch.no_grad():
            t_logits = teacher(batch["hidden_by_layer"], batch["prefix_last"])[0]
        s_logits, conf_logit, s_feature = student(batch["hidden_by_layer"], batch["prefix_last"])

        base, parts = drafter_loss(
            s_logits, s_feature, batch["target_tokens"],
            batch["target_hidden"], conf_logit, config,
        )
        kl = F.kl_div(
            F.log_softmax(s_logits, dim=-1), F.softmax(t_logits, dim=-1),
            reduction="batchmean",
        )
        loss = base + kl_alpha * kl
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    torch.save({"state_dict": student.state_dict(), "config": config.__dict__}, out_path)
    return out_path


def _distill_batches(cache_dir, device, layer_ids, block_size, batch_size=4):
    """Shared collation with Path B: stream the cache and yield training batches."""
    from data.build_cache import stream_cache
    from train.train_drafter import _batches

    yield from _batches(stream_cache(cache_dir), batch_size, device, layer_ids, block_size)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--config", default="configs/drafter_qwen3_4b.yaml")
    ap.add_argument("--cache", default="data/cache")
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--out", default="checkpoints/drafter_fp16.pt")
    args = ap.parse_args()
    cfg = DrafterConfig.from_yaml(args.config)
    distill(cfg, args.teacher, args.cache, args.steps, args.out)


if __name__ == "__main__":
    main()

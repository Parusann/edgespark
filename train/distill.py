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
    """
    import torch  # noqa: F401

    raise NotImplementedError(
        "point this at the DeepSpec release layout for your verifier; "
        "if none loads on ROCm, use train/train_drafter.py (Path B)"
    )


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
    for step, batch in enumerate(_distill_batches(cache_dir, device)):
        if step >= steps:
            break
        with torch.no_grad():
            t_logits = teacher(batch["hidden_by_layer"], batch["prefix_last"])[0]
        s_logits, conf_logit = student(batch["hidden_by_layer"], batch["prefix_last"])

        base, parts = drafter_loss(
            s_logits, batch["block_hidden"], batch["target_tokens"],
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


def _distill_batches(cache_dir, device):
    raise NotImplementedError("share collation with train/train_drafter.py")
    yield  # pragma: no cover


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

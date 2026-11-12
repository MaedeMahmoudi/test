"""IADEPLR diagnostics on the actual VRM/CIFAR-100 pipeline.

Examples
--------
# fast synthetic sanity check (no dataset download)
PYTHONPATH=. python tools/diagnostic_iadeplr.py --synthetic

# real CIFAR-100 batch using the project's VRM teacher/student definitions
PYTHONPATH=. python tools/diagnostic_iadeplr.py --cfg configs/cifar100/vrm/iadeplr_r32x4_r8x4.yaml
"""

import argparse
import copy
import json
import os
import sys

import torch
import torch.nn.functional as F

from mdistiller.distillers.iadeplr import IADEPLR, compute_edge_gradients, compute_interaction_matrix


def synthetic(args):
    """A tiny differentiable graph where the full IADEPLR pipeline is cheap."""
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.nn.Sequential(torch.nn.Linear(8, 16), torch.nn.Tanh(), torch.nn.Linear(16, 4)).to(device)
    xw = torch.randn(args.batch_size, 8, device=device)
    xs = xw + 0.05 * torch.randn_like(xw)
    teacher_w = torch.randn(args.batch_size, 4, device=device)
    teacher_s = torch.randn(args.batch_size, 4, device=device)
    target = torch.randint(0, 4, (args.batch_size,), device=device)

    sw = model(xw)
    ss = model(xs)
    stu_w = F.softmax(sw, dim=1)
    stu_s = F.softmax(ss, dim=1)
    with torch.no_grad():
        tw = F.softmax(teacher_w, dim=1)
        ts = F.softmax(teacher_s, dim=1)
        td = F.normalize(tw.unsqueeze(0) - ts.unsqueeze(1), p=2, dim=2)
    sd = F.normalize(stu_w.unsqueeze(0) - stu_s.unsqueeze(1), p=2, dim=2)
    edge_losses = F.smooth_l1_loss(td, sd, reduction="none").mean(dim=2)
    val_loss = F.cross_entropy(model(torch.randn(args.batch_size, 8, device=device)), target)

    controller = IADEPLR(
        model,
        rank=min(args.rank, 4),
        eta=args.eta,
        prune_fraction=args.prune_fraction,
        interaction_threshold=args.interaction_threshold,
        max_candidates=args.max_candidates,
        damping=args.damping,
        seed=args.seed,
    )
    mask, report = controller.step(edge_losses, val_loss)
    print(json.dumps(report, indent=2, default=str))
    print("mask_keep_ratio={:.4f}".format(mask.mean().item()))


def build_real(cfg_path):
    from mdistiller.engine.cfg import CFG as cfg
    from mdistiller.engine.utils import load_checkpoint
    from mdistiller.models import cifar_model_dict
    from mdistiller.dataset import get_dataset_strong
    from mdistiller.distillers.VRM import VRM

    cfg.defrost()
    cfg.merge_from_file(cfg_path)
    cfg.freeze()
    train_loader, val_loader, _, num_classes = get_dataset_strong(cfg)
    net, teacher_path = cifar_model_dict[cfg.DISTILLER.TEACHER]
    teacher = net(num_classes=num_classes)
    teacher.load_state_dict(load_checkpoint(teacher_path)["model"])
    student = cifar_model_dict[cfg.DISTILLER.STUDENT][0](num_classes=num_classes)
    distiller = VRM(student, teacher, cfg).cuda()
    return cfg, distiller, train_loader, val_loader


def real(args):
    cfg, distiller, train_loader, val_loader = build_real(args.cfg)
    distiller.train()
    train_data = next(iter(train_loader))
    image, target, _ = train_data
    image_w, image_s = image
    image_w = image_w.float().cuda()
    image_s = image_s.float().cuda()
    target = target.cuda()
    _, losses = distiller(image_weak=image_w, image_strong=image_s, target=target)
    edge_losses = losses["__iadeplr_edge_losses__"]

    val_image, val_target = next(iter(val_loader))
    val_image = val_image.float().cuda()
    val_target = val_target.cuda()
    student = distiller.student
    was_training = student.training
    student.eval()
    val_loss = F.cross_entropy(student(val_image)[0], val_target)
    student.train(was_training)

    controller = IADEPLR(
        student,
        rank=args.rank,
        eta=args.eta,
        prune_fraction=args.prune_fraction,
        interaction_threshold=args.interaction_threshold,
        max_candidates=args.max_candidates,
        damping=args.damping,
        seed=args.seed,
    )
    mask, report = controller.step(edge_losses, val_loss)
    report["device"] = str(edge_losses.device)
    report["batch_size"] = int(edge_losses.size(0))
    report["keep_ratio"] = float(mask.mean().detach().cpu())
    print(json.dumps(report, indent=2, default=str))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--cfg", default="configs/cifar100/vrm/iadeplr_r32x4_r8x4.yaml")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--max-candidates", type=int, default=8)
    parser.add_argument("--prune-fraction", type=float, default=0.25)
    parser.add_argument("--interaction-threshold", type=float, default=0.0)
    parser.add_argument("--damping", type=float, default=1e-3)
    parser.add_argument("--eta", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    if args.synthetic:
        synthetic(args)
    else:
        real(args)


if __name__ == "__main__":
    main()

# VRM
import torch
import torch.nn as nn
import torch.nn.functional as F
from ._base import Distiller


def _zs_probs(logits, temperature=4.0):
    x = logits.view(logits.shape[0], -1)
    mean = x.mean(dim=-1, keepdim=True)
    std = x.std(dim=-1, keepdim=True)
    x = (x - mean) / (1e-7 + std)
    return F.softmax(x / temperature, dim=1)


def isv_edge_loss(f_sw, f_ss, f_tw, f_ts, edge_weight=50.0):
    """Return the BxB ISV relation loss, retaining one scalar per edge."""
    stu_w = _zs_probs(f_sw)
    stu_s = _zs_probs(f_ss)
    with torch.no_grad():
        tea_w = _zs_probs(f_tw)
        tea_s = _zs_probs(f_ts)
        td = tea_w.unsqueeze(0) - tea_s.unsqueeze(1)
        norm_td = F.normalize(td, p=2, dim=2)

    sd = stu_w.unsqueeze(0) - stu_s.unsqueeze(1)
    norm_sd = F.normalize(sd, p=2, dim=2)
    # Original VRM used reduction='mean' over [B,B,C]. We first reduce only
    # over C so the exact same global mean is recovered by edge.mean().
    per_feature = F.smooth_l1_loss(norm_td, norm_sd, reduction="none")
    return per_feature.mean(dim=2) * edge_weight


def isv_loss(f_sw, f_ss, f_tw, f_ts, edge_weight=50.0, edge_mask=None):
    edge_losses = isv_edge_loss(f_sw, f_ss, f_tw, f_ts, edge_weight=edge_weight)
    if edge_mask is not None:
        edge_losses = edge_losses * edge_mask
    return edge_losses.mean()


def icv_loss(f_sw, f_ss, f_tw, f_ts, edge_weight=50.0):
    stu_w = _zs_probs(f_sw)
    stu_s = _zs_probs(f_ss)
    with torch.no_grad():
        tea_w = _zs_probs(f_tw)
        tea_s = _zs_probs(f_ts)
        td = (tea_w.unsqueeze(2) - tea_s.unsqueeze(1)).permute(1, 2, 0)
        norm_td = F.normalize(td, p=2, dim=2)
    sd = (stu_w.unsqueeze(2) - stu_s.unsqueeze(1)).permute(1, 2, 0)
    norm_sd = F.normalize(sd, p=2, dim=2)
    loss_a = F.smooth_l1_loss(norm_td, norm_sd)
    return edge_weight * loss_a


class VRM(Distiller):
    def __init__(self, student, teacher, cfg):
        super(VRM, self).__init__(student, teacher)
        self.temperature = cfg.KD.TEMPERATURE
        self.ce_loss_weight = cfg.KD.LOSS.CE_WEIGHT
        self.kd_loss_weight = cfg.KD.LOSS.KD_WEIGHT
        self.isv_scale = 64.0
        self.icv_scale = 32.0

        # IADEPLR is deliberately opt-in. Existing VRM experiments remain
        # numerically unchanged when the config block is absent/disabled.
        self.use_iadeplr = bool(getattr(cfg, "IADEPLR", {}).get("ENABLED", False)) if hasattr(getattr(cfg, "IADEPLR", None), "get") else False
        self.iadeplr_mask = None

    def set_iadeplr_mask(self, mask):
        self.iadeplr_mask = mask

    def clear_iadeplr_mask(self):
        self.iadeplr_mask = None

    def forward_train(self, image_weak, image_strong, target, **kwargs):
        logits_student_weak, _ = self.student(image_weak)
        logits_student_strong, _ = self.student(image_strong)
        with torch.no_grad():
            logits_teacher_weak, _ = self.teacher(image_weak)
            logits_teacher_strong, _ = self.teacher(image_strong)

        loss_ce = F.cross_entropy(logits_student_weak, target) + F.cross_entropy(logits_student_strong, target)
        edge_losses = isv_edge_loss(
            logits_student_weak, logits_student_strong,
            logits_teacher_weak, logits_teacher_strong,
            edge_weight=50.0,
        )
        masked_edge_losses = edge_losses if self.iadeplr_mask is None else edge_losses * self.iadeplr_mask
        loss_isv = self.isv_scale * masked_edge_losses.mean()
        loss_icv = self.icv_scale * icv_loss(
            logits_student_weak, logits_student_strong,
            logits_teacher_weak, logits_teacher_strong,
        )

        losses_dict = {
            "loss_ce": loss_ce,
            "loss_isv": loss_isv,
            "loss_icv": loss_icv,
        }
        # This entry is metadata for AugTrainer; it is intentionally excluded
        # from the scalar training objective to avoid double counting.
        if self.use_iadeplr:
            losses_dict["__iadeplr_edge_losses__"] = edge_losses
        return logits_student_weak, losses_dict

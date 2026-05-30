"""Scene-level assistive metrics (validation): proxies beyond detection mAP."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class AssistiveEvalAccumulator:
    """Image-level proxies using scene urgency logits vs GT scene urgency.

    Per-object hazard recall needs Hungarian matching; these counters stay cheap
    and stable for epoch-over-epoch monitoring.
    """

    hazard_hit: int = 0
    hazard_total: int = 0
    false_alert: int = 0
    safe_total: int = 0
    correct_level: int = 0
    level_total: int = 0

    def reset(self) -> None:
        self.hazard_hit = 0
        self.hazard_total = 0
        self.false_alert = 0
        self.safe_total = 0
        self.correct_level = 0
        self.level_total = 0

    def update(self, pred_urgency_logits: torch.Tensor, gt_scene_urgency: torch.Tensor) -> None:
        """pred: [B, L] logits; gt: [B] integer scene urgency 0..L-1."""
        if pred_urgency_logits.dim() != 2:
            return
        if gt_scene_urgency.dim() != 1:
            return
        b = min(pred_urgency_logits.shape[0], gt_scene_urgency.shape[0])
        if b == 0:
            return
        pred = pred_urgency_logits[:b].argmax(dim=-1)
        gt = gt_scene_urgency[:b].long()
        danger = gt >= 2
        if danger.any():
            self.hazard_total += int(danger.sum().item())
            self.hazard_hit += int((danger & (pred >= 2)).sum().item())
        safe = gt <= 0
        if safe.any():
            self.safe_total += int(safe.sum().item())
            self.false_alert += int((safe & (pred >= 2)).sum().item())
        self.level_total += b
        self.correct_level += int((pred == gt).sum().item())

    def compute(self) -> dict:
        out = {
            "hazard_recall_proxy": (
                self.hazard_hit / self.hazard_total if self.hazard_total > 0 else 0.0
            ),
            "false_alert_rate_proxy": (
                self.false_alert / self.safe_total if self.safe_total > 0 else 0.0
            ),
            "urgency_level_accuracy": (
                self.correct_level / self.level_total if self.level_total > 0 else 0.0
            ),
        }
        return out

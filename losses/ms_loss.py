import torch
from torch import nn

__all__ = ["MultiSimilarityLoss"]

class MultiSimilarityLoss(nn.Module):
    def __init__(self, scale_pos, scale_neg):
        super(MultiSimilarityLoss, self).__init__()
        self.thresh = 0.5
        self.margin = 0.1

        self.scale_pos = scale_pos
        self.scale_neg = scale_neg

    def forward(self, feats, labels):
        assert feats.size(0) == labels.size(0), \
            f"feats.size(0): {feats.size(0)} is not equal to labels.size(0): {labels.size(0)}"
        batch_size = feats.size(0)
        sim_mat = torch.matmul(feats, feats.t())

        labels = labels.view(-1, 1)
        pos_mask = labels == labels.t()
        pos_mask.fill_diagonal_(False)
        neg_mask = labels != labels.t()

        pos_sim = sim_mat.clone()
        pos_sim[~pos_mask] = float('inf')
        # Handle cases where some anchors have no positive pairs
        try:
            min_pos_sim, _ = pos_sim.min(dim=1)
        except Exception:
            min_pos_sim = torch.full((batch_size,), float('inf'), device=feats.device)

        neg_sim = sim_mat.clone()
        neg_sim[~neg_mask] = float('-inf')
        try:
            max_neg_sim, _ = neg_sim.max(dim=1)
        except Exception:
            max_neg_sim = torch.full((batch_size,), float('-inf'), device=feats.device)

        # Apply hard mining conditions
        valid_neg_mask = neg_mask & (sim_mat + self.margin > min_pos_sim.unsqueeze(1))
        valid_pos_mask = pos_mask & (sim_mat - self.margin < max_neg_sim.unsqueeze(1))

        # Calculate exponentials
        pos_exp = torch.exp(-self.scale_pos * (sim_mat - self.thresh))
        pos_exp = pos_exp * valid_pos_mask.float()

        neg_exp = torch.exp(self.scale_neg * (sim_mat - self.thresh))
        neg_exp = neg_exp * valid_neg_mask.float()

        pos_sum = pos_exp.sum(dim=1)
        neg_sum = neg_exp.sum(dim=1)

        valid_anchor_mask = (valid_pos_mask.sum(dim=1) > 0) & (valid_neg_mask.sum(dim=1) > 0)

        pos_loss = (1.0 / self.scale_pos) * torch.log(1 + pos_sum)
        neg_loss = (1.0 / self.scale_neg) * torch.log(1 + neg_sum)

        total_anchor_loss = pos_loss + neg_loss
        valid_losses = total_anchor_loss[valid_anchor_mask]

        if valid_losses.numel() == 0:
            return torch.zeros([], requires_grad=True, device=labels.device)

        loss = valid_losses.sum() / batch_size
        return loss

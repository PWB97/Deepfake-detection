import torch
import torch.nn.functional as F


class AUCLoss(torch.nn.Module):
    def __init__(self, device, gamma=0.15, alpha=0.6, p=2):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.p = p
        self.device = device

    def forward(self, y_pred, y_true):
        pred = torch.sigmoid(y_pred)
        pos = pred[torch.where(y_true == 0)]
        neg = pred[torch.where(y_true == 1)]
        pos = torch.unsqueeze(pos, 0)
        neg = torch.unsqueeze(neg, 1)
        diff = torch.zeros_like(pos * neg, device=self.device) + pos - neg - self.gamma
        masked = diff[torch.where(diff < 0.0)]
        auc = torch.mean(torch.pow(-masked, self.p))
        bce = F.binary_cross_entropy_with_logits(y_pred, y_true)
        if masked.shape[0] == 0:
            loss = bce
        else:
            loss = self.alpha * bce + (1 - self.alpha) * auc
        return loss

import torch
import torch.nn as nn

class BinaryFocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(inputs)
        probs = probs.view(-1)
        targets = targets.view(-1)
        
        pt = targets * probs + (1 - targets) * (1 - probs)
        alpha_t = targets * self.alpha + (1 - targets) * (1.0 - self.alpha)
        
        loss = -alpha_t * (1.0 - pt) ** self.gamma * torch.log(pt + 1e-6)
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss

class TverskyLoss(nn.Module):
    def __init__(self, alpha: float = 0.7, beta: float = 0.3, smooth: float = 1e-6):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
        
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(inputs)
        probs = probs.view(-1)
        targets = targets.view(-1)
        
        tp = (probs * targets).sum()
        fp = (probs * (1.0 - targets)).sum()
        fn = ((1.0 - probs) * targets).sum()
        
        tversky_index = (tp + self.smooth) / (tp + self.alpha * fn + self.beta * fp + self.smooth)
        return 1.0 - tversky_index

class FocalTverskyLoss(nn.Module):
    def __init__(self, alpha: float = 0.7, beta: float = 0.3, gamma: float = 0.75, smooth: float = 1e-6):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth
        self.tversky = TverskyLoss(alpha=alpha, beta=beta, smooth=smooth)
        
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        tversky_loss = self.tversky(inputs, targets)
        return tversky_loss ** self.gamma

class ComboLoss(nn.Module):
    def __init__(self, alpha_focal: float = 0.25, gamma_focal: float = 2.0, 
                 alpha_tversky: float = 0.7, beta_tversky: float = 0.3, 
                 weight_focal: float = 0.5, smooth: float = 1e-6):
        super().__init__()
        self.weight_focal = weight_focal
        self.focal = BinaryFocalLoss(alpha=alpha_focal, gamma=gamma_focal)
        self.tversky = TverskyLoss(alpha=alpha_tversky, beta=beta_tversky, smooth=smooth)
        
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        focal_loss = self.focal(inputs, targets)
        tversky_loss = self.tversky(inputs, targets)
        
        total_loss = self.weight_focal * focal_loss + (1.0 - self.weight_focal) * tversky_loss
        return total_loss

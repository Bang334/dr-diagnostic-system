import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from models.unet import get_segmentation_model
from losses.loss import BinaryFocalLoss, TverskyLoss, FocalTverskyLoss, ComboLoss
from metrics.eval_metrics import compute_pixel_metrics

class RetinalLesionLightningModule(pl.LightningModule):
    """
    LightningModule cho phân đoạn tổn thương võng mạc.
    Quản lý luồng huấn luyện, đánh giá và bộ tối ưu hóa tự động.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.save_hyperparameters()
        
        self.model = get_segmentation_model(
            architecture_name=config.model_architecture,
            backbone_name=config.backbone_name,
            encoder_weights=config.encoder_weights,
            in_channels=1,
            classes=1
        )
        
        self._init_loss_function()
        
    def _init_loss_function(self):
        loss_type = self.config.loss_type
        
        if loss_type == 'focal':
            self.criterion = BinaryFocalLoss(
                alpha=self.config.focal_alpha,
                gamma=self.config.focal_gamma
            )
        elif loss_type == 'tversky':
            self.criterion = TverskyLoss(
                alpha=self.config.tversky_alpha,
                beta=self.config.tversky_beta
            )
        elif loss_type == 'focal_tversky':
            self.criterion = FocalTverskyLoss(
                alpha=self.config.tversky_alpha,
                beta=self.config.tversky_beta,
                gamma=0.75
            )
        elif loss_type == 'combo':
            self.criterion = ComboLoss(
                alpha_focal=self.config.focal_alpha,
                gamma_focal=self.config.focal_gamma,
                alpha_tversky=self.config.tversky_alpha,
                beta_tversky=self.config.tversky_beta,
                weight_focal=self.config.combo_weight_focal
            )
        else:
            print(f"[Warning] Không nhận diện được loss {loss_type}. Mặc định sử dụng Combo Loss.")
            self.criterion = ComboLoss()
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
        
    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        images, masks = batch
        logits = self(images)
        loss = self.criterion(logits, masks)
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss
        
    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        images, masks = batch
        logits = self(images)
        loss = self.criterion(logits, masks)
        
        probs = torch.sigmoid(logits)
        preds = (probs > self.config.threshold).float()
        targets = (masks > 0.5).float()
        
        tp = torch.sum(preds * targets, dim=(1, 2, 3))
        fp = torch.sum(preds * (1.0 - targets), dim=(1, 2, 3))
        fn = torch.sum((1.0 - preds) * targets, dim=(1, 2, 3))
        
        dice = (2.0 * tp) / (2.0 * tp + fp + fn + 1e-6)
        iou = tp / (tp + fp + fn + 1e-6)
        sens = tp / (tp + fn + 1e-6)
        prec = tp / (tp + fp + 1e-6)
        
        dice_mean = torch.mean(dice)
        iou_mean = torch.mean(iou)
        sens_mean = torch.mean(sens)
        prec_mean = torch.mean(prec)
        
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_dice', dice_mean, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_iou', iou_mean, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log('val_sens', sens_mean, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log('val_prec', prec_mean, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        
        return loss
        
    def on_validation_epoch_end(self):
        metrics = self.trainer.callback_metrics
        if 'val_loss' in metrics and 'val_dice' in metrics:
            val_loss = metrics['val_loss'].item()
            val_dice = metrics['val_dice'].item()
            val_sens = metrics.get('val_sens', torch.tensor(0.0)).item()
            val_prec = metrics.get('val_prec', torch.tensor(0.0)).item()
            
            train_loss_tensor = metrics.get('train_loss', None)
            train_loss_str = f"{train_loss_tensor.item():.4f}" if train_loss_tensor is not None else "N/A"
            
            print(f"Epoch {self.current_epoch:02d}: train_loss={train_loss_str} | val_loss={val_loss:.4f} | val_dice={val_dice:.4f} | val_sens={val_sens:.4f} | val_prec={val_prec:.4f}")
        
    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        warmup_epochs = 5
        scheduler_warmup = optim.lr_scheduler.LinearLR(
            optimizer, 
            start_factor=0.01, 
            end_factor=1.0, 
            total_iters=warmup_epochs
        )
        
        scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=self.config.max_epochs - warmup_epochs, 
            eta_min=1e-6
        )
        
        scheduler = optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[scheduler_warmup, scheduler_cosine],
            milestones=[warmup_epochs]
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1
            }
        }

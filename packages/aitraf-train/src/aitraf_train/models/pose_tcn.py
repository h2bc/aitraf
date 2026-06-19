"""LightningModule implementing a lightweight temporal convolutional network."""

import torch
from torch import nn
import pytorch_lightning as pl


class TemporalBlock(nn.Module):
    """Basic dilated temporal convolution block with residual connection."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ) -> None:
        super().__init__()

        padding = (kernel_size - 1) * dilation

        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            dilation=dilation,
            padding=padding,
        )

        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size,
            dilation=dilation,
            padding=padding,
        )

        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()
        self.downsample = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.activation(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.activation(out)
        out = self.dropout(out)

        residual = x if self.downsample is None else self.downsample(x)

        # remove extra elements from padding so residual length matches
        out = out[..., : residual.shape[-1]]

        return out + residual


class TCNBackbone(nn.Module):
    """Shared temporal feature extractor."""

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int,
        num_layers: int,
        kernel_size: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Conv1d(feature_dim, hidden_dim, kernel_size=1)

        layers = []
        in_channels = hidden_dim

        for layer_idx in range(num_layers):
            dilation = 2**layer_idx
            block = TemporalBlock(
                in_channels=in_channels,
                out_channels=hidden_dim,
                kernel_size=kernel_size,
                dilation=dilation,
                dropout=dropout,
            )
            layers.append(block)
            in_channels = hidden_dim

        self.tcn = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, feature_dim)
        x = x.transpose(1, 2)  # -> (batch, feature_dim, seq_len)
        x = self.input_proj(x)  # (batch, hidden_dim, seq_len)
        return self.tcn(x)


class TCNClassificationHead(nn.Module):
    """Pooling + classification head."""

    def __init__(self, hidden_dim: int, num_classes: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


class TCNRegressionHead(nn.Module):
    """Pooling + regression head (single target)."""

    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(-1)


class TCNClassifier(pl.LightningModule):
    """Temporal convolutional classifier over pose sequences."""

    def __init__(
        self,
        feature_dim: int,
        num_classes: int,
        learning_rate: float,
        metrics_fn,
        loss_fn: nn.Module | None = None,
        hidden_dim: int = 128,
        num_layers: int = 3,
        kernel_size: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["loss_fn", "metrics_fn"])

        self.backbone = TCNBackbone(
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            kernel_size=kernel_size,
            dropout=dropout,
        )
        self.head = TCNClassificationHead(
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            dropout=dropout,
        )
        self.loss_fn = loss_fn or nn.CrossEntropyLoss()
        self.metrics_fn = metrics_fn
        self._val_pred_batches: list[torch.Tensor] = []
        self._val_label_batches: list[torch.Tensor] = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.head(features)

    def training_step(self, batch, batch_idx):
        logits = self(batch["inputs"])
        loss = self.loss_fn(logits, batch["labels"])
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def on_validation_epoch_start(self):
        self._val_pred_batches = []
        self._val_label_batches = []

    def validation_step(self, batch, batch_idx):
        logits = self(batch["inputs"])
        loss = self.loss_fn(logits, batch["labels"])
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        preds = logits.argmax(dim=-1).detach().cpu()
        labels = batch["labels"].detach().cpu()
        self._val_pred_batches.append(preds)
        self._val_label_batches.append(labels)

    def on_validation_epoch_end(self):
        if not self._val_pred_batches:
            return

        preds = torch.cat(self._val_pred_batches, dim=0).numpy()
        labels = torch.cat(self._val_label_batches, dim=0).numpy()
        metrics = self.metrics_fn(preds, labels)

        for name, value in metrics.items():
            self.log(
                f"val_{name}",
                float(value),
                prog_bar=False,
                on_step=False,
                on_epoch=True,
            )

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)


class TCNRegressor(pl.LightningModule):
    """Temporal convolutional regressor for pose sequences."""

    def __init__(
        self,
        feature_dim: int,
        learning_rate: float,
        metrics_fn,
        hidden_dim: int = 128,
        num_layers: int = 3,
        kernel_size: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.backbone = TCNBackbone(
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            kernel_size=kernel_size,
            dropout=dropout,
        )
        self.head = TCNRegressionHead(
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
        self.loss_fn = nn.MSELoss()
        self.metrics_fn = metrics_fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.head(features)

    def training_step(self, batch, batch_idx):
        preds = self(batch["inputs"])
        targets = batch["labels"].float()
        loss = self.loss_fn(preds, targets)
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        preds = self(batch["inputs"])
        targets = batch["labels"].float()
        loss = self.loss_fn(preds, targets)
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        metrics = self.metrics_fn(
            preds.detach().cpu().numpy(), targets.detach().cpu().numpy()
        )
        for name, value in metrics.items():
            self.log(
                f"val_{name}",
                value,
                prog_bar=False,
                on_step=False,
                on_epoch=True,
            )

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)


__all__ = [
    "TCNBackbone",
    "TCNClassificationHead",
    "TCNClassifier",
    "TCNRegressionHead",
    "TCNRegressor",
]

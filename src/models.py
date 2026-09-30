"""Tiny CNN for CIFAR-10 with configurable dropout (for MC Dropout at inference).

Deliberately avoids BatchNorm so that `model.train()` at inference only turns
dropout on, which is exactly what Monte-Carlo Dropout wants.
"""
import torch
import torch.nn as nn

from src import config


class CIFAR10CNN(nn.Module):
    def __init__(self, dropout_p: float = config.DROPOUT_P):
        super().__init__()
        self.dropout_p = dropout_p
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout_p),
            nn.Linear(4 * 4 * 256, 256), nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p),
            nn.Linear(256, 10),
        )

    def forward(self, x):
        return self.classifier(self.features(x))

    @torch.no_grad()
    def forward_features(self, x):
        """Returns (features, logits): penultimate features + logits."""
        self.eval()
        feats = self.features(x)
        feats_flat = torch.flatten(self.features(x), 1)
        hidden = self.classifier[2](self.classifier[1](feats_flat))
        feat_out = self.classifier[3](hidden)          # penultimate features
        logits = self.classifier[5](self.classifier[4](feat_out))
        return feat_out, logits


def softmax(logits):
    return torch.softmax(logits, dim=-1)


def build_model():
    return CIFAR10CNN(config.DROPOUT_P)


def load_model(path=config.MODEL_CHECKPOINT, device="cpu"):
    model = CIFAR10CNN(config.DROPOUT_P)
    model.load_state_dict(torch.load(path, map_location=device))
    model = model.to(device)
    model.eval()
    return model
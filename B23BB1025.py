"""
Age Classifier — ResNet-18 with extract_features for embedding distillation.
Clean model compatible with the evaluation script.
"""
import torch
import torch.nn as nn
from torchvision import models


class AgeClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        backbone = models.resnet18(weights=None)

        # ResNet-18 backbone (core layers untouched)
        self.conv1   = backbone.conv1
        self.bn1     = backbone.bn1
        self.relu    = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1  = backbone.layer1
        self.layer2  = backbone.layer2
        self.layer3  = backbone.layer3
        self.layer4  = backbone.layer4
        self.avgpool = backbone.avgpool

        # MLP classification head: 512 → 256 → 64 → 2
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.25),

            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),

            nn.Linear(64, num_classes),
        )

    def extract_features(self, x):
        """Extract 512-d backbone features (used during distillation training)."""
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        return torch.flatten(x, 1)  # (B, 512)

    def forward(self, x):
        feat = self.extract_features(x)
        return self.classifier(feat)


def build_model(num_classes=2):
    """Build the AgeClassifier model (required by evaluation script)."""
    return AgeClassifier(num_classes=num_classes)
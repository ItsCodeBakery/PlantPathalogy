import torch
import torch.nn as nn

class SimCLR_CNNClassifier(nn.Module):
    def __init__(self, num_classes=15, projection_dim=128, mode='classification'):
        super(SimCLR_CNNClassifier, self).__init__()
        self.mode = mode  # 'pretrain' or 'classification'
        
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2),
        )

        # Shared output size after convs: (B, 256, 8, 8)
        self.flatten = nn.Flatten()
        
        self.feature_dim = 256 * 8 * 8

        # ✳️ Projection head (for SimCLR pretraining)
        self.projection_head = nn.Sequential(
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(),
            nn.Linear(512, projection_dim)
        )

        # 🧠 Classification head
        self.classification_head = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.flatten(x)

        if self.mode == 'pretrain':
            return self.projection_head(x)  # for contrastive loss
        else:
            return self.classification_head(x)  # for classification

    def set_mode(self, mode):
        """Switch between 'pretrain' and 'classification'."""
        assert mode in ['pretrain', 'classification'], "Mode must be 'pretrain' or 'classification'"
        self.mode = mode

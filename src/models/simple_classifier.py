#!/usr/bin/env python3

import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim, dropout):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim)
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.block(x)
        out += residual  # Skip connection
        return self.relu(out)


class SimpleAudioClassifier(nn.Module):
    def __init__(self, input_dim: int, num_labels: int, hidden_dim: int = 512, dropout: float = 0.3):
        super().__init__()

        # Project input up to hidden dimension
        self.initial_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Stack of Residual Blocks (ResNet-style MLP)
        self.res_blocks = nn.Sequential(
            ResidualBlock(hidden_dim, dropout),
            ResidualBlock(hidden_dim, dropout),
            ResidualBlock(hidden_dim, dropout)
        )

        # Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_labels)
        )

    def forward(self, x):
        x = self.initial_layer(x)
        x = self.res_blocks(x)
        return self.classifier(x)

if __name__ == "__main__":
    model = SimpleAudioClassifier(input_dim=784, num_labels=8)

    # Print the model summary
    print("Model successfully initialized!")
    print(f"Model type: {type(model)}")
    print(f"Number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
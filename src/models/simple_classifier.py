#!/usr/bin/env python3

import torch
import torch.nn as nn


class CNN1DClassifier(nn.Module):
    def __init__(self, input_dim: int, num_labels: int):
        super().__init__()

        # Keras: Conv1D(256, 5, strides=1, padding='same')
        # PyTorch Conv1d expects input shape: (Batch, Channels, Length)
        # We will reshape input in forward() to (Batch, 1, 162)

        self.features = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=256, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=5, stride=2, padding=2),

            nn.Conv1d(in_channels=256, out_channels=256, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=5, stride=2, padding=2),

            nn.Conv1d(in_channels=256, out_channels=128, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=5, stride=2, padding=2),

            nn.Dropout(0.2),

            nn.Conv1d(in_channels=128, out_channels=64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=5, stride=2, padding=2)
        )

        # Flattening logic:
        # We need to calculate the size after all those pooling layers.
        # For input 162:
        # After Pool 1 (~/2) -> ~81
        # After Pool 2 (~/2) -> ~41
        # After Pool 3 (~/2) -> ~21
        # After Pool 4 (~/2) -> ~11
        # Final shape is roughly (64 channels * 11 length) = 704
        # We use a lazy linear layer or calculate it dynamically.

        self.flatten = nn.Flatten()

        # We use a dummy pass in __init__ to determine the exact flattened size
        # This makes the model flexible to any input_dim (162, 190, etc.)
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, input_dim)
            dummy_output = self.features(dummy_input)
            flattened_size = dummy_output.view(1, -1).shape[1]

        self.classifier = nn.Sequential(
            nn.Linear(flattened_size, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, num_labels)
        )

    def forward(self, x):
        # x comes in as (Batch, 162)
        # Conv1D needs (Batch, Channel, Length) -> (Batch, 1, 162)
        x = x.unsqueeze(1)

        x = self.features(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x

if __name__ == "__main__":
    model = CNN1DClassifier(input_dim=784, num_labels=8)

    # Print the model summary
    print("Model successfully initialized!")
    print(f"Model type: {type(model)}")
    print(f"Number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
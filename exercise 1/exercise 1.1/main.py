import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support
import numpy as np

# --- Setup ---
print(f"Using device: {torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else "Using device: CPU")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
input_size = 784  # 28x28 images flattened
hidden_size = 128  # Size of the hidden layer
num_epochs = 10
batch_size = 64
learning_rate = 0.001

# Load MNIST
transform = transforms.Compose([
    transforms.ToTensor(), # convert to a tensor and scale to [0, 1]
    transforms.Normalize((0.1307,), (0.3081,)) # subtract mean and divide by std dev
])


# Helper function to only flag 2 as true
def binary_label(label):
    """Converts labels: 2 -> 1, all others -> 0"""
    return 1 if label == 2 else 0


# Dataset class for binary labels
class BinaryMNIST(datasets.MNIST):
    def __getitem__(self, index):
        # Get the original image and label
        image, label = super().__getitem__(index)

        # Apply the binary label preprocessing
        binary_label_tensor = torch.tensor(binary_label(label), dtype=torch.float32)

        # Flatten the image to 784
        image_flat = image.view(-1)

        return image_flat, binary_label_tensor


# Load the training and test datasets using our custom class
print("Downloading/Loading MNIST dataset...")
train_data = BinaryMNIST(
    root='../data',
    train=True,
    download=True, # download the dataset in case it's missing
    transform=transform
)

test_data = BinaryMNIST(
    root='../data',
    train=False,
    download=True, # download the dataset in case it's missing
    transform=transform
)

# Create DataLoaders to handle batching and shuffling
train_loader = DataLoader(
    dataset=train_data,
    batch_size=batch_size,
    shuffle=True
)

test_loader = DataLoader(
    dataset=test_data,
    batch_size=batch_size,
    shuffle=False  # No need to shuffle test data
)
print("Dataset loaded.")


# Build the neural network
class BinaryClassifier(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(BinaryClassifier, self).__init__()
        # Input layer to hidden layer
        self.fc1 = nn.Linear(input_size, hidden_size)
        # ReLU activation for the hidden layer
        self.relu = nn.ReLU()
        # Hidden layer to the single output neuron
        self.fc2 = nn.Linear(hidden_size, 1)
        # Sigmoid activation for binary classification (outputs probability 0-1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Forward pass definition
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x


# Initialize the model
model = BinaryClassifier(input_size, hidden_size).to(device)

# Binary cross entropy loss
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
print(f"Starting training for {num_epochs} epochs...")
# List to store loss values for plotting
losses = []

model.train()  # Set model to training mode
for epoch in range(num_epochs):
    total_loss = 0
    for i, (images, labels) in enumerate(train_loader):
        # Move tensors to the configured device
        images = images.to(device)
        labels = labels.to(device).unsqueeze(1)  # [batch_size] -> [batch_size, 1]

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()  # Clear old gradients
        loss.backward()  # Calculate new gradients
        optimizer.step()  # Update model parameters

        total_loss += loss.item()

    # Calculate average loss for the epoch
    avg_loss = total_loss / len(train_loader)
    losses.append(avg_loss)
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}")

print("Training finished.")

# Plot the training loss
print("Plotting training loss...")
plt.figure(figsize=(10, 5))
plt.plot(range(1, num_epochs + 1), losses)
plt.xlabel("Epoch")
plt.ylabel("Average BCELoss")
plt.title("Training Loss Over 10 Epochs")
plt.grid(True)
plt.show()

# Evaluate on the Test Dataset
print("Evaluating on test data...")
model.eval()  # Set model to evaluation mode
all_preds = []
all_labels = []

with torch.no_grad():  # Disable gradient calculation for efficiency
    for images, labels in test_loader:
        images = images.to(device)

        # Forward pass
        outputs = model(images)

        # Convert sigmoid probabilities to binary predictions (0 or 1)
        # (outputs > 0.5) creates a boolean tensor
        # .float() or .int() converts it to 0s and 1s
        preds = (outputs > 0.5).int().cpu().numpy()

        all_preds.extend(preds.flatten())
        all_labels.extend(labels.cpu().numpy().astype(int))

# Calculate and print metrics
print("\n--- Evaluation Metrics ---")
# Use sklearn.metrics.classification_report
# target_names maps the labels 0 and 1 to "Not 2" and "2"
print(classification_report(all_labels, all_preds, target_names=["Not 2", "2"]))

# For individual metrics, you can also do:
accuracy = accuracy_score(all_labels, all_preds)
precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary', pos_label=1)

print(f"Overall Accuracy: {accuracy:.4f}")
print(f"Metrics for '2' (Positive Class):")
print(f"  Precision: {precision:.4f}")
print(f"  Recall:    {recall:.4f}")
print(f"  F1-Score:  {f1:.4f}")

# Visualize Some Predictions
print("\nVisualizing predictions...")

# Get one batch of test data to visualize
images_viz, labels_viz = next(iter(test_loader))
images_viz = images_viz.to(device)
labels_viz = labels_viz.to(device)

# Get model predictions
model.eval()
with torch.no_grad():
    outputs_viz = model(images_viz)
    preds_viz = (outputs_viz > 0.5).int().flatten()  # Flatten for easy indexing

# Reshape flattened images back to 28x28 for plotting
images_viz_2d = images_viz.view(-1, 28, 28).cpu().numpy()
labels_viz_np = labels_viz.cpu().numpy().astype(int)

# Create a 4x4 grid for visualization
fig, axes = plt.subplots(4, 4, figsize=(10, 10))

for i, ax in enumerate(axes.flat):
    if i >= len(labels_viz_np):  # Handle batches smaller than 16
        break

    # Display the image
    # Note: Images are normalized, so they won't look like pure 0-255 grayscale
    ax.imshow(images_viz_2d[i], cmap='gray')

    true_label = labels_viz_np[i]
    pred_label = preds_viz[i].item()

    # Set title with prediction and true label
    title = f"True: {true_label}\nPred: {pred_label}"

    # Color the title green for correct, red for incorrect
    if pred_label == true_label:
        color = "green"
    else:
        color = "red"

    ax.set_title(title, color=color)
    ax.axis('off')  # Hide axes ticks

plt.tight_layout()
plt.suptitle("Sample Test Predictions (1=is a 2, 0=not a 2)", y=1.02, fontsize=16)
plt.show()

print("\nExercise 1.1 complete.")
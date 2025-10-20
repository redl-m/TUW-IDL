# Import necessary libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import numpy as np

# --- Setup ---
print(f"Using device: {torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else "Using device: CPU")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Hyperparameters ---
input_size = 784  # 28x28 images flattened
hidden_size_1 = 256  # Size of the first hidden layer
hidden_size_2 = 128  # Size of the second hidden layer
num_classes = 10  # Output size: 10 neurons for digits 0-9
num_epochs = 20  # Instruction
batch_size = 64

# A good starting point is 0.001.
# - Try a higher rate like 0.01: might converge faster, but could overshoot and become unstable.
# - Try a lower rate like 0.0001: will learn slower, may require more epochs, but can achieve a more precise result.
learning_rate = 0.0001

# Load MNIST Dataset with Original Labels

transform = transforms.Compose([
    transforms.ToTensor(),  # convert to a tensor and scale to [0, 1]
    transforms.Normalize((0.1307,), (0.3081,)),  # subtract mean and divide by std dev
    transforms.Lambda(lambda x: torch.flatten(x))  # create a 1D vector from all flattened vectors
])

# Load the training and test datasets using the original MNIST class
print("Downloading/Loading MNIST dataset...")
train_data = datasets.MNIST(
    root='../data',
    train=True,
    download=True,  # download the dataset in case it's missing
    transform=transform
)

test_data = datasets.MNIST(
    root='../data',
    train=False,
    download=True,  # download the dataset in case it's missing
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
    shuffle=False
)
print("Dataset loaded.")


# Build the neural network
class MulticlassClassifier(nn.Module):
    def __init__(self, input_size, hidden_1, hidden_2, num_classes):
        super(MulticlassClassifier, self).__init__()
        # Create a sequence of layers
        self.network = nn.Sequential(
            # Input layer to first hidden layer
            nn.Linear(input_size, hidden_1),
            nn.ReLU(),
            # First hidden layer to second hidden layer
            nn.Linear(hidden_1, hidden_2),
            nn.ReLU(),
            # Second hidden layer to the output layer
            nn.Linear(hidden_2, num_classes)
        )
        # Note: We do NOT apply Softmax here because nn.CrossEntropyLoss
        # expects raw, unnormalized scores (logits) as input.
        # It applies the equivalent of LogSoftmax internally for better
        # numerical stability and efficiency.

    def forward(self, x):
        # The forward pass simply passes the input through the sequential network
        return self.network(x)


# Initialize the model and move it to the selected device
model = MulticlassClassifier(input_size, hidden_size_1, hidden_size_2, num_classes).to(device)
print("\nModel Architecture:")
print(model)

# Use Cross-Entropy Loss
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train the Model
print(f"\nStarting training for {num_epochs} epochs...")
# List to store loss values for plotting
losses = []

model.train()  # Set model to training mode
for epoch in range(num_epochs):
    total_loss = 0
    for i, (images, labels) in enumerate(train_loader):
        # Move tensors to the configured device
        images = images.to(device)
        labels = labels.to(device)  # Labels are LongTensors for CrossEntropy

        # Forward pass: get raw logit scores
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    # Calculate and store average loss for the epoch
    avg_loss = total_loss / len(train_loader)
    losses.append(avg_loss)
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}")

print("Training finished.")

# Plot the training loss
print("\nPlotting training loss...")
plt.figure(figsize=(10, 5))
plt.plot(range(1, num_epochs + 1), losses, marker='o')
plt.xlabel("Epoch")
plt.ylabel("Average CrossEntropyLoss")
plt.title(f"Training Loss (LR={learning_rate})")
plt.grid(True)
plt.xticks(range(1, num_epochs + 1))
plt.show()

# Evaluate on the Test Dataset
print("\nEvaluating on test data...")
model.eval()  # Set model to evaluation mode
all_preds = []
all_labels = []

with torch.no_grad():  # Disable gradient calculation for efficiency
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)

        # Get the prediction: the index of the max logit is the predicted class
        # torch.max returns (values, indices)
        _, predicted = torch.max(outputs.data, 1)

        # Append batch results to the lists
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Calculate overall accuracy
accuracy = accuracy_score(all_labels, all_preds)
print(f"Overall Test Accuracy: {accuracy * 100:.2f}%")

# Generate and plot the confusion matrix
print("Generating confusion matrix...")
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=range(num_classes), yticklabels=range(num_classes))
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

# Visualize predictions
print("\nVisualizing some sample predictions...")

# Get one batch of test data to visualize
# We need to reload data without the flatten transform to visualize it as 2D
viz_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
viz_test_data = datasets.MNIST(root='../data', train=False, download=True, transform=viz_transform)
viz_loader = DataLoader(dataset=viz_test_data, batch_size=batch_size, shuffle=True)  # Shuffle to get random samples

images_viz, labels_viz = next(iter(viz_loader))

# Get model predictions
model.eval()
with torch.no_grad():
    # We need to flatten the images before passing them to the model
    images_flat = images_viz.view(-1, 784).to(device)
    outputs_viz = model(images_flat)
    _, preds_viz = torch.max(outputs_viz, 1)

preds_viz_np = preds_viz.cpu().numpy()
labels_viz_np = labels_viz.numpy()

# Create a 4x4 grid for visualization
fig, axes = plt.subplots(4, 4, figsize=(10, 10))

for i, ax in enumerate(axes.flat):
    if i >= len(labels_viz_np):  # Handle batches smaller than 16
        break

    # Display the image
    ax.imshow(images_viz[i].squeeze(), cmap='gray')

    true_label = labels_viz_np[i]
    pred_label = preds_viz_np[i]

    # Set title with prediction and true label
    title = f"True: {true_label} | Pred: {pred_label}"

    # Color the title green for correct, red for incorrect
    color = "green" if pred_label == true_label else "red"

    ax.set_title(title, color=color, fontsize=12)
    ax.axis('off')  # Hide axes ticks

plt.tight_layout()
plt.suptitle("Sample Test Predictions", y=1.02, fontsize=16)
plt.show()

print("\nExercise 1.2 complete.")

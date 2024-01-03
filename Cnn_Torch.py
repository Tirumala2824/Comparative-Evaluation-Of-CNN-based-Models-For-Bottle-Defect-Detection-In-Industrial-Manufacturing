import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
from PIL import Image
import os
import torch.nn.functional as F

# Define Custom Dataset class
class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.dataset = datasets.ImageFolder(root=root_dir, transform=transform)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

# Define SimpleCNN model
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.25)  # Adding dropout
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = self.dropout(x)  # Apply dropout
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Set parameters and transformations
root_directory = "D://python//Flask-2//Bottle_Fault_Detection//images"
batch_size = 32
num_classes = 9

transform_train = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.RandomHorizontalFlip(),  # Augmentation: Randomly flip images horizontally
    transforms.RandomRotation(10),      # Augmentation: Randomly rotate images
    transforms.ToTensor(),
])

transform_test = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
])

# Create datasets and data loaders
train_dataset = CustomDataset(root_directory, transform=transform_train)
test_dataset = CustomDataset(root_directory, transform=transform_test)

train_size = int(0.8 * len(train_dataset))
val_size = len(train_dataset) - train_size

train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

# Initialize the model, optimizer, and loss function
model = SimpleCNN(num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Learning rate scheduler
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)  # Decay LR by a factor of 0.1 every 7 epochs

# Training loop with validation and learning rate scheduling
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    scheduler.step()  # Update learning rate

    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0

    for inputs, labels in val_loader:
        outputs = model(inputs)
        _, predicted = outputs.max(1)
        val_total += labels.size(0)
        val_correct += predicted.eq(labels).sum().item()

    val_accuracy = 100 * val_correct / val_total
    print(f"Epoch [{epoch + 1}/{num_epochs}] - Validation Accuracy: {val_accuracy:.2f}%")

# Testing the model on the test set
model.eval()
test_correct = 0
test_total = 0

for inputs, labels in test_loader:
    outputs = model(inputs)
    _, predicted = outputs.max(1)
    test_total += labels.size(0)
    test_correct += predicted.eq(labels).sum().item()

test_accuracy = 100 * test_correct / test_total
print(f"Test Accuracy: {test_accuracy:.2f}%")

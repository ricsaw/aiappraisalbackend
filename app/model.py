import os
import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

# Config
DATA_DIR = 'app/dataset'
NUM_CLASSES = 5
BATCH_SIZE = 16
EPOCHS = 10
LEARNING_RATE = 1e-4
MODEL_SAVE_PATH = 'app/models/grader_classifier.pt'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], 
                         [0.229, 0.224, 0.225])
])

# Dataset
dataset = datasets.ImageFolder(DATA_DIR, transform=transform)

# Train/val split
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# Model
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
model = model.to(DEVICE)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

class LabelSmoothingLoss(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
        
    def forward(self, pred, target):
        mse = nn.MSELoss()(pred, target)
        # Add small random noise to make model less overconfident
        smooth_target = target + torch.randn_like(target) * self.smoothing
        smooth_loss = nn.MSELoss()(pred, smooth_target)
        return 0.8 * mse + 0.2 * smooth_loss
    
# 2. Add dropout for regularization
class GradeRegressionNet(nn.Module):
    def __init__(self, base_model):
        super(GradeRegressionNet, self).__init__()
        self.backbone = nn.Sequential(*list(base_model.children())[:-1])
        in_features = base_model.fc.in_features
        self.dropout = nn.Dropout(0.3)
        self.regressor = nn.Linear(in_features, 1)

    def forward(self, x):
        x = self.backbone(x).view(x.size(0), -1)
        x = self.dropout(x)
        return self.regressor(x).squeeze()

# Training loop
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {avg_loss:.4f}")

    # Validation accuracy
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    acc = correct / total
    print(f"Validation Accuracy: {acc:.2%}")

# Save model
torch.save(model.state_dict(), MODEL_SAVE_PATH)
print(f"Model saved to {MODEL_SAVE_PATH}")

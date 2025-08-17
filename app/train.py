import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from PIL import Image, UnidentifiedImageError
from torch.utils.data import random_split


# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Parameters
data_dir = "./dataset"
batch_size = 32
num_epochs = 10
learning_rate = 0.001

# Transformations
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Custom ImageFolder to skip unreadable files and "ungraded" folder
class SafeImageFolder(ImageFolder):
    def find_classes(self, directory):
        # Override to skip "ungraded" class
        classes = [d.name for d in os.scandir(directory) if d.is_dir() and d.name.lower() != "ungraded"]
        classes.sort()
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx

    def __getitem__(self, index):
        path, target = self.samples[index]
        try:
            sample = self.loader(path)
            if self.transform is not None:
                sample = self.transform(sample)
            if self.target_transform is not None:
                target = self.target_transform(target)
            return sample, target
        except (UnidentifiedImageError, OSError):
            # Skip corrupted/unreadable file
            return self.__getitem__((index + 1) % len(self.samples))

# Dataset
dataset = SafeImageFolder(root=data_dir, transform=transform)
class_names = dataset.classes
class_counts = Counter([label for _, label in dataset])
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(dataset.targets), y=dataset.targets)
weights = [class_weights[label] for _, label in dataset]
sampler = WeightedRandomSampler(weights, len(weights))

# DataLoader
data_loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)

# Model
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, len(class_names))
model = model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Training
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    all_confidences = []

    for images, labels in data_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        probs = F.softmax(outputs, dim=1)
        confidences, preds = torch.max(probs, 1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_confidences.extend(confidences.detach().cpu().numpy())

    epoch_loss = running_loss / len(data_loader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    mean_confidence = np.mean(all_confidences)
    mean_uncertainty = 1.0 - mean_confidence

    print(f"Epoch [{epoch+1}/{num_epochs}] "
          f"- Loss: {epoch_loss:.4f} "
          f"- Accuracy: {epoch_acc:.4f} "
          f"- Confidence: {mean_confidence:.4f} "
          f"- Uncertainty: {mean_uncertainty:.4f}")

# Save the trained model checkpoint
checkpoint = {
    'epoch': num_epochs,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'class_names': class_names
}
torch.save(checkpoint, "models/grader_classifier.pt")
print("Model checkpoint saved as models/grader_classifier.pt")

# Evaluation
print("\nClassification Report:")
print(classification_report(all_labels, all_preds, target_names=class_names))

cm = confusion_matrix(all_labels, all_preds)
sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()
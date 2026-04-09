import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# ------------------- DATASET -------------------
class CryDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.files = []
        self.labels = []
        self.transform = transform
        for label, folder in enumerate(os.listdir(root_dir)):
            folder_path = os.path.join(root_dir, folder)
            for file in os.listdir(folder_path):
                if file.endswith(".png"):
                    self.files.append(os.path.join(folder_path, file))
                    self.labels.append(label)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = self.files[idx]
        label = self.labels[idx]
        image = Image.open(file_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

# ------------------- SETTINGS -------------------
DATA_PATH = r"C:\Users\E-304\Desktop\Baby_cry\merge_data\mel_dataset_fixed"
BATCH_SIZE = 16
EPOCHS = 25
LEARNING_RATE = 1e-4

# ------------------- TRANSFORMS -------------------
train_transforms = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

val_transforms = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
])

# ------------------- DATA LOADERS -------------------
full_dataset = CryDataset(DATA_PATH, transform=train_transforms)
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
val_dataset.dataset.transform = val_transforms

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ------------------- MODEL -------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, len(os.listdir(DATA_PATH)))
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# ------------------- FUNCTIONS -------------------
def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    acc = 100 * correct / total
    return acc, all_labels, all_preds

# ------------------- TRAINING LOOP -------------------
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0
    loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{EPOCHS}]", colour="green")
    for images, labels in loop:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        loop.set_postfix(loss=loss.item())
    val_acc, _, _ = evaluate(model, val_loader)
    print(f"\n✅ Epoch {epoch+1}: Loss={running_loss:.4f}, Val Accuracy={val_acc:.2f}%\n")

# ------------------- FINAL EVALUATION -------------------
val_acc, y_true, y_pred = evaluate(model, val_loader)
print(f"\n🔥 Final Validation Accuracy: {val_acc:.2f}%\n")

# ------------------- CONFUSION MATRIX -------------------
cm = confusion_matrix(y_true, y_pred)
class_names = os.listdir(DATA_PATH)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Greens", xticklabels=class_names, yticklabels=class_names)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# ------------------- CLASSIFICATION REPORT -------------------
print("\n📊 Classification Report:\n")
print(classification_report(y_true, y_pred))

# ------------------- SAVE MODEL -------------------
torch.save(model.state_dict(), "resnet50_baby_cry.pth")
print("\n💾 Model saved as resnet50_baby_cry.pth")
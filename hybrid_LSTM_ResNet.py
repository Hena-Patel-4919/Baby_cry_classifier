import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# ------------------- PATHS -------------------
DATA_PATH = r"C:\Users\E-304\Desktop\Baby_cry\merge_data\mel_dataset_fixed"
REPORT_PATH = r"C:\Users\E-304\Desktop\Baby_cry\reports"
MODEL_PATH = r"C:\Users\E-304\Desktop\Baby_cry\model_merged"

os.makedirs(REPORT_PATH, exist_ok=True)
os.makedirs(MODEL_PATH, exist_ok=True)

# ------------------- SETTINGS -------------------
BATCH_SIZE = 16
EPOCHS = 30
LR = 1e-4

# ------------------- DATASET -------------------
class CryDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.files = []
        self.labels = []
        self.transform = transform

        self.classes = sorted(os.listdir(root_dir))
        for label, folder in enumerate(self.classes):
            folder_path = os.path.join(root_dir, folder)
            for file in os.listdir(folder_path):
                if file.endswith(".png"):
                    self.files.append(os.path.join(folder_path, file))
                    self.labels.append(label)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        image = Image.open(self.files[idx]).convert("RGB")
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

# ------------------- TRANSFORMS -------------------
train_tf = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
])

val_tf = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
])

# ------------------- LOADERS -------------------
dataset = CryDataset(DATA_PATH, transform=train_tf)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])
val_ds.dataset.transform = val_tf

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

# ------------------- FOCAL LOSS -------------------
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss()

    def forward(self, inputs, targets):
        ce_loss = self.ce(inputs, targets)
        pt = torch.exp(-ce_loss)
        loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return loss

# ------------------- ATTENTION -------------------
class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        weights = torch.softmax(self.attn(x), dim=1)
        return torch.sum(weights * x, dim=1)

# ------------------- HYBRID MODEL -------------------
class ResNet_LSTM_Attention(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        base = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.cnn = nn.Sequential(*list(base.children())[:-2])

        self.lstm = nn.LSTM(
            input_size=2048,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )

        self.attn = Attention(512)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.cnn(x)          # (B, 2048, H, W)
        x = x.mean(dim=2)        # (B, 2048, W)
        x = x.permute(0, 2, 1)   # (B, W, 2048)

        lstm_out, _ = self.lstm(x)
        x = self.attn(lstm_out)

        return self.fc(x)

# ------------------- INIT -------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ResNet_LSTM_Attention(len(dataset.classes)).to(device)
criterion = FocalLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# ------------------- EVALUATION -------------------
def evaluate(model, loader):
    model.eval()
    correct, total = 0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            out = model(imgs)
            preds = torch.argmax(out, dim=1)

            total += labels.size(0)
            correct += (preds == labels).sum().item()

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return 100 * correct / total, all_labels, all_preds

# ------------------- TRAIN -------------------
log_file = os.path.join(REPORT_PATH, "training_log.txt")

with open(log_file, "w") as f:
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0

        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for imgs, labels in loop:
            imgs, labels = imgs.to(device), labels.to(device)

            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        acc, _, _ = evaluate(model, val_loader)

        log = f"Epoch {epoch+1} | Loss: {total_loss:.4f} | Val Acc: {acc:.2f}%"
        print(log)
        f.write(log + "\n")

# ------------------- FINAL -------------------
acc, y_true, y_pred = evaluate(model, val_loader)

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Greens",
            xticklabels=dataset.classes,
            yticklabels=dataset.classes)

cm_path = os.path.join(REPORT_PATH, "confusion_matrix.png")
plt.savefig(cm_path)
plt.close()

# Classification report
report = classification_report(y_true, y_pred)
with open(os.path.join(REPORT_PATH, "classification_report.txt"), "w") as f:
    f.write(report)

# Save model
model_name = f"resnet_lstm_attention_focal_{acc:.2f}.pth"
model_path = os.path.join(MODEL_PATH, model_name)

torch.save(model.state_dict(), model_path)

print("\n✅ Saved everything!")
print(f"📊 Reports → {REPORT_PATH}")
print(f"💾 Model → {model_path}")
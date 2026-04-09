import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import tensorflow as tf
from tensorflow.keras import layers, models
from PIL import Image

# =========================
# PATH
# =========================
data_path = r"C:\Users\E-304\Desktop\Baby_cry\merge_data\mel_dataset_fixed"

IMG_SIZE = 128

# =========================
# LOAD DATA
# =========================
X = []
y = []
labels = []

print("Loading dataset...")

for label_idx, label in enumerate(os.listdir(data_path)):
    
    label_path = os.path.join(data_path, label)
    
    if not os.path.isdir(label_path):
        continue
    
    labels.append(label)
    
    for file in os.listdir(label_path):
        if file.endswith(".png"):
            img_path = os.path.join(label_path, file)
            
            img = Image.open(img_path).convert("L")  # grayscale
            img = img.resize((IMG_SIZE, IMG_SIZE))
            img = np.array(img) / 255.0
            
            X.append(img)
            y.append(label_idx)

X = np.array(X)
y = np.array(y)

# Add channel dimension
X = X.reshape(-1, IMG_SIZE, IMG_SIZE, 1)

print("Data shape:", X.shape)

# =========================
# TRAIN TEST SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# =========================
# CNN MODEL
# =========================
model = models.Sequential()

model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(128,128,1)))
model.add(layers.MaxPooling2D((2,2)))

model.add(layers.Conv2D(64, (3,3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))

model.add(layers.Conv2D(128, (3,3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))

model.add(layers.Flatten())

model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dropout(0.3))

model.add(layers.Dense(len(labels), activation='softmax'))

# =========================
# COMPILE
# =========================
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# =========================
# TRAIN
# =========================
history = model.fit(
    X_train, y_train,
    epochs=25,
    batch_size=16,
    validation_data=(X_test, y_test)
)

# =========================
# EVALUATE
# =========================
loss, acc = model.evaluate(X_test, y_test)

print(f"\n✅ Test Accuracy: {acc * 100:.2f}%")

# =========================
# PREDICTIONS
# =========================
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

# =========================
# CONFUSION MATRIX
# =========================
cm = confusion_matrix(y_test, y_pred_classes)

plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=labels, yticklabels=labels)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# =========================
# CLASSIFICATION REPORT
# =========================
print("\n📊 Classification Report:\n")
print(classification_report(y_test, y_pred_classes, target_names=labels))


model.save(r"C:\Users\E-304\Desktop\Baby_cry\model_merged\cnn_model.h5")
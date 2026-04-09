import os
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical

#  NEW: Callbacks for logging
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint

# ==========================
# PATHS
# ==========================
mfcc_path = r"C:\Users\E-304\Desktop\Baby_cry\merge_data\processed\mfcc"

model_save_path = r"C:\Users\E-304\Desktop\Baby_cry\model_merged"
log_save_path = r"C:\Users\E-304\Desktop\Baby_cry\reports"

# Create folders if not exist
os.makedirs(model_save_path, exist_ok=True)
os.makedirs(log_save_path, exist_ok=True)

model_name = "baby_cry_lstm_mfcc"

# ==========================
# LOAD DATA
# ==========================
X = []
y = []

for label in os.listdir(mfcc_path):

    label_path = os.path.join(mfcc_path, label)

    if os.path.isdir(label_path):

        for file in os.listdir(label_path):

            if file.endswith(".npy"):

                file_path = os.path.join(label_path, file)

                mfcc = np.load(file_path)

                X.append(mfcc)
                y.append(label)

X = np.array(X)
y = np.array(y)

print("Total samples:", len(X))

# ==========================
# ENCODE LABELS
# ==========================
le = LabelEncoder()
y_encoded = le.fit_transform(y)

y_cat = to_categorical(y_encoded)

# ==========================
# TRAIN / VALIDATION SPLIT
# ==========================
X_train, X_val, y_train, y_val = train_test_split(
    X, y_cat, test_size=0.2, random_state=42, stratify=y_encoded
)

# ==========================
# RESHAPE FOR LSTM
# ==========================
X_train = np.transpose(X_train, (0, 2, 1))
X_val = np.transpose(X_val, (0, 2, 1))

print("Shape:", X_train.shape)

# ==========================
# MODEL
# ==========================
model = Sequential()

model.add(LSTM(128, return_sequences=True,
               input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.3))

model.add(LSTM(64))
model.add(Dropout(0.3))

model.add(Dense(64, activation='relu'))
model.add(Dense(len(le.classes_), activation='softmax'))

# ==========================
# COMPILE
# ==========================
model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# ==========================
# ✅ LOGGING SETUP
# ==========================

# CSV log file (epoch-wise logs)
csv_log_path = os.path.join(log_save_path, model_name + "_training_log.csv")
csv_logger = CSVLogger(csv_log_path, append=False)

# Save best model checkpoint
checkpoint_path = os.path.join(model_save_path, model_name + "_best.keras")
checkpoint = ModelCheckpoint(
    filepath=checkpoint_path,
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)

# ==========================
# TRAIN
# ==========================
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=25,
    batch_size=32,
    callbacks=[csv_logger, checkpoint]
)

# ==========================
# SAVE FINAL MODEL
# ==========================
final_model_path = os.path.join(model_save_path, model_name + ".keras")
model.save(final_model_path)

# ==========================
# SAVE LABEL CLASSES (IMPORTANT)
# ==========================
label_path = os.path.join(model_save_path, model_name + "_labels.npy")
np.save(label_path, le.classes_)

print(" Model saved at:", final_model_path)
print(" Best model saved at:", checkpoint_path)
print(" Logs saved at:", csv_log_path)
print(" Labels saved at:", label_path)
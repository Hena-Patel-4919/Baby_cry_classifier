# import os
# import random
# import shutil
# import librosa
# import numpy as np
# import soundfile as sf

# # =========================
# # PATHS
# # =========================
# input_path = r"C:\Users\E-304\Desktop\Baby_cry\merge_data\merged_file"
# output_path = r"C:\Users\E-304\Desktop\Baby_cry\merge_data\final_balanced_500"

# TARGET = 500

# os.makedirs(output_path, exist_ok=True)

# # =========================
# # AUGMENTATION FUNCTION
# # =========================
# def augment_audio(y, sr):
#     choice = random.choice(["noise", "pitch", "stretch"])

#     if choice == "noise":
#         y = y + 0.005 * np.random.randn(len(y))

#     elif choice == "pitch":
#         y = librosa.effects.pitch_shift(y, sr=sr, n_steps=random.randint(-2, 2))

#     elif choice == "stretch":
#         y = librosa.effects.time_stretch(y, rate=random.uniform(0.8, 1.2))

#     return y

# # =========================
# # PROCESS EACH CLASS
# # =========================
# for label in os.listdir(input_path):

#     class_path = os.path.join(input_path, label)

#     if not os.path.isdir(class_path):
#         continue

#     files = [f for f in os.listdir(class_path) if f.endswith(".wav")]

#     # 🚨 skip empty classes
#     if len(files) == 0:
#         print(f"⚠️ Skipping {label} (no data)")
#         continue

#     print(f"\nProcessing {label}: {len(files)}")

#     out_class = os.path.join(output_path, label)
#     os.makedirs(out_class, exist_ok=True)

#     # =========================
#     # 🔻 DOWNSAMPLE
#     # =========================
#     if len(files) > TARGET:
#         selected = random.sample(files, TARGET)

#         for f in selected:
#             shutil.copy2(os.path.join(class_path, f),
#                          os.path.join(out_class, f))

#     # =========================
#     # 🔺 UPSAMPLE
#     # =========================
#     else:
#         # copy originals
#         for f in files:
#             shutil.copy2(os.path.join(class_path, f),
#                          os.path.join(out_class, f))

#         count = len(files)

#         while count < TARGET:
#             f = random.choice(files)
#             path = os.path.join(class_path, f)

#             y, sr = librosa.load(path, sr=None)
#             y_aug = augment_audio(y, sr)

#             new_name = f"aug_{count}.wav"
#             new_path = os.path.join(out_class, new_name)

#             sf.write(new_path, y_aug, sr)

#             count += 1

# # =========================
# # FINAL CHECK
# # =========================
# print("\n📊 FINAL DATASET COUNT:\n")

# for label in os.listdir(output_path):
#     count = len(os.listdir(os.path.join(output_path, label)))
#     print(f"{label} : {count}")

import os
from collections import defaultdict

# =========================
# PATH
# =========================
dataset_path = r"C:\Users\E-304\Desktop\Baby_cry\merge_data\final_balanced_500"

# =========================
# GLOBAL COUNTER
# =========================
total_counts = defaultdict(int)

print("\n📊 FILE FORMAT CHECK:\n")

# =========================
# LOOP THROUGH CLASSES
# =========================
for label in os.listdir(dataset_path):
    
    class_path = os.path.join(dataset_path, label)

    if not os.path.isdir(class_path):
        continue

    format_count = defaultdict(int)

    for file in os.listdir(class_path):
        ext = os.path.splitext(file)[1].lower()
        format_count[ext] += 1
        total_counts[ext] += 1

    print(f"\n🔹 {label}")
    for ext, count in format_count.items():
        print(f"{ext} : {count}")

# =========================
# TOTAL SUMMARY
# =========================
print("\n📊 TOTAL FILE FORMAT DISTRIBUTION:\n")

for ext, count in total_counts.items():
    print(f"{ext} : {count}")
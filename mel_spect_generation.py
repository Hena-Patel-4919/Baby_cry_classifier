# import os
# import librosa
# import numpy as np
# import matplotlib.pyplot as plt
# from tqdm import tqdm

# # =========================
# # PATHS
# # =========================
# input_path = r"C:\Users\E-304\Desktop\Baby_cry\merge_data\final_balanced_500"
# output_path = r"C:\Users\E-304\Desktop\Baby_cry\merge_data\mel_dataset"

# os.makedirs(output_path, exist_ok=True)

# # =========================
# # PARAMETERS
# # =========================
# SR = 22050
# DURATION = 3
# IMG_SIZE = (128, 128)

# # =========================
# # PROCESS
# # =========================
# for label in os.listdir(input_path):

#     class_path = os.path.join(input_path, label)
#     out_class = os.path.join(output_path, label)

#     os.makedirs(out_class, exist_ok=True)

#     print(f"\nProcessing {label}...")

#     for file in tqdm(os.listdir(class_path)):

#         if not file.endswith(".wav"):
#             continue

#         file_path = os.path.join(class_path, file)

#         try:
#             # Load
#             y, sr = librosa.load(file_path, sr=SR)

#             # Fix length (3 sec)
#             max_len = SR * DURATION
#             if len(y) > max_len:
#                 y = y[:max_len]
#             else:
#                 y = np.pad(y, (0, max_len - len(y)))

#             # Normalize
#             y = librosa.util.normalize(y)

#             # Mel Spectrogram
#             mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)

#             # Convert to dB (VERY IMPORTANT)
#             mel_db = librosa.power_to_db(mel, ref=np.max)

#             # Resize to 128x128
#             mel_db = np.resize(mel_db, IMG_SIZE)

#             # Save as image
#             save_path = os.path.join(out_class, file.replace(".wav", ".png"))
#             plt.imsave(save_path, mel_db, cmap='viridis')

#         except Exception as e:
#             print("Error:", file, e)

# print("\n✅ Mel Spectrogram Dataset Ready!")


import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm

# =========================
# PATHS
# =========================
input_path = r"C:\Users\E-304\Desktop\Baby_cry\merge_data\final_balanced_500"
output_path = r"C:\Users\E-304\Desktop\Baby_cry\merge_data\mel_dataset_fixed"

os.makedirs(output_path, exist_ok=True)

# =========================
# PARAMETERS
# =========================
SR = 22050
DURATION = 3
IMG_SIZE = (128, 128)

# =========================
# PROCESS
# =========================
for label in os.listdir(input_path):

    class_path = os.path.join(input_path, label)
    out_class = os.path.join(output_path, label)

    os.makedirs(out_class, exist_ok=True)

    print(f"\nProcessing {label}...")

    for file in tqdm(os.listdir(class_path)):

        if not file.endswith(".wav"):
            continue

        file_path = os.path.join(class_path, file)

        try:
            # Load audio
            y, sr = librosa.load(file_path, sr=SR)

            # Fix length
            max_len = SR * DURATION
            if len(y) > max_len:
                y = y[:max_len]
            else:
                y = np.pad(y, (0, max_len - len(y)))

            # Normalize
            y = librosa.util.normalize(y)

            # Mel Spectrogram
            mel = librosa.feature.melspectrogram(
                y=y, sr=sr, n_mels=128, fmax=8000
            )

            mel_db = librosa.power_to_db(mel, ref=np.max)

            # 🔥 NORMALIZE IMAGE (IMPORTANT)
            mel_db = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min())

            # 🔥 PROPER RESIZE (NOT np.resize)
            mel_img = cv2.resize(mel_db, IMG_SIZE)

            # Save image
            save_path = os.path.join(out_class, file.replace(".wav", ".png"))
            plt.imsave(save_path, mel_img, cmap='viridis')

        except Exception as e:
            print("Error:", file, e)

print("\n✅ Fixed Mel Spectrogram Dataset Ready!")
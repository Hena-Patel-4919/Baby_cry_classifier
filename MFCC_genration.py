import os
import numpy as np
import librosa

# ==========================
# PATHS
# ==========================
dataset_path = r"C:\Users\E-304\Desktop\Baby_cry\merge_data\final_balanced_500"
output_path = r"C:\Users\E-304\Desktop\Baby_cry\merge_data\processed\mfcc"

# ==========================
# CREATE OUTPUT FOLDER
# ==========================
os.makedirs(output_path, exist_ok=True)

# ==========================
# MFCC FUNCTION
# ==========================
def extract_mfcc(file_path):
    try:
        signal, sr = librosa.load(file_path, sr=22050)

        mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=40)

        max_len = 100

        if mfcc.shape[1] < max_len:
            pad_width = max_len - mfcc.shape[1]
            mfcc = np.pad(mfcc, pad_width=((0,0),(0,pad_width)), mode='constant')
        else:
            mfcc = mfcc[:, :max_len]

        return mfcc

    except Exception as e:
        print("Error:", file_path)
        return None

# ==========================
# PROCESS DATASET
# ==========================
count = 0

for label in os.listdir(dataset_path):

    label_path = os.path.join(dataset_path, label)

    if os.path.isdir(label_path):

        save_label_path = os.path.join(output_path, label)
        os.makedirs(save_label_path, exist_ok=True)

        for file in os.listdir(label_path):

            if file.endswith((".wav", ".ogg")):

                file_path = os.path.join(label_path, file)

                mfcc = extract_mfcc(file_path)

                if mfcc is not None:

                    save_name = file.replace(".wav", ".npy").replace(".ogg", ".npy")
                    save_path = os.path.join(save_label_path, save_name)

                    np.save(save_path, mfcc)

                    count += 1

print("✅ MFCC SAVED SUCCESSFULLY")
print("Total files processed:", count)
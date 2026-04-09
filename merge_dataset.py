import os
import shutil
from collections import defaultdict

# ==============================
# 📂 DATASET PATHS
# ==============================
dataset_paths = [
    r"C:\Users\E-304\Desktop\Baby_cry\merge_data\Baby Cry Sence Dataset",
    r"C:\Users\E-304\Desktop\Baby_cry\merge_data\baby_crying_sound_1",
    r"C:\Users\E-304\Desktop\Baby_cry\merge_data\cry",
    r"C:\Users\E-304\Desktop\Baby_cry\merge_data\donateacry_corpus"
]

# Output merged dataset path
output_path = r"C:\Users\E-304\Desktop\Baby_cry\merge_data\merged_file"

# ==============================
# 🏷️ LABEL STANDARDIZATION
# ==============================
label_map = {
    "belly_pain": "belly pain",
    "bellypain": "belly pain",
    "burping": "burping",
    "cold": "cold_hot",
    "hot": "cold_hot",
    "cold_hot": "cold_hot",
    "discomfort": "discomfort",
    "hungry": "hungry",
    "laugh": "laugh",
    "noise": "noise",
    "silence": "silence",
    "tired": "tired",
    "scared": "discomfort",
    "lonely": "discomfort"
}

# ==============================
# 📊 CLASS COUNTER
# ==============================
class_counts = defaultdict(int)

# Create output folder
os.makedirs(output_path, exist_ok=True)

# ==============================
# 🔁 MERGING PROCESS
# ==============================
for dataset in dataset_paths:
    for root, dirs, files in os.walk(dataset):
        for file in files:
            if file.endswith((".wav", ".mp3", ".ogg")):
                
                # Get label from folder name
                label = os.path.basename(root).lower().replace("_", "").replace(" ", "")
                
                # Map label to standard label
                mapped_label = label_map.get(label, None)
                
                if mapped_label is None:
                    continue  # skip unknown labels
                
                # Create label folder in output
                label_folder = os.path.join(output_path, mapped_label)
                os.makedirs(label_folder, exist_ok=True)
                
                # Source and destination paths
                src_file = os.path.join(root, file)
                dst_file = os.path.join(label_folder, file)
                
                # Handle duplicate filenames
                base, ext = os.path.splitext(file)
                counter = 1
                while os.path.exists(dst_file):
                    dst_file = os.path.join(label_folder, f"{base}_{counter}{ext}")
                    counter += 1
                
                # Copy file
                shutil.copy2(src_file, dst_file)
                
                # Update count
                class_counts[mapped_label] += 1

# ==============================
# 📊 FINAL OUTPUT
# ==============================
print("\n📊 FINAL MERGED CLASS DISTRIBUTION:\n")

total_files = 0
for label, count in sorted(class_counts.items()):
    print(f"{label} : {count}")
    total_files += count

print("\n✅ TOTAL FILES:", total_files)
print(f"\n📁 Merged dataset created at:\n{output_path}")
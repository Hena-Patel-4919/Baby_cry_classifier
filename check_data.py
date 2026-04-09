import os

# ✅ USE RAW STRING (VERY IMPORTANT)
dataset_path = r"data_1_WO_balancing\raw\baby_crying_sound"

class_counts = {}

for label in os.listdir(dataset_path):
    label_path = os.path.join(dataset_path, label)
    
    if os.path.isdir(label_path):
        count = len(os.listdir(label_path))
        class_counts[label] = count

print("\n📊 Class Distribution:\n")

for k, v in class_counts.items():
    print(f"{k} : {v}")
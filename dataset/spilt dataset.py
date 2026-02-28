import os
import random
import shutil

random.seed(42)

SOURCE_DIR = r"C:\Users\babpl\OneDrive\Desktop\is_project\dataset\images"
TARGET_DIR = r"C:\Users\babpl\OneDrive\Desktop\is_project\dataset"

TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

for breed in os.listdir(SOURCE_DIR):
    breed_path = os.path.join(SOURCE_DIR, breed)
    if not os.path.isdir(breed_path):
        continue

    imgs = os.listdir(breed_path)
    random.shuffle(imgs)

    n_total = len(imgs)
    n_train = int(n_total * TRAIN_RATIO)
    n_val = int(n_total * VAL_RATIO)

    splits = {
        "train": imgs[:n_train],
        "val": imgs[n_train:n_train + n_val],
        "test": imgs[n_train + n_val:]
    }

    for split, files in splits.items():
        target_dir = os.path.join(TARGET_DIR, split, breed)
        os.makedirs(target_dir, exist_ok=True)

        for f in files:
            shutil.copy(
                os.path.join(breed_path, f),
                os.path.join(target_dir, f)
            )

print("✅ แยก dataset สำเร็จแล้ว")
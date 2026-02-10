import os
import shutil
import random
import glob

# Config
BASE_DIR = r"..\datasets"
IMAGES_TRAIN = os.path.join(BASE_DIR, "images", "train")
IMAGES_VAL = os.path.join(BASE_DIR, "images", "val")
LABELS_TRAIN = os.path.join(BASE_DIR, "labels", "train")
LABELS_VAL = os.path.join(BASE_DIR, "labels", "val")
SPLIT_RATIO = 0.2

def split_dataset():
    # Ensure val directories exist
    os.makedirs(IMAGES_VAL, exist_ok=True)
    os.makedirs(LABELS_VAL, exist_ok=True)

    # Get all images in train
    images = glob.glob(os.path.join(IMAGES_TRAIN, "*.*"))
    
    # Filter out label files if they accidentally got into images folder (just to be safe)
    images = [img for img in images if not img.endswith('.txt') and not img.endswith('.classes')]

    random.seed(42) # For reproducibility
    random.shuffle(images)

    num_val = int(len(images) * SPLIT_RATIO)
    val_images = images[:num_val]

    print(f"Total: {len(images)}. Moving {len(val_images)} to validation.")

    count = 0
    for img_path in val_images:
        filename = os.path.basename(img_path)
        name_no_ext = os.path.splitext(filename)[0]

        # 1. Move Image
        dest_img_path = os.path.join(IMAGES_VAL, filename)
        shutil.move(img_path, dest_img_path)

        # 2. Move Label (if valid)
        label_filename = name_no_ext + ".txt"
        src_label_path = os.path.join(LABELS_TRAIN, label_filename)
        if os.path.exists(src_label_path):
            dest_label_path = os.path.join(LABELS_VAL, label_filename)
            shutil.move(src_label_path, dest_label_path)
        
        count += 1

    print(f"Moved {count} items to validaton set.")

if __name__ == "__main__":
    split_dataset()

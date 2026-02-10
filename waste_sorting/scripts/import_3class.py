import os
import glob
import shutil
import random

# Source folders
SOURCE_ROOT = r"C:\Users\hoang\Downloads\trashnet-master\trashnet-master\data\dataset-resized\dataset-resized"

# Destination
TRAIN_IMG = r"..\datasets\images\train"
VAL_IMG = r"..\datasets\images\val"
TRAIN_LABEL = r"..\datasets\labels\train"
VAL_LABEL = r"..\datasets\labels\val"

# Class mapping: folder -> class_id
# plastic = 0, metal = 1, other (glass) = 2
CLASS_MAP = {
    "plastic": 0,
    "metal": 1,
    "glass": 2  # glass bottles -> other
}

# How many images to take from each class
MAX_IMAGES_PER_CLASS = 150

def import_and_label():
    # Create directories
    for d in [TRAIN_IMG, VAL_IMG, TRAIN_LABEL, VAL_LABEL]:
        os.makedirs(d, exist_ok=True)
    
    all_images = []
    
    for folder, class_id in CLASS_MAP.items():
        src_path = os.path.join(SOURCE_ROOT, folder)
        if not os.path.exists(src_path):
            print(f"Warning: {src_path} not found!")
            continue
        
        images = glob.glob(os.path.join(src_path, "*.jpg"))
        # Limit number of images
        images = images[:MAX_IMAGES_PER_CLASS]
        
        print(f"Found {len(images)} images in {folder} (class {class_id})")
        
        for img_path in images:
            all_images.append((img_path, class_id, folder))
    
    # Shuffle
    random.shuffle(all_images)
    
    # 80/20 split
    split_idx = int(len(all_images) * 0.8)
    train_images = all_images[:split_idx]
    val_images = all_images[split_idx:]
    
    print(f"\nTotal: {len(all_images)} images")
    print(f"Train: {len(train_images)}, Val: {len(val_images)}")
    
    # Copy and label
    for img_path, class_id, folder in train_images:
        basename = os.path.splitext(os.path.basename(img_path))[0]
        new_name = f"{folder}_{basename}"
        
        # Copy image
        shutil.copy2(img_path, os.path.join(TRAIN_IMG, f"{new_name}.jpg"))
        
        # Create label
        with open(os.path.join(TRAIN_LABEL, f"{new_name}.txt"), 'w', encoding='utf-8', newline='\n') as f:
            f.write(f"{class_id} 0.5 0.5 0.9 0.9\n")
    
    for img_path, class_id, folder in val_images:
        basename = os.path.splitext(os.path.basename(img_path))[0]
        new_name = f"{folder}_{basename}"
        
        # Copy image
        shutil.copy2(img_path, os.path.join(VAL_IMG, f"{new_name}.jpg"))
        
        # Create label
        with open(os.path.join(VAL_LABEL, f"{new_name}.txt"), 'w', encoding='utf-8', newline='\n') as f:
            f.write(f"{class_id} 0.5 0.5 0.9 0.9\n")
    
    print("\nDone! Images and labels created.")

if __name__ == "__main__":
    import_and_label()

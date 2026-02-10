import os
import glob

# Directories
TRAIN_IMG_DIR = r"..\datasets\images\train"
TRAIN_LABEL_DIR = r"..\datasets\labels\train"
VAL_IMG_DIR = r"..\datasets\images\val"
VAL_LABEL_DIR = r"..\datasets\labels\val"

# Class mapping based on filename prefix
# plastic* -> 0, metal/z* -> 1
def get_class_id(filename):
    if filename.startswith("plastic"):
        return 0
    else:
        return 1  # metal (z* files)

def create_labels(img_dir, label_dir):
    os.makedirs(label_dir, exist_ok=True)
    count = 0
    
    for img_path in glob.glob(os.path.join(img_dir, "*.jpg")):
        basename = os.path.splitext(os.path.basename(img_path))[0]
        class_id = get_class_id(basename)
        
        label_path = os.path.join(label_dir, f"{basename}.txt")
        
        with open(label_path, 'w', encoding='utf-8', newline='\n') as f:
            f.write(f"{class_id} 0.5 0.5 0.9 0.9\n")
        count += 1
    
    return count

if __name__ == "__main__":
    print("Creating labels for training images...")
    train_count = create_labels(TRAIN_IMG_DIR, TRAIN_LABEL_DIR)
    print(f"Created {train_count} labels in train")
    
    print("Creating labels for validation images...")
    val_count = create_labels(VAL_IMG_DIR, VAL_LABEL_DIR)
    print(f"Created {val_count} labels in val")
    
    print("Done!")

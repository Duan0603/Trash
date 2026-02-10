import os
import shutil
import glob

# Config
SOURCE_ROOT = r"C:\Users\hoang\OneDrive\Documents\Archive (3)\Garbage classification\Garbage classification"
DEST_ROOT = r"C:\Users\hoang\OneDrive\Desktop\np\Trash\waste_sorting\datasets\images\train"
# New classes to be mapped to "other"
TARGET_CLASSES = ["cardboard", "glass", "paper", "trash"]

def import_data():
    if not os.path.exists(DEST_ROOT):
        os.makedirs(DEST_ROOT)
        print(f"Created {DEST_ROOT}")

    total_copied = 0

    for cls in TARGET_CLASSES:
        src_path = os.path.join(SOURCE_ROOT, cls)
        # ... logic remains same ...
        if not os.path.exists(src_path):
            print(f"Warning: {src_path} not found!")
            continue

        images = glob.glob(os.path.join(src_path, "*.*")) # Get all files
        print(f"Found {len(images)} images in {cls}")

        for img_path in images:
            filename = os.path.basename(img_path)
            # Rename to avoid collisions: class_filename
            new_filename = f"{cls}_{filename}"
            dest_path = os.path.join(DEST_ROOT, new_filename)
            
            shutil.copy2(img_path, dest_path)
            total_copied += 1
            
            # Optional: Move 20% to val? 
            # For now, let's put everything in train, user can split later or we add logic.
    
    print(f"Successfully copied {total_copied} images to {DEST_ROOT}")

if __name__ == "__main__":
    import_data()

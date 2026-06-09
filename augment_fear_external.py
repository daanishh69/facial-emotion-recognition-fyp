import os, shutil
from PIL import Image

# Path to FER+ fear folder (from old FYP1 dataset)
FERPLUS_FEAR = r"C:\Users\Danish Omar\Downloads\archive (2)\train\fear"

# Path to RAF-DB fear training folder
RAFDB_FEAR   = r"C:\Users\Danish Omar\Desktop\SEM5\FYPPROJECT\data\DATASET\train\fear"

copied = 0
for img_file in os.listdir(FERPLUS_FEAR):
    if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
        src = os.path.join(FERPLUS_FEAR, img_file)
        dst = os.path.join(RAFDB_FEAR, f"ferplus_{img_file}")
        if not os.path.exists(dst):
            # Convert to RGB and save as JPEG to match RAF-DB format
            img = Image.open(src).convert('RGB').resize((100, 100))
            img.save(dst, 'JPEG', quality=90)
            copied += 1

print(f"Copied {copied} FER+ fear images to RAF-DB fear folder")
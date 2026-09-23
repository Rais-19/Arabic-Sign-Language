import os
import shutil
from pathlib import Path

# Your mapping
arabic_map = {
    "aleff": "ا", "bb": "ب", "ta": "ت", "thaa": "ث", "jeem": "ج",
    "haa": "ح", "khaa": "خ", "dal": "د", "thal": "ذ", "ra": "ر",
    "zay": "ز", "seen": "س", "sheen": "ش", "saad": "ص", "dhad": "ض",
    "taa": "ط", "dha": "ظ", "ain": "ع", "ghain": "غ", "fa": "ف",
    "gaaf": "ق", "kaaf": "ك", "laam": "لام", "la": "لا", "meem": "ميم",
    "nun": "ن", "ha": "ه", "waw": "و", "ya": "ي", "yaa": "ي",
    "toot": "ة", "al": "ال"
}

source_dir = r"C:\Users\LENOVO\Desktop\Computer Vision\arabic data\datasets\train\images"
output_dir = r"C:\Users\LENOVO\Desktop\Computer Vision\Arabic sign language\data"

os.makedirs(output_dir, exist_ok=True)

# Get all images
all_files = os.listdir(source_dir)
organized_count = 0

for img_file in all_files:
    if not img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
        continue
    
    # Extract letter name from filename: {ID}_M_{letter_name}_{number}.jpg
    parts = img_file.rsplit('_', 1)[0].split('_')
    letter_name = parts[-1]  # Last part before the number
    
    # Get Arabic letter from map
    if letter_name in arabic_map:
        arabic_letter = arabic_map[letter_name]
        
        # Create letter folder
        letter_folder = os.path.join(output_dir, arabic_letter)
        os.makedirs(letter_folder, exist_ok=True)
        
        # Copy image
        src = os.path.join(source_dir, img_file)
        dst = os.path.join(letter_folder, img_file)
        shutil.copy(src, dst)
        organized_count += 1
        
        if organized_count % 500 == 0:
            print(f"Organized {organized_count} images...")
    else:
        print(f"Unknown letter in {img_file}: {letter_name}")

print(f"\n✓ Done! Organized {organized_count} images into {len(os.listdir(output_dir))} letter folders")
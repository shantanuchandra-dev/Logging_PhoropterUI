
import os
import shutil
import random
from pathlib import Path
from collections import defaultdict

# Configuration
SOURCE_DIRS = [
    "chart_dataset",
    "Charts_Processed",
    "all_the_charts"
]
OUTPUT_DIR = "total_final_charts"
TRAIN_RATIO = 0.8

# Typos and normalization mapping
CLASS_MAPPING = {
    # Fix typos
    "smellen": "snellen",
    "snellenchart": "snellen_chart",
    "snellen _chart": "snellen_chart",
    "snellen chart": "snellen_chart",
    "number chart": "number_chart",
    "echart": "echart",
    "e charts": "echart",
    "alphabetic chart": "snellen_chart", 
    "near vision": "near_vision",
    "pictorial chart": "pictorial_chart",
    "jcc chart": "jcc_chart",
    
    # Fix awkward names discovered during QA
    "echart<600": "echart_600",
    "echartlt600": "echart_600",
    
    # Standardize
    "duochrome": "duochrome",
    
    # Handle spaces
    " ": "_",
    "__": "_",
}

def normalize_class_name(name):
    """
    Normalizes class names by fixing typos and standardizing formatting.
    """
    name = name.lower().strip()
    
    # Specific substitutions
    for bad, good in CLASS_MAPPING.items():
        if bad in name:
            name = name.replace(bad, good)
            
    # Fix spacing/formatting
    name = name.replace(" ", "_").replace("__", "_")
    
    # Handle numeric values with < or >
    name = name.replace("<", "lt").replace(">", "gt")
    
    # Post-process check: if we created echartlt600, force it to echart_600
    if "echartlt600" in name:
        name = name.replace("echartlt600", "echart_600")

    # Final cleanup
    while "__" in name:
        name = name.replace("__", "_")
        
    return name

def consolidate_data():
    if os.path.exists(OUTPUT_DIR):
        print(f"Removing existing output directory: {OUTPUT_DIR}")
        shutil.rmtree(OUTPUT_DIR)
    
    os.makedirs(os.path.join(OUTPUT_DIR, "train"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "val"), exist_ok=True)
    
    # Collect all file paths per normalized class
    files_by_class = defaultdict(list)
    
    for source in SOURCE_DIRS:
        if not os.path.exists(source):
            print(f"Skipping missing source: {source}")
            continue
            
        print(f"Processing source: {source}")
        
        for root, dirs, files in os.walk(source):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                    rel_path = os.path.relpath(root, source)
                    parts = rel_path.split(os.sep)
                    
                    class_candidate = parts[-1]
                    if class_candidate.lower() in ['train', 'val'] and len(parts) > 1:
                        class_candidate = parts[-2]
                        
                    normalized_class = normalize_class_name(class_candidate)
                    
                    if normalized_class == "." or normalized_class == "":
                        continue
                        
                    full_path = os.path.join(root, file)
                    files_by_class[normalized_class].append(full_path)

    print(f"\nFound {len(files_by_class)} unique classes.")
    
    total_images = 0
    for cls, files in files_by_class.items():
        unique_files = list(set(files)) # Remove duplicates by path
        random.shuffle(unique_files)
        
        count = len(unique_files)
        total_images += count
        
        # Ensure at least one image in train and val if possible, or duplicate
        if count == 0:
            continue
            
        if count == 1:
            # Duplicate for robust training/verification
            train_files = unique_files
            val_files = unique_files
        else:
            split_idx = int(count * TRAIN_RATIO)
            if split_idx == 0: split_idx = 1
            if split_idx == count: split_idx = count - 1
                 
            train_files = unique_files[:split_idx]
            val_files = unique_files[split_idx:]
            
        train_dest = os.path.join(OUTPUT_DIR, "train", cls)
        val_dest = os.path.join(OUTPUT_DIR, "val", cls)
        
        os.makedirs(train_dest, exist_ok=True)
        os.makedirs(val_dest, exist_ok=True)
        
        for f in train_files:
            shutil.copy2(f, os.path.join(train_dest, os.path.basename(f)))
            
        for f in val_files:
            shutil.copy2(f, os.path.join(val_dest, os.path.basename(f)))
            
    print(f"\nConsolidation complete. Total images: {total_images}")
    print(f"Output saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    consolidate_data()

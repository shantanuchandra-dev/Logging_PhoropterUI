import os
import shutil
import random
from pathlib import Path
from PIL import Image

def prepare_data(source_dir="_Charts", output_dir="chart_dataset", train_split=0.8):
    """
    Organizes images from _Charts into a standard train/val split.
    Normalizes folder names.
    """
    source_path = Path(source_dir)
    output_path = Path(output_dir)
    
    if output_path.exists():
        shutil.rmtree(output_path)
    
    classes = [d for d in source_path.iterdir() if d.is_dir()]
    print(f"Found {len(classes)} classes.")
    
    for class_path in classes:
        # Normalize class name: strip spaces and convert to lowercase
        class_name = class_path.name.strip().lower().replace(" ", "_")
        print(f"Processing class: {class_path.name} -> {class_name}")
        
        # Get all images
        images = [f for f in class_path.iterdir() if f.is_file() and f.suffix.lower() in ['.png', '.jpg', '.jpeg']]
        random.shuffle(images)
        
        if len(images) == 1:
            train_images = images
            val_images = images # Use same image for both train and val if only 1 available
        else:
            split_idx = int(len(images) * train_split)
            # Ensure at least one in train if possible
            if len(images) > 0 and split_idx == 0:
                split_idx = 1
            # Ensure at least one in val if possible
            if len(images) > 1 and split_idx == len(images):
                split_idx = len(images) - 1
            train_images = images[:split_idx]
            val_images = images[split_idx:]
        
        # Create directories
        (output_path / "train" / class_name).mkdir(parents=True, exist_ok=True)
        (output_path / "val" / class_name).mkdir(parents=True, exist_ok=True)
        
        # Copy files
        for img in train_images:
            shutil.copy(img, output_path / "train" / class_name / img.name)
        for img in val_images:
            shutil.copy(img, output_path / "val" / class_name / img.name)
            
    print(f"Data preparation complete. Dataset located at: {output_dir}")

if __name__ == "__main__":
    prepare_data()

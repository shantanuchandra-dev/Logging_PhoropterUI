import os
import shutil
import random
from pathlib import Path
from PIL import Image

def prepare_data(source_dir="all_the_charts", output_dir="chart_dataset", train_split=0.8):
    """
    Organizes images from source_dir into a standard train/val split.
    Classes are defined by the immediate parent folder of each image.
    """
    source_path = Path(source_dir)
    output_path = Path(output_dir)
    
    if output_path.exists():
        shutil.rmtree(output_path)
    
    # Collect all images and their immediate parent (class)
    image_groups = {}
    
    for root, dirs, files in os.walk(source_dir):
        # Skip if the directory itself is hidden or special
        if any(part.startswith('.') for part in Path(root).parts):
            continue
            
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = Path(root) / file
                # Class name is the immediate parent directory
                # Sanitize class name: lowercase, replace spaces with underscores, remove special chars
                orig_class_name = Path(root).name.strip().lower()
                class_name = "".join([c if c.isalnum() or c == "_" else "_" for c in orig_class_name.replace(" ", "_")])
                # Collapse multiple underscores
                while "__" in class_name:
                    class_name = class_name.replace("__", "_")
                class_name = class_name.strip("_")
                
                if class_name not in image_groups:
                    image_groups[class_name] = []
                image_groups[class_name].append(img_path)

    print(f"Found {len(image_groups)} granular classes.")
    
    for class_name, images in image_groups.items():
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

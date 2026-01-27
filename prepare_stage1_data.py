"""
Stage 1 Dataset Preparation
Creates a binary classification dataset: filled vs jcc_pattern
"""

import os
import shutil
import cv2

# Configuration
SOURCE_DATASET = 'jcc_occluder_dataset'
STAGE1_DATASET = 'stage1_dataset'

def create_stage1_dataset():
    """
    Reorganize existing dataset into 2 classes for Stage 1:
    - filled: grey_filled + blue_filled
    - jcc_pattern: green_axis_refine + red_axis_refine + green_power_refine + red_power_refine
    """
    
    print("=" * 60)
    print("Stage 1 Dataset Preparation")
    print("=" * 60)
    
    if not os.path.exists(SOURCE_DATASET):
        print(f"Error: Source dataset {SOURCE_DATASET} not found.")
        print("Please run prepare_jcc_data.py first.")
        return
    
    # Remove existing stage1 dataset if it exists
    if os.path.exists(STAGE1_DATASET):
        shutil.rmtree(STAGE1_DATASET)
    
    # Create stage1 dataset structure
    os.makedirs(STAGE1_DATASET)
    filled_dir = os.path.join(STAGE1_DATASET, 'filled')
    jcc_pattern_dir = os.path.join(STAGE1_DATASET, 'jcc_pattern')
    pinhole_dir = os.path.join(STAGE1_DATASET, 'pinhole')
    os.makedirs(filled_dir)
    os.makedirs(jcc_pattern_dir)
    os.makedirs(pinhole_dir)
    
    # Mapping of source classes to stage1 classes
    class_mapping = {
        'grey_filled': 'filled',
        'blue_filled': 'filled',
        'pinhole': 'pinhole',
        'green_axis_refine': 'jcc_pattern',
        'red_axis_refine': 'jcc_pattern',
        'green_power_refine': 'jcc_pattern',
        'red_power_refine': 'jcc_pattern'
    }
    
    # Copy files
    total_copied = 0
    for source_class, target_class in class_mapping.items():
        source_dir = os.path.join(SOURCE_DATASET, source_class)
        target_dir = os.path.join(STAGE1_DATASET, target_class)
        
        if not os.path.exists(source_dir):
            print(f"Warning: {source_dir} not found, skipping...")
            continue
        
        files = [f for f in os.listdir(source_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        print(f"\nCopying {len(files)} images from {source_class} to {target_class}...")
        
        for filename in files:
            src_path = os.path.join(source_dir, filename)
            # Rename to avoid conflicts
            new_filename = f"{source_class}_{filename}"
            dst_path = os.path.join(target_dir, new_filename)
            shutil.copy2(src_path, dst_path)
            total_copied += 1
    
    print("\n" + "=" * 60)
    print("Dataset preparation complete!")
    print("=" * 60)
    
    # Print summary
    print("\nStage 1 Dataset Summary:")
    for class_name in ['filled', 'jcc_pattern', 'pinhole']:
        class_dir = os.path.join(STAGE1_DATASET, class_name)
        count = len([f for f in os.listdir(class_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        print(f"  {class_name}: {count} images")
    
    print(f"\nTotal images: {total_copied}")

if __name__ == "__main__":
    create_stage1_dataset()

#!/usr/bin/env python3
"""
Create ROI_5 bottom half images from ROI_0 full screenshots
"""

import cv2
import os
from pathlib import Path

def create_bottom_half(input_dir, output_dir):
    """Crop bottom half from ROI_0 images and save to ROI_5"""
    
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Find all roi0_*.png files (not roi0_box_*.png)
    input_path = Path(input_dir)
    roi0_files = sorted([f for f in input_path.glob('roi0_*.png') if 'box' not in f.name])
    
    print(f"Found {len(roi0_files)} ROI_0 files to process")
    
    for img_file in roi0_files:
        # Read the image
        img = cv2.imread(str(img_file))
        if img is None:
            print(f"Error reading {img_file}")
            continue
            
        height, width = img.shape[:2]
        
        # Crop bottom half
        mid_height = height // 2
        bottom_half = img[mid_height:, :]
        
        # Generate output filename
        # Extract timestamp from original filename
        # roi0_20260115_085323.png -> roi0_bottom_half_20260115_085323.png
        timestamp = img_file.stem.replace('roi0_', '')
        output_filename = f"roi0_bottom_half_{timestamp}.png"
        output_path = Path(output_dir) / output_filename
        
        # Save the bottom half
        cv2.imwrite(str(output_path), bottom_half)
        print(f"Created: {output_path}")
        print(f"  Original size: {width}x{height}")
        print(f"  Bottom half size: {width}x{height - mid_height}")

if __name__ == "__main__":
    roi0_dir = "ROI_0"
    roi5_dir = "ROI_5"
    
    create_bottom_half(roi0_dir, roi5_dir)
    print(f"\n✓ Bottom half images saved to {roi5_dir}/")

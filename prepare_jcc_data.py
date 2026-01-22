"""
Improved JCC Data Preparation Script
Extracts the actual occluder circles from JCC screenshots before training.
This ensures the model learns from cropped ROIs, not full screenshots.
"""

import os
import shutil
import cv2
import numpy as np

# Configuration
DATASET_DIR = 'jcc_occluder_dataset'
CLASSES = [
    'grey_filled',
    'blue_filled', 
    'green_axis_refine',
    'red_axis_refine',
    'green_power_refine',
    'red_power_refine'
]

TRAIN_SIZE = (64, 64)

def create_dirs():
    """Create dataset directory structure"""
    if os.path.exists(DATASET_DIR):
        shutil.rmtree(DATASET_DIR)
    os.makedirs(DATASET_DIR)
    
    for cls in CLASSES:
        cls_dir = os.path.join(DATASET_DIR, cls)
        os.makedirs(cls_dir)
    print(f"Created dataset directory: {DATASET_DIR}")

def extract_circle_roi(img):
    """
    Extract the main circle from a JCC screenshot.
    Returns the cropped circle region.
    """
    h, w = img.shape[:2]
    
    # Find the largest circle
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_blurred = cv2.GaussianBlur(gray, (9, 9), 2)
    
    circles = cv2.HoughCircles(gray_blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=50,
                               param1=50, param2=30, minRadius=20, maxRadius=min(h, w)//2)
    
    if circles is None:
        # If no circle detected, return center crop
        print("  Warning: No circle detected, using center crop")
        center_x, center_y = w // 2, h // 2
        crop_size = min(h, w) // 2
        x1 = max(0, center_x - crop_size)
        y1 = max(0, center_y - crop_size)
        x2 = min(w, center_x + crop_size)
        y2 = min(h, center_y + crop_size)
        return img[y1:y2, x1:x2]
    
    # Get the largest circle
    circles = np.uint16(np.around(circles))
    largest_circle = max(circles[0, :], key=lambda c: c[2])
    
    cx, cy, r = int(largest_circle[0]), int(largest_circle[1]), int(largest_circle[2])
    
    # Expand radius slightly to include circumference
    r_expanded = int(r * 1.1)
    
    # Crop the circle region (ensure proper bounds)
    x1 = max(0, cx - r_expanded)
    y1 = max(0, cy - r_expanded)
    x2 = min(w, cx + r_expanded)
    y2 = min(h, cy + r_expanded)
    
    roi = img[y1:y2, x1:x2]
    
    # Validate ROI is not empty
    if roi.size == 0:
        print("  Warning: Empty ROI, using center crop")
        center_x, center_y = w // 2, h // 2
        crop_size = min(h, w) // 2
        x1 = max(0, center_x - crop_size)
        y1 = max(0, center_y - crop_size)
        x2 = min(w, center_x + crop_size)
        y2 = min(h, center_y + crop_size)
        roi = img[y1:y2, x1:x2]
    
    return roi

def detect_color_type(img):
    """
    Detect if the circumference is red or green based on color analysis.
    Returns: 'red', 'green', or 'unknown'
    """
    # Convert to HSV for better color detection
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Define color ranges in HSV
    # Red has two ranges in HSV (wraps around 180)
    red_lower1 = np.array([0, 100, 100])
    red_upper1 = np.array([10, 255, 255])
    red_lower2 = np.array([170, 100, 100])
    red_upper2 = np.array([180, 255, 255])
    
    # Green range
    green_lower = np.array([40, 100, 100])
    green_upper = np.array([80, 255, 255])
    
    # Create masks
    red_mask1 = cv2.inRange(hsv, red_lower1, red_upper1)
    red_mask2 = cv2.inRange(hsv, red_lower2, red_upper2)
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)
    
    green_mask = cv2.inRange(hsv, green_lower, green_upper)
    
    # Count pixels
    red_pixels = cv2.countNonZero(red_mask)
    green_pixels = cv2.countNonZero(green_mask)
    
    # Determine dominant color
    if red_pixels > green_pixels and red_pixels > 50:
        return 'red'
    elif green_pixels > red_pixels and green_pixels > 50:
        return 'green'
    else:
        return 'unknown'

def process_jcc_images(source_dir, refine_type):
    """
    Process JCC images from Axis_refine or Power_refine folders.
    NOW extracts the circle ROI first before saving.
    
    Args:
        source_dir: Path to source directory (_JCC/Axis_refine or _JCC/Power_refine)
        refine_type: 'axis' or 'power'
    """
    if not os.path.exists(source_dir):
        print(f"Directory {source_dir} not found.")
        return
    
    files = [f for f in os.listdir(source_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    print(f"\nProcessing {len(files)} images from {source_dir}...")
    
    for filename in files:
        src_path = os.path.join(source_dir, filename)
        img = cv2.imread(src_path)
        if img is None:
            print(f"  Could not read {filename}")
            continue
        
        # Extract the circle ROI first
        roi = extract_circle_roi(img)
        
        # Detect color
        color = detect_color_type(roi)
        
        if color == 'unknown':
            print(f"  {filename}: Could not determine color, skipping")
            continue
        
        # Determine class name
        class_name = f"{color}_{refine_type}_refine"
        
        # Resize to training size
        img_resized = cv2.resize(roi, TRAIN_SIZE)
        
        # Save to appropriate class folder
        dst_dir = os.path.join(DATASET_DIR, class_name)
        dst_path = os.path.join(dst_dir, filename)
        cv2.imwrite(dst_path, img_resized)
        print(f"  {filename} -> {class_name}")

def process_filled_images(source_dir, class_name, max_count=20):
    """
    Process blue_filled or grey_filled images from ROI_3/ROI_4 folders.
    
    Args:
        source_dir: Path to ROI_3 or ROI_4
        class_name: 'blue_filled' or 'grey_filled'
        max_count: Maximum number of images to process
    """
    if not os.path.exists(source_dir):
        print(f"Directory {source_dir} not found.")
        return
    
    files = [f for f in os.listdir(source_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    files = files[:max_count]  # Limit number of images
    
    print(f"\nProcessing {len(files)} images from {source_dir} as {class_name}...")
    
    dst_dir = os.path.join(DATASET_DIR, class_name)
    
    for filename in files:
        src_path = os.path.join(source_dir, filename)
        img = cv2.imread(src_path)
        if img is None:
            continue
        
        # Resize to training size
        img_resized = cv2.resize(img, TRAIN_SIZE)
        
        # Save
        dst_path = os.path.join(dst_dir, filename)
        cv2.imwrite(dst_path, img_resized)
        print(f"  {filename} -> {class_name}")

def generate_grey_filled(count=20):
    """Generate synthetic grey_filled images"""
    print(f"\nGenerating {count} synthetic grey_filled images...")
    dst_dir = os.path.join(DATASET_DIR, 'grey_filled')
    
    for i in range(count):
        # Create a grey circle on grey background
        img = np.full((TRAIN_SIZE[1], TRAIN_SIZE[0], 3), 160, dtype=np.uint8)
        
        # Draw a filled grey circle
        center = (TRAIN_SIZE[0] // 2, TRAIN_SIZE[1] // 2)
        radius = TRAIN_SIZE[0] // 2 - 5
        grey_value = np.random.randint(120, 180)
        cv2.circle(img, center, radius, (grey_value, grey_value, grey_value), -1)
        
        # Add slight noise
        noise = np.random.normal(0, 5, img.shape).astype(np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        cv2.imwrite(os.path.join(dst_dir, f'synth_grey_{i}.png'), img)
    print(f"  Generated {count} synthetic grey images")

def augment_jcc_classes(count_per_image=10):
    """
    Augment JCC classes (which have very few samples) by creating variations.
    This helps balance the dataset.
    """
    print(f"\nAugmenting JCC classes with {count_per_image} variations per image...")
    
    jcc_classes = ['green_axis_refine', 'red_axis_refine', 'green_power_refine', 'red_power_refine']
    
    for cls in jcc_classes:
        cls_dir = os.path.join(DATASET_DIR, cls)
        if not os.path.exists(cls_dir):
            continue
        
        original_files = [f for f in os.listdir(cls_dir) if f.lower().endswith('.png')]
        
        for orig_file in original_files:
            orig_path = os.path.join(cls_dir, orig_file)
            img = cv2.imread(orig_path)
            if img is None:
                continue
            
            # Create augmented versions
            for i in range(count_per_image):
                aug_img = img.copy()
                
                # Random rotation
                angle = np.random.randint(-30, 30)
                h, w = aug_img.shape[:2]
                M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
                aug_img = cv2.warpAffine(aug_img, M, (w, h))
                
                # Random brightness/contrast
                alpha = np.random.uniform(0.8, 1.2)  # Contrast
                beta = np.random.randint(-20, 20)    # Brightness
                aug_img = cv2.convertScaleAbs(aug_img, alpha=alpha, beta=beta)
                
                # Random flip
                if np.random.random() > 0.5:
                    aug_img = cv2.flip(aug_img, 1)  # Horizontal flip
                
                # Save
                base_name = os.path.splitext(orig_file)[0]
                aug_path = os.path.join(cls_dir, f'{base_name}_aug_{i}.png')
                cv2.imwrite(aug_path, aug_img)
        
        total_count = len([f for f in os.listdir(cls_dir) if f.lower().endswith('.png')])
        print(f"  {cls}: {total_count} images (after augmentation)")

def main():
    print("=" * 60)
    print("JCC Occluder Dataset Preparation (Improved)")
    print("=" * 60)
    
    create_dirs()
    
    # Process JCC images (with circle extraction)
    process_jcc_images('_JCC/Axis_refine', 'axis')
    process_jcc_images('_JCC/Power_refine', 'power')
    
    # Augment JCC classes to balance dataset
    augment_jcc_classes(count_per_image=15)
    
    # Process blue filled from ROI_3 and ROI_4
    process_filled_images('ROI_3', 'blue_filled', max_count=15)
    process_filled_images('ROI_4', 'blue_filled', max_count=15)
    
    # Generate synthetic grey filled
    generate_grey_filled(count=20)
    
    print("\n" + "=" * 60)
    print("Dataset preparation complete!")
    print("=" * 60)
    
    # Print summary
    print("\nDataset Summary:")
    for cls in CLASSES:
        cls_dir = os.path.join(DATASET_DIR, cls)
        count = len([f for f in os.listdir(cls_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        print(f"  {cls}: {count} images")

if __name__ == "__main__":
    main()

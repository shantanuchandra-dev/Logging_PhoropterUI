import os
import shutil
import cv2
import numpy as np

# Configuration
SOURCE_DIRS = {
    'ROI_3': 'ROI_3',
    'ROI_4': 'ROI_4',
    'JCC_AXIS': 'JCC/Axis_refine',
    'JCC_POWER': 'JCC/Power_refine'
}
DATASET_DIR = 'occluder_dataset'
CLASSES = ['blue_filled', 'grey_filled', 'grey_unfilled']

# ROI Size guess based on typically small ROIs, but we will resize for training anyway.
# We'll use 64x64 for training.
TRAIN_SIZE = (64, 64)

def create_dirs():
    if not os.path.exists(DATASET_DIR):
        os.makedirs(DATASET_DIR)
    for cls in CLASSES:
        cls_dir = os.path.join(DATASET_DIR, cls)
        if os.path.exists(cls_dir):
            shutil.rmtree(cls_dir)
        os.makedirs(cls_dir)

def generate_grey_unfilled(count=20):
    print(f"Generating {count} synthetic grey_unfilled images...")
    dst_dir = os.path.join(DATASET_DIR, 'grey_unfilled')
    
    for i in range(count):
        # Create a grey image (approx 128-160 range)
        # 160 is light grey, 100 is dark grey.
        # User said "unfilled is classified by grey color".
        base_grey = np.random.randint(140, 180)
        img = np.full((TRAIN_SIZE[1], TRAIN_SIZE[0], 3), base_grey, dtype=np.uint8)
        
        # Add noise
        noise = np.random.normal(0, 10, img.shape).astype(np.uint8)
        img = cv2.add(img, noise)
        
        cv2.imwrite(os.path.join(dst_dir, f'synth_grey_{i}.png'), img)

def process_and_resize(src_dir, dest_class):
    print(f"Processing {src_dir} -> {dest_class}...")
    if not os.path.exists(src_dir):
        print(f"Directory {src_dir} not found.")
        return

    dst_dir = os.path.join(DATASET_DIR, dest_class)
    
    files = [f for f in os.listdir(src_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    for filename in files:
        src_path = os.path.join(src_dir, filename)
        img = cv2.imread(src_path)
        if img is None:
            continue
            
        # Resize to training size
        img_resized = cv2.resize(img, TRAIN_SIZE)
        
        # Unique name
        new_name = f"{os.path.basename(src_dir)}_{filename}"
        cv2.imwrite(os.path.join(dst_dir, new_name), img_resized)
        print(f"Processed {filename}")

def main():
    create_dirs()
    
    # ROI_3 and ROI_4 are Blue Filled (based on analysis and user context)
    process_and_resize('ROI_3', 'blue_filled')
    process_and_resize('ROI_4', 'blue_filled')
    
    # JCC folders are Grey Filled
    process_and_resize('JCC/Axis_refine', 'grey_filled')
    process_and_resize('JCC/Power_refine', 'grey_filled')
    
    # Generate Synthetic Grey Unfilled
    # We generate about as many as we have for others (approx 10-15)
    generate_grey_unfilled(count=15)
    
    print("Dataset preparation complete.")

if __name__ == "__main__":
    main()

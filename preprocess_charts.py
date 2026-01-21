import cv2
import os
import glob
import numpy as np
import extract_roi7

def preprocess_images(input_dir, output_dir, target_size=(64, 64)):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    image_paths = glob.glob(os.path.join(input_dir, "*.png"))
    image_paths += glob.glob(os.path.join(input_dir, "*.jpg"))
    image_paths += glob.glob(os.path.join(input_dir, "*.jpeg"))

    print(f"Found {len(image_paths)} images in {input_dir}")

    processed_count = 0
    for img_path in image_paths:
        try:
            img = cv2.imread(img_path)
            if img is None:
                print(f"Failed to read: {img_path}")
                continue
            
            # Use the actual extraction logic to get the crop
            bbox, _ = extract_roi7.extract_roi7_from_roi0(img)
            
            if bbox:
                x, y, w, h = bbox
                roi7_crop = img[y:y+h, x:x+w]
                
                # Convert to grayscale
                gray = cv2.cvtColor(roi7_crop, cv2.COLOR_BGR2GRAY)
                
                # Resize
                resized = cv2.resize(gray, target_size, interpolation=cv2.INTER_AREA)
                
                # Save
                filename = os.path.basename(img_path)
                output_path = os.path.join(output_dir, filename)
                cv2.imwrite(output_path, resized)
                processed_count += 1
            else:
                print(f"ROI7 not detected in {os.path.basename(img_path)}. Skipping.")
        except Exception as e:
            print(f"Error processing {img_path}: {e}")

    print(f"Preprocessing complete. {processed_count} images saved to {output_dir}")

if __name__ == "__main__":
    input_folder = "Charts"
    output_folder = "Charts_Processed"
    preprocess_images(input_folder, output_folder)

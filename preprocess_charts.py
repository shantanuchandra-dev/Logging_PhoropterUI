import cv2
import os
import glob

def preprocess_images(input_dir, output_dir, target_size=(64, 64)):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    image_paths = glob.glob(os.path.join(input_dir, "*.png"))
    image_paths += glob.glob(os.path.join(input_dir, "*.jpg"))
    image_paths += glob.glob(os.path.join(input_dir, "*.jpeg"))

    print(f"Found {len(image_paths)} images in {input_dir}")

    for img_path in image_paths:
        try:
            img = cv2.imread(img_path)
            if img is None:
                print(f"Failed to read: {img_path}")
                continue
            
            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Resize
            resized = cv2.resize(gray, target_size, interpolation=cv2.INTER_AREA)
            
            # Save
            filename = os.path.basename(img_path)
            output_path = os.path.join(output_dir, filename)
            cv2.imwrite(output_path, resized)
            # print(f"Processed: {filename}")
        except Exception as e:
            print(f"Error processing {img_path}: {e}")

    print(f"Preprocessing complete. Processed images saved to {output_dir}")

if __name__ == "__main__":
    input_folder = "/Users/chirayumaru/Desktop/Charts"
    output_folder = "/Users/chirayumaru/Desktop/Charts_Processed"
    preprocess_images(input_folder, output_folder)

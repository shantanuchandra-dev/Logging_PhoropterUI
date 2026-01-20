import cv2
import os
import shutil

def label_images(processed_dir):
    if not os.path.exists(processed_dir):
        print(f"Directory not found: {processed_dir}")
        return

    images = [f for f in os.listdir(processed_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not images:
        print("No images found to label.")
        return

    print("--- Chart Labeling Tool ---")
    print("Instructions:")
    print("1. An image will appear.")
    print("2. Type the label (e.g., 'VA_Chart', 'Landolt_C', etc.) in the console and press Enter.")
    print("3. Type 'skip' to skip the image.")
    print("4. Type 'quit' to exit.")
    print("---------------------------")

    for filename in images:
        img_path = os.path.join(processed_dir, filename)
        img = cv2.imread(img_path)
        
        if img is None:
            continue

        # Show image (User must be able to see windows on their machine)
        cv2.imshow("Labeling", img)
        cv2.moveWindow("Labeling", 100, 100)
        
        print(f"Labeling: {filename}")
        label = input("Enter label: ").strip()
        
        if label.lower() in ['quit', 'exit']:
            break
        elif label.lower() == 'skip':
            continue
        elif label:
            label_dir = os.path.join(processed_dir, label)
            if not os.path.exists(label_dir):
                os.makedirs(label_dir)
            
            shutil.move(img_path, os.path.join(label_dir, filename))
            print(f"Moved to {label_dir}")
        
    cv2.destroyAllWindows()
    print("Labeling session finished.")

if __name__ == "__main__":
    processed_folder = "Charts_Processed"
    label_images(processed_folder)

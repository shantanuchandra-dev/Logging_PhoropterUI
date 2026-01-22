import os
import cv2
import numpy as np

def analyze_dir(directory):
    print(f"--- Analyzing {directory} ---")
    if not os.path.exists(directory):
        print("Dir not found")
        return

    for filename in os.listdir(directory):
        if not filename.endswith(('.png', '.jpg')):
            continue
            
        path = os.path.join(directory, filename)
        img = cv2.imread(path)
        if img is None:
            continue
            
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # Calculate mean Saturation and Value
        mean_s = np.mean(hsv[:, :, 1])
        mean_v = np.mean(hsv[:, :, 2])
        mean_h = np.mean(hsv[:, :, 0])
        
        # Check for blue pixels (rough check)
        lower_blue = np.array([90, 50, 50])
        upper_blue = np.array([130, 255, 255])
        mask = cv2.inRange(hsv, lower_blue, upper_blue)
        blue_pixel_count = np.sum(mask > 0)
        
        print(f"{filename}: H={mean_h:.1f}, S={mean_s:.1f}, V={mean_v:.1f}, BluePixels={blue_pixel_count}")

def main():
    analyze_dir('ROI_3')
    analyze_dir('ROI_4')
    analyze_dir('JCC/Axis_refine')

if __name__ == "__main__":
    main()

"""Debug script to visualize what the model sees"""
import cv2
import numpy as np

def extract_circle_roi(img):
    """Extract the main circle from a JCC screenshot."""
    h, w = img.shape[:2]
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_blurred = cv2.GaussianBlur(gray, (9, 9), 2)
    
    circles = cv2.HoughCircles(gray_blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=50,
                               param1=50, param2=30, minRadius=20, maxRadius=min(h, w)//2)
    
    if circles is None:
        center_x, center_y = w // 2, h // 2
        crop_size = min(h, w) // 2
        x1 = max(0, center_x - crop_size)
        y1 = max(0, center_y - crop_size)
        x2 = min(w, center_x + crop_size)
        y2 = min(h, center_y + crop_size)
        return img[y1:y2, x1:x2]
    
    circles = np.uint16(np.around(circles))
    largest_circle = max(circles[0, :], key=lambda c: c[2])
    
    cx, cy, r = int(largest_circle[0]), int(largest_circle[1]), int(largest_circle[2])
    r_expanded = int(r * 1.1)
    
    x1 = max(0, cx - r_expanded)
    y1 = max(0, cy - r_expanded)
    x2 = min(w, cx + r_expanded)
    y2 = min(h, cy + r_expanded)
    
    roi = img[y1:y2, x1:x2]
    
    if roi.size == 0:
        center_x, center_y = w // 2, h // 2
        crop_size = min(h, w) // 2
        x1 = max(0, center_x - crop_size)
        y1 = max(0, center_y - crop_size)
        x2 = min(w, center_x + crop_size)
        y2 = min(h, center_y + crop_size)
        roi = img[y1:y2, x1:x2]
    
    return roi

# Test on red axis image
img = cv2.imread('_JCC/Axis_refine/Screenshot 2026-01-21 at 13.31.40.png')
roi = extract_circle_roi(img)
cv2.imwrite('debug_red_axis_roi.png', roi)
print(f"Red axis ROI shape: {roi.shape}")
print(f"Red axis ROI mean color (BGR): {cv2.mean(roi)[:3]}")

# Test on green axis image  
img = cv2.imread('_JCC/Axis_refine/Screenshot 2026-01-21 at 13.31.54.png 14-16-13-503.png')
roi = extract_circle_roi(img)
cv2.imwrite('debug_green_axis_roi.png', roi)
print(f"Green axis ROI shape: {roi.shape}")
print(f"Green axis ROI mean color (BGR): {cv2.mean(roi)[:3]}")

# Check what's in the dataset
dataset_red = cv2.imread('jcc_occluder_dataset/red_axis_refine/Screenshot 2026-01-21 at 13.31.40.png')
print(f"\nDataset red axis shape: {dataset_red.shape}")
print(f"Dataset red axis mean color (BGR): {cv2.mean(dataset_red)[:3]}")

dataset_green = cv2.imread('jcc_occluder_dataset/green_axis_refine/Screenshot 2026-01-21 at 13.31.54.png 14-16-13-503.png')
print(f"Dataset green axis shape: {dataset_green.shape}")
print(f"Dataset green axis mean color (BGR): {cv2.mean(dataset_green)[:3]}")

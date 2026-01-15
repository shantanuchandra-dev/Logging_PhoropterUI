
import cv2
import numpy as np
import datetime
import os

# Load the image
input_path = 'topcon_ui_001.png'  # Replace with your actual image filename
img = cv2.imread(input_path)
if img is None:
    raise FileNotFoundError(f'Image not found: {input_path}')

# Convert to grayscale and blur
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 0)

# Edge detection
edges = cv2.Canny(blur, 50, 150)

# Find contours
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Find the largest rectangle-like contour (ROI-0)
roi_contour = None
max_area = 0
for cnt in contours:
    approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
    area = cv2.contourArea(cnt)
    if len(approx) == 4 and area > max_area:
        roi_contour = approx
        max_area = area

if roi_contour is None:
    raise Exception('ROI-0 (main window) not found.')

# Get bounding rect and crop
x, y, w, h = cv2.boundingRect(roi_contour)
roi0 = img[y:y+h, x:x+w]

# Prepare output directory
output_dir = 'ROI_0'
os.makedirs(output_dir, exist_ok=True)

# Save with timestamp in ROI_0 folder
now = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
output_path = os.path.join(output_dir, f'roi0_{now}.png')
cv2.imwrite(output_path, roi0)
print(f'ROI-0 saved to {output_path}')

# Optionally, draw the detected ROI-0 on the original image for visualization
vis = img.copy()
cv2.rectangle(vis, (x, y), (x+w, y+h), (0, 255, 0), 3)
vis_path = os.path.join(output_dir, f'roi0_box_{now}.png')
cv2.imwrite(vis_path, vis)
print(f'ROI-0 bounding box visualization saved to {vis_path}')

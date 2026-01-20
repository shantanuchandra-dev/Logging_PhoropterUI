import cv2
import json
import os
from extract_roi1_ocr import extract_roi1_ocr

# Verification script for ROI1 OCR improvements
img_path = '/Users/shantanuchandra/Downloads/Logging_PhoropterUI/ROI_0/3ym8_2001_162651.png'
json_path = '/Users/shantanuchandra/Downloads/Logging_PhoropterUI/MatchedScreens/3ym80YNRSvOOPQjDTAu7wg_coords.json'

if not os.path.exists(img_path) or not os.path.exists(json_path):
    print(f"Error: Required files not found.\nImg: {img_path}\nJson: {json_path}")
    exit(1)

img = cv2.imread(img_path)
with open(json_path, 'r') as f:
    data = json.load(f)

bboxes = data['rois']['roi1']['cell_bboxes_on_roi0']

print("Running improved OCR...")
results = extract_roi1_ocr(img, bboxes)

print("\nFinal Results:")
print(json.dumps(results, indent=2))

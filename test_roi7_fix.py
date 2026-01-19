import cv2
import extract_roi7
import os

# Paths
roi0_img_path = "/Users/shantanuchandra/Downloads/Logging_PhoropterUI/ROI_0/_-wu_2001_034240.png"
output_path = "/Users/shantanuchandra/Downloads/Logging_PhoropterUI/ROI_7/test_fix_output.png"

# Load img
img = cv2.imread(roi0_img_path)
if img is None:
    print(f"Error: Could not load {roi0_img_path}")
    exit(1)

# Run extraction
result = extract_roi7.extract(img, save_debug=True, filename=roi0_img_path, debug=True)
print(f"Extraction result: {result}")

if 'debug_image' in result:
    print(f"Debug image saved to: {result['debug_image']}")
else:
    print("Warning: No debug image saved.")

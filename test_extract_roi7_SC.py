import cv2
from extract_roi7 import extract_roi7_from_roi0

# Path to the test image
img_path = 'ROI_0/3ym80YNRSvOOPQjDTAu7wg_14.png'
img = cv2.imread(img_path)

if img is None:
    print(f"Failed to load image: {img_path}")
else:
    bbox, labeled = extract_roi7_from_roi0(img, debug=True)
    if bbox:
        print(f"ROI7 bounding box: {bbox}")
        # Save the labeled image in the ROI_7 folder
        import os
        output_dir = 'ROI_7'
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, 'labeled_3ym80YNRSvOOPQjDTAu7wg_14.png')
        cv2.imwrite(output_path, labeled)
        print(f"Labeled image saved to {output_path}")
    else:
        print("ROI7 not found.")

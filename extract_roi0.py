import cv2
import numpy as np
import datetime
import os

def extract_roi0(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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
    x, y, w, h = cv2.boundingRect(roi_contour)
    roi0 = img[y:y+h, x:x+w]
    return {'roi0': roi0, 'bbox': (x, y, w, h)}


# If run as a script, keep original behavior
if __name__ == '__main__':
    # Only process the first matched frame from MatchedScreens (or firstFrame)
    import glob
    import os
    matched_dir = 'firstFrame'
    matched_files = sorted(glob.glob(os.path.join(matched_dir, '*.png')))
    if not matched_files:
        print(f'No matched frames found in {matched_dir}')
        exit(1)
    input_path = matched_files[-1]  # Fetch the latest image
    print(f'Processing latest matched frame: {input_path}')
    img = cv2.imread(input_path)
    if img is None:
        raise FileNotFoundError(f'Image not found: {input_path}')
    result = extract_roi0(img)
    roi0 = result['roi0']
    x, y, w, h = result['bbox']
    output_dir = 'ROI_0'
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    output_path = os.path.join(output_dir, f'crop_{base_name}.png')
    cv2.imwrite(output_path, roi0)
    print(f'ROI-0 saved to {output_path}')
    vis = img.copy()
    cv2.rectangle(vis, (x, y), (x+w, y+h), (0, 255, 0), 3)
    vis_path = os.path.join(output_dir, f'{base_name}_box.png')
    cv2.imwrite(vis_path, vis)
    print(f'ROI-0 bounding box visualization saved to {vis_path}')

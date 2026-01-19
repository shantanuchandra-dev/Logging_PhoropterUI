import cv2
import numpy as np
import datetime
import os

def extract_roi0(img, filename=None, save_dir='ROI_0', save=False):
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
    # Save ROI-0 and bounding box visualization if requested
    if save:
        os.makedirs(save_dir, exist_ok=True)
        if filename:
            base_name = os.path.splitext(os.path.basename(filename))[0]
        else:
            now = datetime.datetime.now().strftime('%d%m_%H%M%S')
            base_name = f'roi0_{now}'
        output_path = os.path.join(save_dir, f'{base_name}.png')
        cv2.imwrite(output_path, roi0)
        vis = img.copy()
        cv2.rectangle(vis, (x, y), (x+w, y+h), (0, 255, 0), 3)
        vis_path = os.path.join(save_dir, f'{base_name}_box.png')
        cv2.imwrite(vis_path, vis)
    return {'roi0': roi0, 'bbox': (x, y, w, h)}


# If run as a script, keep original behavior
if __name__ == '__main__':
    import glob
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
    result = extract_roi0(img, filename=input_path, save_dir='ROI_0', save=True)
    print(f'ROI-0 and bounding box saved for {input_path}')

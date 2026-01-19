
import cv2
import numpy as np
import datetime

def extract_roi7_from_roi0(roi0_img, debug=False):
    """
    Detects the acuity chart (ROI7) in the bottom-right region of ROI0.
    Uses morphological blob merging to isolate the slightly landscape chart area.
    """
    h, w = roi0_img.shape[:2]
    # Tightened search region: Focus more precisely on the chart area
    y_start = int(0.65 * h)
    y_end = int(0.93 * h)
    x_start = int(0.62 * w)
    x_end = int(0.90 * w)
    search_area = roi0_img[y_start:y_end, x_start:x_end]

    # Preprocess
    gray = cv2.cvtColor(search_area, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Use adaptive thresholding to get better separation of text on background
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 11, 2)
    
    # Morphological closing to merge letters into a single block
    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 10))
    kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 40))
    
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_h)
    closed = cv2.morphologyEx(closed, cv2.MORPH_CLOSE, kernel_v)

    # Find contours on the merged blobs
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    candidates = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 8000:
            x, y, rw, rh = cv2.boundingRect(cnt)
            aspect = rw / rh if rh > 0 else 0
            
            # Preference for slightly landscape (w > h just slightly)
            # Ideal aspect around 1.2 - 1.3
            ideal_aspect = 1.25
            aspect_score = 1.0 / (1.0 + abs(ideal_aspect - aspect))
            
            # Final score prioritizing area and ideal proportion
            score = area * (aspect_score ** 2)
            
            candidates.append({
                'bbox': (x, y, rw, rh),
                'area': area,
                'aspect': aspect,
                'aspect_score': aspect_score,
                'score': score
            })

    if not candidates:
        if debug:
            print("No suitable chart blobs found in ROI7 search area.")
        return None, roi0_img

    # Pick the best candidate (biggest rectangle with preferred proportion)
    candidates.sort(key=lambda x: x['score'], reverse=True)
    best = candidates[0]
    x, y, rw, rh = best['bbox']
    
    # Adjust coordinates to original ROI0
    x_abs = x + x_start
    y_abs = y + y_start
    
    labeled_img = roi0_img.copy()
    cv2.rectangle(labeled_img, (x_abs, y_abs), (x_abs+rw, y_abs+rh), (0, 255, 0), 2)
    label_text = f'ROI7 (asp={best["aspect"]:.2f})'
    cv2.putText(labeled_img, label_text, (x_abs, y_abs-10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    if debug:
        print(f"ROI7 found at: x={x_abs}, y={y_abs}, w={rw}, h={rh}, aspect={best['aspect']:.2f}, score={best['score']:.0f}")
        
    return (x_abs, y_abs, rw, rh), labeled_img


def extract(roi0_img, save_debug=False, filename=None, debug=False):
    """
    Extract function for ROI7, returns a result dict for test_extractors.
    """
    bbox, labeled_img = extract_roi7_from_roi0(roi0_img, debug=debug)
    result = {
        'roi_id': 'ROI7',
        'bbox': bbox,
    }
    if bbox and save_debug:
        import os
        out_dir = 'ROI_7'
        os.makedirs(out_dir, exist_ok=True)
        now = datetime.datetime.now().strftime('%d%m_%H%M%S')
        if filename:
            base = os.path.splitext(os.path.basename(filename))[0]
            prefix = base[:4]
        else:
            prefix = 'roi7'
        out_path = os.path.join(out_dir, f'{prefix}_{now}_labeled_roi7.png')
        cv2.imwrite(out_path, labeled_img)
        result['debug_image'] = out_path
    if not bbox:
        result['error'] = 'ROI7 not found'
    return result

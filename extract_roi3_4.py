
import cv2
import numpy as np
import os
import datetime

def extract(roi0_img, save_debug=False, output_dir='ROI_3'):
    """
    Extract occluders (ROI-3 and ROI-4) from ROI-0 image.
    
    Args:
        roi0_img: ROI-0 image (numpy array)
        save_debug: Whether to save debug images
        output_dir: Directory to save debug images (will create ROI_3 and ROI_4 subdirs)
    
    Returns:
        dict: {
            'roi_id': 'ROI_3_4',
            'bboxes': [
                {'label': 'left_occluder', 'box': [x, y, w, h], 'state': 'filled/active (blue)' or 'unfilled/inactive (gray)'},
                {'label': 'right_occluder', 'box': [x, y, w, h], 'state': '...'}
            ],
            'image_paths': ['path/to/roi3.png', 'path/to/roi4.png'] (if save_debug=True)
        }
    """
    # Resize to expected resolution for ROI 3/4
    img = cv2.resize(roi0_img, (929, 823))
    h, w = img.shape[:2]

    # Find Circles (Occluders)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_blurred = cv2.GaussianBlur(gray, (9, 9), 2)

    # Hough Circles
    circles = cv2.HoughCircles(gray_blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=200,
                               param1=50, param2=35, minRadius=30, maxRadius=70)

    if circles is None:
        return {
            'roi_id': 'ROI_3_4',
            'bboxes': [],
            'error': 'No circles detected'
        }

    circles = np.uint16(np.around(circles))
    detected_circles = sorted(circles[0, :], key=lambda x: x[0])
    
    if len(detected_circles) < 2:
        return {
            'roi_id': 'ROI_3_4',
            'bboxes': [],
            'error': f'Found {len(detected_circles)} circles, need at least 2'
        }

    # Find the two circles closest to the center vertically
    mid_y = h / 2
    candidates = sorted(detected_circles, key=lambda c: abs(c[1] - mid_y))
    left_right = sorted(candidates[:2], key=lambda c: c[0])
    
    left_circle = left_right[0]
    right_circle = left_right[1]

    # Classification logic (Heuristic: check blue channel dominance)
    def classify_state(circle_img):
        avg_color = cv2.mean(circle_img)[:3]
        # In BGR, blue is index 0
        if avg_color[0] > avg_color[1] + 20 and avg_color[0] > avg_color[2] + 20:
            return "filled/active (blue)"
        else:
            return "unfilled/inactive (gray)"

    # Left Occluder (ROI-3)
    lx, ly, lr = left_circle
    roi3 = img[ly-lr:ly+lr, lx-lr:lx+lr]
    state3 = classify_state(roi3)
    
    # Right Occluder (ROI-4)
    rx, ry, rr = right_circle
    roi4 = img[ry-rr:ry+rr, rx-rr:rx+rr]
    state4 = classify_state(roi4)
    
    result = {
        'roi_id': 'ROI_3_4',
        'bboxes': [
            {
                'label': 'left_occluder',
                'box': [int(lx - lr), int(ly - lr), int(2 * lr), int(2 * lr)],  # x, y, w, h
                'state': state3
            },
            {
                'label': 'right_occluder',
                'box': [int(rx - rr), int(ry - rr), int(2 * rr), int(2 * rr)],
                'state': state4
            }
        ]
    }
    
    if save_debug:
        now = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save ROI-3
        roi3_dir = 'ROI_3'
        os.makedirs(roi3_dir, exist_ok=True)
        roi3_path = os.path.join(roi3_dir, f'roi3_{now}.png')
        cv2.imwrite(roi3_path, roi3)
        
        # Save ROI-4
        roi4_dir = 'ROI_4'
        os.makedirs(roi4_dir, exist_ok=True)
        roi4_path = os.path.join(roi4_dir, f'roi4_{now}.png')
        cv2.imwrite(roi4_path, roi4)
        
        result['image_paths'] = [roi3_path, roi4_path]
    
    return result


if __name__ == "__main__":
    # Fallback to loading from ROI_0 directory
    roi0_dir = 'ROI_0'
    roi0_files = [f for f in os.listdir(roi0_dir) if f.startswith('roi0_') and f.endswith('.png') and 'box' not in f]
    if not roi0_files:
        print('No ROI-0 images found in ROI_0 directory.')
        exit(1)
    roi0_files.sort()
    roi0_path = os.path.join(roi0_dir, roi0_files[-1])

    img = cv2.imread(roi0_path)
    if img is None:
        print(f'Could not load {roi0_path}')
        exit(1)

    # Call the extract function with debug saving enabled
    result = extract(img, save_debug=True)
    print(f'Occluders extracted: {result}')

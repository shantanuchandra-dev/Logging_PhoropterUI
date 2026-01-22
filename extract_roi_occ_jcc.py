"""
JCC Occluder State Detection and Classification
Extracts ROI 3/4 (occluders) and classifies them into 6 visual states,
then maps to phoropter states (BINO, Occluded, JCC Flips).
"""

import cv2
import numpy as np
import torch
from torchvision import transforms
from occluder_model import OccluderNet
import os

# Model configuration
MODEL_PATH = 'jcc_occluder_model.pth'
CLASSES_FILE = 'jcc_occluder_classes.txt'

# Load class mapping
def load_class_mapping(classes_file):
    """Load class index to name mapping"""
    class_map = {}
    with open(classes_file, 'r') as f:
        for line in f:
            line = line.strip()
            if ':' in line:
                idx, name = line.split(':', 1)
                class_map[int(idx)] = name
    return class_map

# Initialize model (lazy loading)
_model = None
_device = None
_class_map = None
_transform = None

def get_model():
    """Lazy load model and related objects"""
    global _model, _device, _class_map, _transform
    
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model file {MODEL_PATH} not found. Train the model first.")
        if not os.path.exists(CLASSES_FILE):
            raise FileNotFoundError(f"Classes file {CLASSES_FILE} not found.")
        
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _class_map = load_class_mapping(CLASSES_FILE)
        
        _model = OccluderNet(num_classes=len(_class_map))
        _model.load_state_dict(torch.load(MODEL_PATH, map_location=_device))
        _model.to(_device)
        _model.eval()
        
        _transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        print(f"Model loaded from {MODEL_PATH}")
        print(f"Classes: {_class_map}")
    
    return _model, _device, _class_map, _transform

def classify_occluder(roi_img):
    """
    Classify a single occluder ROI image.
    
    Args:
        roi_img: OpenCV image (BGR) of the occluder
    
    Returns:
        str: Class name (e.g., 'blue_filled', 'green_axis_refine', etc.)
    """
    model, device, class_map, transform = get_model()
    
    # Preprocess
    img_tensor = transform(roi_img).unsqueeze(0).to(device)
    
    # Inference
    with torch.no_grad():
        outputs = model(img_tensor)
        _, predicted = torch.max(outputs, 1)
        class_idx = predicted.item()
    
    return class_map[class_idx]

def extract_occluders(roi0_img):
    """
    Extract left and right occluder regions from ROI-0 image.
    Uses the same circle detection logic as extract_roi3_4.py.
    
    Args:
        roi0_img: OpenCV image (BGR) of ROI-0
    
    Returns:
        tuple: (left_roi, right_roi, left_bbox, right_bbox) or (None, None, None, None) if detection fails
               bbox format: [x, y, w, h]
    """
    if roi0_img is None:
        return None, None, None, None
    
    # Resize to expected resolution
    img = cv2.resize(roi0_img, (929, 823))
    h, w = img.shape[:2]
    
    # Find Circles (Occluders)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_blurred = cv2.GaussianBlur(gray, (9, 9), 2)
    
    # Hough Circles
    circles = cv2.HoughCircles(gray_blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=200,
                               param1=50, param2=35, minRadius=30, maxRadius=70)
    
    if circles is None:
        print("Warning: No circles detected")
        return None, None, None, None
    
    circles = np.uint16(np.around(circles))
    detected_circles = sorted(circles[0, :], key=lambda x: x[0])
    
    if len(detected_circles) < 2:
        print(f"Warning: Found {len(detected_circles)} circles, need at least 2")
        return None, None, None, None
    
    # Find the two circles closest to the center vertically
    mid_y = h / 2
    candidates = sorted(detected_circles, key=lambda c: abs(c[1] - mid_y))
    left_right = sorted(candidates[:2], key=lambda c: c[0])
    
    left_circle = left_right[0]
    right_circle = left_right[1]
    
    # Extract ROIs
    lx, ly, lr = left_circle
    rx, ry, rr = right_circle
    
    left_roi = img[ly-lr:ly+lr, lx-lr:lx+lr]
    right_roi = img[ry-rr:ry+rr, rx-rr:rx+rr]
    
    # Scale coordinates back to original image resolution
    orig_h, orig_w = roi0_img.shape[:2]
    scale_x = orig_w / 929.0
    scale_y = orig_h / 823.0
    
    left_bbox = [
        int((lx - lr) * scale_x),
        int((ly - lr) * scale_y),
        int((2 * lr) * scale_x),
        int((2 * lr) * scale_y)
    ]
    
    right_bbox = [
        int((rx - rr) * scale_x),
        int((ry - rr) * scale_y),
        int((2 * rr) * scale_x),
        int((2 * rr) * scale_y)
    ]
    
    return left_roi, right_roi, left_bbox, right_bbox

def map_to_phoropter_state(left_class, right_class):
    """
    Map visual classes to phoropter state based on user-defined logic.
    
    Args:
        left_class: Visual class of left occluder (ROI-3)
        right_class: Visual class of right occluder (ROI-4)
    
    Returns:
        str: Phoropter state
    """
    # Both Blue -> BINO
    if left_class == 'blue_filled' and right_class == 'blue_filled':
        return 'BINO'
    
    # Both Grey -> Both_Occluded
    if left_class == 'grey_filled' and right_class == 'grey_filled':
        return 'Both_Occluded'
    
    # Left Grey, Right Blue -> Left_Occluded
    if left_class == 'grey_filled' and right_class == 'blue_filled':
        return 'Left_Occluded'
    
    # Right Grey, Left Blue -> Right_Occluded
    if left_class == 'blue_filled' and right_class == 'grey_filled':
        return 'Right_Occluded'
    
    # JCC States (Left eye testing)
    if left_class == 'green_axis_refine':
        return 'Left_Axis_Flip1'
    
    if left_class == 'red_axis_refine':
        return 'Left_Axis_Flip2'
    
    # Right eye JCC (Power refine on right means testing left power)
    if right_class == 'green_power_refine':
        return 'Left_Power_Flip1'
    
    if right_class == 'red_power_refine':
        return 'Left_Power_Flip2'
    
    # Default: Unknown state
    return f'Unknown({left_class},{right_class})'

def extract(roi0_img, save_debug=False, output_dir='ROI_3', filename=None):
    """
    Main extraction function compatible with existing pipeline.
    
    Args:
        roi0_img: OpenCV image (BGR) of ROI-0
        save_debug: Whether to save debug images
        output_dir: Directory to save debug images
        filename: Original filename for naming debug outputs
    
    Returns:
        dict: {
            'roi_id': 'ROI_3_4',
            'bboxes': [
                {'label': 'left_occluder', 'box': [x, y, w, h], 'state': 'visual_class'},
                {'label': 'right_occluder', 'box': [x, y, w, h], 'state': 'visual_class'}
            ],
            'phoropter_state': 'BINO' | 'Left_Occluded' | 'Right_Occluded' | etc.,
            'image_paths': [...] (if save_debug=True)
        }
    """
    # Extract occluder ROIs
    left_roi, right_roi, left_bbox, right_bbox = extract_occluders(roi0_img)
    
    if left_roi is None or right_roi is None:
        return {
            'roi_id': 'ROI_3_4',
            'bboxes': [],
            'phoropter_state': 'Detection_Failed',
            'error': 'Failed to detect occluders'
        }
    
    # Classify each occluder
    left_class = classify_occluder(left_roi)
    right_class = classify_occluder(right_roi)
    
    # Map to phoropter state
    phoropter_state = map_to_phoropter_state(left_class, right_class)
    
    result = {
        'roi_id': 'ROI_3_4',
        'bboxes': [
            {
                'label': 'left_occluder',
                'box': left_bbox,
                'state': left_class
            },
            {
                'label': 'right_occluder',
                'box': right_bbox,
                'state': right_class
            }
        ],
        'phoropter_state': phoropter_state
    }
    
    # Save debug images if requested
    if save_debug:
        import datetime
        now = datetime.datetime.now().strftime('%d%m_%H%M%S')
        
        if filename:
            input_base = os.path.splitext(os.path.basename(filename))[0]
            prefix = input_base[:4]
        else:
            prefix = 'roi0'
        
        # Save ROI-3 (left)
        roi3_dir = 'ROI_3'
        os.makedirs(roi3_dir, exist_ok=True)
        roi3_path = os.path.join(roi3_dir, f'{prefix}_{now}_roi3_{left_class}.png')
        cv2.imwrite(roi3_path, left_roi)
        
        # Save ROI-4 (right)
        roi4_dir = 'ROI_4'
        os.makedirs(roi4_dir, exist_ok=True)
        roi4_path = os.path.join(roi4_dir, f'{prefix}_{now}_roi4_{right_class}.png')
        cv2.imwrite(roi4_path, right_roi)
        
        result['image_paths'] = [roi3_path, roi4_path]
    
    return result

if __name__ == "__main__":
    # Test with a sample ROI-0 image
    roi0_dir = 'ROI_0'
    roi0_files = [f for f in os.listdir(roi0_dir) if f.endswith('.png') and 'box' not in f]
    
    if not roi0_files:
        print('No ROI-0 images found in ROI_0 directory.')
        exit(1)
    
    roi0_files.sort()
    roi0_path = os.path.join(roi0_dir, roi0_files[-1])
    
    print(f"Testing with: {roi0_path}")
    img = cv2.imread(roi0_path)
    
    if img is None:
        print(f'Could not load {roi0_path}')
        exit(1)
    
    # Run extraction
    result = extract(img, save_debug=True, filename=roi0_path)
    
    print("\n" + "="*60)
    print("RESULT:")
    print("="*60)
    print(f"ROI ID: {result['roi_id']}")
    print(f"Phoropter State: {result.get('phoropter_state', 'N/A')}")
    print("\nOccluders:")
    for bbox_info in result.get('bboxes', []):
        print(f"  {bbox_info['label']}: {bbox_info['state']}")
        print(f"    Box: {bbox_info['box']}")
    
    if 'image_paths' in result:
        print(f"\nDebug images saved:")
        for path in result['image_paths']:
            print(f"  {path}")
    
    if 'error' in result:
        print(f"\nError: {result['error']}")

"""
JCC Occluder State Detection with Two-Stage Classification
Stage 1: ML-based pattern type detection (filled vs jcc_pattern)
Stage 2: Rule-based detailed classification
"""

import cv2
import numpy as np
import torch
from torchvision import transforms
from occluder_model import OccluderNet
import os

# Model configuration
STAGE1_MODEL_PATH = 'stage1_model.pth'
STAGE1_CLASSES_FILE = 'stage1_classes.txt'

# Import Stage 2 classifiers
from stage2_classifier import classify_filled, classify_jcc_pattern

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

# Initialize Stage 1 model (lazy loading)
_stage1_model = None
_device = None
_stage1_class_map = None
_transform = None

def get_stage1_model():
    """Lazy load Stage 1 model and related objects"""
    global _stage1_model, _device, _stage1_class_map, _transform
    
    if _stage1_model is None:
        if not os.path.exists(STAGE1_MODEL_PATH):
            raise FileNotFoundError(f"Stage 1 model file {STAGE1_MODEL_PATH} not found. Train the model first.")
        if not os.path.exists(STAGE1_CLASSES_FILE):
            raise FileNotFoundError(f"Stage 1 classes file {STAGE1_CLASSES_FILE} not found.")
        
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _stage1_class_map = load_class_mapping(STAGE1_CLASSES_FILE)
        
        _stage1_model = OccluderNet(num_classes=len(_stage1_class_map))
        _stage1_model.load_state_dict(torch.load(STAGE1_MODEL_PATH, map_location=_device))
        _stage1_model.to(_device)
        _stage1_model.eval()
        
        _transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        print(f"Stage 1 model loaded from {STAGE1_MODEL_PATH}")
        print(f"Stage 1 classes: {_stage1_class_map}")
    
    return _stage1_model, _device, _stage1_class_map, _transform

def stage1_classify(roi_img):
    """
    Stage 1: Classify as 'filled' or 'jcc_pattern'
    
    Args:
        roi_img: OpenCV image (BGR) of the occluder
    
    Returns:
        str: 'filled' or 'jcc_pattern'
    """
    model, device, class_map, transform = get_stage1_model()
    
    # Preprocess
    img_tensor = transform(roi_img).unsqueeze(0).to(device)
    
    # Inference
    with torch.no_grad():
        outputs = model(img_tensor)
        _, predicted = torch.max(outputs, 1)
        class_idx = predicted.item()
    
    return class_map[class_idx]

def classify_occluder_two_stage(roi_img):
    """
    Two-stage classification pipeline with color-first priority.
    """
    # 1. Check basic color (Grey vs Blue)
    basic_state = classify_filled(roi_img)
    
    # 2. If it's Blue, it could be a JCC pattern
    if basic_state == 'blue_filled':
        # Stage 1: ML-based pattern type detection (structural)
        pattern_type = stage1_classify(roi_img)
        
        if pattern_type == 'jcc_pattern':
            # Stage 2: Detailed rule-based classification
            jcc_result = classify_jcc_pattern(roi_img)
            # If classify_jcc_pattern didn't find clear red/green, 
            # it returns blue_filled (handled inside classify_jcc_pattern)
            return jcc_result
            
    return basic_state

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
    
    # Add bounds checking to prevent empty ROI errors
    y1_l, y2_l = max(0, ly-lr), min(h, ly+lr)
    x1_l, x2_l = max(0, lx-lr), min(w, lx+lr)
    left_roi = img[y1_l:y2_l, x1_l:x2_l]
    
    y1_r, y2_r = max(0, ry-rr), min(h, ry+rr)
    x1_r, x2_r = max(0, rx-rr), min(w, rx+rr)
    right_roi = img[y1_r:y2_r, x1_r:x2_r]
    
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

def map_to_phoropter_state(os_class, od_class):
    """
    Map visual classes to phoropter state.
    os_class: Left Eye (OS) - Right side of UI
    od_class: Right Eye (OD) - Left side of UI
    """
    # 1. JCC States - RIGHT EYE (OD)
    if 'refine' in od_class:
        pattern = 'Axis' if 'axis' in od_class else 'Power'
        suffix = 'Flip1' if 'green' in od_class else 'Flip2'
        return f'Right_{pattern}_{suffix}'

    # 2. JCC States - LEFT EYE (OS)
    if 'refine' in os_class:
        pattern = 'Axis' if 'axis' in os_class else 'Power'
        suffix = 'Flip1' if 'green' in os_class else 'Flip2'
        return f'Left_{pattern}_{suffix}'
    
    # 3. Standard Occlusion States
    if os_class == 'blue_filled' and od_class == 'blue_filled':
        return 'BINO'
    
    if os_class == 'grey_filled' and od_class == 'grey_filled':
        return 'Both_Occluded'
    
    if os_class == 'grey_filled':
        return 'Left_Occluded'
    
    if od_class == 'grey_filled':
        return 'Right_Occluded'
    
    return f'Unknown(OS:{os_class},OD:{od_class})'


def extract(roi0_img, save_debug=False, output_dir='ROI_3', filename=None):
    """
    Main extraction function compatible with existing pipeline.
    Uses two-stage classification.
    
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
    # ui_left = icon on LEFT side of UI (Right Eye / OD)
    # ui_right = icon on RIGHT side of UI (Left Eye / OS)
    od_class = classify_occluder_two_stage(left_roi)
    os_class = classify_occluder_two_stage(right_roi)
    
    # Map to phoropter state
    phoropter_state = map_to_phoropter_state(os_class, od_class)
    
    result = {
        'roi_id': 'ROI_3_4',
        'bboxes': [
            {
                'label': 'right_eye_occluder', # Left side icon
                'box': left_bbox,
                'state': od_class
            },
            {
                'label': 'left_eye_occluder',  # Right side icon
                'box': right_bbox,
                'state': os_class
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
        
        # Save ROI-3 (left side icon - Right Eye)
        roi3_dir = 'ROI_3'
        os.makedirs(roi3_dir, exist_ok=True)
        roi3_path = os.path.join(roi3_dir, f'{prefix}_{now}_roi3_{od_class}.png')
        cv2.imwrite(roi3_path, left_roi)
        
        # Save ROI-4 (right side icon - Left Eye)
        roi4_dir = 'ROI_4'
        os.makedirs(roi4_dir, exist_ok=True)
        roi4_path = os.path.join(roi4_dir, f'{prefix}_{now}_roi4_{os_class}.png')
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

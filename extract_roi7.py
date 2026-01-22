
import cv2
import numpy as np
import datetime
import os
import json
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# Global variables to cache model and class mapping
_CHART_MODEL = None
_CLASS_MAPPING = None

def load_chart_model():
    """Loads the trained chart classification model and class mapping."""
    global _CHART_MODEL, _CLASS_MAPPING
    
    if _CHART_MODEL is not None:
        return _CHART_MODEL, _CLASS_MAPPING
    
    model_path = "chart_classifier.pth"
    mapping_path = "class_mapping.json"
    
    if not os.path.exists(model_path) or not os.path.exists(mapping_path):
        return None, None
        
    try:
        with open(mapping_path, 'r') as f:
            _CLASS_MAPPING = json.load(f)
        
        num_classes = len(_CLASS_MAPPING)
        model = models.resnet18()
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
        
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()
        _CHART_MODEL = model
        return _CHART_MODEL, _CLASS_MAPPING
    except Exception as e:
        print(f"[ROI7] Error loading model: {e}")
        return None, None

def classify_chart(roi_img, threshold=0.1):
    """Classifies the given chart image using the trained model.
    Returns empty string if below confidence threshold or error occurs.
    """
    if roi_img is None or roi_img.size == 0:
        return ""
        
    model, mapping = load_chart_model()
    if model is None:
        return ""
        
    try:
        preprocess = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        img_pil = Image.fromarray(cv2.cvtColor(roi_img, cv2.COLOR_BGR2RGB))
        input_tensor = preprocess(img_pil)
        input_batch = input_tensor.unsqueeze(0)
        
        with torch.no_grad():
            output = model(input_batch)
            probabilities = torch.nn.functional.softmax(output[0], dim=0)
            conf, preds = torch.max(probabilities, 0)
            
            if conf.item() < threshold:
                return ""
                
            class_idx = str(preds.item())
            return mapping.get(class_idx, "")
    except Exception as e:
        print(f"[ROI7] Error during classification: {e}")
        return ""

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
        return None, roi0_img, ""

    # Pick the best candidate (biggest rectangle with preferred proportion)
    candidates.sort(key=lambda x: x['score'], reverse=True)
    best = candidates[0]
    x, y, rw, rh = best['bbox']
    
    # Crop the chart region for classification
    chart_crop = search_area[y:y+rh, x:x+rw]
    chart_info = classify_chart(chart_crop)
    
    # Adjust coordinates to original ROI0
    x_abs = x + x_start
    y_abs = y + y_start
    
    labeled_img = roi0_img.copy()
    cv2.rectangle(labeled_img, (x_abs, y_abs), (x_abs+rw, y_abs+rh), (0, 255, 0), 2)
    label_text = f'ROI7: {chart_info}'
    cv2.putText(labeled_img, label_text, (x_abs, y_abs-10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    if debug:
        print(f"ROI7 found at: x={x_abs}, y={y_abs}, w={rw}, h={rh}, chart={chart_info}")
        
    return (x_abs, y_abs, rw, rh), labeled_img, chart_info


def extract(roi0_img, save_debug=False, filename=None, debug=False, bbox=None):
    """
    Extract function for ROI7, returns a result dict for test_extractors.
    If bbox is provided, skips detection and only performs classification.
    """
    chart_info = ""
    labeled_img = roi0_img.copy()
    
    if bbox is not None and len(bbox) == 4:
        x_abs, y_abs, rw, rh = bbox
        # Crop the chart region for classification
        # Need to convert absolute ROI0 coordinates back to search_area relative if we were using search_area
        # But here we can just crop from roi0_img directly
        chart_crop = roi0_img[y_abs:y_abs+rh, x_abs:x_abs+rw]
        chart_info = classify_chart(chart_crop)
        
        cv2.rectangle(labeled_img, (x_abs, y_abs), (x_abs+rw, y_abs+rh), (0, 255, 0), 2)
        label_text = f'ROI7: {chart_info}'
        cv2.putText(labeled_img, label_text, (x_abs, y_abs-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    else:
        bbox, labeled_img, chart_info = extract_roi7_from_roi0(roi0_img, debug=debug)

    result = {
        'roi_id': 'ROI7',
        'bbox': bbox,
        'chart_info': chart_info
    }
    
    if bbox and save_debug:
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

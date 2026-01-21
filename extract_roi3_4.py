import cv2
import numpy as np
import os
import datetime
import torch
import torch.nn.functional as F
from torchvision import transforms
from occluder_model import OccluderNet

# Function to load model once ideally, but here we load inside or use a global
MODEL_PATH = 'occluder_model.pth'
CLASSES_PATH = 'occluder_classes.txt'

class OccluderPredictor:
    def __init__(self, model_path, classes_path):
        self.device = torch.device("cpu") # Use CPU for inference to avoid complexity
        self.classes = self._load_classes(classes_path)
        self.model = OccluderNet(num_classes=len(self.classes))
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        else:
            print(f"Warning: Model file {model_path} not found. Predictions will be random.")
        self.model.to(self.device)
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def _load_classes(self, path):
        classes = {}
        if os.path.exists(path):
            with open(path, 'r') as f:
                for line in f:
                    idx, name = line.strip().split(':')
                    classes[int(idx)] = name
        else:
             # Fallback default
             return {0: 'blue_filled', 1: 'grey_filled', 2: 'grey_unfilled'}
        return classes

    def predict(self, img_bgr):
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        tensor = self.transform(img_rgb).unsqueeze(0).to(self.device)
        with torch.no_grad():
            outputs = self.model(tensor)
            _, predicted = torch.max(outputs, 1)
            idx = predicted.item()
            return self.classes.get(idx, "unknown")

# Global predictor instance
predictor = None

def get_predictor():
    global predictor
    if predictor is None:
        predictor = OccluderPredictor(MODEL_PATH, CLASSES_PATH)
    return predictor

def extract(roi0_img, save_debug=False, output_dir='ROI_3', filename=None):
    """
    Extract occluders and classify using OccluderNet.
    """
    if roi0_img is None:
        raise ValueError('Input image is None')

    if filename:
        input_base = os.path.splitext(os.path.basename(filename))[0]
        prefix = input_base[:4]
    else:
        prefix = 'roi0'

    # Resize to expected resolution
    img = cv2.resize(roi0_img, (929, 823))
    h, w = img.shape[:2]

    # Find Circles
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_blurred = cv2.GaussianBlur(gray, (9, 9), 2)
    circles = cv2.HoughCircles(gray_blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=200,
                               param1=50, param2=35, minRadius=30, maxRadius=70)

    if circles is None:
        return {'roi_id': 'ROI_3_4', 'bboxes': [], 'error': 'No circles detected'}

    circles = np.uint16(np.around(circles))
    detected_circles = sorted(circles[0, :], key=lambda x: x[0])
    
    if len(detected_circles) < 2:
        return {'roi_id': 'ROI_3_4', 'bboxes': [], 'error': f'Found {len(detected_circles)} circles'}

    # Sort vertically to find center ones (heuristic from before)
    mid_y = h / 2
    candidates = sorted(detected_circles, key=lambda c: abs(c[1] - mid_y))
    left_right = sorted(candidates[:2], key=lambda c: c[0])
    
    left_circle = left_right[0]
    right_circle = left_right[1]

    pred = get_predictor()

    # Process Left
    lx, ly, lr = left_circle
    # Pad crop slightly
    pad = 5
    roi3 = img[max(0, ly-lr-pad):min(h, ly+lr+pad), max(0, lx-lr-pad):min(w, lx+lr+pad)]
    state3_class = pred.predict(roi3) if roi3.size > 0 else "unknown"

    # Process Right
    rx, ry, rr = right_circle
    roi4 = img[max(0, ry-rr-pad):min(h, ry+rr+pad), max(0, rx-rr-pad):min(w, rx+rr+pad)]
    state4_class = pred.predict(roi4) if roi4.size > 0 else "unknown"

    # Scale back
    orig_h, orig_w = roi0_img.shape[:2]
    scale_x = orig_w / 929.0
    scale_y = orig_h / 823.0
    
    # Coordinates
    lx_abs = int((lx - lr) * scale_x)
    ly_abs = int((ly - lr) * scale_y)
    lw_abs = int(2 * lr * scale_x)
    lh_abs = int(2 * lr * scale_y)
    
    rx_abs = int((rx - rr) * scale_x)
    ry_abs = int((ry - rr) * scale_y)
    rw_abs = int(2 * rr * scale_x)
    rh_abs = int(2 * rr * scale_y)

    # Determine Aggregate State
    # Logic:
    # blue_filled / grey_filled / refine_filled -> OPEN
    # grey_unfilled -> OCCLUDED
    
    def is_open(cls_name):
        return cls_name in ['blue_filled', 'grey_filled', 'refine_filled'] # Treat refines as open/filled
    
    left_open = is_open(state3_class)
    right_open = is_open(state4_class)
    
    if left_open and right_open:
        agg_state = "BINO"
    elif not left_open and right_open:
        agg_state = "Left_Occluded"
    elif left_open and not right_open:
        agg_state = "Right_Occluded"
    else:
        agg_state = "Both_Occluded"

    result = {
        'roi_id': 'ROI_3_4',
        'bboxes': [
            {'label': 'left_occluder', 'box': [lx_abs, ly_abs, lw_abs, lh_abs], 'state': state3_class},
            {'label': 'right_occluder', 'box': [rx_abs, ry_abs, rw_abs, rh_abs], 'state': state4_class}
        ],
        'occluder_state': agg_state
    }
    
    if save_debug:
        now = datetime.datetime.now().strftime('%d%m_%H%M%S')
        roi3_dir = 'ROI_3'
        os.makedirs(roi3_dir, exist_ok=True)
        roi3_path = os.path.join(roi3_dir, f'{prefix}_{now}_roi3_{state3_class}.png')
        if roi3.size > 0: cv2.imwrite(roi3_path, roi3)
        
        roi4_dir = 'ROI_4'
        os.makedirs(roi4_dir, exist_ok=True)
        roi4_path = os.path.join(roi4_dir, f'{prefix}_{now}_roi4_{state4_class}.png')
        if roi4.size > 0: cv2.imwrite(roi4_path, roi4)
        
        result['image_paths'] = [roi3_path, roi4_path]

    return result

if __name__ == "__main__":
    roi0_dir = 'ROI_0'
    if os.path.exists(roi0_dir):
        files = sorted([f for f in os.listdir(roi0_dir) if f.endswith('.png') and 'box' not in f])
        if files:
            path = os.path.join(roi0_dir, files[-1]) # Last one
            img = cv2.imread(path)
            res = extract(img, save_debug=True, filename=path)
            print(res)
        else:
            print("No images in ROI_0")
    else:
        print("ROI_0 dir not found")

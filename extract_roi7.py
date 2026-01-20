import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
import os

def extract_roi7_from_roi0(roi0_img, debug=False):
    """
    Detects the acuity chart (ROI7) in the lower right region of ROI0.
    Returns the bounding box coordinates (x, y, w, h) and the labeled image.
    """
    h, w = roi0_img.shape[:2]
    y_start = int(0.6 * h)
    x_start = int(0.6 * w)
    search_area = roi0_img[y_start:, x_start:]

    # Preprocess
    gray = cv2.cvtColor(search_area, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rectangles = []
    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            x, y, rw, rh = cv2.boundingRect(approx)
            area = rw * rh
            aspect = rw / rh if rh > 0 else 0
            # Heuristic: reasonable size and aspect ratio
            if area > 1000 and 0.5 < aspect < 2.5:
                rectangles.append((x, y, rw, rh, area, approx))

    if not rectangles:
        if debug:
            print("No rectangles found in ROI7 search area.")
        return None, roi0_img

    # Pick the largest rectangle
    rectangles.sort(key=lambda x: x[4], reverse=True)
    x, y, rw, rh, _, approx = rectangles[0]
    # Adjust coordinates to original ROI0
    x_abs = x + x_start
    y_abs = y + y_start
    labeled_img = roi0_img.copy()
    cv2.rectangle(labeled_img, (x_abs, y_abs), (x_abs+rw, y_abs+rh), (0, 255, 0), 2)
    cv2.putText(labeled_img, 'ROI7', (x_abs, y_abs-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
    if debug:
        print(f"ROI7 found at: x={x_abs}, y={y_abs}, w={rw}, h={rh}")
    return (x_abs, y_abs, rw, rh), labeled_img


# Define the same CNN structure for loading
class ChartCNN(nn.Module):
    def __init__(self, num_classes):
        super(ChartCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def predict_chart(roi7_img):
    model_path = "chart_model.pth"
    classes_path = "chart_classes.txt"
    
    if not os.path.exists(model_path) or not os.path.exists(classes_path):
        return "Unknown (Model not found)"

    with open(classes_path, "r") as f:
        classes = [line.strip() for line in f.readlines()]
    
    num_classes = len(classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize and load model
    model = ChartCNN(num_classes).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Preprocess image
    gray = cv2.cvtColor(roi7_img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (64, 64), interpolation=cv2.INTER_AREA)
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    input_tensor = transform(resized).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(input_tensor)
        _, predicted = torch.max(outputs, 1)
        return classes[predicted.item()]


def extract(roi0_img, save_debug=False, filename=None, debug=False):
    """
    Extract function for ROI7, returns a result dict for test_extractors.
    """
    bbox, labeled_img = extract_roi7_from_roi0(roi0_img, debug=debug)
    
    chart_name = None
    if bbox:
        x, y, w, h = bbox
        roi7_crop = roi0_img[y:y+h, x:x+w]
        chart_name = predict_chart(roi7_crop)
        if debug:
            print(f"Predicted chart: {chart_name}")

    result = {
        'roi_id': 'ROI7',
        'bbox': bbox,
        'chart_name': chart_name
    }
    if bbox and save_debug:
        import os
        out_dir = 'ROI_7'
        os.makedirs(out_dir, exist_ok=True)
        # Use filename if provided, else default name
        if filename:
            base = os.path.splitext(os.path.basename(filename))[0]
            out_path = os.path.join(out_dir, f'labeled_{base}.png')
        else:
            out_path = os.path.join(out_dir, 'labeled_roi7.png')
        cv2.imwrite(out_path, labeled_img)
        result['debug_image'] = out_path
    if not bbox:
        result['error'] = 'ROI7 not found'
    return result

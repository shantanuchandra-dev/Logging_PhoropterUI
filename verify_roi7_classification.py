import cv2
import os
import json
from extract_roi7 import extract

def test_on_charts():
    chart_dir = "_Charts/e charts "
    # Get first image
    images = [f for f in os.listdir(chart_dir) if f.lower().endswith('.png') or f.lower().endswith('.jpg')]
    if not images:
        print("No images found in alphabetic chart dir")
        return
        
    img_path = os.path.join(chart_dir, images[0])
    img = cv2.imread(img_path)
    
    # We need to simulate ROI0 because extract_roi7 expects a full UI frame to search in.
    # However, the images in _Charts might be just the chart area or something else.
    # Let's see what's in ROI_0 or Sample to find a real full UI frame.
    
    # Alternatively, if the _Charts images are full UI frames, we can use them directly.
    # From the list_dir earlier, they look like screenshots.
    
    print(f"Testing on {img_path}")
    from extract_roi7 import classify_chart
    chart_info = classify_chart(img)
    print(f"Classified as: {chart_info}")

if __name__ == "__main__":
    test_on_charts()

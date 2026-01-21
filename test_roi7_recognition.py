import cv2
import extract_roi7
import os
import glob

def test_roi7_classification():
    # Define a test image path - picking one from the ROI_0 folder as seen in test_extractors.py
    test_images = glob.glob("MatchedScreens/*.png")
    if not test_images:
        print("No test images found in ROI_0/")
        return

    print(f"Testing ROI7 classification on {len(test_images)} images...")
    
    for img_path in test_images[:5]:  # Test first 5 images
        print(f"\nProcessing {img_path}...")
        img = cv2.imread(img_path)
        if img is None:
            print(f"Failed to load {img_path}")
            continue
            
        result = extract_roi7.extract(img, debug=True)
        
        if 'bbox' in result and result['bbox']:
            print(f"✓ ROI7 detected at {result['bbox']}")
            print(f"✓ Predicted Chart Name: {result.get('chart_name')}")
        else:
            print(f"✗ ROI7 not detected. Error: {result.get('error')}")

if __name__ == "__main__":
    test_roi7_classification()

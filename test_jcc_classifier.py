"""
Test script to validate JCC occluder detection on sample images.
Tests the model on the original JCC images to verify classification accuracy.
"""

import cv2
import os
import numpy as np
from extract_roi_occ_jcc_v2 import classify_occluder_two_stage as classify_occluder

def extract_circle_roi(img):
    """
    Extract the main circle from a JCC screenshot.
    Same logic as in prepare_jcc_data.py
    """
    h, w = img.shape[:2]
    
    # Find the largest circle
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_blurred = cv2.GaussianBlur(gray, (9, 9), 2)
    
    circles = cv2.HoughCircles(gray_blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=50,
                               param1=50, param2=30, minRadius=20, maxRadius=min(h, w)//2)
    
    if circles is None:
        # If no circle detected, return center crop
        center_x, center_y = w // 2, h // 2
        crop_size = min(h, w) // 2
        x1 = max(0, center_x - crop_size)
        y1 = max(0, center_y - crop_size)
        x2 = min(w, center_x + crop_size)
        y2 = min(h, center_y + crop_size)
        return img[y1:y2, x1:x2]
    
    # Get the largest circle
    circles = np.uint16(np.around(circles))
    largest_circle = max(circles[0, :], key=lambda c: c[2])
    
    cx, cy, r = int(largest_circle[0]), int(largest_circle[1]), int(largest_circle[2])
    
    # Expand radius slightly to include circumference
    r_expanded = int(r * 1.1)
    
    # Crop the circle region
    x1 = max(0, cx - r_expanded)
    y1 = max(0, cy - r_expanded)
    x2 = min(w, cx + r_expanded)
    y2 = min(h, cy + r_expanded)
    
    roi = img[y1:y2, x1:x2]
    
    # Validate ROI is not empty
    if roi.size == 0:
        center_x, center_y = w // 2, h // 2
        crop_size = min(h, w) // 2
        x1 = max(0, center_x - crop_size)
        y1 = max(0, center_y - crop_size)
        x2 = min(w, center_x + crop_size)
        y2 = min(h, center_y + crop_size)
        roi = img[y1:y2, x1:x2]
    
    return roi

def test_jcc_images():
    """Test classification on JCC sample images"""
    
    test_cases = [
        {
            'path': '_JCC/Axis_refine/Screenshot 2026-01-21 at 13.31.40.png',
            'expected': 'red_axis_refine',
            'description': 'Red circumference, no line (axis refine)'
        },
        {
            'path': '_JCC/Axis_refine/Screenshot 2026-01-21 at 13.31.54.png 14-16-13-503.png',
            'expected': 'green_axis_refine',
            'description': 'Green circumference, no line (axis refine)'
        },
        {
            'path': '_JCC/Power_refine/Screenshot 2026-01-21 at 13.32.35.png',
            'expected': 'red_power_refine',
            'description': 'Red circumference, with line (power refine)'
        },
        {
            'path': '_JCC/Power_refine/Screenshot 2026-01-21 at 13.33.04.png',
            'expected': 'green_power_refine',
            'description': 'Green circumference, with line (power refine)'
        }
    ]
    
    print("="*70)
    print("JCC OCCLUDER CLASSIFICATION TEST")
    print("="*70)
    
    correct = 0
    total = 0
    
    for test_case in test_cases:
        path = test_case['path']
        expected = test_case['expected']
        description = test_case['description']
        
        if not os.path.exists(path):
            print(f"\n❌ SKIP: {path} not found")
            continue
        
        img = cv2.imread(path)
        if img is None:
            print(f"\n❌ SKIP: Could not read {path}")
            continue
        
        # Extract circle ROI first (to match training data)
        roi = extract_circle_roi(img)
        
        # Classify the extracted ROI
        predicted = classify_occluder(roi)
        
        # Check result
        is_correct = (predicted == expected)
        total += 1
        if is_correct:
            correct += 1
            status = "✅ PASS"
        else:
            status = "❌ FAIL"
        
        print(f"\n{status}")
        print(f"  File: {os.path.basename(path)}")
        print(f"  Description: {description}")
        print(f"  Expected: {expected}")
        print(f"  Predicted: {predicted}")
    
    print("\n" + "="*70)
    print(f"RESULTS: {correct}/{total} correct ({100*correct/total:.1f}%)")
    print("="*70)

def test_blue_grey_filled():
    """Test blue and grey filled detection"""
    
    print("\n" + "="*70)
    print("BLUE/GREY FILLED CLASSIFICATION TEST")
    print("="*70)
    
    # Test a few blue filled from ROI_3/ROI_4
    blue_samples = [
        'ROI_3/cWLM_2201_135636_roi3.png',
        'ROI_4/cWLM_2201_135636_roi4.png'
    ]
    
    for path in blue_samples:
        if not os.path.exists(path):
            continue
        
        img = cv2.imread(path)
        if img is None:
            continue
        
        predicted = classify_occluder(img)
        is_correct = (predicted == 'blue_filled')
        status = "✅ PASS" if is_correct else "❌ FAIL"
        
        print(f"\n{status}")
        print(f"  File: {os.path.basename(path)}")
        print(f"  Expected: blue_filled")
        print(f"  Predicted: {predicted}")
    
    # Test synthetic grey filled
    grey_dir = 'jcc_occluder_dataset/grey_filled'
    if os.path.exists(grey_dir):
        grey_samples = [f for f in os.listdir(grey_dir) if f.endswith('.png')][:3]
        
        for filename in grey_samples:
            path = os.path.join(grey_dir, filename)
            img = cv2.imread(path)
            if img is None:
                continue
            
            predicted = classify_occluder(img)
            is_correct = (predicted == 'grey_filled')
            status = "✅ PASS" if is_correct else "❌ FAIL"
            
            print(f"\n{status}")
            print(f"  File: {filename}")
            print(f"  Expected: grey_filled")
            print(f"  Predicted: {predicted}")

if __name__ == "__main__":
    test_jcc_images()
    test_blue_grey_filled()
    
    print("\n" + "="*70)
    print("Testing complete!")
    print("="*70)

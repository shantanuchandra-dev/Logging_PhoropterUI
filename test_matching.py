import cv2
import os
import numpy as np
from pathlib import Path

def test_matching():
    # 1. Load the debug crop from the video
    query_img_path = 'debug_roi7_crop.png'
    if not os.path.exists(query_img_path):
        print(f"Error: {query_img_path} not found. Please verify the previous steps.")
        return

    query_img = cv2.imread(query_img_path)
    if query_img is None:
        print("Error loading query image.")
        return
    
    print(f"Query Image Size: {query_img.shape}")
    query_gray = cv2.cvtColor(query_img, cv2.COLOR_BGR2GRAY)

    # 2. Iterate through Reference Database
    charts_dir = "Charts_Processed"
    best_score = -1
    best_match = None
    
    results = []

    print("\nscanning reference images...")
    for root, dirs, files in os.walk(charts_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                ref_path = os.path.join(root, file)
                
                # Load reference
                ref_img = cv2.imread(ref_path)
                if ref_img is None: continue
                
                # Get the immediate parent folder name as the class
                class_name = Path(root).name
                
                # Strategy: Resize Query to Reference Size (Downsampling is better than upsampling)
                try:
                    query_resized = cv2.resize(query_gray, (ref_img.shape[1], ref_img.shape[0]))
                    ref_gray = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
                    
                    # Method 1: Template Matching (normalized) - closest to 1.0 is best
                    res = cv2.matchTemplate(query_resized, ref_gray, cv2.TM_CCOEFF_NORMED)
                    score = res[0][0]
                    
                    results.append((score, class_name, file))
                    
                    if score > best_score:
                        best_score = score
                        best_match = class_name
                        
                except Exception as e:
                    print(f"Error processing {file}: {e}")

    # 3. Sort and Show Results
    results.sort(key=lambda x: x[0], reverse=True)
    
    print("\n--- Top 5 Matches (Template Matching Score) ---")
    for i in range(min(5, len(results))):
        score, name, fname = results[i]
        print(f"{i+1}. {name} : {score:.4f} (File: {fname})")

if __name__ == "__main__":
    test_matching()

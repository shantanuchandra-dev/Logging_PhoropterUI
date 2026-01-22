import cv2
import os
import numpy as np
from pathlib import Path

def test_feature_matching():
    # 1. Load the debug crop from the video
    query_img_path = 'debug_roi7_crop.png'
    if not os.path.exists(query_img_path):
        print(f"Error: {query_img_path} not found.")
        return

    query_img = cv2.imread(query_img_path, cv2.IMREAD_GRAYSCALE)
    if query_img is None: return

    # Init ORB
    orb = cv2.ORB_create(nfeatures=1000)
    kp1, des1 = orb.detectAndCompute(query_img, None)
    
    if des1 is None:
        print("No features found in query image.")
        return

    # 2. Iterate through Reference Database
    charts_dir = "Charts_Processed"
    results = []
    
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    print(f"Scanning reference images using ORB (Query features: {len(kp1)})...")
    
    for root, dirs, files in os.walk(charts_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                ref_path = os.path.join(root, file)
                
                # Load reference
                ref_img = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE)
                if ref_img is None: continue
                
                class_name = Path(root).name
                
                try:
                    # Detect features
                    kp2, des2 = orb.detectAndCompute(ref_img, None)
                    
                    if des2 is None or len(kp2) < 2:
                        continue
                        
                    # Match
                    matches = bf.match(des1, des2)
                    
                    # Sort matches by distance
                    matches = sorted(matches, key=lambda x: x.distance)
                    
                    # Score: Number of "good" matches (distance < 50 is very strict for HAMMING, maybe 60-70)
                    # Or just sum of top 10 matches?
                    # Let's count matches with distance < 64
                    good_matches = [m for m in matches if m.distance < 64]
                    score = len(good_matches)
                    
                    # Normalize by number of features in reference to avoid bias towards complex images?
                    # No, we want absolute best match number.
                    
                    results.append((score, class_name, file))
                        
                except Exception as e:
                    print(f"Error processing {file}: {e}")

    # 3. Sort and Show Results
    results.sort(key=lambda x: x[0], reverse=True)
    
    print("\n--- Top 5 Matches (Number of Good Features) ---")
    for i in range(min(5, len(results))):
        score, name, fname = results[i]
        print(f"{i+1}. {name} : {score} matches (File: {fname})")

if __name__ == "__main__":
    test_feature_matching()

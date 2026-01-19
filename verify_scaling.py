import cv2
import os
import json
import extract_roi2
import extract_roi3_4

def verify():
    roi0_path = '/Users/shantanuchandra/Downloads/Logging_PhoropterUI/ROI_0/_-wu_2001_032132.png'
    img = cv2.imread(roi0_path)
    if img is None:
        print(f"Error: {roi0_path} not found")
        return

    print(f"Original image size: {img.shape}")

    # Run ROI2
    res2 = extract_roi2.extract(img, save_debug=True, output_dir='SCALING_VERIF')
    print(f"ROI2 bbox: {res2['pd_value_bbox']}")

    # Run ROI3_4
    res3_4 = extract_roi3_4.extract(img, save_debug=True, output_dir='SCALING_VERIF')
    print(f"ROI3_4 bboxes: {[b['box'] for b in res3_4['bboxes']]}")

    # Create visualization
    viz = img.copy()
    
    # Draw ROI2
    bx, by, bw, bh = res2['pd_value_bbox']
    cv2.rectangle(viz, (bx, by), (bx+bw, by+bh), (0, 255, 0), 2)
    cv2.putText(viz, "ROI2", (bx, by-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Draw ROI3_4
    for b in res3_4['bboxes']:
        bx, by, bw, bh = b['box']
        cv2.rectangle(viz, (bx, by), (bx+bw, by+bh), (255, 0, 0), 2)
        cv2.putText(viz, b['label'], (bx, by-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    out_path = 'SCALING_VERIF/all_bboxes_orig.png'
    os.makedirs('SCALING_VERIF', exist_ok=True)
    cv2.imwrite(out_path, viz)
    print(f"Verification image saved to: {out_path}")

if __name__ == "__main__":
    verify()

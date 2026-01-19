import cv2
import os
import extract_roi2

def test_extraction():
    roi0_path = '/Users/shantanuchandra/Downloads/Logging_PhoropterUI/ROI_0/_-wu_2001_032132.png'
    if not os.path.exists(roi0_path):
        print(f"Error: {roi0_path} not found.")
        return

    img = cv2.imread(roi0_path)
    if img is None:
        print(f"Error: Could not load {roi0_path}")
        return

    # Run extraction with debug saving
    result = extract_roi2.extract(img, save_debug=True, output_dir='ROI_2_TEST')
    
    print(f"Result for {os.path.basename(roi0_path)}:")
    print(f"  PD Value: {result.get('pd_value')}")
    print(f"  PD Value BBox: {result.get('pd_value_bbox')}")
    
    if result.get('pd_value_bbox'):
        bx, by, bw, bh = result['pd_value_bbox']
        print(f"  BBox Height: {bh}")
        
    print(f"  Debug Image: {result.get('image_path')}")

if __name__ == "__main__":
    test_extraction()

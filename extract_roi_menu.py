"""
This script extracts the menu panel (top bar) from the latest ROI-0 image and saves it as ROI_Menu.
"""
import cv2
import numpy as np
import pytesseract
import os
import datetime


def extract(roi0_img, save_debug=False, output_dir='ROI_Menu'):
    """
    Extract the menu panel (top bar) from ROI-0 image.
    
    Args:
        roi0_img: ROI-0 image (numpy array)
        save_debug: Whether to save debug images
        output_dir: Directory to save debug images
    
    Returns:
        dict: {
            'roi_id': 'ROI_Menu',
            'bbox': [x, y, w, h],  # Coordinates relative to ROI-0
            'ocr_text': 'extracted text',
            'image_path': 'path/to/debug_image.png' (if save_debug=True)
        }
    """
    img_h, img_w = roi0_img.shape[:2]
    menu_height = max(40, img_h // 10)  # At least 40px, or top 1/10th
    menu = roi0_img[0:menu_height, :]
    
    # OCR the Menu panel
    ocr_config = '--psm 6'
    txt = pytesseract.image_to_string(menu, config=ocr_config).strip()
    
    result = {
        'roi_id': 'ROI_Menu',
        'bbox': [0, 0, img_w, menu_height],  # x, y, w, h relative to ROI-0
        'ocr_text': txt
    }
    
    if save_debug:
        os.makedirs(output_dir, exist_ok=True)
        now = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        menu_path = os.path.join(output_dir, f'menu_{now}.png')
        cv2.imwrite(menu_path, menu)
        
        ocr_txt_path = os.path.join(output_dir, f'menu_{now}_ocr.txt')
        with open(ocr_txt_path, 'w') as f:
            f.write(txt)
        
        result['image_path'] = menu_path
    
    return result


if __name__ == '__main__':
    # Accept direct image path as argument
    import sys
    if len(sys.argv) > 1:
        roi0_path = sys.argv[1]
        if not os.path.isfile(roi0_path):
            raise FileNotFoundError(f'Provided ROI-0 image not found: {roi0_path}')
    else:
        roi0_dir = 'ROI_0'
        roi0_files = [f for f in os.listdir(roi0_dir) if f.endswith('.png') and 'box' not in f]
        if not roi0_files:
            raise FileNotFoundError('No ROI-0 images found in ROI_0 directory.')
        roi0_files.sort()
        roi0_path = os.path.join(roi0_dir, roi0_files[-1])

    img = cv2.imread(roi0_path)
    if img is None:
        raise FileNotFoundError(f'Could not load {roi0_path}')
    
    # Call the extract function with debug saving enabled
    result = extract(img, save_debug=True)
    print(f'Menu panel extracted: {result}')

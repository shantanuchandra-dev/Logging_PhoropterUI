"""
This script extracts the menu panel (top bar) from the latest ROI-0 image and saves it as ROI_Menu.
"""
import cv2
import numpy as np
# import pytesseract
import os
import datetime



if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        roi0_path = sys.argv[1]
        if not os.path.isfile(roi0_path):
            raise FileNotFoundError(f'Provided ROI-0 image not found: {roi0_path}')
    else:
        roi0_dir = 'ROI_0'
        if not os.path.exists(roi0_dir):
            raise FileNotFoundError('ROI_0 directory does not exist.')
        roi0_files = [f for f in os.listdir(roi0_dir) if f.endswith('.png') and 'box' not in f]
        if not roi0_files:
            raise FileNotFoundError('No ROI-0 images found in ROI_0 directory.')
        roi0_files.sort()
        roi0_path = os.path.join(roi0_dir, roi0_files[-1])

    img = cv2.imread(roi0_path)
    if img is None:
        raise FileNotFoundError(f'Could not load {roi0_path}')

    # Robust: Crop top 1/10th of image as menu panel
    img_h, img_w = img.shape[:2]
    menu_height = max(40, img_h // 10)  # At least 40px, or top 1/10th
    menu = img[0:menu_height, :]

    # Save Menu crop
    now = datetime.datetime.now().strftime('%d%m_%H%M%S')
    output_dir = 'ROI_Menu'
    os.makedirs(output_dir, exist_ok=True)
    input_base = os.path.splitext(os.path.basename(roi0_path))[0]
    prefix = input_base[:4]
    menu_path = os.path.join(output_dir, f'{prefix}_{now}_menu.png')
    cv2.imwrite(menu_path, menu)
    print(f'Menu panel saved to {menu_path}')

    # OCR the Menu panel (optional)
    # ocr_config = '--psm 6'
    # txt = pytesseract.image_to_string(menu, config=ocr_config)
    # ocr_txt_path = os.path.join(output_dir, f'{prefix}_{now}_ocr.txt')
    # with open(ocr_txt_path, 'w') as f:
    #     f.write(txt)
    # print(f'OCR result saved to {ocr_txt_path}')

def extract(roi0_img, save_debug=False, output_dir='ROI_Menu', filename=None):
    """
    Extracts the menu panel (top bar) from a given ROI-0 image.
    Returns a dict with ROI ID, bounding box, and OCR text.
    """
    img_h, img_w = roi0_img.shape[:2]
    menu_height = max(40, img_h // 10)
    menu = roi0_img[0:menu_height, :]
    bbox = [0, 0, img_w, menu_height]
    # ocr_config = '--psm 6'
    # ocr_text = pytesseract.image_to_string(menu, config=ocr_config)
    ocr_text = ''
    if save_debug:
        now = datetime.datetime.now().strftime('%d%m_%H%M%S')
        os.makedirs(output_dir, exist_ok=True)
        if filename:
            input_base = os.path.splitext(os.path.basename(filename))[0]
            prefix = input_base[:4]
        else:
            prefix = 'menu'
        menu_path = os.path.join(output_dir, f'{prefix}_{now}_menu.png')
        cv2.imwrite(menu_path, menu)
        # ocr_txt_path = os.path.join(output_dir, f'{prefix}_{now}_ocr.txt')
        # with open(ocr_txt_path, 'w') as f:
        #     f.write(ocr_text)
    return {
        'roi_id': 'ROI_Menu',
        'bbox': bbox,
        'ocr_text': ocr_text
    }

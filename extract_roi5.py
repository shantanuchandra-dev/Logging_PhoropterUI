import cv2
import numpy as np
import os
import pytesseract
import datetime

# 1. Broadly Crop the "Chart" area (User requested 60-75%)
def crop_tab_band(roi0_img, output_dir, prefix):
    h = roi0_img.shape[0]
    y1 = int(h * 0.60)
    y2 = int(h * 0.75)
    tab_band = roi0_img[y1:y2, :]
    os.makedirs(output_dir, exist_ok=True)
    now = datetime.datetime.now().strftime('%d%m_%H%M%S')
    crop_path = os.path.join(output_dir, f'{prefix}_{now}_tab_band.png')
    cv2.imwrite(crop_path, tab_band)
    return tab_band, crop_path, y1, y2

# Step 2: Preprocess (grayscale, enhanced)
def preprocess_image_enhanced(img):
    # Scale up for better OCR on small text
    img_scaled = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(img_scaled, cv2.COLOR_BGR2GRAY)
    
    # Simple Otsu thresholding
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # If the image is mostly dark (text is light), invert it
    # But usually text is dark on light background.
    if np.mean(thresh) < 127:
        thresh = cv2.bitwise_not(thresh)
        
    return thresh

def is_chart_like(word):
    import difflib
    import re
    word = word.upper().strip()
    # Remove any non-alphanumeric characters
    word = re.sub(r'[^A-Z0-9]', '', word)
    
    if "CHART" in word: # Added check for substring
        return True
    
    # Check for fuzzy match against CHART or CHART1-5
    variants = ["CHART", "CHART1", "CHART2", "CHART3", "CHART4", "CHART5"]
    for v in variants:
        if difflib.SequenceMatcher(None, word, v).ratio() > 0.6:
            return True
            
    # Check if word contains something like 'CHART' with some OCR errors
    # e.g., 'Chari', 'Chait', 'Cbart'
    if len(word) >= 4:
        if difflib.SequenceMatcher(None, word[:5], "CHART").ratio() > 0.6:
            return True
            
    return False

def extract_roi5_sc_v2(roi0_img, output_dir, prefix):
    """
    Revised logic for ROI5 SC extraction:
    1. Crop 60-75% height.
    2. Fuzzy match "Chart" on preprocessed crop.
    3. Crop horizontally with 10px vertical buffer.
    4. Aggressive contour matching (width > 2*height) for 5 tabs.
    5. Label and return coordinates.
    """
    now = datetime.datetime.now().strftime('%d%m_%H%M%S')
    
    # 1. Initial Height Crop
    tab_band, band_path, band_y1, band_y2 = crop_tab_band(roi0_img, output_dir, prefix)
    
    # 2. Preprocess for OCR
    proc_band = preprocess_image_enhanced(tab_band)
    cv2.imwrite(os.path.join(output_dir, f"{prefix}_{now}_proc_band.png"), proc_band)
    
    # 3. Fuzzy Match "Chart"
    chart_word_idx = -1
    psm_modes = ['--psm 11', '--psm 6', '--psm 3'] # Updated PSM modes
    
    # Scale factor used in preprocessing
    scale_factor = 2.0
    
    for psm in psm_modes:
        data = pytesseract.image_to_data(proc_band, config=psm, output_type=pytesseract.Output.DICT)
        print(f"Trying PSM: {psm}, found words: {[t for t in data['text'] if t.strip()]}")
        for i, text in enumerate(data['text']):
            if is_chart_like(text):
                chart_word_idx = i
                break
        if chart_word_idx != -1:
            print(f"Found 'Chart' using {psm}: '{data['text'][chart_word_idx]}'")
            break
            
    if chart_word_idx == -1:
        # Fallback: Try CLAHE on original size
        print("Fallback: Trying CLAHE preprocessing...")
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        gray_orig = cv2.cvtColor(tab_band, cv2.COLOR_BGR2GRAY)
        proc_band_clahe = clahe.apply(gray_orig)
        scale_factor = 1.0 # No scaling in fallback for now
        for psm in psm_modes:
            data = pytesseract.image_to_data(proc_band_clahe, config=psm, output_type=pytesseract.Output.DICT)
            for i, text in enumerate(data['text']):
                if is_chart_like(text):
                    chart_word_idx = i
                    break
            if chart_word_idx != -1:
                print(f"Found 'Chart' with CLAHE using {psm}: '{data['text'][chart_word_idx]}'")
                proc_band = proc_band_clahe # Update proc_band if CLAHE worked
                break

    if chart_word_idx == -1:
        print("Error: Could not find 'Chart' on cropped image after multiple attempts.")
        return [], None

    # Get OCR coordinates and scale back if needed
    ox = data['left'][chart_word_idx] / scale_factor
    oy = data['top'][chart_word_idx] / scale_factor
    ow = data['width'][chart_word_idx] / scale_factor
    oh = data['height'][chart_word_idx] / scale_factor
    
    print(f"Chart text found at band-relative coords: x={ox}, y={oy}, w={ow}, h={oh}")
    
    # Use the exact vertical bounds of the detected "Chart" text (no buffers)
    y_start = int(oy)
    y_end = int(oy + oh)
    
    line_crop = tab_band[y_start:y_end, :]
    cv2.imwrite(os.path.join(output_dir, f"{prefix}_{now}_line_crop.png"), line_crop)
    
    # 5. Aggressive Contour Detection
    gray_line = cv2.cvtColor(line_crop, cv2.COLOR_BGR2GRAY)
    
    # Use Canny edge detection
    edges = cv2.Canny(gray_line, 80, 200)
    cv2.imwrite(os.path.join(output_dir, f"{prefix}_{now}_edges.png"), edges)
    
    # Use Morphological Closing to connect edges of same tab but not between tabs
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    morphed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    cv2.imwrite(os.path.join(output_dir, f"{prefix}_{now}_morphed.png"), morphed)
    
    contours, _ = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    valid_contours = []
    for cnt in contours:
        bx, by, bw, bh = cv2.boundingRect(cnt)
        # Tabs are horizontally wide. Target width is ~70-100px.
        if bw > 50 and bh > 15:
            valid_contours.append((bx, by, bw, bh))
            
    # Sort by X
    valid_contours.sort(key=lambda b: b[0])

    # Merge/Split Logic:
    # If a contour is very wide, it contains multiple tabs. 
    # A single tab in this ROI0 seems to be ~92px based on previous run.
    tab_w_estimated = 92
    new_valid = []
    for bx, by, bw, bh in valid_contours:
        if bw > 130: # Likely 2 or more tabs
            num_tabs = int(round(bw / tab_w_estimated))
            if num_tabs < 2: num_tabs = 2
            actual_split_w = bw / num_tabs
            for i in range(num_tabs):
                new_valid.append((int(bx + i*actual_split_w), by, int(actual_split_w), bh))
        else:
            # Check if it's wide enough to be a tab
            if bw > 60:
                new_valid.append((bx, by, bw, bh))
    
    # Final cleanup: remove duplicates or highly overlapping ones
    final_valid = []
    if new_valid:
        new_valid.sort(key=lambda b: b[0])
        curr = new_valid[0]
        final_valid.append(curr)
        for i in range(1, len(new_valid)):
            next_box = new_valid[i]
            # If overlap > 50%, ignore
            if next_box[0] < curr[0] + curr[2] * 0.5:
                continue
            else:
                curr = next_box
                final_valid.append(curr)

    valid_contours = final_valid[:5] # Take first 5 from left
        
    # Labeling
    labeled_contours = []
    viz = line_crop.copy()
    for i, (bx, by, bw, bh) in enumerate(valid_contours):
        label = i + 1
        labeled_contours.append({
            'label': label,
            'x': bx,
            'y': by + y_start + band_y1, # Absolute Y in ROI0
            'w': bw,
            'h': bh,
            'rel_x': bx,
            'rel_y': by
        })
        cv2.rectangle(viz, (bx, by), (bx + bw, by + bh), (0, 255, 0), 2)
        cv2.putText(viz, str(label), (bx + 2, by + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
    viz_path = os.path.join(output_dir, f'{prefix}_{now}_final_contours.png')
    cv2.imwrite(viz_path, viz)
    
    print(f"Final contours detected: {len(labeled_contours)}")
    for lc in labeled_contours:
        print(f"Label {lc['label']}: x={lc['x']}, y={lc['y']}, w={lc['w']}, h={lc['h']}")
        
    return labeled_contours, viz_path


def extract(image, save_debug=False, output_dir='ROI_5', filename=None):
    """
    Extract ROI-5 tab info from image.
    Args:
        image: np.ndarray, input image
        filename: str, base filename (used for prefix)
        debug: bool, whether to print debug info
    Returns:
        results: list of labeled contours
        final_viz: path to final debug image
    """
    # Always save ROI-5 outputs to ROI_5 folder, not pipeline output_dir
    output_dir = 'ROI_5'
    if filename:
        prefix = os.path.splitext(os.path.basename(filename))[0][:4]
    else:
        prefix = 'roi5'
    results, final_viz = extract_roi5_sc_v2(image, output_dir, prefix)
    # Compute yellow ratio for each tab and select the one with max yellow
    def get_yellow_ratio(image, bbox):
        x, y, w, h = bbox
        roi = image[y:y+h, x:x+w]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        lower_yellow = np.array([20, 80, 80])
        upper_yellow = np.array([40, 255, 255])
        mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
        yellow_ratio = np.sum(mask > 0) / (w * h) if (w * h) > 0 else 0
        return yellow_ratio
    max_yellow = 0
    selected_tab = -1
    for i, tab in enumerate(results):
        bbox = (tab['x'], tab['y'], tab['w'], tab['h'])
        yellow_ratio = get_yellow_ratio(image, bbox)
        if yellow_ratio > max_yellow:
            max_yellow = yellow_ratio
            selected_tab = i
    # Prepare output dict for pipeline compatibility
    out = {
        'selected_tab': selected_tab,
        'bboxes': results,
        'viz_path': final_viz
    }
    if save_debug:
        if results:
            print(f"Success! Final debug image: {final_viz}")
            for lc in results:
                print(f"Label {lc['label']}: x={lc['x']}, y={lc['y']}, w={lc['w']}, h={lc['h']}")
            print(f"Selected tab (max yellow): {selected_tab}")
        else:
            print("Failed to extract contours.")
    return out

# For standalone usage
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        roi0_path = sys.argv[1]
    else:
        roi0_path = "ROI_0/3ym80YNRSvOOPQjDTAu7wg_14.png"
    img = cv2.imread(roi0_path)
    if img is None:
        print(f"Could not load {roi0_path}")
        exit(1)
    extract(img, roi0_path, debug=True)

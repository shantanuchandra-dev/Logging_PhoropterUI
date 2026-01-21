import cv2
import numpy as np
import pytesseract
import os
import datetime
import re

def extract_roi1_ocr(img, bboxes):
    """
    Extracts OCR values from ROI-1 table cells using absolute bboxes on ROI0.
    img: The full ROI0 image (not a cropped table image!)
    bboxes: List of (x1, y1, x2, y2) absolute coordinates on ROI0.
    """
    results = {}
    cell_labels = [
        ['R_Sph', 'S_Anchor', 'L_Sph'],
        ['R_Cyl', 'C_Anchor', 'L_Cyl'],
        ['R_Axis', 'A_Anchor', 'L_Axis'],
        ['R_Add', 'ADD_Anchor', 'L_Add'],
        ['R_blank', 'blank_anchor', 'L_blank']
    ]
    
    # Filter bboxes if we have more than 15 cells
    n_cols = 3
    n_expected = 15
    if len(bboxes) > n_expected:
        # Group by rows and find starting row
        n_rows_detected = len(bboxes) // n_cols
        rows = []
        for i in range(n_rows_detected):
            row_bboxes = bboxes[i * n_cols:(i + 1) * n_cols]
            rows.append(row_bboxes)
        
        # Find starting row where 2nd cell has 'S'
        start_row_idx = None
        for i, row_bboxes in enumerate(rows):
            if len(row_bboxes) >= 2:
                x1, y1, x2, y2 = row_bboxes[1]
                cell_img = img[y1:y2, x1:x2]
                gray = cv2.cvtColor(cell_img, cv2.COLOR_BGR2GRAY)
                _, bin_img = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
                text = pytesseract.image_to_string(bin_img, config='--oem 3 --psm 10 -c tessedit_char_whitelist=S')
                text = text.strip().upper()
                if 'S' in text:
                    start_row_idx = i
                    break
        
        # If no 'S' found, skip rows with 'R' in 1st or 'L' in 3rd cell
        if start_row_idx is None:
            for i, row_bboxes in enumerate(rows):
                if len(row_bboxes) >= 3:
                    # Check 1st cell for 'R'
                    x1, y1, x2, y2 = row_bboxes[0]
                    cell_img = img[y1:y2, x1:x2]
                    gray = cv2.cvtColor(cell_img, cv2.COLOR_BGR2GRAY)
                    _, bin_img = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
                    text1 = pytesseract.image_to_string(bin_img, config='--oem 3 --psm 10 -c tessedit_char_whitelist=R')
                    text1 = text1.strip().upper()
                    
                    # Check 3rd cell for 'L'
                    x1, y1, x2, y2 = row_bboxes[2]
                    cell_img = img[y1:y2, x1:x2]
                    gray = cv2.cvtColor(cell_img, cv2.COLOR_BGR2GRAY)
                    _, bin_img = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
                    text3 = pytesseract.image_to_string(bin_img, config='--oem 3 --psm 10 -c tessedit_char_whitelist=L')
                    text3 = text3.strip().upper()
                    
                    if 'R' in text1 or 'L' in text3:
                        continue
                    else:
                        start_row_idx = i
                        break
        
        if start_row_idx is None:
            start_row_idx = 0
        
        # Take exactly 5 rows
        filtered_rows = rows[start_row_idx:start_row_idx + 5]
        bboxes = []
        for row_bboxes in filtered_rows:
            if len(row_bboxes) == n_cols:
                bboxes.extend(row_bboxes)
        
        # ...existing code...
    
        
    def preprocess_image(cell_img, scale=4):
        """
        Preprocesses a cell image for OCR. Optimized for speed and clarity.
        """
        if cell_img is None or cell_img.size == 0:
            return []
            
        # 1. Upscale (Scale 4x is the sweet spot for speed vs accuracy)
        h, w = cell_img.shape[:2]
        new_size = (int(w * scale), int(h * scale))
        upscaled = cv2.resize(cell_img, new_size, interpolation=cv2.INTER_CUBIC)
        
        # 2. Grayscale
        gray = cv2.cvtColor(upscaled, cv2.COLOR_BGR2GRAY)
        
        # 3. CLAHE (Contrast Enhancement)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        versions = []
        # Version A: CLAHE (Best for most cases)
        versions.append(enhanced)
        
        # Version B: Simple binary threshold (Otsu)
        _, bin_simple = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        versions.append(bin_simple)
        
        # Version C: Adaptive threshold
        bin_adaptive = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        versions.append(bin_adaptive)
        
        return versions

    def quantize_0_25(val_str, force_sign=False):
        """Quantizes to nearest 0.25 increment, standardizing zeros."""
        try:
            val = float(val_str)
            quantized = round(val * 4) / 4
            if abs(quantized) < 0.001:
                return "0.00"
            return f"{quantized:+.2f}"
        except:
            return val_str

    def extract_value_with_ensemble(cell_img, label):
        """
        Optimized OCR ensemble: uses skeptical early exit and weighted voting for signs.
        """
        is_sph_cyl = 'Sph' in label or 'Cyl' in label
        is_add = 'Add' in label

        if is_sph_cyl or is_add:
            target_type = 'float'
            pattern = r'([+-]?\s*\d+\.\d{1,2})'
            whitelist = '+-0123456789. '
        elif 'Axis' in label:
            target_type = 'int'
            pattern = r'(\d{1,3})'
            whitelist = '0123456789'
        elif 'Anchor' in label:
            target_type = 'anchor'
            anchor_map = {'S_Anchor': 'S', 'C_Anchor': 'C', 'A_Anchor': 'A', 'ADD_Anchor': 'ADD'}
            whitelist = anchor_map.get(label, '')
            pattern = rf'({whitelist})' if whitelist else ''
        else:
            return None

        # Tiered processing: Start with fast 4x scale
        current_scale = 4
        image_versions = preprocess_image(cell_img, scale=current_scale)
        # Added PSM 4 for Axis (sometimes handled better as block)
        psm_modes = [7, 8, 11, 13, 4]
        
        candidates = []
        for psm in psm_modes:
            config = f'--oem 3 --psm {psm}'
            if whitelist:
                config += f' -c tessedit_char_whitelist={whitelist}'
            
            for img_version in image_versions:
                text = pytesseract.image_to_string(img_version, config=config)
                text = text.strip().replace(' ', '').replace('\n', '').replace('\r', '')
                
                res = None
                if target_type == 'float':
                    text = text.replace(',', '.')
                    match = re.search(pattern, text)
                    if match:
                        val_str = match.group(1).replace(' ', '')
                        if '.' not in val_str: val_str += '.00'
                        elif len(val_str.split('.')[1]) == 1: val_str += '0'
                        res = quantize_0_25(val_str, force_sign=is_sph_cyl)
                elif target_type == 'int':
                    match = re.search(pattern, text)
                    if match:
                        val_int = int(match.group(1))
                        # Common phoropter misreads for Axis 180
                        if val_int in [1, 18]: val_int = 180
                        if val_int <= 180: res = str(val_int)
                elif target_type == 'anchor':
                    if whitelist.upper() in text.upper(): res = whitelist

                if res:
                    candidates.append(res)
                    # SKEPTICAL EARLY EXIT:
                    # Negative/Zero values are common/safe - return immediately.
                    # Positive (+) values are skeptical - always require ensemble verification.
                    if psm in [7, 8]:
                        if res.startswith('-') or res == "0.00" or target_type != 'float':
                            return res
                
                if len(candidates) >= 6: break
            if len(candidates) >= 6: break

        if not candidates:
            return None
            
        # Consensus with HYPER-SKEPTICAL sign weights
        from collections import Counter
        counts = Counter(candidates)
        
        if is_sph_cyl:
             groups = {}
             for val, count in counts.items():
                 abs_val = val.replace('+', '').replace('-', '')
                 if abs_val not in groups: groups[abs_val] = []
                 groups[abs_val].append((val, count))
             
             best_abs = max(groups.keys(), key=lambda k: sum(c for v, c in groups[k]))
             variants = groups[best_abs]
             
             # SIGN SKEPTICISM: Heuristic preference for '-' in Phoropter UI
             # HALLUCINATION GUARD: '+' requires 2.5x more weight to win against a '-' or ambiguity.
             scores = {}
             has_minus = any(v.startswith('-') for v, c in variants)
             for val, count in variants:
                 if val.startswith('-'):
                     score = count * 3.0 # Heavy preference for negative
                 elif val == "0.00":
                     score = count * 2.0
                 else:
                     # Positive values must be very consistent to win
                     score = count * 1.0
                 scores[val] = score
             
             return max(scores.keys(), key=lambda k: scores[k])

        return counts.most_common(1)[0][0]

    for row in range(5):
        for col in range(3):
            idx = row * 3 + col
            label = cell_labels[row][col]
            if idx >= len(bboxes):
                results[label] = None
                continue
                
            x1, y1, x2, y2 = bboxes[idx]
            cell_img = img[y1:y2, x1:x2]
            
            value = extract_value_with_ensemble(cell_img, label)
            results[label] = value

    # Only print summary if there is a warning (too many blanks)
    main_keys = ['R_Sph', 'L_Sph', 'R_Cyl', 'L_Cyl', 'R_Axis', 'L_Axis', 'R_Add', 'L_Add']
    blank_count = sum(1 for k in main_keys if not results.get(k))
    if blank_count >= 5:
        # Save crops for debugging
        try:
            os.makedirs('ROI_1', exist_ok=True)
            for k in ['R_Sph', 'L_Sph']:
                k_idx = 0 if k == 'R_Sph' else 2
                if k_idx < len(bboxes):
                    bx1, by1, bx2, by2 = bboxes[k_idx]
                    cv2.imwrite(f'ROI_1/{k}_failed_crop.png', img[by1:by2, bx1:bx2])
        except:
            pass
        print(' | '.join(str(results.get(k) or '') for k in main_keys))
        print(f"[WARNING] extract_roi1_ocr: {blank_count}/8 fields blank. Accuracy might be low.")
        # We don't exit in Phase 3 to avoid stopping the whole pipeline, 
        # but we should definitely log it.
    
    return results
    return results


# Keep original main for testing
if __name__ == '__main__':
    import sys
    import json
    if len(sys.argv) > 1:
        img_path = sys.argv[1]
        img = cv2.imread(img_path)
        if img is not None:
            # If it's a full ROI-0 image
            if img.shape[0] > 600:
                res = extract(img, save_debug=True)
                # Removed debug print
            else:
                # If it's just the table crop
                if len(sys.argv) > 2:
                    bbox_path = sys.argv[2]
                    def parse_bbox_line(line):
                        nums = re.findall(r'(?:np\.int64\()?(-?\d+)(?:\))?', line)
                        return tuple(map(int, nums))
                    with open(bbox_path, 'r') as f:
                        bboxes = [parse_bbox_line(line) for line in f]
                    res = extract_roi1_ocr(img, bboxes)
                    # Removed debug print



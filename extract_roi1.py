import cv2
import numpy as np
import os
import datetime


def draw_and_save_grid_visualization(roi1_img, col_peaks, row_peaks, bboxes, output_dir='ROI_1', basename='roi1'):
    """
    Draws the grid and bounding boxes on the table image and saves the visualization.
    Returns the visualization image path.
    """
    vis_img = roi1_img.copy()
    height, width = vis_img.shape[:2]
    # Draw vertical lines
    for px in col_peaks:
        cv2.line(vis_img, (px, 0), (px, height), (0, 0, 255), 2)
    # Draw horizontal lines
    for py in row_peaks:
        cv2.line(vis_img, (0, py), (width, py), (0, 0, 255), 2)
    # Draw bounding boxes and cell numbers in top left
    cell_num = 1
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = min(height, width) / 600.0
    thickness = max(1, int(min(height, width) / 300))
    margin = int(5 * font_scale)
    for bbox in bboxes:
        x1, y1, x2, y2 = bbox
        cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 255, 0), 1)
        # Write cell number in the top left of the cell
        cv2.putText(vis_img, str(cell_num), (x1 + margin, y1 + int(20 * font_scale)), font, font_scale, (255, 0, 0), thickness, cv2.LINE_AA)
        cell_num += 1
    now = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    vis_path = os.path.join(output_dir, f'{basename}_roi1_grid_{now}.png')
    cv2.imwrite(vis_path, vis_img)
    print(f'ROI-1 with grid lines saved to {vis_path}')
    return vis_path
def calculate_and_save_bboxes(col_peaks, row_peaks, output_dir='ROI_1', basename='roi1'):
    """
    Calculates bounding boxes for each cell in the grid and saves them to a text file.
    Returns the list of bounding boxes and the file path.
    """
    bboxes = []
    for i in range(len(row_peaks)-1):
        for j in range(len(col_peaks)-1):
            x1 = col_peaks[j]
            x2 = col_peaks[j+1]
            y1 = row_peaks[i]
            y2 = row_peaks[i+1]
            bboxes.append((x1, y1, x2, y2))
    now = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    bbox_path = os.path.join(output_dir, f'{basename}_roi1_bboxes_{now}.txt')
    with open(bbox_path, 'w') as f:
        for bbox in bboxes:
            f.write(f'{bbox}\n')
    print(f'ROI-1 cell bounding boxes saved to {bbox_path}')
    return bboxes, bbox_path
def detect_grid_lines_hough(roi1_img, n_cols=3, n_rows=5):
    """
    Detects grid lines in the ROI-1 table image using Hough Line Transform.
    Returns column and row positions (peaks) for grid lines.
    """
    gray = cv2.cvtColor(roi1_img, cv2.COLOR_BGR2GRAY)
    # Adaptive threshold to binarize
    bin_img = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, 10)
    # Morphological operations to enhance lines
    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (roi1_img.shape[1]//12, 1))
    kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, roi1_img.shape[0]//12))
    morph_h = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, kernel_h)
    morph_v = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, kernel_v)
    # Save intermediate images
    now = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    os.makedirs('ROI_1', exist_ok=True)
    cv2.imwrite(f'ROI_1/intermediate_bin_{now}.png', bin_img)
    cv2.imwrite(f'ROI_1/intermediate_hlines_{now}.png', morph_h)
    cv2.imwrite(f'ROI_1/intermediate_vlines_{now}.png', morph_v)
    # Projection profiles for line detection
    h_proj = np.sum(morph_h, axis=1)
    v_proj = np.sum(morph_v, axis=0)
    # Find peaks in projection profiles
    from scipy.signal import find_peaks
    h_peaks, _ = find_peaks(h_proj, height=np.max(h_proj)*0.5, distance=roi1_img.shape[0]//10)
    v_peaks, _ = find_peaks(v_proj, height=np.max(v_proj)*0.5, distance=roi1_img.shape[1]//10)
    # Do NOT remove border lines; keep all detected peaks
    height, width = roi1_img.shape[:2]
    # h_peaks and v_peaks now include border lines if detected
    # Sort and ensure width > height for cells
    h_peaks = sorted(h_peaks)
    v_peaks = sorted(v_peaks)
    # Overlay detected lines for visualization
    overlay = roi1_img.copy()
    for py in h_peaks:
        cv2.line(overlay, (0, py), (width, py), (0, 255, 255), 2)
    for px in v_peaks:
        cv2.line(overlay, (px, 0), (px, height), (255, 255, 0), 2)
    cv2.imwrite(f'ROI_1/intermediate_grid_overlay_{now}.png', overlay)
    print(f'Intermediate images saved: bin, hlines, vlines, grid overlay')
    # Return peaks as grid lines
    return v_peaks, h_peaks
def extract_and_save_table_region(cropped_img, table_rect, output_dir='ROI_1', basename='roi1'):
    """
    Extracts the table region (ROI-1) from the cropped image and saves it as an image.
    Returns the ROI-1 image and its path.
    """
    x, y, w, h = table_rect
    # Add 5 pixels to the right and bottom
    roi1 = cropped_img[y:y+h+5, x:x+w+5]
    # Further crop from the top until the first complete end-to-end horizontal line is found
    # Use adaptive threshold and morphology to find hlines
    gray = cv2.cvtColor(roi1, cv2.COLOR_BGR2GRAY)
    bin_img = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, 10)
    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (roi1.shape[1]//12, 1))
    morph_h = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, kernel_h)
    # Scan rows for a complete horizontal line (all white pixels)
    found_top = 0
    for i in range(morph_h.shape[0]):
        if np.sum(morph_h[i, :] > 200) > 0.95 * morph_h.shape[1]:
            found_top = i
            break
    # Crop from found_top if a line is found and it's not too far down
    if found_top > 0 and found_top < roi1.shape[0]//3:
        roi1 = roi1[found_top:, :]
    now = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    roi1_path = os.path.join(output_dir, f'{basename}_roi1_{now}.png')
    cv2.imwrite(roi1_path, roi1)
    print(f'ROI-1 (table) saved to {roi1_path}')
    return roi1, roi1_path
def detect_centered_table(cropped_img):
    """
    Detect a centered 3x5 table in the cropped image using contour filtering and heuristics.
    Returns the bounding box (x, y, w, h) of the detected table region.
    """
    import pytesseract
    h_img, w_img = cropped_img.shape[:2]
    # Focus search on central region (60% width, 80% height)
    cx, cy = w_img // 2, h_img // 2
    search_w, search_h = int(w_img * 0.6), int(h_img * 0.8)
    x0, y0 = cx - search_w // 2, cy - search_h // 2
    central_crop = cropped_img[y0:y0+search_h, x0:x0+search_w]
    # OCR to find S, C, A, ADD, Blank in the central column
    ocr_config = '--psm 6'
    ocr_result = pytesseract.image_to_data(central_crop, config=ocr_config, output_type=pytesseract.Output.DICT)
    # Find all row label positions and heights
    row_labels = ['S', 'C', 'A', 'ADD', '']
    found_rows = []
    for i, text in enumerate(ocr_result['text']):
        t = text.strip().upper()
        if t in {'S', 'C', 'A', 'ADD'} or t == '':
            found_rows.append((t, ocr_result['left'][i], ocr_result['top'][i], ocr_result['width'][i], ocr_result['height'][i]))
    # Try to order by y (top)
    found_rows = sorted(found_rows, key=lambda x: x[2])
    # Try to match the sequence S, C, A, ADD, Blank (allow missing Blank)
    sequence = ['S', 'C', 'A', 'ADD']
    matched = []
    for label in sequence:
        for row in found_rows:
            if row[0] == label:
                matched.append(row)
                break
    # If at least S, C, A, ADD found in order, use their heights
    if len(matched) == 4:
        # Use the first row's height as canonical row height
        row_height = matched[0][4]
        # Use the y of S as the top
        y_start = matched[0][2]
        # Use the x and width of the leftmost label as the left
        x_left = min([row[1] for row in matched])
        x_right = max([row[1] + row[3] for row in matched])
        # Expand horizontally to cover the table (add padding)
        pad_x = int(search_w * 0.15)
        x1 = max(x0 + x_left - pad_x, 0)
        x2 = min(x0 + x_right + pad_x, w_img)
        # Table is 5 rows (S, C, A, ADD, Blank)
        n_rows = 5
        y1 = max(y0 + y_start, 0)
        y2 = min(y1 + n_rows * row_height, h_img)
        w = x2 - x1
        h = y2 - y1
        return (x1, y1, w, h)
    # Fallback: use previous grid/contour heuristics
    gray = cv2.cvtColor(central_crop, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 10)
    edges = cv2.Canny(thresh, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=central_crop.shape[1]//6, maxLineGap=15)
    verticals = []
    horizontals = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if abs(x1 - x2) < 10:
                verticals.append((x1, y1, x2, y2))
            elif abs(y1 - y2) < 10:
                horizontals.append((x1, y1, x2, y2))
        from sklearn.cluster import KMeans
        if len(verticals) >= 4 and len(horizontals) >= 6:
            v_coords = np.array([v[0] for v in verticals] + [v[2] for v in verticals]).reshape(-1, 1)
            kmeans_v = KMeans(n_clusters=4, n_init=10).fit(v_coords)
            col_peaks = sorted([int(c[0]) for c in kmeans_v.cluster_centers_])
            h_coords = np.array([h[1] for h in horizontals] + [h[3] for h in horizontals]).reshape(-1, 1)
            kmeans_h = KMeans(n_clusters=6, n_init=10).fit(h_coords)
            row_peaks = sorted([int(c[0]) for c in kmeans_h.cluster_centers_])
            # Use the outermost grid lines to define the table region
            x1, x2 = x0 + col_peaks[0], x0 + col_peaks[-1]
            y1, y2 = y0 + row_peaks[0], y0 + row_peaks[-1]
            w, h = x2 - x1, y2 - y1
            return (x1, y1, w, h)
    # Final fallback: use central region as table
    return (x0, y0, search_w, search_h)

def crop_and_subtract_menu(img_path, roi_menu_path=None, output_dir='ROI_1'):
    """
    Crop the input image in half, subtract the height of ROI_Menu, and save to ROI_1 folder.
    Returns the cropped image for further processing.
    """
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f'Could not load {img_path}')
    roi_menu_y2 = 0
    if roi_menu_path:
        roi_menu_img = cv2.imread(roi_menu_path)
        if roi_menu_img is not None:
            roi_menu_y2 = roi_menu_img.shape[0]
        else:
            print(f'Warning: Could not load ROI_Menu image: {roi_menu_path}')
    # Subtract menu height and crop top half
    img_cropped = img[roi_menu_y2:, :]
    h_full, w_full = img_cropped.shape[:2]
    cropped_half = img_cropped[:h_full//2, :]
    os.makedirs(output_dir, exist_ok=True)
    now = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    basename = os.path.splitext(os.path.basename(img_path))[0]
    crop_path = os.path.join(output_dir, f'half_{basename}_{now}.png')
    cv2.imwrite(crop_path, cropped_half)
    print(f'Cropped image saved to {crop_path}')
    return cropped_half, crop_path


if __name__ == '__main__':
    # Test pipeline: run on latest ROI_0 image and corresponding ROI_Menu image
    roi0_dir = 'ROI_0'
    roi_menu_dir = 'ROI_Menu'
    output_dir = 'ROI_1'
    # Find latest crop_ image in ROI_0
    roi0_files = [f for f in os.listdir(roi0_dir) if f.startswith('crop_') and f.endswith('.png')]
    roi0_files.sort()
    roi0_path = os.path.join(roi0_dir, roi0_files[-1]) if roi0_files else None
    if not roi0_path or not os.path.isfile(roi0_path):
        raise FileNotFoundError('No crop_ ROI_0 image found.')
    # Find corresponding ROI_Menu image (same prefix)
    prefix = os.path.splitext(os.path.basename(roi0_path))[0].replace('crop_', '').split('_')[0]
    roi_menu_files = [f for f in os.listdir(roi_menu_dir) if f.endswith('.png') and prefix in f]
    roi_menu_files.sort()
    roi_menu_path = os.path.join(roi_menu_dir, roi_menu_files[-1]) if roi_menu_files else None

    # Step 1: Crop and subtract menu
    cropped_img, crop_path = crop_and_subtract_menu(roi0_path, roi_menu_path, output_dir)
    # Step 2: Detect centered table
    table_rect = detect_centered_table(cropped_img)
    # Step 3: Extract and save table region
    roi1_img, roi1_path = extract_and_save_table_region(cropped_img, table_rect, output_dir, basename=prefix)
    # Step 4: Detect grid lines
    col_peaks, row_peaks = detect_grid_lines_hough(roi1_img, n_cols=3, n_rows=5)
    # Step 5: Calculate and save bounding boxes
    bboxes, bbox_path = calculate_and_save_bboxes(col_peaks, row_peaks, output_dir, basename=prefix)
    # Step 6: Draw and save grid visualization
    vis_path = draw_and_save_grid_visualization(roi1_img, col_peaks, row_peaks, bboxes, output_dir, basename=prefix)
    print('Test pipeline completed.')
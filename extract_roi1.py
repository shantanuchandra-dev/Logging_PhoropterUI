
import cv2
import numpy as np

def extract_roi1(img, roi_menu_img=None):
    # Optionally crop out ROI_Menu if provided
    if roi_menu_img is not None:
        roi_menu_y2 = roi_menu_img.shape[0]
        img_cropped = img[roi_menu_y2:, :]
    else:
        img_cropped = img

    h_full, w_full = img_cropped.shape[:2]
    roi0_top_half = img_cropped[:h_full//2, :]

    # --- Improved 3x5 Table Detection ---
    gray = cv2.cvtColor(roi0_top_half, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, 10)
    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
    kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))
    horizontal = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_h, iterations=2)
    vertical = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_v, iterations=2)
    grid = cv2.add(horizontal, vertical)
    contours, _ = cv2.findContours(grid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_area = 0
    table_rect = None
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        aspect = w / float(h)
        if area > max_area and 1.5 < aspect < 3.5 and w > 100 and h > 50:
            max_area = area
            table_rect = (x, y, w, h)
    if table_rect is None:
        raise Exception('No 3x5 table-like rectangle found in ROI-0 top half.')
    x, y, w, h = table_rect
    roi1 = roi0_top_half[y:y+h, x:x+w]

    # --- Robust grid detection using Hough Line Transform ---
    vis_grid = roi1.copy()
    gray_roi1 = cv2.cvtColor(roi1, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray_roi1, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=80, minLineLength=30, maxLineGap=10)
    verticals = []
    horizontals = []
    height, width = vis_grid.shape[:2]
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if abs(x1 - x2) < 10:
                verticals.append((x1, y1, x2, y2))
            elif abs(y1 - y2) < 10:
                horizontals.append((x1, y1, x2, y2))
        from sklearn.cluster import KMeans
        if len(verticals) >= 4:
            v_coords = np.array([v[0] for v in verticals] + [v[2] for v in verticals]).reshape(-1, 1)
            kmeans_v = KMeans(n_clusters=4, n_init=10).fit(v_coords)
            col_peaks = sorted([int(c[0]) for c in kmeans_v.cluster_centers_])
        else:
            col_peaks = np.linspace(0, width, 4, dtype=int)
        if len(horizontals) >= 6:
            h_coords = np.array([h[1] for h in horizontals] + [h[3] for h in horizontals]).reshape(-1, 1)
            kmeans_h = KMeans(n_clusters=6, n_init=10).fit(h_coords)
            row_peaks = sorted([int(c[0]) for c in kmeans_h.cluster_centers_])
        else:
            row_peaks = np.linspace(0, height, 6, dtype=int)
    else:
        col_peaks = np.linspace(0, width, 4, dtype=int)
        row_peaks = np.linspace(0, height, 6, dtype=int)
    bboxes = []
    for i in range(5):
        for j in range(3):
            x1 = col_peaks[j]
            x2 = col_peaks[j+1]
            y1 = row_peaks[i]
            y2 = row_peaks[i+1]
            bboxes.append((x1, y1, x2, y2))
    return {'roi1': roi1, 'bboxes': bboxes}

# If run as a script, keep original behavior
if __name__ == '__main__':
    import sys
    import os
    import datetime
    roi0_dir = 'ROI_0'
    roi0_files = [f for f in os.listdir(roi0_dir) if f.startswith('roi0_') and f.endswith('.png') and 'box' not in f]
    if not roi0_files:
        raise FileNotFoundError('No ROI-0 images found in ROI_0 directory.')
    roi0_files.sort()
    roi0_path = os.path.join(roi0_dir, roi0_files[-1])
    img = cv2.imread(roi0_path)
    if img is None:
        raise FileNotFoundError(f'Could not load {roi0_path}')
    result = extract_roi1(img)
    roi1 = result['roi1']
    bboxes = result['bboxes']
    output_dir = 'ROI_1'
    os.makedirs(output_dir, exist_ok=True)
    now = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    roi1_path = os.path.join(output_dir, f'roi1_{now}.png')
    cv2.imwrite(roi1_path, roi1)
    print(f'ROI-1 (table) saved to {roi1_path}')
    # Optionally, draw grid and save
    vis_grid = roi1.copy()
    height, width = vis_grid.shape[:2]
    for bbox in bboxes:
        x1, y1, x2, y2 = bbox
        cv2.rectangle(vis_grid, (x1, y1), (x2, y2), (0, 255, 0), 1)
    vis_path = os.path.join(output_dir, f'roi1_grid_{now}.png')
    cv2.imwrite(vis_path, vis_grid)
    print(f'ROI-1 with grid lines saved to {vis_path}')
    bbox_path = os.path.join(output_dir, f'roi1_bboxes_{now}.txt')
    with open(bbox_path, 'w') as f:
        for bbox in bboxes:
            f.write(f'{bbox}\n')
    print(f'ROI-1 cell bounding boxes saved to {bbox_path}')
import cv2
import numpy as np
import pytesseract
import os
import datetime


# Path to the latest ROI-0 image (assume most recent file in ROI_0)
roi0_dir = 'ROI_0'
roi0_files = [f for f in os.listdir(roi0_dir) if f.startswith('roi0_') and f.endswith('.png') and 'box' not in f]
if not roi0_files:
    raise FileNotFoundError('No ROI-0 images found in ROI_0 directory.')
roi0_files.sort()
roi0_path = os.path.join(roi0_dir, roi0_files[-1])

img = cv2.imread(roi0_path)
if img is None:
    raise FileNotFoundError(f'Could not load {roi0_path}')

# --- Find ROI_Menu to crop it out completely and track offset ---
roi_menu_dir = 'ROI_MENU'
roi_menu_files = [f for f in os.listdir(roi_menu_dir) if f.startswith('roi_menu_') and f.endswith('.png')]
roi_menu_files.sort()
roi_menu_path = os.path.join(roi_menu_dir, roi_menu_files[-1]) if roi_menu_files else None
roi_menu_y2 = 0
if roi_menu_path:
    menu_img = cv2.imread(roi_menu_path)
    if menu_img is not None:
        roi_menu_y2 = menu_img.shape[0]
        # Crop out the menu from the image for processing, but keep offset for output
        img_cropped = img[roi_menu_y2:, :]
    else:
        print('Warning: Could not load ROI_Menu image, not cropping menu.')
        img_cropped = img
else:
    print('Warning: No ROI_Menu image found, not cropping menu.')
    img_cropped = img



# Crop ROI-0 to top half (minus ROI_Menu)
h_full, w_full = img_cropped.shape[:2]
roi0_top_half = img_cropped[:h_full//2, :]

# Save the cropped top half image
output_dir = 'ROI_1'
os.makedirs(output_dir, exist_ok=True)
now = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
top_half_path = os.path.join(output_dir, f'roi0_top_half_{now}.png')
cv2.imwrite(top_half_path, roi0_top_half)
print(f'Cropped ROI-0 top half saved to {top_half_path}')


# --- Improved 3x5 Table Detection ---
gray = cv2.cvtColor(roi0_top_half, cv2.COLOR_BGR2GRAY)
thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, 10)

# Morphological operations to find horizontal and vertical lines
kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))
horizontal = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_h, iterations=2)
vertical = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_v, iterations=2)

# Combine lines to get grid intersections
grid = cv2.add(horizontal, vertical)
contours, _ = cv2.findContours(grid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Find the largest rectangle that could be a table
max_area = 0
table_rect = None
for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)
    area = w * h
    aspect = w / float(h)
    # Heuristic: Table is wide, not too thin, and large
    if area > max_area and 1.5 < aspect < 3.5 and w > 100 and h > 50:
        max_area = area
        table_rect = (x, y, w, h)


if table_rect is None:
    raise Exception('No 3x5 table-like rectangle found in ROI-0 top half.')

# Crop the detected table
x, y, w, h = table_rect
roi1 = roi0_top_half[y:y+h, x:x+w]
roi1_path = os.path.join(output_dir, f'roi1_{now}.png')
cv2.imwrite(roi1_path, roi1)
print(f'ROI-1 (table) saved to {roi1_path}')



# --- Robust grid detection using Hough Line Transform ---
vis_grid = roi1.copy()
gray_roi1 = cv2.cvtColor(roi1, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray_roi1, 50, 150, apertureSize=3)
lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=80, minLineLength=30, maxLineGap=10)

verticals = []
horizontals = []
height, width = vis_grid.shape[:2]
if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if abs(x1 - x2) < 10:  # vertical
            verticals.append((x1, y1, x2, y2))
        elif abs(y1 - y2) < 10:  # horizontal
            horizontals.append((x1, y1, x2, y2))

    # Cluster verticals and horizontals
    from sklearn.cluster import KMeans
    if len(verticals) >= 4:
        v_coords = np.array([v[0] for v in verticals] + [v[2] for v in verticals]).reshape(-1, 1)
        kmeans_v = KMeans(n_clusters=4, n_init=10).fit(v_coords)
        col_peaks = sorted([int(c[0]) for c in kmeans_v.cluster_centers_])
    else:
        col_peaks = np.linspace(0, width, 4, dtype=int)
    if len(horizontals) >= 6:
        h_coords = np.array([h[1] for h in horizontals] + [h[3] for h in horizontals]).reshape(-1, 1)
        kmeans_h = KMeans(n_clusters=6, n_init=10).fit(h_coords)
        row_peaks = sorted([int(c[0]) for c in kmeans_h.cluster_centers_])
    else:
        row_peaks = np.linspace(0, height, 6, dtype=int)
else:
    col_peaks = np.linspace(0, width, 4, dtype=int)
    row_peaks = np.linspace(0, height, 6, dtype=int)

# Draw vertical lines
for px in col_peaks:
    cv2.line(vis_grid, (px, 0), (px, height), (0, 0, 255), 2)
# Draw horizontal lines
for py in row_peaks:
    cv2.line(vis_grid, (0, py), (width, py), (0, 0, 255), 2)

# Save bounding boxes for each cell
bboxes = []
for i in range(5):
    for j in range(3):
        x1 = col_peaks[j]
        x2 = col_peaks[j+1]
        y1 = row_peaks[i]
        y2 = row_peaks[i+1]
        bboxes.append((x1, y1, x2, y2))
        cv2.rectangle(vis_grid, (x1, y1), (x2, y2), (0, 255, 0), 1)

roi1_path = os.path.join(output_dir, f'roi1_{now}.png')
cv2.imwrite(roi1_path, vis_grid)
print(f'ROI-1 with grid lines saved to {roi1_path}')

# Save bounding boxes to file
bbox_path = os.path.join(output_dir, f'roi1_bboxes_{now}.txt')
with open(bbox_path, 'w') as f:
    for bbox in bboxes:
        f.write(f'{bbox}\n')
print(f'ROI-1 cell bounding boxes saved to {bbox_path}')

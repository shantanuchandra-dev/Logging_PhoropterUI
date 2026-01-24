"""
JCC Occluder State Detection with Two-Stage Classification - V3
V3 Changes:
- No resizing of ROI-0
- Search only in 40-60% vertical region
- High contrast circle detection (20% minimum)
- Equidistant circle selection from VERTICAL center
- Similar size validation
- Debug images at each processing stage

Stage 1: ML-based pattern type detection (filled vs jcc_pattern)
Stage 2: Rule-based detailed classification
"""

import cv2
import numpy as np
import torch
from torchvision import transforms
from occluder_model import OccluderNet
import os

# Model configuration
STAGE1_MODEL_PATH = 'stage1_model.pth'
STAGE1_CLASSES_FILE = 'stage1_classes.txt'

# Import Stage 2 classifiers
import stage2_classifier
from stage2_classifier import classify_filled

# Load class mapping
def load_class_mapping(classes_file):
    """Load class index to name mapping"""
    class_map = {}
    with open(classes_file, 'r') as f:
        for line in f:
            line = line.strip()
            if ':' in line:
                idx, name = line.split(':', 1)
                class_map[int(idx)] = name
    return class_map

# Initialize Stage 1 model (lazy loading)
_stage1_model = None
_device = None
_stage1_class_map = None
_transform = None

def get_stage1_model():
    """Lazy load Stage 1 model and related objects"""
    global _stage1_model, _device, _stage1_class_map, _transform
    
    if _stage1_model is None:
        if not os.path.exists(STAGE1_MODEL_PATH):
            raise FileNotFoundError(f"Stage 1 model file {STAGE1_MODEL_PATH} not found. Train the model first.")
        if not os.path.exists(STAGE1_CLASSES_FILE):
            raise FileNotFoundError(f"Stage 1 classes file {STAGE1_CLASSES_FILE} not found.")
        
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _stage1_class_map = load_class_mapping(STAGE1_CLASSES_FILE)
        
        _stage1_model = OccluderNet(num_classes=len(_stage1_class_map))
        _stage1_model.load_state_dict(torch.load(STAGE1_MODEL_PATH, map_location=_device))
        _stage1_model.to(_device)
        _stage1_model.eval()
        
        _transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        print(f"Stage 1 model loaded from {STAGE1_MODEL_PATH}")
        print(f"Stage 1 classes: {_stage1_class_map}")
    
    return _stage1_model, _device, _stage1_class_map, _transform

def stage1_classify(roi_img):
    """
    Stage 1: Classify as 'filled' or 'jcc_pattern'
    
    Args:
        roi_img: OpenCV image (BGR) of the occluder
    
    Returns:
        str: 'filled' or 'jcc_pattern'
    """
    model, device, class_map, transform = get_stage1_model()
    
    # Preprocess
    img_tensor = transform(roi_img).unsqueeze(0).to(device)
    
    # Inference
    with torch.no_grad():
        outputs = model(img_tensor)
        _, predicted = torch.max(outputs, 1)
        class_idx = predicted.item()
    
    return class_map[class_idx]

def classify_occluder_two_stage(roi_img, cyl_axis=None):
    """
    Two-stage classification pipeline with color-first priority.
    """
    # 1. Check basic color (Grey vs Blue)
    basic_state = classify_filled(roi_img)
    
    # 2. If it's Blue, it could be a JCC pattern
    if basic_state == 'blue_filled':
        # Stage 1: ML-based pattern type detection (structural)
        pattern_type = stage1_classify(roi_img)
        
        if pattern_type == 'jcc_pattern':
            # Stage 2: Detailed rule-based classification
            # Pass cylinder axis to make it rotation-aware
            jcc_result = stage2_classifier.classify_jcc_pattern(roi_img, cyl_axis=cyl_axis)
            return jcc_result
            
    return basic_state

def extract_occluders(roi0_img, save_debug=False, debug_prefix='debug'):
    """
    Extract left and right occluder regions from ROI-0 image.
    V3: No resizing, search in 40-60% vertical region, intelligent contrast-based detection.
    
    Args:
        roi0_img: OpenCV image (BGR) of ROI-0
        save_debug: Whether to save debug images at each stage
        debug_prefix: Prefix for debug image filenames
    
    Returns:
        tuple: (left_roi, right_roi, left_bbox, right_bbox, debug_images) or (None, None, None, None, {}) if detection fails
               bbox format: [x, y, w, h]
               debug_images: dict of stage_name -> image_path
    """
    debug_images = {}
    
    if roi0_img is None:
        return None, None, None, None, debug_images
    
    # Work with original image - NO RESIZING
    img = roi0_img.copy()
    h, w = img.shape[:2]
    
    # Define search region: 40-60% of height, 30-70% of width (middle region)
    y_start = int(h * 0.4)
    y_end = int(h * 0.6)
    x_start = int(w * 0.3)
    x_end = int(w * 0.7)
    search_region = img[y_start:y_end, x_start:x_end]
    search_h, search_w = search_region.shape[:2]
    mid_y = search_h / 2
    
    # Debug: Save search region with vertical center line
    if save_debug:
        debug_dir = 'debug_stages'
        os.makedirs(debug_dir, exist_ok=True)
        search_region_vis = img.copy()
        # Draw search region rectangle
        cv2.rectangle(search_region_vis, (x_start, y_start), (x_end, y_end), (0, 255, 0), 2)
        # Draw vertical center line in the search region
        vertical_center_y = y_start + int(mid_y)
        cv2.line(search_region_vis, (x_start, vertical_center_y), (x_end, vertical_center_y), (0, 255, 255), 2)
        cv2.putText(search_region_vis, '40-60% Vert, 30-70% Horiz', (x_start + 10, y_start - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(search_region_vis, 'Vertical Center', (x_start + 10, vertical_center_y - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        path = os.path.join(debug_dir, f'{debug_prefix}_1_search_region.png')
        cv2.imwrite(path, search_region_vis)
        debug_images['search_region'] = path
    
    # Step 1: Convert to grayscale
    gray = cv2.cvtColor(search_region, cv2.COLOR_BGR2GRAY)
    
    if save_debug:
        path = os.path.join(debug_dir, f'{debug_prefix}_2a_grayscale.png')
        cv2.imwrite(path, gray)
        debug_images['grayscale'] = path
    
    # Step 2: Apply CLAHE for better contrast
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    if save_debug:
        path = os.path.join(debug_dir, f'{debug_prefix}_2b_contrast_enhanced.png')
        cv2.imwrite(path, enhanced)
        debug_images['contrast_enhanced'] = path
    
    # Step 3: Apply Gaussian blur
    blurred = cv2.GaussianBlur(enhanced, (9, 9), 2)
    
    if save_debug:
        path = os.path.join(debug_dir, f'{debug_prefix}_2c_blurred.png')
        cv2.imwrite(path, blurred)
        debug_images['blurred'] = path
    
    # Step 4: Detect edges for visualization
    edges = cv2.Canny(blurred, 50, 150)
    
    if save_debug:
        path = os.path.join(debug_dir, f'{debug_prefix}_2d_edges.png')
        cv2.imwrite(path, edges)
        debug_images['edges'] = path
    
    # Step 5: Try multiple parameter sets for circle detection
    circles = None
    param_sets = [
        # (param1, param2, minRadius, maxRadius, minDist)
        (100, 30, 25, 100, 100),  # High contrast, strict
        (80, 25, 25, 100, 100),   # Medium-high contrast
        (60, 20, 25, 100, 100),   # Medium contrast
        (50, 18, 20, 120, 80),    # Lower contrast, wider range
    ]
    
    for idx, (p1, p2, minR, maxR, minD) in enumerate(param_sets):
        circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=minD,
                                   param1=p1, param2=p2, minRadius=minR, maxRadius=maxR)
        
        if save_debug:
            # Visualize circles found with this parameter set
            test_vis = search_region.copy()
            if circles is not None:
                test_circles = np.uint16(np.around(circles))
                for i, (cx, cy, r) in enumerate(test_circles[0, :]):
                    cv2.circle(test_vis, (cx, cy), r, (255, 0, 0), 2)
                    cv2.circle(test_vis, (cx, cy), 2, (0, 0, 255), 3)
                    cv2.putText(test_vis, f'{i+1}', (cx - 10, cy - r - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                cv2.putText(test_vis, f'Params: p1={p1}, p2={p2}, Found={len(test_circles[0])} (unfiltered)', 
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            else:
                cv2.putText(test_vis, f'Params: p1={p1}, p2={p2}, Found=0', 
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            path = os.path.join(debug_dir, f'{debug_prefix}_3{chr(97+idx)}_circles_p1_{p1}_p2_{p2}.png')
            cv2.imwrite(path, test_vis)
            debug_images[f'circles_attempt_{idx+1}'] = path
        
        if circles is not None and len(circles[0]) >= 2:
            print(f"✓ Found {len(circles[0])} circles (unfiltered) with params: p1={p1}, p2={p2}")
            break
    
    if circles is None:
        print("Warning: No circles detected in 40-60% region with any parameter set")
        return None, None, None, None, debug_images
    
    circles = np.uint16(np.around(circles))
    detected_circles = circles[0, :]
    
    # Step 6: Filter circles based on edge validation
    # Only keep circles that have sufficient edge pixels on their circumference
    validated_circles = []
    
    for cx, cy, r in detected_circles:
        # Sample points on the circle circumference
        edge_count = 0
        total_samples = 36  # Sample every 10 degrees
        
        for angle in np.linspace(0, 2 * np.pi, total_samples, endpoint=False):
            # Calculate point on circumference
            x = int(cx + r * np.cos(angle))
            y = int(cy + r * np.sin(angle))
            
            # Check if point is within bounds and has an edge
            if 0 <= x < edges.shape[1] and 0 <= y < edges.shape[0]:
                # Check 3x3 neighborhood for edge pixels
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < edges.shape[1] and 0 <= ny < edges.shape[0]:
                            if edges[ny, nx] > 0:
                                edge_count += 1
                                break
        
        # Calculate edge density (percentage of circumference with edges)
        edge_density = edge_count / total_samples
        
        # Keep circles with at least 30% edge coverage
        if edge_density >= 0.30:
            validated_circles.append((cx, cy, r, edge_density))
    
    print(f"✓ After edge validation: {len(validated_circles)} circles (from {len(detected_circles)})")
    
    if save_debug:
        # Visualize validated circles
        validated_vis = search_region.copy()
        edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        
        for i, (cx, cy, r, density) in enumerate(validated_circles):
            cv2.circle(validated_vis, (cx, cy), r, (0, 255, 0), 2)
            cv2.circle(validated_vis, (cx, cy), 2, (0, 255, 0), 3)
            cv2.putText(validated_vis, f'{i+1}:{density:.0%}', (cx - 20, cy - r - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Also draw on edges
            cv2.circle(edges_colored, (cx, cy), r, (0, 255, 0), 2)
            cv2.circle(edges_colored, (cx, cy), 2, (0, 255, 0), 3)
        
        cv2.putText(validated_vis, f'Edge-Validated Circles: {len(validated_circles)}', 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        path = os.path.join(debug_dir, f'{debug_prefix}_3z_validated_circles.png')
        cv2.imwrite(path, validated_vis)
        debug_images['validated_circles'] = path
        
        path_edges = os.path.join(debug_dir, f'{debug_prefix}_3z_validated_on_edges.png')
        cv2.imwrite(path_edges, edges_colored)
        debug_images['validated_on_edges'] = path_edges
    
    if len(validated_circles) < 2:
        print(f"Warning: Found {len(validated_circles)} validated circles, need at least 2")
        return None, None, None, None, debug_images
    
    # NEW: Filter by expected radius (PD labels are often ~60-80px, occluders are ~25-45px)
    # Range broadened to 20-55 to ensure capture of smaller/blurry circles
    clinical_circles = []
    for cx, cy, r, density in validated_circles:
        if 20 <= r <= 55: # Clinical range for occluder circles
             clinical_circles.append((cx, cy, r))
        else:
             print(f"  > Ignoring circle with r={r} (outside clinical range 20-55)")
             
    if len(clinical_circles) < 2:
        print(f"Warning: Found {len(clinical_circles)} clinical circles after radius filtering, need at least 2")
        return None, None, None, None, debug_images

    # Convert back to simple format
    detected_circles = np.array(clinical_circles, dtype=np.uint16)
    
    # Filter circles that are roughly equidistant from VERTICAL center
    # and have similar sizes
    candidates = []
    for circle in detected_circles:
        cx, cy, r = circle
        dist_from_center = abs(cy - mid_y)
        candidates.append((circle, dist_from_center, r))
    
    # Sort by distance from vertical center
    candidates.sort(key=lambda x: x[1])
    
    # Find pairs with similar distance from vertical center and similar radius
    best_pair = None
    for i in range(len(candidates) - 1):
        c1, dist1, r1 = candidates[i]
        c2, dist2, r2 = candidates[i + 1]
        
        # Check if distances are similar (within 30%)
        dist_ratio = min(dist1, dist2) / max(dist1, dist2) if max(dist1, dist2) > 0 else 0
        
        # New: Use absolute tolerance if they are very close to center
        # If both are within 15 pixels of vertical center, the ratio doesn't matter much
        dist_ok = (dist_ratio > 0.7) or (dist1 < 15 and dist2 < 15)
        
        # Check if radii are similar (within 20%)
        radius_ratio = min(r1, r2) / max(r1, r2) if max(r1, r2) > 0 else 0
        
        if dist_ok and radius_ratio > 0.8:
            best_pair = (c1, c2)
            print(f"✓ Selected pair: dist_ratio={dist_ratio:.2f} (ok={dist_ok}), radius_ratio={radius_ratio:.2f}")
            break
    
    if best_pair is None:
        print("Warning: Could not find equidistant circles with similar sizes")
        return None, None, None, None, debug_images
    
    # Sort by x-coordinate (left to right)
    left_circle, right_circle = sorted(best_pair, key=lambda c: c[0])
    
    # Debug: Save selected circles
    if save_debug:
        selected_vis = search_region.copy()
        # Draw vertical center line
        cv2.line(selected_vis, (0, int(mid_y)), (search_w, int(mid_y)), (0, 255, 255), 2)
        cv2.putText(selected_vis, 'Vertical Center', (10, int(mid_y) - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Draw all candidates in blue
        for circle, dist, r in candidates:
            cx, cy, cr = circle
            cv2.circle(selected_vis, (cx, cy), cr, (255, 200, 0), 1)
        
        # Draw selected circles in green
        lx, ly, lr = left_circle
        rx, ry, rr = right_circle
        cv2.circle(selected_vis, (lx, ly), lr, (0, 255, 0), 3)
        cv2.circle(selected_vis, (lx, ly), 2, (0, 255, 0), 3)
        cv2.putText(selected_vis, 'ROI3 (Right Eye)', (lx - 50, ly - lr - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        cv2.circle(selected_vis, (rx, ry), rr, (0, 255, 0), 3)
        cv2.circle(selected_vis, (rx, ry), 2, (0, 255, 0), 3)
        cv2.putText(selected_vis, 'ROI4 (Left Eye)', (rx - 50, ry + rr + 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        path = os.path.join(debug_dir, f'{debug_prefix}_4_final_selection.png')
        cv2.imwrite(path, selected_vis)
        debug_images['final_selection'] = path
    
    # Extract ROIs
    lx, ly, lr = left_circle
    rx, ry, rr = right_circle
    
    # Adjust coordinates back to full image space
    lx += x_start
    ly += y_start
    rx += x_start
    ry += y_start
    
    # Add bounds checking to prevent empty ROI errors
    y1_l, y2_l = max(0, ly-lr), min(h, ly+lr)
    x1_l, x2_l = max(0, lx-lr), min(w, lx+lr)
    left_roi = img[y1_l:y2_l, x1_l:x2_l]
    
    y1_r, y2_r = max(0, ry-rr), min(h, ry+rr)
    x1_r, x2_r = max(0, rx-rr), min(w, rx+rr)
    right_roi = img[y1_r:y2_r, x1_r:x2_r]
    
    # Bounding boxes in original ROI0 resolution (no scaling needed)
    left_bbox = [
        int(lx - lr),
        int(ly - lr),
        int(2 * lr),
        int(2 * lr)
    ]
    
    right_bbox = [
        int(rx - rr),
        int(ry - rr),
        int(2 * rr),
        int(2 * rr)
    ]
    
    return left_roi, right_roi, left_bbox, right_bbox, debug_images

def map_to_phoropter_state(os_class, od_class):
    """
    Map visual classes to phoropter state.
    os_class: Left Eye (OS) - Right side of UI
    od_class: Right Eye (OD) - Left side of UI
    """
    # 1. JCC States - RIGHT EYE (OD)
    if 'refine' in od_class:
        pattern = 'Axis' if 'axis' in od_class else 'Power'
        suffix = 'Flip1' if 'green' in od_class else 'Flip2'
        return f'Right_{pattern}_{suffix}'

    # 2. JCC States - LEFT EYE (OS)
    if 'refine' in os_class:
        pattern = 'Axis' if 'axis' in os_class else 'Power'
        suffix = 'Flip1' if 'green' in os_class else 'Flip2'
        return f'Left_{pattern}_{suffix}'
    
    # 3. Standard Occlusion States
    if os_class == 'blue_filled' and od_class == 'blue_filled':
        return 'BINO'
    
    if os_class == 'grey_filled' and od_class == 'grey_filled':
        return 'Both_Occluded'
    
    if os_class == 'grey_filled':
        return 'Left_Occluded'
    
    if od_class == 'grey_filled':
        return 'Right_Occluded'
    
    return f'Unknown(OS:{os_class},OD:{od_class})'


def extract(roi0_img, save_debug=False, output_dir='ROI_3', filename=None, timestamp_str=None, stored_bboxes=None, right_axis=None, left_axis=None):
    """
    Main extraction function compatible with existing pipeline.
    Uses two-stage classification.
    
    Args:
        roi0_img: OpenCV image (BGR) of ROI-0
        save_debug: Whether to save debug images
        output_dir: Directory to save debug images
        filename: Original filename for naming debug outputs
        timestamp_str: Optional timestamp string (e.g., '02:38') for debug filenames
    
    Returns:
        dict: {
            'roi_id': 'ROI_3_4',
            'bboxes': [
                {'label': 'left_occluder', 'box': [x, y, w, h], 'state': 'visual_class'},
                {'label': 'right_occluder', 'box': [x, y, w, h], 'state': 'visual_class'}
            ],
            'phoropter_state': 'BINO' | 'Left_Occluded' | 'Right_Occluded' | etc.,
            'image_paths': [...] (if save_debug=True)
        }
    """
    # Generate debug prefix
    import datetime
    now = datetime.datetime.now().strftime('%d%m_%H%M%S')
    prefix_parts = []
    if filename:
        input_base = os.path.splitext(os.path.basename(filename))[0]
        prefix_parts.append(input_base[:4])
    
    if timestamp_str:
        # Clean timestamp (replace : with _)
        t_clean = timestamp_str.replace(':', '_')
        prefix_parts.append(f"t{t_clean}")
    
    prefix_parts.append(now)
    debug_prefix = "_".join(prefix_parts)

    # Initialize variables
    left_roi = None
    right_roi = None
    left_bbox = None
    right_bbox = None
    debug_images = {}

    # USE STORED BBOXES IF PROVIDED (Phase 3 persistence)
    if stored_bboxes and len(stored_bboxes) == 2:
        try:
            # ROI3 = left side icon (Right Eye / OD)
            # ROI4 = right side icon (Left Eye / OS)
            b1 = stored_bboxes[0]['box']
            b2 = stored_bboxes[1]['box']
            
            lx, ly, lw, lh = b1
            rx, ry, rw, rh = b2

            h, w = roi0_img.shape[:2]
            left_roi = roi0_img[max(0,ly):min(h,ly+lh), max(0,lx):min(w,lx+lw)]
            right_roi = roi0_img[max(0,ry):min(h,ry+rh), max(0,rx):min(w,rx+rw)]
            
            left_bbox = b1
            right_bbox = b2
        except Exception as e:
            print(f"Error in manual ROI3/4 crop: {e}")
            return {'roi_id': 'ROI_3_4', 'phoropter_state': 'Error', 'error': str(e)}

    # OTHERWISE RUN DETECTION (Phase 2 or redetection)
    else:
        left_roi, right_roi, left_bbox, right_bbox, debug_images = extract_occluders(
            roi0_img, save_debug=save_debug, debug_prefix=debug_prefix
        )
    
    if left_roi is None or right_roi is None:
        return {
            'roi_id': 'ROI_3_4',
            'bboxes': [],
            'phoropter_state': 'Detection_Failed',
            'error': 'Failed to detect occluders'
        }
    
    # Classify each occluder
    # Use eye-specific axis for rotation-aware JCC
    od_class = classify_occluder_two_stage(left_roi, cyl_axis=right_axis)
    os_class = classify_occluder_two_stage(right_roi, cyl_axis=left_axis)
    
    # Map to phoropter state
    phoropter_state = map_to_phoropter_state(os_class, od_class)
    
    result = {
        'roi_id': 'ROI_3_4',
        'bboxes': [
            {
                'label': 'right_occluder',  # ROI3 - Left side icon (Right Eye / OD)
                'box': left_bbox,
                'state': od_class
            },
            {
                'label': 'left_occluder',   # ROI4 - Right side icon (Left Eye / OS)
                'box': right_bbox,
                'state': os_class
            }
        ],
        'phoropter_state': phoropter_state
    }
    
    # Save debug images if requested
    if save_debug:
        # Save ROI-3 (left side icon - Right Eye)
        roi3_dir = 'ROI_3'
        os.makedirs(roi3_dir, exist_ok=True)
        roi3_path = os.path.join(roi3_dir, f'{debug_prefix}_4_roi3_{od_class}.png')
        cv2.imwrite(roi3_path, left_roi)
        
        # Save ROI-4 (right side icon - Left Eye)
        roi4_dir = 'ROI_4'
        os.makedirs(roi4_dir, exist_ok=True)
        roi4_path = os.path.join(roi4_dir, f'{debug_prefix}_5_roi4_{os_class}.png')
        cv2.imwrite(roi4_path, right_roi)
        
        # Combine all debug images
        all_debug_paths = list(debug_images.values()) + [roi3_path, roi4_path]
        result['image_paths'] = all_debug_paths
        result['debug_stages'] = {
            **debug_images,
            'roi3_extracted': roi3_path,
            'roi4_extracted': roi4_path
        }
    
    return result

if __name__ == "__main__":
    # Test with a sample ROI-0 image
    roi0_dir = 'ROI_0'
    roi0_files = [f for f in os.listdir(roi0_dir) if f.endswith('.png') and 'box' not in f]
    
    if not roi0_files:
        print('No ROI-0 images found in ROI_0 directory.')
        exit(1)
    
    roi0_files.sort()
    roi0_path = os.path.join(roi0_dir, roi0_files[-1])
    
    print(f"Testing with: {roi0_path}")
    img = cv2.imread(roi0_path)
    
    if img is None:
        print(f'Could not load {roi0_path}')
        exit(1)
    
    # Run extraction - save_debug=False to avoid saving debug images
    result = extract(img, save_debug=False, filename=roi0_path)
    
    print("\n" + "="*60)
    print("RESULT:")
    print("="*60)
    print(f"ROI ID: {result['roi_id']}")
    print(f"Phoropter State: {result.get('phoropter_state', 'N/A')}")
    print("\nOccluders:")
    for bbox_info in result.get('bboxes', []):
        print(f"  {bbox_info['label']}: {bbox_info['state']}")
        print(f"    Box: {bbox_info['box']}")
    
    if 'image_paths' in result:
        print(f"\nDebug images saved:")
        for path in result['image_paths']:
            print(f"  {path}")
    
    if 'error' in result:
        print(f"\nError: {result['error']}")

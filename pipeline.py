#!/usr/bin/env python3
"""
Automated ROI Extraction Pipeline

This script automates the extraction of all available ROIs from a video file.
It performs periodic ROI detection, data extraction, CSV logging, and visualization.
"""

import cv2
import numpy as np
import json
import csv
import os
import sys
import glob
import datetime
from pathlib import Path

# Import ROI extractors
import extract_roi0
import extract_roi_menu
import extract_roi1_ocr
import extract_roi2  # Using extract_roi2 instead of extract_roi2_temp
import extract_roi3_4
import extract_roi5
import extract_roi6
import extract_roi7


def detect_gpu():
    """
    Detect if GPU is available for acceleration.
    Returns: dict with 'available', 'device', and 'backend' keys
    """
    gpu_info = {
        'available': False,
        'device': 'CPU',
        'backend': None
    }
    
    # Check PyTorch CUDA
    try:
        import torch
        if torch.cuda.is_available():
            gpu_info['available'] = True
            gpu_info['device'] = torch.cuda.get_device_name(0)
            gpu_info['backend'] = 'PyTorch CUDA'
            return gpu_info
    except ImportError:
        pass
    
    # Check OpenCV CUDA
    try:
        if cv2.cuda.getCudaEnabledDeviceCount() > 0:
            gpu_info['available'] = True
            gpu_info['device'] = 'CUDA Device'
            gpu_info['backend'] = 'OpenCV CUDA'
            return gpu_info
    except:
        pass
    
    return gpu_info


def load_config(config_path="config.json"):
    """Load pipeline configuration from JSON file."""
    default_config = {
        "video_source_dir": "Sample/videos",
        "reference_image": "topcon_ui_001.png",
        "output_dir": "roi_all",
        "sampling_interval_seconds": 2,
        "match_threshold": 0.8,
        "max_consecutive_failures": 5,
        "save_debug_images": True
    }
    
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            user_config = json.load(f)
            default_config.update(user_config)
    
    # Ensure results paths are relative to the final output_dir
    out_dir = default_config["output_dir"]
    default_config["csv_output"] = os.path.join(out_dir, "results.csv")
    default_config["json_output"] = os.path.join(out_dir, "results.json")
    
    return default_config


def verify_ui_present(frame, reference_template, threshold=0.8):
    """
    Verify that the medical UI is present in the frame using template matching.
    Returns: (is_present, match_score)
    """
    h_frame, w_frame = frame.shape[:2]
    h_template, w_template = reference_template.shape[:2]
    
    # Resize template if needed
    if h_frame != h_template or w_frame != w_template:
        resized_template = cv2.resize(reference_template, (w_frame, h_frame))
    else:
        resized_template = reference_template
    
    # Match on top half
    process_frame = frame[0:h_frame//2, 0:w_frame]
    process_template = resized_template[0:h_frame//2, 0:w_frame]
    
    res = cv2.matchTemplate(process_frame, process_template, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    
    return (max_val >= threshold, max_val)


import scan_video

# ... (detect_gpu, load_config, verify_ui_present remain same)

def extract_all_rois(roi0_img, gpu_available=False, save_debug=True, output_dir='MatchedScreens', filename=None):
    """
    Extract all ROIs from ROI-0 image.
    """
    results = {
        'timestamp': datetime.datetime.now().isoformat(),
        'rois': {}
    }
    
    # 1. Menu
    try:
        menu_result = extract_roi_menu.extract(roi0_img, save_debug=save_debug, output_dir='ROI_Menu', filename=filename)
        results['rois']['menu'] = menu_result
    except Exception as e:
        results['rois']['menu'] = {'error': str(e)}
    # 2. ROI1 (Table)
    try:
        roi1_result = extract_roi1_ocr.extract(roi0_img, save_debug=save_debug, output_dir='ROI_1', filename=filename)
        results['rois']['roi1'] = roi1_result
    except Exception as e:
        results['rois']['roi1'] = {'error': str(e)}
    # 3. ROI2 (PD)
    try:
        roi2_result = extract_roi2.extract(roi0_img, save_debug=save_debug, output_dir='ROI_2', filename=filename)
        results['rois']['roi2'] = roi2_result
    except Exception as e:
        results['rois']['roi2'] = {'error': str(e)}
    # 4. ROI3/ROI4 (Occluders)
    try:
        roi3_4_result = extract_roi3_4.extract(roi0_img, save_debug=save_debug, output_dir='ROI_3', filename=filename)
        results['rois']['roi3_4'] = roi3_4_result
    except Exception as e:
        results['rois']['roi3_4'] = {'error': str(e)}
    # 5. ROI5 (Chart Tabs)
    try:
        roi5_result = extract_roi5.extract(roi0_img, save_debug=save_debug, output_dir='ROI_5', filename=filename)
        results['rois']['roi5'] = roi5_result
    except Exception as e:
        results['rois']['roi5'] = {'error': str(e)}
    # 6. ROI6 (Chart Grid)
    try:
        roi6_result = extract_roi6.extract(roi0_img, save_debug=save_debug, output_dir='ROI_6', filename=filename)
        results['rois']['roi6'] = roi6_result
    except Exception as e:
        results['rois']['roi6'] = {'error': str(e)}
    # 7. ROI7 (Big Chart)
    try:
        roi6_data = results['rois'].get('roi6', {})
        roi7_result = extract_roi7.extract(roi0_img, roi6_data=roi6_data, save_debug=save_debug, output_dir='ROI_7', filename=filename)
        results['rois']['roi7'] = roi7_result
    except Exception as e:
        results['rois']['roi7'] = {'error': str(e)}
    
    return results


def save_visualization(frame, roi0_bbox, all_roi_data, output_path):
    """
    Draw all ROI bounding boxes on the frame and save visualization.
    """
    viz = frame.copy()
    
    # Draw ROI-0 bbox
    if roi0_bbox:
        x, y, w, h = roi0_bbox
        cv2.rectangle(viz, (x, y), (x+w, y+h), (0, 255, 0), 3)
        cv2.putText(viz, "ROI-0", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Draw other ROIs (relative to ROI-0)
    roi0_x, roi0_y = roi0_bbox[0], roi0_bbox[1] if roi0_bbox else (0, 0)
    
    for roi_name, roi_data in all_roi_data.get('rois', {}).items():
        if 'error' in roi_data or 'bbox' not in roi_data:
            continue
        
        bbox = roi_data['bbox']
        if not bbox or len(bbox) < 4:
            continue
        
        # Convert relative to absolute coordinates
        abs_x = roi0_x + bbox[0]
        abs_y = roi0_y + bbox[1]
        abs_w = bbox[2]
        abs_h = bbox[3]
        
        # Draw bbox
        color = (255, 0, 0)  # Blue
        cv2.rectangle(viz, (abs_x, abs_y), (abs_x+abs_w, abs_y+abs_h), color, 2)
        cv2.putText(viz, roi_name, (abs_x, abs_y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    cv2.imwrite(output_path, viz)
    return output_path


def append_to_csv(csv_path, frame_data):
    """
    Append extracted data to CSV file with specific headers:
    R_SPH, R_CYL, R_AXIS, R_ADD, L_SPH, L_CYL, L_AXIS, L_ADD, PD, Chart_Number, Occluder_State, Chart_Display
    """
    rois = frame_data.get('rois', {})
    
    # 1. Extraction from Table (ROI-1)
    table = rois.get('table', {}).get('data', {})
    
    # 2. PD (ROI-2)
    pd_val = rois.get('pd', {}).get('pd_value', '')
    
    # 3. Chart Number (ROI-5)
    chart_num = rois.get('chart_tabs', {}).get('selected_tab', -1)
    if chart_num != -1:
        chart_num += 1  # 1-based index
    
    # 4. Occluder State (ROI-3/4)
    # Logic:
    # Both Blue (filled) -> BINO
    # Left Grey (unfilled) -> Left_Occluded
    # Right Grey (unfilled) -> Right_Occluded
    # Both Grey (unfilled) -> Both_Occluded
    
    occ_state = "Unknown"
    if 'occluders' in rois and 'bboxes' in rois['occluders']:
        occs = rois['occluders']['bboxes']
        left_active = False
        right_active = False
        
        for occ in occs:
            state = occ.get('state', '').lower()
            # Use '(blue)' to strictly match the filled state
            is_active = "(blue)" in state
            if occ.get('label') == 'left_occluder':
                left_active = is_active
            elif occ.get('label') == 'right_occluder':
                right_active = is_active
        
        if left_active and right_active:
            occ_state = "BINO"
        elif not left_active and not right_active:
            occ_state = "Both_Occluded"
        elif not left_active:
            occ_state = "Left_Occluded"
        elif not right_active:
            occ_state = "Right_Occluded"

    # 5. Chart Display (ROI-7)
    chart_display = rois.get('big_chart', {}).get('chart_info', '')

    # Prepare Row
    row = {
        'R_SPH': table.get('R_Sph', ''),
        'R_CYL': table.get('R_Cyl', ''),
        'R_AXIS': table.get('R_Axis', ''),
        'R_ADD': table.get('R_Add', ''),
        'L_SPH': table.get('L_Sph', ''),
        'L_CYL': table.get('L_Cyl', ''),
        'L_AXIS': table.get('L_Axis', ''),
        'L_ADD': table.get('L_Add', ''),
        'PD': pd_val,
        'Chart_Number': chart_num,
        'Occluder_State': occ_state,
        'Chart_Display': chart_display
    }
    
    # Write to CSV
    headers = ['R_SPH', 'R_CYL', 'R_AXIS', 'R_ADD', 'L_SPH', 'L_CYL', 'L_AXIS', 'L_ADD', 'PD', 'Chart_Number', 'Occluder_State', 'Chart_Display']
    
    file_exists = os.path.exists(csv_path)
    with open(csv_path, 'a', newline='') as f:
        import csv
        writer = csv.DictWriter(f, fieldnames=headers)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def main():
    """Main pipeline execution."""
    print("=" * 60)
    print("Automated ROI Extraction Pipeline")
    print("=" * 60)
    
    # 1. GPU Detection
    gpu_info = detect_gpu()
    if gpu_info['available']:
        print(f"✓ GPU Available: {gpu_info['device']} ({gpu_info['backend']})")
    else:
        print("ℹ GPU Not Detected - Running on CPU")
    
    # 2. Load Configuration
    config = load_config()
    print(f"\nConfiguration:")
    print(f"  Video Source: {config['video_source_dir']}")
    print(f"  Output Directory: {config['output_dir']}")
    
    # 3. Setup Output Directories
    os.makedirs(config['output_dir'], exist_ok=True)
    os.makedirs("firstFrame", exist_ok=True)
    
    # 4. Find Video File
    video_files = glob.glob(os.path.join(config['video_source_dir'], "*"))
    video_files = [f for f in video_files if os.path.isfile(f) and not f.lower().endswith('.ds_store')]
    
    if not video_files:
        print(f"\n✗ No videos found in {config['video_source_dir']}")
        return
    
    print(f"\nFound {len(video_files)} videos to process.")
    
    for video_path in video_files:
        video_filename = os.path.basename(video_path)
        video_basename = os.path.splitext(video_filename)[0]
        
        # Performance overrides for CSV/JSON outputs
        video_csv_path = os.path.join(config['output_dir'], f"{video_basename}.csv")
        video_json_path = os.path.join(config['output_dir'], f"{video_basename}.json")
        
        print(f"\n{'='*60}")
        print(f"Processing Video: {video_filename}")
        print(f"{'='*60}")
        print(f"✓ Output CSV: {video_csv_path}")
        
        # Close capture if left open
        cap = None
        
        # --- PHASE 1: Find First UI Frame ---
        print("\n[PHASE 1] Scanning for first UI frame...")
        first_frame, first_time_sec, first_frame_idx = scan_video.find_first_ui_frame(video_path, config)
        
        if first_frame is None:
            print(f"✗ No UI found in {video_filename}. Skipping.")
            continue
        
        # --- PHASE 2: Fix Coordinates on First Frame ---
        print(f"\n[PHASE 2] Setting reference coordinates on first frame (t={first_time_sec:.2f}s)...")
        try:
            roi0_result = extract_roi0.extract_roi0(first_frame,save_dir='ROI_0',save=True)
            roi0_img = roi0_result['roi0']
            
            # Extract all ROIs to establish baseline coordinates
            ref_data = extract_all_rois(
                roi0_img,
                gpu_available=gpu_info['available'],
                save_debug=True,
                output_dir=config['output_dir'],
                filename=video_filename
            )
            ref_data['frame_id'] = first_frame_idx
            ref_data['time_seconds'] = first_time_sec
            
            # Store Reference Coordinates
            ref_path = os.path.join(config['output_dir'], f"{video_basename}_coords.json")
            with open(ref_path, 'w') as f:
                json.dump(ref_data, f, indent=2)
            print(f"  ✓ Reference coordinates stored: {ref_path}")
            
        except Exception as e:
            print(f"✗ Failed to set reference on first frame: {e}")
            continue

        # --- PHASE 3: Process Video and Store Every Change ---
        print("\n[PHASE 3] Processing video and logging changes...")
        cap = cv2.VideoCapture(video_path)
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if video_fps <= 0:
            print("✗ Invalid video FPS. Skipping.")
            cap.release()
            continue

        sampling_interval_frames = int(video_fps * config['sampling_interval_seconds'])
        all_results = [ref_data]
        append_to_csv(video_csv_path, ref_data)
        
        # Start processing from first_frame_idx
        frame_count = first_frame_idx
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
        
        extraction_count = 1
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            time_seconds = frame_count / video_fps
            
            # Periodic sampling
            if frame_count % sampling_interval_frames != 0:
                continue
                
            print(f"\r→ Frame {frame_count} / {total_frames} (t={time_seconds:.2f}s)", end="")
            
            try:
                # Check UI presence still
                is_present, _ = verify_ui_present(frame, cv2.imread(config['reference_image']), config['match_threshold'])
                if not is_present:
                    continue
                    
                roi0_res = extract_roi0.extract_roi0(frame)
                current_roi_data = extract_all_rois(
                    roi0_res['roi0'],
                    gpu_available=gpu_info['available'],
                    save_debug=False,
                    output_dir=config['output_dir']
                )
                
                current_roi_data['frame_id'] = frame_count
                current_roi_data['time_seconds'] = time_seconds
                
                # Save results
                all_results.append(current_roi_data)
                append_to_csv(video_csv_path, current_roi_data)
                
                extraction_count += 1
                
            except Exception:
                continue

        cap.release()
        
        # Save final JSON
        with open(video_json_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        print(f"\n✓ Video {video_filename} complete. Extractions: {extraction_count}")

    print(f"\n\n{'='*60}")
    print("All Pipeline Tasks Complete!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()

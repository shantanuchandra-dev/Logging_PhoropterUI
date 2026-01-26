#!/usr/bin/env python3
"""
Automated ROI Extraction Pipeline (Parallel Version)
"""

import cv2
import numpy as np
import json
import csv
import os
import sys
import glob
import datetime
import multiprocessing
import traceback
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

# Import ROI extractors
import extract_roi0
import extract_roi_menu
import extract_roi1, extract_roi1_ocr
import extract_roi2 
import extract_roi3_4_jcc_SC as extract_roi3_4 
import extract_roi5
import extract_roi6
import extract_roi7
import scan_video

def detect_gpu():
    """Detect if GPU is available (MPS or CUDA)."""
    gpu_info = {'available': False, 'device': 'CPU', 'backend': None}
    try:
        import torch
        if torch.cuda.is_available():
            gpu_info.update({'available': True, 'device': torch.cuda.get_device_name(0), 'backend': 'PyTorch CUDA'})
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            gpu_info.update({'available': True, 'device': 'Apple Metal (MPS)', 'backend': 'PyTorch MPS'})
    except ImportError: pass
    if not gpu_info['available']:
        try:
            if cv2.cuda.getCudaEnabledDeviceCount() > 0:
                gpu_info.update({'available': True, 'device': 'CUDA Device', 'backend': 'OpenCV CUDA'})
        except: pass
    return gpu_info

def load_config(config_path="config.json"):
    """Load pipeline configuration."""
    default_config = {
        "video_source_dir": "Sample/videos",
        "reference_image": "topcon_ui_001.png",
        "output_dir": "MatchedScreens",
        "sampling_interval_seconds": 2,
        "match_threshold": 0.35,
        "max_consecutive_failures": 5,
        "save_debug_images": False,
        "frame_diff_threshold": 0.0002,
        "roi0_diff_threshold": 0.0005
    }
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            default_config.update(json.load(f))
    return default_config

def verify_ui_present(frame, reference_template, threshold=0.8):
    """Verify UI presence."""
    h_f, w_f = frame.shape[:2]
    h_t, w_t = reference_template.shape[:2]
    res_t = cv2.resize(reference_template, (w_f, h_f)) if (h_f, w_f) != (h_t, w_t) else reference_template
    res = cv2.matchTemplate(frame[0:h_f//2, 0:w_f], res_t[0:h_f//2, 0:w_f], cv2.TM_CCOEFF_NORMED)
    _, max_val, _, _ = cv2.minMaxLoc(res)
    return (max_val >= threshold, max_val)

def calculate_image_difference(img1, img2):
    """Calculates normalized difference."""
    if img1 is None or img2 is None: return 1.0
    if img1.shape != img2.shape: img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    diff = cv2.absdiff(img1, img2)
    if len(diff.shape) == 3: diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    return np.mean(diff) / 255.0

def extract_all_rois(roi0_img, gpu_available=False, save_debug=True, output_dir='MatchedScreens', filename=None, timestamp_str=None, video_basename=None):
    """Extract all ROIs."""
    results = {'timestamp': datetime.datetime.now().isoformat(), 'rois': {}}
    coords_data = {}
    
    if not save_debug and output_dir:
        v_base = video_basename or (filename and os.path.splitext(os.path.basename(filename))[0])
        if v_base:
            for d in [output_dir, '.']:
                c = os.path.join(d, f'{v_base}_coords.json')
                if os.path.isfile(c):
                    try:
                        with open(c, 'r') as f: coords_data = json.load(f)
                        break
                    except: pass

    # Set contextual logging info for sub-modules
    for module in [extract_roi1_ocr, extract_roi2, extract_roi3_4, extract_roi5, extract_roi7]:
        if hasattr(module, 'set_log_context'):
            module.set_log_context(filename or video_basename, timestamp_str)

    # 1. Menu
    try: results['rois']['menu'] = extract_roi_menu.extract(roi0_img, save_debug=save_debug, output_dir='ROI_Menu', filename=filename)
    except Exception as e: results['rois']['menu'] = {'error': str(e)}

    # 2. ROI1 (OCR)
    try:
        bboxes = []
        if save_debug:
            roi0_base = os.path.splitext(os.path.basename(filename))[0] if filename else 'roi0'
            roi0_path = os.path.join('ROI_0', f'{roi0_base}.png')
            if not os.path.isfile(roi0_path):
                os.makedirs('ROI_0', exist_ok=True)
                cv2.imwrite(roi0_path, roi0_img)
            roi1_res = extract_roi1.extract(roi0_path, roi0_dir='ROI_0', roi_menu_dir='ROI_Menu', output_dir='ROI_1')
            results['rois']['roi1'] = roi1_res
            bboxes = roi1_res.get('cell_bboxes_on_roi0', [])
        else:
            bboxes = coords_data.get('rois', {}).get('roi1', {}).get('cell_bboxes_on_roi0', [])
            bboxes = [tuple(bb) for bb in bboxes if len(bb) == 4]
            results['rois']['roi1'] = {'cell_bboxes_on_roi0': bboxes}
        
        if bboxes:
            ocr_res = extract_roi1_ocr.extract_roi1_ocr(roi0_img, bboxes)
            results['rois']['roi1_ocr'] = ocr_res
        else:
            results['rois']['roi1_ocr'] = {'error': 'No bboxes for OCR'}
    except Exception as e:
        results['rois']['roi1_ocr'] = {'error': str(e)}
        print(f"[OCR ERROR] {e}")

    # 3. ROI2 (PD) - Static placeholder or extraction logic
    results['rois']['roi2'] = {'pd_value': ''}

    # 4. ROI3/ROI4 (Occluders)
    try:
        stored = coords_data.get('rois', {}).get('roi3_4', {}).get('bboxes', None) if not save_debug else None
        r_ax = results.get('rois', {}).get('roi1_ocr', {}).get('R_Axis') or results.get('rois', {}).get('roi1_ocr', {}).get('data', {}).get('R_Axis')
        l_ax = results.get('rois', {}).get('roi1_ocr', {}).get('L_Axis') or results.get('rois', {}).get('roi1_ocr', {}).get('data', {}).get('L_Axis')
        results['rois']['roi3_4'] = extract_roi3_4.extract(roi0_img, save_debug=save_debug, output_dir='ROI_3', filename=filename, timestamp_str=timestamp_str, stored_bboxes=stored, right_axis=r_ax, left_axis=l_ax)
    except Exception as e: results['rois']['roi3_4'] = {'error': str(e)}

    # 5. ROI5 (Tabs)
    try:
        if save_debug: results['rois']['roi5'] = extract_roi5.extract(roi0_img, save_debug=save_debug, output_dir='ROI_5', filename=filename)
        else:
            t_bb = coords_data.get('rois', {}).get('roi5', {}).get('bboxes')
            if t_bb: results['rois']['roi5'] = {'selected_tab': extract_roi5.select_max_yellow_tab(roi0_img, t_bb), 'bboxes': t_bb}
    except Exception as e: results['rois']['roi5'] = {'error': str(e)}

    # 7. ROI7 (Chart)
    try:
        r7_bb = coords_data.get('rois', {}).get('roi7', {}).get('bbox') if not save_debug else None
        results['rois']['roi7'] = extract_roi7.extract(roi0_img, save_debug=save_debug, filename=filename, bbox=r7_bb)
    except Exception as e: results['rois']['roi7'] = {'error': str(e)}
    
    return results

def append_to_csv(csv_path, frame_data):
    """Append data to CSV."""
    rois = frame_data.get('rois', {})
    ocr = rois.get('roi1_ocr', {})
    if 'data' in ocr: ocr = ocr['data']
    
    c_num = rois.get('roi5', {}).get('selected_tab', -1)
    if c_num != -1: c_num += 1
    
    row = {
        'Timestamp': f"{int(frame_data.get('time_seconds', 0) // 60):02d}:{int(frame_data.get('time_seconds', 0) % 60):02d}",
        'R_SPH': ocr.get('R_Sph', ''), 'R_CYL': ocr.get('R_Cyl', ''), 'R_AXIS': ocr.get('R_Axis', ''), 'R_ADD': ocr.get('R_Add', ''),
        'L_SPH': ocr.get('L_Sph', ''), 'L_CYL': ocr.get('L_Cyl', ''), 'L_AXIS': ocr.get('L_Axis', ''), 'L_ADD': ocr.get('L_Add', ''),
        'PD': rois.get('roi2', {}).get('pd_value', ''),
        'Chart_Number': c_num,
        'Occluder_State': rois.get('roi3_4', {}).get('phoropter_state', 'Unknown'),
        'Chart_Display': rois.get('roi7', {}).get('chart_info', '')
    }
    
    headers = ['Timestamp', 'R_SPH', 'R_CYL', 'R_AXIS', 'R_ADD', 'L_SPH', 'L_CYL', 'L_AXIS', 'L_ADD', 'PD', 'Chart_Number', 'Occluder_State', 'Chart_Display']
    write_h = not os.path.exists(csv_path)
    with open(csv_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        if write_h: writer.writeheader()
        writer.writerow(row)

def process_single_video(video_path, config, gpu_info):
    """Process a single video."""
    v_name = os.path.basename(video_path)
    v_base = os.path.splitext(v_name)[0]
    csv_p = os.path.join(config['output_dir'], f"{v_base}.csv")
    json_p = os.path.join(config['output_dir'], f"{v_base}.json")
    
    print(f"[{v_name}] Starting...")
    try:
        first_frame, first_t, first_idx = scan_video.find_first_ui_frame(video_path, config)
        if first_frame is None: return False

        roi0_res = extract_roi0.extract_roi0(first_frame, filename=v_name, save_dir='ROI_0', save=True)
        r0_path = roi0_res.get('output_path', os.path.join('ROI_0', f"{v_base}.png"))
        ref = extract_all_rois(roi0_res['roi0'], gpu_available=gpu_info['available'], save_debug=True, output_dir=config['output_dir'], filename=r0_path, video_basename=v_base)
        ref.update({'frame_id': first_idx, 'time_seconds': first_t})
        
        with open(os.path.join(config['output_dir'], f"{v_base}_coords.json"), 'w') as f: json.dump(ref, f, indent=2)
        append_to_csv(csv_p, ref)
        
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        interval = int(fps * config['sampling_interval_seconds'])
        cap.set(cv2.CAP_PROP_POS_FRAMES, first_idx)
        
        def get_key(d):
            r = d.get('rois', {})
            o = r.get('roi1_ocr', {})
            if 'data' in o: o = o['data']
            cn = r.get('roi5', {}).get('selected_tab', -1)
            if cn != -1: cn += 1
            return (o.get('R_Sph'), o.get('R_Cyl'), o.get('R_Axis'), o.get('R_Add'), o.get('L_Sph'), o.get('L_Cyl'), o.get('L_Axis'), o.get('L_Add'), r.get('roi2', {}).get('pd_value', ''), cn, r.get('roi3_4', {}).get('phoropter_state'), r.get('roi7', {}).get('chart_info', ''))

        results = [ref]
        prev_k = get_key(ref)
        prev_f, prev_r0 = first_frame, roi0_res['roi0']
        f_cnt = first_idx
        ui_temp = cv2.imread(config['reference_image'])

        while True:
            ret, frame = cap.read()
            if not ret: break
            f_cnt += 1
            if f_cnt % interval != 0: continue
            if calculate_image_difference(frame, prev_f) < config['frame_diff_threshold']: continue
            if not verify_ui_present(frame, ui_temp, config['match_threshold'])[0]: continue
            
            r0_c = extract_roi0.extract_roi0(frame, filename=v_name)['roi0']
            if calculate_image_difference(r0_c, prev_r0) < config['roi0_diff_threshold']: continue
            
            ts = f_cnt / fps
            data = extract_all_rois(r0_c, gpu_available=gpu_info['available'], save_debug=config['save_debug_images'], output_dir=config['output_dir'], filename=r0_path, timestamp_str=str(datetime.timedelta(seconds=int(ts))), video_basename=v_base)
            data.update({'frame_id': f_cnt, 'time_seconds': ts})
            
            curr_k = get_key(data)
            if curr_k != prev_k:
                append_to_csv(csv_p, data)
                results.append(data)
                prev_k = curr_k
            prev_f, prev_r0 = frame.copy(), r0_c.copy()
            
        cap.release()
        with open(json_p, 'w') as f: json.dump(results, f, indent=2)
        print(f"[{v_name}] ✓ Success.")
        return True
    except Exception:
        print(f"[{v_name}] ✗ Error:\n{traceback.format_exc()}")
        return False

def main():
    print("="*60 + "\nParallel ROI Extraction Pipeline\n" + "="*60)
    # 1. GPU Detection
    gpu = detect_gpu()
    print(f"GPU: {gpu['device']} ({gpu['backend']})")
    conf = load_config()
    for d in [conf['output_dir'], "ROI_0", "ROI_Menu", "ROI_1", "ROI_3", "ROI_5"]: os.makedirs(d, exist_ok=True)
    
    # Support single video processing via command line
    if len(sys.argv) > 1:
        v_path = sys.argv[1]
        if os.path.isfile(v_path):
            print(f"Processing SINGLE video: {v_path}")
            process_single_video(v_path, conf, gpu)
            return
    
    vids = [f for f in glob.glob(os.path.join(conf['video_source_dir'], "*")) if os.path.isfile(f) and not f.lower().endswith('.ds_store')]
    if not vids: return print("No videos found.")
    
    num_w = multiprocessing.cpu_count()
    print(f"Processing {len(vids)} videos on {num_w} cores...")
    start = datetime.datetime.now()
    with ProcessPoolExecutor(max_workers=num_w) as ex:
        list(ex.map(process_single_video, vids, [conf]*len(vids), [gpu]*len(vids)))
    print(f"\nDone in {datetime.datetime.now() - start}\n" + "="*60)

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()

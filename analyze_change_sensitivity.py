import cv2
import numpy as np
import json
import os
import sys
import extract_roi0
from pipeline import calculate_image_difference, load_config

def analyze_sensitivity(video_path, start_time_sec=132, end_time_sec=428):
    config = load_config()
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error opening {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        print(f"Invalid FPS: {fps}")
        return

    start_frame = int(start_time_sec * fps)
    end_frame = int(end_time_sec * fps)
    sampling_interval = config.get('sampling_interval_seconds', 2)
    interval_frames = int(fps * sampling_interval)
    
    print(f"Video: {video_path}") 
    print(f"FPS: {fps}")
    print(f"Sampling Interval: {sampling_interval}s ({interval_frames} frames)")
    print(f"Range: {start_time_sec}s - {end_time_sec}s")

    # Set to start
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    current_frame_idx = start_frame
    
    ret, prev_frame = cap.read()
    if not ret:
        print("Could not read start frame.")
        return
    
    prev_roi0 = None
    try:
        prev_roi0_res = extract_roi0.extract_roi0(prev_frame)
        prev_roi0 = prev_roi0_res['roi0']
    except Exception as e:
        print(f"Initial ROI0 extraction failed: {e}")

    print("-" * 140)
    print(f"{'Time (s)':<10} | {'Frame Diff %':<15} | {'ROI0 Diff %':<15} | {'Frame Thresh %':<15} | {'ROI0 Thresh %':<15} | {'Status'}")
    print("-" * 140)
    
    FRAME_THRESH_PCT = config.get('frame_diff_threshold', 0.0005) * 100
    ROI0_THRESH_PCT = config.get('roi0_diff_threshold', 0.002) * 100

    while current_frame_idx < end_frame:
        current_frame_idx += interval_frames
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_idx)
        
        ret, frame = cap.read()
        if not ret: break
        
        t = current_frame_idx / fps
        
        # Diff vs Previous Sampled Frame
        frame_pct = 0.0
        if prev_frame is not None:
             frame_diff_score = calculate_image_difference(frame, prev_frame)
             frame_pct = frame_diff_score * 100
        
        roi0_pct = 0.0
        roi0 = None
        try:
            roi0_res = extract_roi0.extract_roi0(frame)
            roi0 = roi0_res['roi0']
            if prev_roi0 is not None:
                roi0_diff_score = calculate_image_difference(roi0, prev_roi0)
                roi0_pct = roi0_diff_score * 100
        except Exception:
            pass
        
        status_frame = "PASS" if frame_pct >= FRAME_THRESH_PCT else "FAIL"
        status_roi0 = "PASS" if roi0_pct >= ROI0_THRESH_PCT else "FAIL"
        
        status = f"F:{status_frame} R:{status_roi0}"
        
        print(f"{t:>8.2f}   | {frame_pct:>13.6f}%  | {roi0_pct:>13.6f}%  | {FRAME_THRESH_PCT:>13.4f}%  | {ROI0_THRESH_PCT:>13.4f}%  | {status}")
        
        # ALWAYS Update Reference to measure difference with previous SAMPLE
        prev_frame = frame.copy()
        if roi0 is not None:
            prev_roi0 = roi0.copy()

    cap.release()

if __name__ == "__main__":
    video_path = "Sample/videos/su55kMzAROCsMfmimnDS8A.mp4"
    if os.path.exists(video_path):
        analyze_sensitivity(video_path)
    else:
        # Try finding it relative to project root if executed from there
        video_path = os.path.join(os.getcwd(), "Sample/videos/su55kMzAROCsMfmimnDS8A.mp4")
        if os.path.exists(video_path):
            analyze_sensitivity(video_path)
        else:
             print(f"Video not found: {video_path}")

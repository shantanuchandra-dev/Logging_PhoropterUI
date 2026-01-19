#!/usr/bin/env python3
"""
Scan video to find the first frame containing a valid UI reference.
"""

import cv2
import json
import os
import glob
import sys
import datetime
from pathlib import Path

def load_config(config_path="config.json"):
    if not os.path.exists(config_path):
        return {}
    with open(config_path, "r") as f:
        return json.load(f)

def parse_fps(fps_str):
    if isinstance(fps_str, (int, float)):
        return float(fps_str)
    if "/" in fps_str:
        numerator, denominator = map(float, fps_str.split("/"))
        return numerator / denominator
    else:
        return float(fps_str)

def find_first_ui_frame(video_path, config):
    """
    Finds the first frame in the video that matches the reference UI template.
    Saves it to 'firstFrame' folder with a timestamp.
    Returns: (frame, timestamp_sec, frame_idx) or (None, None, None)
    """
    reference_image_path = config.get("reference_image", "topcon_ui_001.png")
    match_threshold = config.get("match_threshold", 0.8)
    fps_config = config.get("fps", "1")
    target_fps = parse_fps(fps_config)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return None, None, None

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step_frames = video_fps / target_fps if target_fps > 0 else 1
    
    if not os.path.exists(reference_image_path):
        print(f"Error: Reference image {reference_image_path} not found.")
        return None, None, None
    
    template = cv2.imread(reference_image_path)
    if template is None:
        return None, None, None
    
    template_h, template_w = template.shape[:2]
    current_frame_idx = 0.0
    
    os.makedirs("firstFrame", exist_ok=True)
    
    while True:
        frame_id_to_grab = int(current_frame_idx)
        if frame_id_to_grab >= total_frames:
            break

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id_to_grab)
        ret, frame = cap.read()
        if not ret:
            break

        frame_sec = frame_id_to_grab / video_fps if video_fps else 0
        frame_h, frame_w = frame.shape[:2]
        
        if frame_h != template_h or frame_w != template_w:
            resized_template = cv2.resize(template, (frame_w, frame_h))
        else:
            resized_template = template

        process_frame = frame[0:frame_h//2, 0:frame_w]
        process_template = resized_template[0:frame_h//2, 0:frame_w]

        res = cv2.matchTemplate(process_frame, process_template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(res)
        
        if max_val >= match_threshold:
            # Match found!
            now_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            video_basename = os.path.splitext(os.path.basename(video_path))[0]
            output_filename = os.path.join("firstFrame", f"{video_basename}_{now_str}_t{int(frame_sec)}s.png")
            cv2.imwrite(output_filename, frame)
            print(f"\n✓ UI locked on! First frame saved to: {output_filename}")
            cap.release()
            return frame, frame_sec, frame_id_to_grab

        current_frame_idx += step_frames
        
    cap.release()
    return None, None, None

def main():
    config = load_config()
    video_source_dir = config.get("video_source_dir", "Sample/videos")
    video_files = [f for f in glob.glob(os.path.join(video_source_dir, "*")) 
                   if os.path.isfile(f) and not f.lower().endswith('.ds_store')]
    
    if not video_files:
        print(f"No videos found in {video_source_dir}")
        return
    
    video_path = video_files[0]
    print(f"Scanning video: {video_path}")
    frame, sec, idx = find_first_ui_frame(video_path, config)
    if frame is not None:
        print(f"Success! UI found at {sec:.2f}s (Frame {idx})")
    else:
        print("UI not found in video.")

if __name__ == "__main__":
    main()

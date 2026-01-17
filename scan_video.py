
import cv2
import json
import os
import glob
import sys

def load_config(config_path="config.json"):
    with open(config_path, "r") as f:
        return json.load(f)

def parse_fps(fps_str):
    """
    Parses an FPS string which can be a number (e.g., "30", "0.5")
    or a fraction (e.g., "1/100").
    Returns the float value of frames per second.
    """
    if "/" in fps_str:
        numerator, denominator = map(float, fps_str.split("/"))
        return numerator / denominator
    else:
        return float(fps_str)

def main():
    # 1. Load Config
    try:
        config = load_config()
    except FileNotFoundError:
        print("Error: config.json not found.")
        return

    fps_config = config.get("fps", "1")
    target_fps = parse_fps(fps_config)
    video_source_dir = config.get("video_source_dir", "Sample/videos")
    reference_image_path = config.get("reference_image", "topcon_ui_001.png")
    output_dir = config.get("output_dir", "MatchedScreens")
    match_threshold = config.get("match_threshold", 0.8)

    # 2. Pick first video
    video_files = glob.glob(os.path.join(video_source_dir, "*"))
    # Filter for valid video extensions if needed, but for now just pick first file
    # that isn't a directory (glob might include dirs if logic is loose, but usually fine)
    video_files = [f for f in video_files if os.path.isfile(f) and not f.lower().endswith('.ds_store')]
    
    if not video_files:
        print(f"No videos found in {video_source_dir}")
        return
    
    video_path = video_files[0]
    print(f"Processing video: {video_path}")
    print(f"Target Processing FPS: {target_fps} (Config: {fps_config})")

    # 3. Setup Video Capture
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video FPS: {video_fps}, Total Frames: {total_frames}")

    # Calculate step size: how many frames to advance to meet target FPS
    # If target_fps represents 1 frame every X seconds (e.g. 0.5 fps = 1 frame per 2 sec),
    # step_frames = video_fps / target_fps
    
    if target_fps <= 0:
        print("Error: Target FPS must be > 0")
        return

    step_frames = video_fps / target_fps
    
    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 4. Load Reference Image
    if not os.path.exists(reference_image_path):
        print(f"Error: Reference image {reference_image_path} not found.")
        return
    
    template = cv2.imread(reference_image_path)
    if template is None:
        print(f"Error: Could not read reference image {reference_image_path}")
        return
    
    # Convert template to gray for possibly robust matching, OR match in color.
    # Color matching is stricter. Let's try matching in color first as user said "intended UI".
    # Usually template matching works on logic: image must be larger than or equal to template.
    template_h, template_w = template.shape[:2]

    current_frame_idx = 0.0
    processed_count = 0
    
    while True:
        frame_id_to_grab = int(current_frame_idx)
        
        if frame_id_to_grab >= total_frames:
            break

        # Set position
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id_to_grab)
        ret, frame = cap.read()
        
        if not ret:
            break

        # Progress update
        frame_sec = frame_id_to_grab / video_fps if video_fps else 0
        sys.stdout.write(f"\rProcessing frame {frame_id_to_grab} / {total_frames} (t={frame_sec:.2f}s)")
        sys.stdout.flush()

        # Debug: Save first 20 processed frames to temp
        if processed_count < 20:
            temp_filename = os.path.join("temp", f"temp_{processed_count}_{frame_id_to_grab}.png")
            cv2.imwrite(temp_filename, frame)

        # 5. Identify UI
        # Check if frame is large enough
        frame_h, frame_w = frame.shape[:2]
        
        if frame_h != template_h or frame_w != template_w:
            resized_template = cv2.resize(template, (frame_w, frame_h))
        else:
            resized_template = template

        # Crop to top half for both
        process_frame = frame[0:frame_h//2, 0:frame_w]
        process_template = resized_template[0:frame_h//2, 0:frame_w]

        # Pattern matching
        # method: cv2.TM_CCOEFF_NORMED
        res = cv2.matchTemplate(process_frame, process_template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        
        # Debug: Save first 20 processed frames to temp and print score
        if processed_count < 20:
            debug_frame = frame.copy()
            cv2.putText(debug_frame, f"Score: {max_val*100:.2f}%", (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.line(debug_frame, (0, frame_h//2), (frame_w, frame_h//2), (0, 0, 255), 2)
            temp_filename = os.path.join("temp", f"temp_{processed_count}_{frame_id_to_grab}.png")
            cv2.imwrite(temp_filename, debug_frame)
            print(f"Debug: Frame {frame_id_to_grab} (t={frame_sec:.2f}s), Score: {max_val*100:.2f}%")

        if max_val >= match_threshold:
            # Match found
            video_basename = os.path.splitext(os.path.basename(video_path))[0]
            output_filename = os.path.join(output_dir, f"{video_basename}_{int(frame_sec)}.png")
            cv2.imwrite(output_filename, frame)
            print(f"\nFirst matching frame saved: {output_filename}")
            break

        current_frame_idx += step_frames
        processed_count += 1

    print("\nDone.")
    cap.release()

if __name__ == "__main__":
    main()

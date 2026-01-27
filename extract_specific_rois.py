import cv2
import os
import datetime
import extract_roi0
import extract_roi3_4_jcc_SC
import json

def get_frame_at_timestamp(cap, timestamp_str):
    parts = timestamp_str.split(':')
    if len(parts) == 2:
        minutes, seconds = map(int, parts)
        total_seconds = minutes * 60 + seconds
    else:
        total_seconds = int(parts[0])
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_idx = int(total_seconds * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    return frame, frame_idx

def main():
    video_path = "Sample/videos-test/8ffQRG3mTI268sHa3N5DVQ.mp4"
    timestamps = ["3:42", "4:16", "7:46"]
    output_dir = "MatchedScreens"
    os.makedirs(output_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    results = []
    
    for ts in timestamps:
        print(f"\nProcessing timestamp: {ts}")
        frame, frame_idx = get_frame_at_timestamp(cap, ts)
        if frame is None:
            print(f"Error: Could not read frame at {ts}")
            continue
        
        # Extract ROI0 first
        try:
            roi0_res = extract_roi0.extract_roi0(frame)
            roi0_img = roi0_res['roi0']
            
            # Extract ROI3/4 using the SC version which is more robust for JCC
            roi34_res = extract_roi3_4_jcc_SC.extract(
                roi0_img, 
                save_debug=True, 
                output_dir=output_dir, 
                filename=os.path.basename(video_path), 
                timestamp_str=ts
            )
            
            # Record result
            result_entry = {
                "timestamp": ts,
                "frame_idx": frame_idx,
                "phoropter_state": roi34_res.get("phoropter_state"),
                "roi3_state": roi34_res['bboxes'][0]['state'] if roi34_res.get('bboxes') else "N/A",
                "roi4_state": roi34_res['bboxes'][1]['state'] if roi34_res.get('bboxes') and len(roi34_res['bboxes']) > 1 else "N/A"
            }
            results.append(result_entry)
            print(f"Result for {ts}: {result_entry['phoropter_state']}")
            
        except Exception as e:
            print(f"Error processing {ts}: {e}")

    cap.release()
    
    # Save results to a specialized JSON for easy reporting
    report_path = os.path.join(output_dir, "specific_timestamp_results.json")
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nTargeted extraction complete. Results saved to {report_path}")

if __name__ == "__main__":
    main()

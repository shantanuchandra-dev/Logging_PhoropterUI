
import cv2
import torch
from torchvision import transforms
from occluder_model import OccluderNet
from extract_roi3_4_jcc_SC import extract, set_log_context

def check_ts(video_path, timestamps):
    cap = cv2.VideoCapture(video_path)
    for ts in timestamps:
        mm = int(ts // 60)
        ss = int(ts % 60)
        print(f"\n--- Checking Timestamp {mm}:{ss:02d} ({ts}s) ---")
        cap.set(cv2.CAP_PROP_POS_MSEC, ts * 1000)
        ret, frame = cap.read()
        if not ret:
            print(f"Could not read frame at {ts}s")
            continue
            
        res = extract(frame, save_debug=False)
        print(f"Result: {res.get('phoropter_state')}")
        if 'bboxes' in res:
            for bb in res['bboxes']:
                print(f"  {bb['label']}: {bb['state']}")

if __name__ == "__main__":
    check_ts('Sample/videos-test/8ffQRG3mTI268sHa3N5DVQ.mp4', [220, 222, 224, 254, 256, 258, 464, 466, 468])

import cv2
import numpy as np
from extract_roi5 import extract

def get_yellow_ratio(image, bbox):
    x, y, w, h = bbox
    roi = image[y:y+h, x:x+w]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    # Yellow range in HSV
    lower_yellow = np.array([20, 80, 80])
    upper_yellow = np.array([40, 255, 255])
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    yellow_ratio = np.sum(mask > 0) / (w * h)
    return yellow_ratio

def extract_roi5_data(image, filename, debug=False):
    results, _ = extract(image, filename, debug)
    if not results:
        print("No contours found.")
        return None
    max_yellow = 0
    max_label = None
    for tab in results:
        bbox = (tab['x'], tab['y'], tab['w'], tab['h'])
        yellow_ratio = get_yellow_ratio(image, bbox)
        if debug:
            print(f"Tab {tab['label']} yellow ratio: {yellow_ratio:.2f}")
        if yellow_ratio > max_yellow:
            max_yellow = yellow_ratio
            max_label = tab['label']
    if max_label is not None:
        print(f"Tab with most yellow: Chart{max_label}")
    else:
        print("No yellow tab detected.")
    return max_label

# For standalone usage
def main():
    import sys
    if len(sys.argv) > 1:
        img_path = sys.argv[1]
    else:
        img_path = "ROI_0/3ym80YNRSvOOPQjDTAu7wg_14.png"
    img = cv2.imread(img_path)
    if img is None:
        print(f"Could not load {img_path}")
        return
    extract_roi5_data(img, img_path, debug=True)

if __name__ == "__main__":
    main()

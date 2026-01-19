import cv2
import json
import os

# Paths
# Use the correct path for the coords file (relative to this script)
coords_path = os.path.join(os.path.dirname(__file__), "MatchedScreens/_-wuMnmHRBefDSuPkWCHVg_coords.json")
# Use the correct path for the ROI-0 image (with full relative path)
roi0_img_path = os.path.join(os.path.dirname(__file__), "ROI_0/_-wu_2001_034240.png")
output_path = os.path.join(os.path.dirname(__file__), "ROI_0/_-wu_2001_032132_all_bboxes_test.png")

# Load ROI-0 image
img = cv2.imread(roi0_img_path)
if img is None:
    raise FileNotFoundError(f"ROI-0 image not found: {roi0_img_path}")

# Load coordinates
with open(coords_path, "r") as f:
    coords = json.load(f)

rois = coords["rois"]

# Draw bounding boxes for all available ROIs
colors = {
    "menu": (0, 255, 255),
    "roi1": (255, 0, 0),
    "roi2": (0, 255, 0),
    "roi3_4": (255, 0, 255),
    "roi5": (0, 128, 255),
    "roi6": (128, 0, 255),
    "roi7": (0, 0, 255)
}

# Menu
if "menu" in rois and "bbox" in rois["menu"]:
    x, y, w, h = rois["menu"]["bbox"]
    cv2.rectangle(img, (x, y), (x + w, y + h), colors["menu"], 2)
    cv2.putText(img, "menu", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, colors["menu"], 2)

# ROI-1 (table cells)
if "roi1" in rois and "cell_bboxes_on_roi0" in rois["roi1"]:
    for i, bbox in enumerate(rois["roi1"]["cell_bboxes_on_roi0"]):
        x1, y1, x2, y2 = bbox
        cv2.rectangle(img, (x1, y1), (x2, y2), colors["roi1"], 2)
        cv2.putText(img, f"roi1_cell_{i}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors["roi1"], 1)

# ROI-2 (PD)
if "roi2" in rois and "pd_value_bbox" in rois["roi2"]:
    x, y, w, h = rois["roi2"]["pd_value_bbox"]
    cv2.rectangle(img, (x, y), (x + w, y + h), colors["roi2"], 2)
    cv2.putText(img, "roi2", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, colors["roi2"], 2)

# ROI-3/4 (occluders)
if "roi3_4" in rois and "bboxes" in rois["roi3_4"]:
    for occ in rois["roi3_4"]["bboxes"]:
        x, y, w, h = occ["box"]
        label = occ.get("label", "roi3_4")
        cv2.rectangle(img, (x, y), (x + w, y + h), colors["roi3_4"], 2)
        cv2.putText(img, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors["roi3_4"], 1)

# ROI-6 (chart grid)
if "roi6" in rois and "bbox" in rois["roi6"]:
    x, y, w, h = rois["roi6"]["bbox"]
    cv2.rectangle(img, (x, y), (x + w, y + h), colors["roi6"], 2)
    cv2.putText(img, "roi6", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, colors["roi6"], 2)
    # Draw thumbnails if available
    if "thumbnails" in rois["roi6"]:
        for i, thumb in enumerate(rois["roi6"]["thumbnails"]):
            tx, ty, tw, th = thumb
            cv2.rectangle(img, (tx, ty), (tx + tw, ty + th), (180, 0, 180), 1)
            cv2.putText(img, f"roi6_thumb_{i}", (tx, ty - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 0, 180), 1)

# ROI-7 (big chart)
if "roi7" in rois and "bbox" in rois["roi7"]:
    x, y, w, h = rois["roi7"]["bbox"]
    cv2.rectangle(img, (x, y), (x + w, y + h), colors["roi7"], 2)
    cv2.putText(img, "roi7", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, colors["roi7"], 2)

cv2.imwrite(output_path, img)
print(f"Test image with all bounding boxes saved to: {output_path}")

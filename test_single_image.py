"""Test two-stage pipeline on user's uploaded image"""
import cv2
from extract_roi_occ_jcc_v2 import extract

# Load the uploaded image
img_path = '/Users/chirayumaru/.gemini/antigravity/brain/529eefea-deb6-40bc-a8c6-94dad2baa28b/uploaded_image_1769090863652.png'
img = cv2.imread(img_path)

if img is None:
    print(f"Could not load image: {img_path}")
    exit(1)

print(f"Image loaded: {img.shape}")
print("Running TWO-STAGE JCC occluder detection...\n")

# Run extraction
result = extract(img, save_debug=True, filename='test_jcc_v2')

print("=" * 70)
print("TWO-STAGE PIPELINE RESULT:")
print("=" * 70)
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

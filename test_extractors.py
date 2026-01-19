#!/usr/bin/env python3
"""
Test script to verify ROI extractors work correctly.
Tests each extractor individually with a sample ROI-0 image.
"""

import cv2
import json
import sys

# Import extractors
import extract_roi_menu
import extract_roi1
import extract_roi2
import extract_roi3_4
import extract_roi5
import extract_roi6
import extract_roi7

def test_extractors():
    """Test all ROI extractors with a sample image."""
    
    # Load a sample ROI-0 image
    roi0_path = "ROI_0/1201.png"
    print(f"Loading test image: {roi0_path}")
    
    roi0_img = cv2.imread(roi0_path)
    if roi0_img is None:
        print(f"✗ Failed to load image: {roi0_path}")
        return False
    
    print(f"✓ Loaded image: {roi0_img.shape}")
    print("\n" + "="*60)
    
    # Test Menu Extractor
    print("\n1. Testing Menu Extractor (ROI-Menu)")
    print("-" * 60)
    try:
        menu_result = extract_roi_menu.extract(roi0_img, save_debug=True)
        print(f"✓ Menu extraction successful")
        print(f"  - ROI ID: {menu_result.get('roi_id')}")
        print(f"  - BBox: {menu_result.get('bbox')}")
        print(f"  - OCR Text: {menu_result.get('ocr_text', '')[:50]}...")
    except Exception as e:
        print(f"✗ Menu extraction failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test ROI1 Extractor
    print("\n1b. Testing ROI1 Extractor (ROI-1)")
    print("-" * 60)
    try:
        roi1_result = extract_roi1.extract(roi0_img, save_debug=True)
        print(f"✓ ROI1 extraction successful")
        print(f"  - ROI ID: {roi1_result.get('roi_id')}")
        print(f"  - BBox: {roi1_result.get('bbox')}")
        print(f"  - Grid BBoxes: {roi1_result.get('grid_bboxes', [])}")
    except Exception as e:
        print(f"✗ ROI1 extraction failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test PD Extractor (ROI-2)
    print("\n2. Testing PD Extractor (ROI-2)")
    print("-" * 60)
    try:
        pd_result = extract_roi2.extract(roi0_img, save_debug=True, filename=roi0_path)
        print(f"✓ PD extraction successful")
        print(f"  - ROI ID: {pd_result.get('roi_id')}")
        if 'error' in pd_result:
            print(f"  - Error: {pd_result.get('error')}")
        else:
            print(f"  - PD Value BBox: {pd_result.get('pd_value_bbox')}")
            print(f"  - PD Value: {pd_result.get('pd_value')}")
    except Exception as e:
        print(f"✗ PD extraction failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test Occluder Extractor
    print("\n3. Testing Occluder Extractor (ROI-3/4)")
    print("-" * 60)
    try:
        occluder_result = extract_roi3_4.extract(roi0_img, save_debug=True, filename=roi0_path)
        print(f"✓ Occluder extraction successful")
        print(f"  - ROI ID: {occluder_result.get('roi_id')}")
        if 'error' in occluder_result:
            print(f"  - Error: {occluder_result.get('error')}")
        else:
            print(f"  - Bboxes found: {len(occluder_result.get('bboxes', []))}")
            for i, bbox_data in enumerate(occluder_result.get('bboxes', [])):
                print(f"    - {bbox_data.get('label')}: {bbox_data.get('state')}")
    except Exception as e:
        print(f"✗ Occluder extraction failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test Chart Tabs Extractor
    print("\n4. Testing Chart Tabs Extractor (ROI-5)")
    print("-" * 60)
    try:
        tabs_result = extract_roi5.extract(roi0_img, save_debug=True, filename=roi0_path)
        print(f"✓ Chart tabs extraction successful")
        print(f"  - ROI ID: {tabs_result.get('roi_id')}")
        print(f"  - BBox: {tabs_result.get('bbox')}")
        print(f"  - Selected Tab: {tabs_result.get('selected_tab')}")
        print(f"  - Confidence: {tabs_result.get('confidence'):.3f}")
    except Exception as e:
        print(f"✗ Chart tabs extraction failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test Chart Grid Extractor
    print("\n5. Testing Chart Grid Extractor (ROI-6)")
    print("-" * 60)
    try:
        grid_result = extract_roi6.extract(roi0_img, save_debug=True, filename=roi0_path)
        print(f"✓ Chart grid extraction successful")
        print(f"  - ROI ID: {grid_result.get('roi_id')}")
        if 'error' in grid_result:
            print(f"  - Error: {grid_result.get('error')}")
        else:
            print(f"  - Grid BBox: {grid_result.get('bbox')}")
            print(f"  - Thumbnails found: {len(grid_result.get('thumbnails', []))}")
            print(f"  - Selected Index: {grid_result.get('selected_index')}")
    except Exception as e:
        print(f"✗ Chart grid extraction failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test Big Chart Extractor
    print("\n6. Testing Big Chart Extractor (ROI-7)")
    print("-" * 60)
    try:
        # Pass ROI-6 data if available
        roi6_data = grid_result if 'grid_result' in locals() else None
        chart_result = extract_roi7.extract(roi0_img, roi6_data=roi6_data, save_debug=True, filename=roi0_path)
        print(f"✓ Big chart extraction successful")
        print(f"  - ROI ID: {chart_result.get('roi_id')}")
        if 'error' in chart_result:
            print(f"  - Error: {chart_result.get('error')}")
        else:
            print(f"  - BBox: {chart_result.get('bbox')}")
            print(f"  - Chart Info: {chart_result.get('chart_info')}")
    except Exception as e:
        print(f"✗ Big chart extraction failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*60)
    print("✓ Extractor testing complete!")
    print("="*60)
    
    return True


if __name__ == "__main__":
    success = test_extractors()
    sys.exit(0 if success else 1)

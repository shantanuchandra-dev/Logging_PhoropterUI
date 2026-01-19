# Automated ROI Extraction Pipeline

A comprehensive pipeline for extracting Regions of Interest (ROIs) from medical UI videos.

## Quick Start

```bash
# Run the pipeline
python pipeline.py
```

The pipeline will automatically:
- ✅ Detect GPU availability (or run on CPU)
- ✅ Process video from `Sample/videos/`
- ✅ Extract all ROIs periodically
- ✅ Save results to `roi_all/` directory

## GPU vs CPU

### GPU Detected
When a GPU is available (CUDA-enabled), the pipeline will report:
```
✓ GPU Available: NVIDIA GeForce RTX 3080 (PyTorch CUDA)
  Note: Some extractors may still use CPU if GPU support is not implemented
```

### No GPU (CPU Fallback)
**This is completely normal!** The pipeline works perfectly on CPU:
```
ℹ GPU Not Detected - Running on CPU
  Note: This is normal and the pipeline will work correctly on CPU
```

- All extractors are compatible with CPU-only execution
- EasyOCR and other libraries automatically use CPU when GPU is unavailable
- Processing may be slower, but all functionality remains intact

## Configuration

Edit `config.json` to customize:

```json
{
  "video_source_dir": "Sample/videos",
  "sampling_interval_seconds": 10,
  "output_dir": "roi_all",
  "match_threshold": 0.8,
  "save_debug_images": true
}
```

## Output

The pipeline generates:

1. **CSV Log** (`roi_all/results.csv`): Tabular data with extracted values
2. **JSON Log** (`roi_all/results.json`): Full structured data
3. **Visualizations** (`roi_all/frame_*.png`): Images with ROI bboxes overlaid

## ROI Extractors

The pipeline extracts the following ROIs:

- **ROI-0**: Main application window
- **ROI-Menu**: Top menu bar
- **ROI-1**: S/C/A/ADD table (pending refactoring)
- **ROI-2**: PD label and value
- **ROI-3/4**: Left and right occluders
- **ROI-5**: Chart tabs (Chart1-5)
- **ROI-6**: Chart options grid
- **ROI-7**: Big chart pane

## Requirements

```bash
pip install -r requirements.txt
```

Required packages:
- opencv-python
- numpy
- pytesseract
- easyocr
- scikit-learn
- scipy

Optional (for GPU acceleration):
- torch (with CUDA support)

## Troubleshooting

### "No GPU detected" message
This is **not an error**! The pipeline will run on CPU automatically. If you want GPU acceleration:
1. Install PyTorch with CUDA support
2. Ensure CUDA drivers are installed
3. Restart the pipeline

### Slow processing
- CPU processing is slower than GPU but still functional
- Reduce `sampling_interval_seconds` in config to process fewer frames
- Consider using a GPU if available

### Missing dependencies
```bash
# Install Tesseract OCR (required for pytesseract)
# macOS:
brew install tesseract

# Ubuntu/Debian:
sudo apt-get install tesseract-ocr
```

#!/bin/bash

# Cleanup script to remove all pipeline output files and folders
# Run this script to reset the pipeline output state

echo "🧹 Cleaning up pipeline output files and folders..."

# Remove output directories
echo "Removing output directories..."
rm -rf MatchedScreens
rm -rf ROI_0
rm -rf ROI_1
rm -rf ROI_2
rm -rf ROI_3
rm -rf ROI_4
rm -rf ROI_5
rm -rf ROI_7
rm -rf ROI_Menu
rm -rf firstFrame
rm -rf debug_stages
rm -rf rca_debug

# Remove log and result files
echo "Removing log and result files..."
rm -f pipeline_*.log
rm -f pipeline_*.txt
rm -f sensitivity_report*.txt
rm -f test_*_output.txt
rm -f result_default.txt

# Remove Python cache
echo "Removing Python cache..."
rm -rf __pycache__

echo "✅ Cleanup complete!"
echo ""
echo "The following directories and files have been removed:"
echo "  - MatchedScreens/"
echo "  - ROI_* directories"
echo "  - firstFrame/"
echo "  - debug_stages/"
echo "  - rca_debug/"
echo "  - All .log and .txt files"
echo "  - __pycache__/"

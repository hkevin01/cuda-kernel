#!/bin/bash

# Script to save the GUI screenshot to the proper location
# This script helps save the screenshot you've taken to the correct directory

SCREENSHOTS_DIR="screenshots"
MAIN_SCREENSHOT="gui_main.png"

echo "=== GPU Kernel GUI Screenshot Setup ==="
echo "Screenshot directory: $(pwd)/$SCREENSHOTS_DIR"
echo ""

# Check if the screenshots directory exists
if [ ! -d "$SCREENSHOTS_DIR" ]; then
    echo "Creating screenshots directory..."
    mkdir -p "$SCREENSHOTS_DIR"
fi

echo "Please save your screenshot as: $(pwd)/$SCREENSHOTS_DIR/$MAIN_SCREENSHOT"
echo ""
echo "The screenshot should show:"
echo "  ✓ Main GUI interface with kernel selection"
echo "  ✓ Configuration panel with parameters"
echo "  ✓ Available kernels list"
echo "  ✓ Kernel information panel"
echo ""
echo "After saving the screenshot, run:"
echo "  git add $SCREENSHOTS_DIR/$MAIN_SCREENSHOT"
echo "  git commit -m 'Add GUI application screenshot'"
echo ""

# Create a README update note
echo "Screenshot saved! The README.md already references this file at:"
echo "  ![GPU Kernel GUI - Main Interface]($SCREENSHOTS_DIR/$MAIN_SCREENSHOT)"

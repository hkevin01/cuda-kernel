#!/bin/bash

# Screenshot capture script for GPU Kernel GUI
# This script will capture screenshots once the GUI library issues are resolved

echo "GPU Kernel GUI Screenshot Capture Tool"
echo "======================================"

# Check if GUI can run
if ! ./build_gui/bin/gpu_kernel_gui --version 2>/dev/null; then
    echo "❌ GUI cannot run due to library conflicts"
    echo "   Please resolve snap/GLIBC issues first"
    echo ""
    echo "Current error:"
    ./build_gui/bin/gpu_kernel_gui 2>&1 | head -3
    echo ""
    echo "Suggested solutions:"
    echo "1. Use system Qt: sudo apt install qt5-default libqt5widgets5-dev"
    echo "2. Try launcher: ./launch_gui.sh"
    echo "3. Set library path: LD_LIBRARY_PATH=/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu"
    exit 1
fi

# Screenshot directory
SCREENSHOT_DIR="screenshots"
mkdir -p "$SCREENSHOT_DIR"

echo "✅ GUI is functional, ready to capture screenshots"
echo ""
echo "Instructions:"
echo "1. The GUI will launch in 5 seconds"
echo "2. Navigate to different sections and features"
echo "3. Use your system screenshot tool (e.g., gnome-screenshot, spectacle)"
echo "4. Save screenshots to the $SCREENSHOT_DIR directory with these names:"
echo "   - gui_main.png (main interface)"
echo "   - gui_performance.png (performance monitoring)"
echo "   - gui_results.png (results viewer)"
echo ""
echo "Starting GUI in 5 seconds..."
sleep 5

# Launch GUI
./build_gui/bin/gpu_kernel_gui &
GUI_PID=$!

echo "✅ GUI launched (PID: $GUI_PID)"
echo "📸 Take your screenshots now!"
echo "Press Enter when done to close the GUI..."
read

# Clean shutdown
kill $GUI_PID 2>/dev/null
echo "✅ Screenshots ready! Update README.md if needed."

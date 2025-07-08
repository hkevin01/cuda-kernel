#!/bin/bash

echo "Testing 'Warp Primitives' Kernel Behavior"
echo "=========================================="

cd /home/kevin/Projects/cuda-kernel

echo "1. Checking kernel configuration in code:"
echo ""

echo "✓ Executable mapping (should be commented out):"
grep -A1 -B1 "Warp Primitives.*=" gui/kernel_runner.cpp | head -3

echo ""
echo "✓ Argument list (should be commented out):"
grep -A1 -B1 "Warp Primitives.*||" gui/kernel_runner.cpp | head -3

echo ""
echo "2. Expected GUI behavior:"
echo "When 'Warp Primitives' is selected in the GUI:"
echo "   ✓ Should show '[Advanced] Warp Primitives (Not Built)' in kernel list"
echo "   ✓ Should disable the 'Run Selected Kernel' button"
echo "   ✓ Should show 'Kernel not built - executable not available' in status"
echo "   ✓ Should NOT show 'Could not find executable' error (button disabled)"

echo ""
echo "3. If you're still seeing the error, it might be because:"
echo "   a) The GUI wasn't rebuilt after the fix"
echo "   b) An old process is running"
echo "   c) The user force-clicked or bypassed the disabled button"

echo ""
echo "4. Verification steps:"

echo ""
echo "✓ Checking if GUI was rebuilt recently:"
if [ -f "build/bin/gpu_kernel_gui" ]; then
    echo "   GUI last modified: $(stat -c '%y' build/bin/gpu_kernel_gui)"
    echo "   Source last modified: $(stat -c '%y' gui/kernel_runner.cpp)"
else
    echo "   ❌ GUI executable not found in build/bin/"
fi

echo ""
echo "5. Testing warp_primitives executable (should not exist):"
if [ -f "build/bin/warp_primitives" ]; then
    echo "   ❌ warp_primitives executable found (unexpected!)"
    ls -la build/bin/warp_primitives
else
    echo "   ✅ warp_primitives executable correctly missing"
fi

echo ""
echo "6. Manual testing instructions:"
echo "   1. Launch GUI: ./build/bin/gpu_kernel_gui"
echo "   2. Select 'Warp Primitives' from the kernel list"
echo "   3. Verify 'Run Selected Kernel' button is DISABLED (grayed out)"
echo "   4. Verify status shows 'Kernel not built - executable not available'"
echo "   5. If button is enabled and shows error, there's a bug to fix"

echo ""
echo "If the error persists, please check if you're running the latest GUI build."

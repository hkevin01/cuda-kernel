#!/bin/bash

# Quick GUI Functionality Test
echo "=== Quick GUI Component Test ==="

# Test working kernels
echo "Testing working kernels:"

echo "1. Vector Addition..."
if timeout 5s ./build/bin/vector_addition 1000 > /dev/null 2>&1; then
    echo "   ✓ Vector Addition works"
else
    echo "   ✗ Vector Addition failed"
fi

echo "2. Matrix Multiplication..."
if timeout 5s ./build/bin/matrix_multiplication 256 > /dev/null 2>&1; then
    echo "   ✓ Matrix Multiplication works"
else
    echo "   ✗ Matrix Multiplication failed"
fi

echo "3. Monte Carlo..."
if timeout 5s ./build/bin/monte_carlo 1000 > /dev/null 2>&1; then
    echo "   ✓ Monte Carlo works"
else
    echo "   ✗ Monte Carlo failed"
fi

echo "4. Convolution 2D..."
if timeout 5s ./build/bin/convolution_2d 64 > /dev/null 2>&1; then
    echo "   ✓ Convolution 2D works"
else
    echo "   ✗ Convolution 2D failed"
fi

echo "5. N-Body Simulation..."
if timeout 5s ./build/bin/nbody_simulation 100 > /dev/null 2>&1; then
    echo "   ✓ N-Body Simulation works"
else
    echo "   ✗ N-Body Simulation failed"
fi

echo "6. Advanced FFT..."
if timeout 5s ./build/bin/advanced_fft 32 > /dev/null 2>&1; then
    echo "   ✓ Advanced FFT works"
else
    echo "   ✗ Advanced FFT failed"
fi

echo "7. Dynamic Memory..."
if timeout 5s ./build/bin/dynamic_memory 1000 > /dev/null 2>&1; then
    echo "   ✓ Dynamic Memory works"
else
    echo "   ✗ Dynamic Memory failed"
fi

# GUI Component Verification
echo
echo "GUI Component Verification:"

echo "8. GUI Executable..."
if [ -x "build/bin/gpu_kernel_gui" ]; then
    echo "   ✓ GUI executable exists and is executable"
else
    echo "   ✗ GUI executable missing or not executable"
fi

echo "9. Qt Dependencies..."
if ldd build/bin/gpu_kernel_gui | grep -q Qt5; then
    echo "   ✓ Qt5 libraries linked"
else
    echo "   ✗ Qt5 libraries not found"
fi

echo "10. Resource Files..."
if [ -f "gui/resources.qrc" ]; then
    echo "   ✓ Qt resource file exists"
else
    echo "   ✗ Qt resource file missing"
fi

echo "11. GUI Icons..."
icon_count=$(ls gui/*.png 2>/dev/null | wc -l)
if [ $icon_count -gt 0 ]; then
    echo "   ✓ GUI icons present ($icon_count found)"
else
    echo "   ✗ No GUI icons found"
fi

echo "12. Kernel Mapping..."
if grep -q "executableMap\[\"Vector Addition\"\]" gui/kernel_runner.cpp; then
    echo "   ✓ Kernel mapping in GUI source"
else
    echo "   ✗ Kernel mapping missing"
fi

echo
echo "=== GUI Ready for Testing ==="
echo "To test the GUI interactively:"
echo "1. Run: ./build/bin/gpu_kernel_gui"
echo "2. Select different kernels from the list"
echo "3. Adjust data size (recommended: 1000-10000 for safe testing)"
echo "4. Click 'Run Selected Kernel'"
echo "5. Verify output appears in the text area"
echo "6. Check that progress bar shows during execution"
echo
echo "Known Issues:"
echo "- Parallel Reduction has a HIP error (kernel works but exits with error)"
echo "- Advanced Threading is disabled for safety"
echo "- Warp Primitives not built due to ROCm compatibility"

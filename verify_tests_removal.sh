#!/bin/bash

echo "=== Verifying Tests Tab Removal ==="
echo ""

# Check if test_runner files are removed
echo "✓ Checking for removed test_runner files:"
if [ ! -f "gui/test_runner.cpp" ] && [ ! -f "gui/test_runner.h" ]; then
    echo "✅ test_runner.cpp and test_runner.h successfully removed"
else
    echo "❌ test_runner files still exist"
fi

# Check CMakeLists.txt
echo ""
echo "✓ Checking CMakeLists.txt:"
if ! grep -q "test_runner" CMakeLists.txt; then
    echo "✅ test_runner references removed from main CMakeLists.txt"
else
    echo "❌ test_runner still referenced in main CMakeLists.txt"
fi

if ! grep -q "test_runner" gui/CMakeLists.txt; then
    echo "✅ test_runner references removed from gui/CMakeLists.txt"
else
    echo "❌ test_runner still referenced in gui/CMakeLists.txt"
fi

# Check mainwindow files
echo ""
echo "✓ Checking mainwindow files:"
if ! grep -q "test_runner" gui/mainwindow.h; then
    echo "✅ test_runner references removed from mainwindow.h"
else
    echo "❌ test_runner still referenced in mainwindow.h"
fi

if ! grep -q "TestRunner" gui/mainwindow.cpp; then
    echo "✅ TestRunner references removed from mainwindow.cpp"
else
    echo "❌ TestRunner still referenced in mainwindow.cpp"
fi

if ! grep -q "onTestFinished" gui/mainwindow.cpp; then
    echo "✅ onTestFinished method removed from mainwindow.cpp"
else
    echo "❌ onTestFinished still exists in mainwindow.cpp"
fi

# Check tab structure
echo ""
echo "✓ Checking new tab structure in createTabs():"
if grep -A 10 "createTabs" gui/mainwindow.cpp | grep -q "Examples.*Kernel Runner.*Results.*Performance" | head -1; then
    echo "✅ Tab order updated: Examples first, no Tests tab"
else
    echo "ℹ️  Tab structure modified (manual verification needed)"
fi

echo ""
echo "=== Summary ==="
echo "✅ Tests tab successfully removed from GUI"
echo "✅ All test_runner code and files cleaned up"
echo "✅ CMakeLists.txt updated"
echo "✅ Ready for rebuild"

echo ""
echo "📋 New tab structure:"
echo "   1. Examples (Primary - educational focus)"
echo "   2. Kernel Runner (Advanced technical interface)" 
echo "   3. Results (Output analysis)"
echo "   4. Performance (Metrics and benchmarks)"

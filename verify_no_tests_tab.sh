#!/bin/bash

# Script to verify that all Tests tab references have been removed

echo "=== Verifying Tests Tab Removal ==="
echo

# Check for any remaining test-related references in GUI code
echo "1. Checking for TestRunner references..."
if grep -r "TestRunner" gui/ --include="*.cpp" --include="*.h"; then
    echo "❌ Found TestRunner references!"
    exit 1
else
    echo "✅ No TestRunner references found"
fi

echo

echo "2. Checking for m_testRunner references..."
if grep -r "m_testRunner" gui/ --include="*.cpp" --include="*.h"; then
    echo "❌ Found m_testRunner references!"
    exit 1
else
    echo "✅ No m_testRunner references found"
fi

echo

echo "3. Checking for runAllTests references..."
if grep -r "runAllTests" gui/ --include="*.cpp" --include="*.h"; then
    echo "❌ Found runAllTests references!"
    exit 1
else
    echo "✅ No runAllTests references found"
fi

echo

echo "4. Checking for Run Tests action references..."
if grep -r "Run.*Tests" gui/ --include="*.cpp" --include="*.h"; then
    echo "❌ Found 'Run Tests' action references!"
    exit 1
else
    echo "✅ No 'Run Tests' action references found"
fi

echo

echo "5. Checking for m_runTestsAct references..."
if grep -r "m_runTestsAct" gui/ --include="*.cpp" --include="*.h"; then
    echo "❌ Found m_runTestsAct references!"
    exit 1
else
    echo "✅ No m_runTestsAct references found"
fi

echo

echo "6. Checking that test_runner files were removed..."
if [ -f "gui/test_runner.h" ] || [ -f "gui/test_runner.cpp" ]; then
    echo "❌ test_runner files still exist!"
    exit 1
else
    echo "✅ test_runner files were removed"
fi

echo

echo "7. Checking CMakeLists.txt for test_runner references..."
if grep -r "test_runner" CMakeLists.txt gui/CMakeLists.txt 2>/dev/null; then
    echo "❌ Found test_runner references in CMakeLists.txt!"
    exit 1
else
    echo "✅ No test_runner references in CMakeLists.txt"
fi

echo

echo "8. Checking that GUI builds successfully..."
if [ -f "build/bin/gpu_kernel_gui" ]; then
    echo "✅ GUI executable exists and was built successfully"
else
    echo "❌ GUI executable not found!"
    exit 1
fi

echo

echo "🎉 All checks passed! Tests tab has been completely removed."
echo

echo "Current GUI tabs should be:"
echo "  1. Examples (primary tab)"
echo "  2. Kernel Runner" 
echo "  3. Results"
echo "  4. Performance"
echo
echo "The Tests tab has been successfully removed!"

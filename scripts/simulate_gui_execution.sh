#!/bin/bash

# Test script to simulate GUI kernel execution
echo "=== Simulating GUI Kernel Execution ==="

cd /home/kevin/Projects/cuda-kernel/build/bin

echo "Testing Vector Addition execution as GUI would do it:"
echo "Executable: $(pwd)/vector_addition"
echo "Arguments: 1000000"
echo "Working directory: $(pwd)"
echo ""

echo "Starting execution..."
timeout 10s ./vector_addition 1000000 2>&1

echo ""
echo "Exit code: $?"

echo ""
echo "Testing Advanced Threading:"
timeout 10s ./advanced_threading 1000000 2>&1 | head -10

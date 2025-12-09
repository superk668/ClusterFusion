#!/bin/bash
# Build and test script for ClusterFusion Pythia kernel

set -e  # Exit on error

echo "========================================"
echo "ClusterFusion Pythia Build & Test"
echo "========================================"

# Check CUDA availability
if ! command -v nvcc &> /dev/null; then
    echo "Error: nvcc not found. Please ensure CUDA is installed."
    exit 1
fi

echo "CUDA version:"
nvcc --version

# Clean previous builds
echo ""
echo "Cleaning previous builds..."
rm -rf build/ dist/ *.egg-info
rm -rf clusterfusion/_clusterfusion*.so

# Build the extension
echo ""
echo "Building ClusterFusion with Pythia support..."
python setup.py build_ext --inplace

if [ $? -ne 0 ]; then
    echo "Error: Build failed!"
    exit 1
fi

echo ""
echo "✓ Build successful!"

# Run tests
echo ""
echo "========================================"
echo "Running Tests"
echo "========================================"

# Test 1: Basic correctness test
echo ""
echo "Test 1: Basic Correctness Test (Small)"
echo "----------------------------------------"
python tests/test_pythia.py

if [ $? -ne 0 ]; then
    echo "Error: Correctness test failed!"
    exit 1
fi

echo ""
echo "✓ Correctness test passed!"

# Test 2: Model integration test (if model is available)
echo ""
echo "Test 2: Model Integration Test"
echo "----------------------------------------"
echo "Note: This requires Pythia-2.8b model to be downloaded."
echo "Run manually: python tests/test_pythia_model.py --layer 0 --seq-len 128"

# Optional: Run model test if --full flag is provided
if [ "$1" == "--full" ]; then
    echo ""
    echo "Running full model integration test..."
    python tests/test_pythia_model.py --layer 0 --seq-len 128
    
    if [ $? -eq 0 ]; then
        echo "✓ Model integration test passed!"
    else
        echo "⚠ Model integration test failed (this may be expected if model is not downloaded)"
    fi
fi

# Optional: Run benchmark if --benchmark flag is provided
if [ "$1" == "--benchmark" ] || [ "$2" == "--benchmark" ]; then
    echo ""
    echo "========================================"
    echo "Running Benchmark"
    echo "========================================"
    python tests/test_pythia_model.py --layer 0 --seq-len 2048 --benchmark --num-runs 100
fi

echo ""
echo "========================================"
echo "All tests completed successfully!"
echo "========================================"
echo ""
echo "Next steps:"
echo "  1. Run full model test: python tests/test_pythia_model.py --layer 0 --seq-len 128"
echo "  2. Run benchmark: python tests/test_pythia_model.py --benchmark --seq-len 2048"
echo "  3. Test generation: python tests/test_pythia_model.py --test-generation"

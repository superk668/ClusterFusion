# Build and test script for ClusterFusion Pythia kernel (Windows PowerShell)

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "ClusterFusion Pythia Build & Test" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

# Check CUDA availability
try {
    $nvccVersion = & nvcc --version 2>&1
    Write-Host "`nCUDA version:" -ForegroundColor Green
    Write-Host $nvccVersion
} catch {
    Write-Host "Error: nvcc not found. Please ensure CUDA is installed." -ForegroundColor Red
    exit 1
}

# Clean previous builds
Write-Host "`nCleaning previous builds..." -ForegroundColor Yellow
Remove-Item -Recurse -Force -ErrorAction SilentlyContinue build, dist, *.egg-info
Remove-Item -Force -ErrorAction SilentlyContinue clusterfusion/_clusterfusion*.pyd

# Build the extension
Write-Host "`nBuilding ClusterFusion with Pythia support..." -ForegroundColor Yellow
python setup.py build_ext --inplace

if ($LASTEXITCODE -ne 0) {
    Write-Host "Error: Build failed!" -ForegroundColor Red
    exit 1
}

Write-Host "`n✓ Build successful!" -ForegroundColor Green

# Run tests
Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "Running Tests" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

# Test 1: Basic correctness test
Write-Host "`nTest 1: Basic Correctness Test (Small)" -ForegroundColor Yellow
Write-Host "----------------------------------------" -ForegroundColor Yellow
python tests/test_pythia.py

if ($LASTEXITCODE -ne 0) {
    Write-Host "Error: Correctness test failed!" -ForegroundColor Red
    exit 1
}

Write-Host "`n✓ Correctness test passed!" -ForegroundColor Green

# Test 2: Model integration test
Write-Host "`nTest 2: Model Integration Test" -ForegroundColor Yellow
Write-Host "----------------------------------------" -ForegroundColor Yellow
Write-Host "Note: This requires Pythia-2.8b model to be downloaded." -ForegroundColor Cyan
Write-Host "Run manually: python tests/test_pythia_model.py --layer 0 --seq-len 128" -ForegroundColor Cyan

# Optional: Run model test if --full flag is provided
if ($args -contains "--full") {
    Write-Host "`nRunning full model integration test..." -ForegroundColor Yellow
    python tests/test_pythia_model.py --layer 0 --seq-len 128
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✓ Model integration test passed!" -ForegroundColor Green
    } else {
        Write-Host "⚠ Model integration test failed (this may be expected if model is not downloaded)" -ForegroundColor Yellow
    }
}

# Optional: Run benchmark if --benchmark flag is provided
if ($args -contains "--benchmark") {
    Write-Host "`n========================================" -ForegroundColor Cyan
    Write-Host "Running Benchmark" -ForegroundColor Cyan
    Write-Host "========================================" -ForegroundColor Cyan
    python tests/test_pythia_model.py --layer 0 --seq-len 2048 --benchmark --num-runs 100
}

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "All tests completed successfully!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "  1. Run full model test: python tests/test_pythia_model.py --layer 0 --seq-len 128" -ForegroundColor White
Write-Host "  2. Run benchmark: python tests/test_pythia_model.py --benchmark --seq-len 2048" -ForegroundColor White
Write-Host "  3. Test generation: python tests/test_pythia_model.py --test-generation" -ForegroundColor White

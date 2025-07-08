# Scripts Directory Organization

This directory contains all shell scripts organized by functionality. The main `run.sh` remains in the project root for easy access.

## Directory Structure

### `/scripts/` (Root Level)
- `build.sh` - Main build script
- `profile.sh` - Performance profiling script  
- `setup_gpu_platform.sh` - GPU platform configuration
- `test.sh` - Main test script

### `/scripts/build/`
Build-related scripts:
- `build_kernels_safely.sh` - Safe kernel building with error handling
- `build_hip.sh` - HIP-specific build script
- `fix_hip_builds.sh` - HIP build issue fixes

### `/scripts/gui/`
GUI-related scripts:
- `launch_gui.sh` - Launch GUI application
- `launch_working_gui.sh` - Launch GUI with working configuration
- `setup_gui.sh` - GUI setup and configuration

### `/scripts/testing/`
Testing and validation scripts:
- `quick_gui_test.sh` - Quick GUI functionality test
- `simulate_gui_execution.sh` - Simulate GUI execution for testing
- `test_gui_components.sh` - Test individual GUI components
- `test_gui_comprehensive.sh` - Comprehensive GUI testing
- `test_gui_final.sh` - Final GUI validation
- `test_gui_final_check.sh` - Final GUI check
- `test_gui_fixes_complete.sh` - Test GUI fixes completion
- `test_gui_functionality.sh` - GUI functionality testing
- `test_gui_kernel_mapping.sh` - Test kernel mapping in GUI
- `test_gui_kernels.sh` - Test GUI kernel execution
- `test_kernel_mapping.sh` - Test kernel mapping
- `test_kernel_mapping_fix.sh` - Test kernel mapping fixes
- `test_rebuilt_gui.sh` - Test rebuilt GUI
- `test_updated_gui_kernels.sh` - Test updated GUI kernels

### `/scripts/verification/`
Verification and final validation scripts:
- `final_gui_report.sh` - Generate final GUI report
- `final_status_check.sh` - Final project status check
- `final_verification_complete.sh` - Complete verification
- `fix_verification_final.sh` - Fix verification issues
- `test_safe_advanced_threading.sh` - Test safe advanced threading
- `verify_safe_advanced_threading.sh` - Verify advanced threading safety

### `/scripts/debugging/`
Debugging and troubleshooting scripts:
- `debug_gui_paths.sh` - Debug GUI path issues
- `debug_warp_primitives.sh` - Debug warp primitives issues

### `/scripts/screenshots/`
Screenshot and documentation scripts:
- `capture_additional_screenshots.sh` - Capture additional screenshots
- `save_screenshot.sh` - Save screenshot utility
- `take_screenshots.sh` - Take project screenshots

## Usage

All scripts maintain their original functionality. Update any references in documentation or other scripts to use the new paths.

### Examples:
```bash
# Build scripts
./scripts/build/build_kernels_safely.sh
./scripts/build/build_hip.sh

# GUI scripts  
./scripts/gui/launch_gui.sh
./scripts/gui/setup_gui.sh

# Testing scripts
./scripts/testing/quick_gui_test.sh
./scripts/testing/test_gui_comprehensive.sh

# Verification scripts
./scripts/verification/verify_safe_advanced_threading.sh
./scripts/verification/final_status_check.sh
```

## Migration Notes

- Main `run.sh` remains in project root for convenience
- All scripts retain their original functionality
- Update any hardcoded paths in scripts or documentation
- Consider updating CI/CD pipelines if they reference moved scripts

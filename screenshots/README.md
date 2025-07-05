# Screenshots Directory

This directory contains screenshots of the GPU Kernel GUI application.

## Current Status

The screenshots are currently placeholders due to snap/GLIBC library conflicts preventing the GUI from launching properly. Once the library issues are resolved, this directory will contain:

- `gui_main.png` - Main interface showing kernel selection and controls
- `gui_performance.png` - Performance monitoring charts and metrics
- `gui_results.png` - Results visualization and analysis tools

## GUI Features to Screenshot

### Main Interface
- Kernel selection dropdown/list
- Parameter configuration panel
- Run/Stop controls
- Status indicators

### Performance Monitor
- Real-time execution time charts
- Memory usage graphs
- Throughput metrics
- Performance comparison tools

### Results Viewer
- Output data visualization
- Statistical analysis
- Export options
- Comparison views

## How to Take Screenshots

Once the library conflicts are resolved:

1. Launch the GUI: `./launch_gui.sh`
2. Navigate through different features
3. Capture screenshots using system screenshot tools
4. Save them in this directory with descriptive names
5. Update the main README.md with actual screenshot paths

## Library Conflict Resolution

The current issue is a snap/GLIBC symbol lookup error:
```
./build_gui/bin/gpu_kernel_gui: symbol lookup error: /snap/core20/current/lib/x86_64-linux-gnu/libpthread.so.0: undefined symbol: __libc_pthread_init, version GLIBC_PRIVATE
```

Potential solutions:
- Use system Qt libraries instead of snap versions
- Set appropriate LD_LIBRARY_PATH
- Build in a clean environment without snap conflicts
- Use AppImage or flatpak distribution

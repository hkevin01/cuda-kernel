# GPU Kernel Examples GUI

A modern Qt-based graphical user interface for the GPU Kernel Examples project, providing an intuitive way to run, test, and analyze CUDA/HIP kernels.

## Features

### 🚀 Kernel Runner
- **Visual Kernel Selection**: Browse and select from available GPU kernels
- **Configuration Panel**: Adjust iterations, data size, and platform settings
- **Real-time Output**: View kernel execution output with syntax highlighting
- **Progress Tracking**: Monitor kernel execution progress

### 📊 Results Viewer
- **Test Results Table**: View all kernel execution results in a tabular format
- **Detailed Results**: Click on any result to see detailed output
- **Statistics**: Track success rates and execution statistics
- **Export Functionality**: Export results to CSV format

### 📈 Performance Widget
- **Real-time Charts**: Visualize kernel execution time and throughput
- **Performance Monitoring**: Start/stop performance data collection
- **Statistical Analysis**: View min, max, and average performance metrics
- **Interactive Charts**: Zoom, pan, and explore performance data

### 🧪 Test Runner
- **Batch Testing**: Run all tests or selected tests
- **Test Categories**: Organize tests by unit, integration, and performance
- **Test Information**: View test descriptions and dependencies
- **Progress Tracking**: Monitor test execution progress

### 🎛️ System Features
- **System Tray Support**: Minimize to system tray with notifications
- **Settings Persistence**: Remember window state and preferences
- **Cross-platform**: Works on Linux, Windows, and macOS
- **Modern UI**: Clean, responsive interface with dark/light themes

## Requirements

### Qt Dependencies
- **Qt6** (preferred) or **Qt5** (fallback)
- **Qt Core**: Basic Qt functionality
- **Qt Widgets**: UI components
- **Qt Charts**: Performance visualization (optional)

### System Requirements
- **Linux**: Ubuntu 20.04+, Fedora 33+, or similar
- **Windows**: Windows 10+ with Visual Studio 2019+
- **macOS**: macOS 10.15+ with Xcode 12+
- **GPU**: NVIDIA GPU (CUDA) or AMD GPU (ROCm/HIP)

## Installation

### Automatic Setup (Linux)
```bash
# Run the setup script
./scripts/gui/setup_gui.sh
```

### Manual Installation

#### Ubuntu/Debian
```bash
sudo apt update
sudo apt install qt6-base-dev qt6-charts-dev qt6-tools-dev
```

#### Fedora/RHEL/CentOS
```bash
sudo dnf install qt6-qtbase-devel qt6-qtcharts-devel qt6-qttools-devel
```

#### Arch Linux
```bash
sudo pacman -S qt6-base qt6-charts qt6-tools
```

### Building the GUI
```bash
# Configure with GUI enabled
cmake -DBUILD_GUI=ON -DUSE_HIP=ON ..

# Build
make -j$(nproc)

# Run the GUI
./bin/gpu_kernel_gui
```

## Usage

### Starting the Application
```bash
# Basic usage
./bin/gpu_kernel_gui

# With platform specification
./bin/gpu_kernel_gui --platform hip

# Test mode
./bin/gpu_kernel_gui --test-mode
```

### Running Kernels
1. **Select a Kernel**: Choose from the list of available kernels
2. **Configure Parameters**: Set iterations, data size, and platform
3. **Run**: Click "Run Selected Kernel" or use the toolbar
4. **Monitor**: Watch real-time output and progress
5. **View Results**: Check the Results tab for execution details

### Running Tests
1. **Select Tests**: Choose individual tests or "Run All Tests"
2. **Monitor Progress**: Watch test execution in the Tests tab
3. **View Results**: Check test results and any failures
4. **Export**: Save test results for analysis

### Performance Analysis
1. **Start Monitoring**: Click "Start Monitoring" in Performance tab
2. **Run Kernels**: Execute kernels while monitoring is active
3. **View Charts**: Observe real-time performance data
4. **Analyze**: Use the performance table for detailed statistics

## Configuration

### Command Line Options
- `--platform <cuda|hip>`: Specify GPU platform
- `--test-mode`: Run in test mode
- `--help`: Show help information
- `--version`: Show version information

### Settings
The GUI automatically saves:
- Window geometry and state
- System tray preferences
- Platform selection
- Recent kernel configurations

## Architecture

### Widget Structure
```
MainWindow
├── KernelRunner (Tab 1)
│   ├── Kernel List
│   ├── Configuration Panel
│   └── Output Display
├── ResultViewer (Tab 2)
│   ├── Results Table
│   ├── Statistics Panel
│   └── Detail View
├── PerformanceWidget (Tab 3)
│   ├── Control Buttons
│   ├── Time Chart
│   ├── Throughput Chart
│   └── Performance Table
└── TestRunner (Tab 4)
    ├── Test List
    ├── Test Information
    └── Test Output
```

### Key Classes
- **MainWindow**: Main application window and coordination
- **KernelRunner**: Kernel execution and management
- **ResultViewer**: Results display and export
- **PerformanceWidget**: Performance monitoring and visualization
- **TestRunner**: Test execution and management

## Development

### Building from Source
```bash
# Clone the repository
git clone <repository-url>
cd cuda-kernel

# Install dependencies
./scripts/gui/setup_gui.sh

# Build
cd build
make -j$(nproc)
```

### Adding New Kernels
1. Add kernel executable to the build system
2. Update `KernelRunner::loadKernelList()` with kernel information
3. Ensure kernel accepts standard command-line arguments

### Adding New Tests
1. Create test executable
2. Update `TestRunner::loadTestList()` with test information
3. Ensure test follows standard output format

### Customization
- **Themes**: Modify Qt stylesheets in the source code
- **Icons**: Replace icons in `gui/resources.qrc`
- **Layout**: Adjust widget layouts in the UI files
- **Functionality**: Extend widget classes for additional features

## Troubleshooting

### Common Issues

#### Qt Not Found
```bash
# Check Qt installation
qmake --version
# or
qt6-qmake --version

# Install Qt if missing
sudo apt install qt6-base-dev  # Ubuntu/Debian
```

#### Build Errors
```bash
# Clean build directory
rm -rf build
mkdir build
cd build

# Reconfigure
cmake -DBUILD_GUI=ON -DUSE_HIP=ON ..
make -j$(nproc)
```

#### Runtime Errors
```bash
# Check GPU drivers
nvidia-smi  # NVIDIA
rocm-smi    # AMD

# Check library dependencies
ldd ./bin/gpu_kernel_gui
```

### Debug Mode
```bash
# Build with debug information
cmake -DCMAKE_BUILD_TYPE=Debug -DBUILD_GUI=ON ..
make -j$(nproc)

# Run with debug output
./bin/gpu_kernel_gui --debug
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This GUI is part of the GPU Kernel Examples project and follows the same license terms.

## Support

For issues and questions:
- Check the troubleshooting section
- Review the main project documentation
- Open an issue on the project repository 
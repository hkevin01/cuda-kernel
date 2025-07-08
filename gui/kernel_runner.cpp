#include "kernel_runner.h"
#include <QDir>
#include <QFileInfo>
#include <QMessageBox>
#include <QApplication>
#include <QStandardPaths>
#include <QJsonDocument>
#include <QJsonObject>
#include <QJsonArray>
#include <QTextStream>
#include <QScrollBar>
#include <QRegularExpression>
#include <QPalette>

KernelRunner::KernelRunner(QWidget *parent)
    : QWidget(parent), m_currentProcess(nullptr), m_progressTimer(new QTimer(this)), m_isRunning(false)
{
    setupUI();
    loadKernelList();

    // Set up progress timer
    m_progressTimer->setInterval(100);
    connect(m_progressTimer, &QTimer::timeout, this, &KernelRunner::updateProgress);
}

KernelRunner::~KernelRunner()
{
    // Stop progress timer
    if (m_progressTimer && m_progressTimer->isActive())
    {
        m_progressTimer->stop();
    }
    
    // Clean up process using QPointer
    if (m_currentProcess)
    {
        if (m_currentProcess->state() == QProcess::Running)
        {
            m_currentProcess->terminate();
            m_currentProcess->waitForFinished(3000);
        }
        
        // Let Qt handle the cleanup naturally
        m_currentProcess->deleteLater();
    }
}

void KernelRunner::setupUI()
{
    QVBoxLayout *mainLayout = new QVBoxLayout(this);

    // Top section with kernel list and info
    QHBoxLayout *topLayout = new QHBoxLayout();

    // Left side - Kernel list
    QGroupBox *kernelGroup = new QGroupBox(tr("Available Kernels"));
    QVBoxLayout *kernelLayout = new QVBoxLayout(kernelGroup);

    m_kernelList = new QListWidget();
    m_kernelList->setSelectionMode(QAbstractItemView::SingleSelection);
    kernelLayout->addWidget(m_kernelList);

    QHBoxLayout *buttonLayout = new QHBoxLayout();
    m_runButton = new QPushButton(tr("Run Selected Kernel"));
    m_refreshButton = new QPushButton(tr("Refresh"));
    buttonLayout->addWidget(m_runButton);
    buttonLayout->addWidget(m_refreshButton);
    buttonLayout->addStretch();
    kernelLayout->addLayout(buttonLayout);

    topLayout->addWidget(kernelGroup);

    // Right side - Kernel info and configuration
    QVBoxLayout *rightLayout = new QVBoxLayout();

    // Kernel information
    QGroupBox *infoGroup = new QGroupBox(tr("Kernel Information"));
    QVBoxLayout *infoLayout = new QVBoxLayout(infoGroup);

    m_kernelNameLabel = new QLabel(tr("Select a kernel"));
    m_kernelNameLabel->setStyleSheet("font-weight: bold; font-size: 14px;");
    infoLayout->addWidget(m_kernelNameLabel);

    m_kernelDescriptionLabel = new QLabel();
    m_kernelDescriptionLabel->setWordWrap(true);
    infoLayout->addWidget(m_kernelDescriptionLabel);

    m_kernelParametersLabel = new QLabel();
    m_kernelParametersLabel->setWordWrap(true);
    infoLayout->addWidget(m_kernelParametersLabel);

    rightLayout->addWidget(infoGroup);

    // Configuration
    QGroupBox *configGroup = new QGroupBox(tr("Configuration"));
    QVBoxLayout *configLayout = new QVBoxLayout(configGroup);

    QHBoxLayout *iterationsLayout = new QHBoxLayout();
    iterationsLayout->addWidget(new QLabel(tr("Iterations:")));
    m_iterationsSpinBox = new QSpinBox();
    m_iterationsSpinBox->setRange(1, 1000);
    m_iterationsSpinBox->setValue(10);
    iterationsLayout->addWidget(m_iterationsSpinBox);
    iterationsLayout->addStretch();
    configLayout->addLayout(iterationsLayout);

    QHBoxLayout *dataSizeLayout = new QHBoxLayout();
    dataSizeLayout->addWidget(new QLabel(tr("Data Size:")));
    m_dataSizeSpinBox = new QSpinBox();
    m_dataSizeSpinBox->setRange(1024, 1000000);
    m_dataSizeSpinBox->setValue(10000);
    m_dataSizeSpinBox->setSuffix(tr(" elements"));
    dataSizeLayout->addWidget(m_dataSizeSpinBox);
    dataSizeLayout->addStretch();
    configLayout->addLayout(dataSizeLayout);

    QHBoxLayout *platformLayout = new QHBoxLayout();
    platformLayout->addWidget(new QLabel(tr("Platform:")));
    m_platformComboBox = new QComboBox();
    m_platformComboBox->addItems({"HIP", "CUDA"});
    platformLayout->addWidget(m_platformComboBox);
    platformLayout->addStretch();
    configLayout->addLayout(platformLayout);

    rightLayout->addWidget(configGroup);
    rightLayout->addStretch();

    topLayout->addLayout(rightLayout);
    mainLayout->addLayout(topLayout);

    // Bottom section - Output and progress
    QGroupBox *outputGroup = new QGroupBox(tr("Output"));
    QVBoxLayout *outputLayout = new QVBoxLayout(outputGroup);

    m_outputText = new QTextEdit();
    m_outputText->setReadOnly(true);
    m_outputText->setMaximumHeight(200);
    
    // Set monospace font for better code readability
    QFont monospaceFont("Consolas", 9);
    if (!monospaceFont.exactMatch()) {
        monospaceFont.setFamily("Monaco");  // macOS
        if (!monospaceFont.exactMatch()) {
            monospaceFont.setFamily("Liberation Mono");  // Linux
            if (!monospaceFont.exactMatch()) {
                monospaceFont.setFamily("Courier New");  // Fallback
            }
        }
    }
    monospaceFont.setStyleHint(QFont::Monospace);
    m_outputText->setFont(monospaceFont);
    
    // Set background color and improve readability - force override
    m_outputText->setStyleSheet(
        "QTextEdit {"
        "    background-color: #d0d0d0 !important;"
        "    border: 1px solid #999999 !important;"
        "    padding: 5px !important;"
        "    line-height: 1.2 !important;"
        "    color: #000000 !important;"
        "}"
    );
    
    // Force palette update to override system themes
    QPalette palette = m_outputText->palette();
    palette.setColor(QPalette::Base, QColor("#d0d0d0"));
    palette.setColor(QPalette::Window, QColor("#d0d0d0"));
    m_outputText->setPalette(palette);
    
    outputLayout->addWidget(m_outputText);

    QHBoxLayout *statusLayout = new QHBoxLayout();
    m_statusLabel = new QLabel(tr("Ready"));
    m_progressBar = new QProgressBar();
    m_progressBar->setVisible(false);
    statusLayout->addWidget(m_statusLabel);
    statusLayout->addWidget(m_progressBar);
    outputLayout->addLayout(statusLayout);

    mainLayout->addWidget(outputGroup);

    // Connect signals
    connect(m_runButton, &QPushButton::clicked, this, &KernelRunner::onRunButtonClicked);
    connect(m_refreshButton, &QPushButton::clicked, this, &KernelRunner::onRefreshButtonClicked);
    connect(m_kernelList, &QListWidget::currentItemChanged, this, &KernelRunner::onKernelSelectionChanged);
}

void KernelRunner::loadKernelList()
{
    m_kernelList->clear();
    m_kernels.clear();

    // Define available kernels (only include working ones)
    QStringList kernelNames = {
        "Vector Addition",
        "Matrix Multiplication",
        "Parallel Reduction",
        "2D Convolution",
        "Monte Carlo",
        "Advanced FFT",
        "Advanced Threading",
        "Dynamic Memory",
        "3D FFT",
        "N-Body Simulation"};

    QStringList descriptions = {
        "Vector Addition: Adds two arrays element by element. The simplest GPU operation - like having thousands of calculators working in parallel to add corresponding numbers from two lists.",
        "Matrix Multiplication: Multiplies two matrices together. Used in machine learning, graphics, and scientific computing. Shows how GPUs excel at mathematical operations with smart memory usage.",
        "Parallel Reduction: Finds the sum, maximum, or minimum of a large array. Demonstrates how to combine results from thousands of parallel threads efficiently.",
        "2D Convolution: Applies filters to images (like blur, sharpen, edge detection). The foundation of image processing and computer vision - shows how GPUs process pixels in parallel.",
        "Monte Carlo: Uses random sampling to solve mathematical problems. Like throwing darts at a dartboard to calculate pi - demonstrates GPU's power for statistical simulations.",
        "Advanced FFT: Fast Fourier Transform for signal processing. Converts signals between time and frequency domains - used in audio processing, compression, and scientific analysis.",
        "Advanced Threading: Shows sophisticated thread cooperation patterns. Demonstrates how thousands of GPU threads can work together safely without conflicts.",
        "Dynamic Memory: Shows how to allocate and manage memory on the GPU during execution. Important for applications that don't know memory requirements beforehand.",
        "3D FFT: Three-dimensional Fast Fourier Transform for volumetric data. Used in medical imaging, weather simulation, and 3D signal processing.",
        "N-Body Simulation: Simulates gravitational forces between particles (like planets, stars, or molecules). Shows GPU's power for physics simulations and scientific computing."};

    QStringList categories = {
        "Basic", "Basic", "Basic", "Basic", "Basic",
        "Advanced", "Advanced", "Advanced", "Advanced", "Advanced"};

    // Map display names to actual executable names
    QMap<QString, QString> executableMap;
    executableMap["Vector Addition"] = "vector_addition";
    executableMap["Matrix Multiplication"] = "matrix_multiplication";
    executableMap["Parallel Reduction"] = "parallel_reduction";
    executableMap["2D Convolution"] = "convolution_2d";
    executableMap["Monte Carlo"] = "monte_carlo";
    executableMap["Advanced FFT"] = "advanced_fft";
    executableMap["Advanced Threading"] = "advanced_threading";
    executableMap["Dynamic Memory"] = "dynamic_memory";
    executableMap["3D FFT"] = "advanced_fft";
    executableMap["N-Body Simulation"] = "nbody_simulation";

    for (int i = 0; i < kernelNames.size(); ++i)
    {
        KernelInfo info;
        info.name = kernelNames[i];
        info.description = descriptions[i];
        info.category = categories[i];
        info.executableName = executableMap.value(kernelNames[i], "");

        // IMPROVED: Check for executable existence *before* adding to the list
        info.executablePath = findKernelExecutable(info.executableName);

        m_kernels[info.name] = info;

        QString displayName = QString("[%1] %2").arg(info.category, info.name);
        if (info.executablePath.isEmpty())
        {
            displayName += " (Not Built)";
        }
        QListWidgetItem *item = new QListWidgetItem(displayName);
        item->setData(Qt::UserRole, info.name);
        m_kernelList->addItem(item);
    }

    if (m_kernelList->count() > 0)
    {
        m_kernelList->setCurrentRow(0);
    }
}

void KernelRunner::runSelectedKernel()
{
    QListWidgetItem *currentItem = m_kernelList->currentItem();
    if (!currentItem)
    {
        QMessageBox::warning(this, tr("No Kernel Selected"),
                             tr("Please select a kernel to run."));
        return;
    }

    QString kernelName = currentItem->data(Qt::UserRole).toString();
    runKernel(kernelName);
}

void KernelRunner::runKernel(const QString &kernelName)
{
    if (m_isRunning)
    {
        QMessageBox::warning(this, tr("Kernel Running"),
                             tr("A kernel is already running. Please wait for it to complete."));
        return;
    }

    if (!m_kernels.contains(kernelName))
    {
        QMessageBox::critical(this, tr("Kernel Not Found"),
                              tr("Kernel '%1' not found.").arg(kernelName));
        return;
    }

    m_currentKernel = kernelName;
    m_isRunning = true;

    // Update UI
    m_runButton->setEnabled(false);
    m_statusLabel->setText(tr("Running %1...").arg(kernelName));
    m_progressBar->setVisible(true);
    m_progressBar->setRange(0, 0); // Indeterminate progress
    m_outputText->clear();

    // Start progress timer
    m_progressTimer->start();
    emit progressUpdated(0);

    // Get executable path from our stored info
    QString executable = m_kernels[kernelName].executablePath;
    if (executable.isEmpty())
    {
        QMessageBox::critical(this, tr("Executable Not Found"),
                              tr("Could not find executable for kernel '%1'").arg(kernelName));
        m_isRunning = false;
        m_runButton->setEnabled(true);
        m_progressBar->setVisible(false);
        m_statusLabel->setText(tr("Ready"));
        return;
    }

    // Create process with minimal signal connections
    if (m_currentProcess) {
        // Schedule cleanup of existing process
        QTimer::singleShot(0, this, &KernelRunner::cleanupProcess);
    }
    
    m_currentProcess = new QProcess();  // Don't set parent to avoid automatic cleanup

    // Use only essential signal connections
    connect(m_currentProcess, QOverload<int, QProcess::ExitStatus>::of(&QProcess::finished),
            this, &KernelRunner::onProcessFinished, Qt::QueuedConnection);
    connect(m_currentProcess, &QProcess::readyReadStandardOutput,
            this, &KernelRunner::onProcessOutput, Qt::QueuedConnection);

    // Prepare arguments based on kernel type
    QStringList arguments;
    KernelInfo &info = m_kernels[kernelName];

    // Check if this kernel uses the new simplified argument format
    if (kernelName == "Vector Addition" ||
        kernelName == "Matrix Multiplication" ||
        kernelName == "Parallel Reduction" ||
        kernelName == "2D Convolution" ||
        kernelName == "Monte Carlo" ||
        kernelName == "Advanced Threading" ||
        kernelName == "Advanced FFT" ||
        kernelName == "3D FFT" ||
        kernelName == "Dynamic Memory" ||
        kernelName == "N-Body Simulation")
    {
        // Use only the size parameter as positional argument
        arguments << QString::number(m_dataSizeSpinBox->value());
    }
    else
    {
        // Use the old format for future kernels
        arguments << "--iterations" << QString::number(m_iterationsSpinBox->value());
        arguments << "--size" << QString::number(m_dataSizeSpinBox->value());
        arguments << "--platform" << m_platformComboBox->currentText();
        arguments << info.parameters;
    }

    // Start process
    m_outputText->append(tr("<span style='color: #0066CC; font-weight: bold; font-size: 12px;'>🚀 Starting %1...</span>").arg(QFileInfo(executable).baseName()));
    m_outputText->append(tr("<span style='color: #666666; font-style: italic;'>📁 Executable: %1</span>").arg(executable));
    m_outputText->append(tr("<span style='color: #666666; font-style: italic;'>⚙️  Arguments: %1</span>").arg(arguments.join(" ")));
    m_outputText->append(""); // Empty line for spacing

    m_currentProcess->start(executable, arguments);
}

QString KernelRunner::findKernelExecutable(const QString &executableName)
{
    if (executableName.isEmpty())
    {
        return QString();
    }

    // --- DYNAMIC PROJECT ROOT SEARCH ---
    QDir currentDir(QApplication::applicationDirPath());
    QString projectRootPath;

    // Search upwards for a project marker (e.g., CMakeLists.txt or .git folder)
    while (currentDir.cdUp()) {
        if (QFileInfo(currentDir.filePath("CMakeLists.txt")).exists() || QFileInfo(currentDir.filePath(".git")).isDir()) {
            projectRootPath = currentDir.absolutePath();
            break;
        }
        if (currentDir.isRoot()) {
            break;
        }
    }

    QStringList searchPaths;

    if (!projectRootPath.isEmpty()) {
        // Prioritize HIP build directories, then standard build directories
        searchPaths << projectRootPath + "/build_gui_hip/bin/" + executableName;
        searchPaths << projectRootPath + "/build_simple/bin/" + executableName;
        searchPaths << projectRootPath + "/build/bin/" + executableName;
        searchPaths << projectRootPath + "/build_hip/bin/" + executableName;
        searchPaths << projectRootPath + "/build_hip/" + executableName + "_hip";
    }

    // --- FALLBACK: ORIGINAL RELATIVE PATHS ---
    searchPaths << QApplication::applicationDirPath() + "/" + executableName;
    searchPaths << QApplication::applicationDirPath() + "/" + executableName + "_hip";
    searchPaths << QApplication::applicationDirPath() + "/../bin/" + executableName;
    searchPaths << QApplication::applicationDirPath() + "/../../build/bin/" + executableName;
    searchPaths << QApplication::applicationDirPath() + "/../../build_simple/bin/" + executableName;
    searchPaths << "./build/bin/" + executableName;
    searchPaths << "./build_simple/bin/" + executableName;

    for (const QString &path : searchPaths)
    {
        QFileInfo fileInfo(path);
        if (fileInfo.exists() && fileInfo.isExecutable())
        {
            return fileInfo.absoluteFilePath(); // Return the full, canonical path
        }
    }

    return QString();
}

void KernelRunner::updateKernelInfo(const QString &kernelName)
{
    if (!m_kernels.contains(kernelName))
    {
        m_kernelNameLabel->setText(tr("Select a kernel"));
        m_kernelDescriptionLabel->clear();
        m_kernelParametersLabel->clear();
        return;
    }

    const KernelInfo &info = m_kernels[kernelName];
    m_kernelNameLabel->setText(info.name);
    m_kernelDescriptionLabel->setText(info.description);

    // Update run button based on whether executable was found
    m_runButton->setEnabled(!info.executablePath.isEmpty() && !m_isRunning);

    if (info.executablePath.isEmpty())
    {
        m_statusLabel->setText(tr("Kernel not built - executable not available"));
    }
    else
    {
        m_statusLabel->setText(tr("Ready"));
    }
}

void KernelRunner::parseKernelOutput(const QString &output)
{
    // Parse performance metrics from output with markdown-style syntax highlighting
    QTextStream stream(const_cast<QString *>(&output), QIODevice::ReadOnly);
    QString line;

    while (stream.readLineInto(&line))
    {
        QString formattedLine = line;
        
        // Apply syntax highlighting based on content patterns
        if (line.contains("===", Qt::CaseInsensitive))
        {
            // Section headers - bold blue
            formattedLine = tr("<span style='color: #0066CC; font-weight: bold; font-size: 14px;'>%1</span>").arg(line);
        }
        else if (line.contains("Time:", Qt::CaseInsensitive) || line.contains("GPU time:", Qt::CaseInsensitive))
        {
            // Timing information - bold green
            formattedLine = tr("<span style='color: #009900; font-weight: bold;'>%1</span>").arg(line);
        }
        else if (line.contains("Bandwidth:", Qt::CaseInsensitive) || line.contains("GB/s", Qt::CaseInsensitive))
        {
            // Performance metrics - bold orange
            formattedLine = tr("<span style='color: #FF6600; font-weight: bold;'>%1</span>").arg(line);
        }
        else if (line.contains("Error:", Qt::CaseInsensitive) || line.contains("FAIL", Qt::CaseInsensitive) || 
                 line.contains("failed", Qt::CaseInsensitive) || line.contains("✗", Qt::CaseInsensitive))
        {
            // Errors and failures - bold red
            formattedLine = tr("<span style='color: #CC0000; font-weight: bold; background-color: #FFE6E6;'>%1</span>").arg(line);
        }
        else if (line.contains("Success:", Qt::CaseInsensitive) || line.contains("PASS", Qt::CaseInsensitive) || 
                 line.contains("✓", Qt::CaseInsensitive) || line.contains("completed successfully", Qt::CaseInsensitive))
        {
            // Success messages - bold green
            formattedLine = tr("<span style='color: #009900; font-weight: bold; background-color: #E6FFE6;'>%1</span>").arg(line);
        }
        else if (line.contains("Result:", Qt::CaseInsensitive) || line.contains("estimate:", Qt::CaseInsensitive))
        {
            // Results - bold purple
            formattedLine = tr("<span style='color: #6600CC; font-weight: bold;'>%1</span>").arg(line);
        }
        else if (line.contains("Device", Qt::CaseInsensitive) || line.contains("GPU", Qt::CaseInsensitive) || 
                 line.contains("HIP", Qt::CaseInsensitive) || line.contains("CUDA", Qt::CaseInsensitive))
        {
            // GPU/Device information - blue
            formattedLine = tr("<span style='color: #0066CC;'>%1</span>").arg(line);
        }
        else if (line.contains("Grid size:", Qt::CaseInsensitive) || line.contains("Block size:", Qt::CaseInsensitive) ||
                 line.contains("threads", Qt::CaseInsensitive) || line.contains("blocks", Qt::CaseInsensitive))
        {
            // Configuration parameters - dark cyan
            formattedLine = tr("<span style='color: #006666;'>%1</span>").arg(line);
        }
        else if (line.contains("Starting", Qt::CaseInsensitive) || line.contains("Arguments:", Qt::CaseInsensitive))
        {
            // Process information - gray
            formattedLine = tr("<span style='color: #666666; font-style: italic;'>%1</span>").arg(line);
        }
        else if (line.trimmed().isEmpty())
        {
            // Keep empty lines as is
            formattedLine = line;
        }
        else
        {
            // Default text - apply subtle highlighting for numbers and units
            formattedLine = line;
            
            // Highlight numbers with units and common patterns
            if (line.contains("ms") || line.contains("GB/s") || line.contains("MB") || 
                line.contains("Hz") || line.contains("%") || line.contains("seconds"))
            {
                // Apply basic number highlighting for performance metrics
                formattedLine = tr("<span style='color: #FF6600;'>%1</span>").arg(line);
            }
        }
        
        m_outputText->append(formattedLine);
    }

    // Auto-scroll to bottom
    QScrollBar *scrollBar = m_outputText->verticalScrollBar();
    scrollBar->setValue(scrollBar->maximum());
}

void KernelRunner::onRunButtonClicked()
{
    runSelectedKernel();
}

void KernelRunner::onRefreshButtonClicked()
{
    loadKernelList();
}

void KernelRunner::onKernelSelectionChanged()
{
    QListWidgetItem *currentItem = m_kernelList->currentItem();
    if (currentItem)
    {
        QString kernelName = currentItem->data(Qt::UserRole).toString();
        updateKernelInfo(kernelName);
    }
}

void KernelRunner::onProcessFinished(int exitCode, QProcess::ExitStatus exitStatus)
{
    m_isRunning = false;
    m_progressTimer->stop();

    // Update UI
    m_runButton->setEnabled(true);
    m_progressBar->setVisible(false);

    bool success = (exitCode == 0 && exitStatus == QProcess::NormalExit);
    QString result;
    
    if (m_currentProcess && !m_currentProcess.isNull()) {
        // Read any remaining output and process it
        QString remainingOutput = m_currentProcess->readAllStandardOutput();
        QString remainingError = m_currentProcess->readAllStandardError();
        
        // Process any remaining output through the normal output parser
        if (!remainingOutput.isEmpty()) {
            parseKernelOutput(remainingOutput);
        }
        
        // Handle any remaining stderr (with filtering)
        if (!remainingError.isEmpty()) {
            QString filteredError = remainingError;
            filteredError.remove(QRegularExpression(".*Warning: Resource leak detected by SharedSignalPool.*\\n?"));
            
            if (!filteredError.trimmed().isEmpty()) {
                m_outputText->append(tr("<span style='color: red; font-weight: bold;'>[STDERR] %1</span>").arg(filteredError.trimmed()));
            }
        }
        
        result = remainingOutput + remainingError;
    }

    if (success)
    {
        m_statusLabel->setText(tr("Kernel completed successfully"));
        m_outputText->append(tr("\n<b>Kernel completed successfully!</b>"));
    }
    else
    {
        m_statusLabel->setText(tr("Kernel failed"));
        m_outputText->append(tr("\n<b>Kernel failed with exit code %1</b>").arg(exitCode));
    }

    // Schedule deferred cleanup of process
    QTimer::singleShot(100, this, &KernelRunner::cleanupProcess);

    // Emit signal
    emit kernelFinished(m_currentKernel, success, result);
    emit progressUpdated(-1);
}

void KernelRunner::onProcessError(QProcess::ProcessError error)
{
    m_isRunning = false;
    m_progressTimer->stop();

    m_runButton->setEnabled(true);
    m_progressBar->setVisible(false);

    QString errorMsg;
    switch (error)
    {
    case QProcess::FailedToStart:
        errorMsg = tr("Failed to start process");
        break;
    case QProcess::Crashed:
        errorMsg = tr("Process crashed");
        break;
    case QProcess::Timedout:
        errorMsg = tr("Process timed out");
        break;
    case QProcess::WriteError:
        errorMsg = tr("Write error");
        break;
    case QProcess::ReadError:
        errorMsg = tr("Read error");
        break;
    case QProcess::UnknownError:
    default:
        errorMsg = tr("Unknown error");
        break;
    }

    m_statusLabel->setText(tr("Error: %1").arg(errorMsg));
    m_outputText->append(tr("\n<b>Error: %1</b>").arg(errorMsg));

    // Schedule deferred cleanup of process
    QTimer::singleShot(100, this, &KernelRunner::cleanupProcess);

    emit kernelFinished(m_currentKernel, false, errorMsg);
    emit progressUpdated(-1);
}

void KernelRunner::onProcessOutput()
{
    if (!m_currentProcess)
        return;

    QString output = m_currentProcess->readAllStandardOutput();
    QString error = m_currentProcess->readAllStandardError();

    if (!output.isEmpty())
    {
        parseKernelOutput(output);
    }

    if (!error.isEmpty())
    {
        // Filter out Qt SharedSignalPool warnings
        QString filteredError = error;
        filteredError.remove(QRegularExpression(".*Warning: Resource leak detected by SharedSignalPool.*\\n?"));
        
        if (!filteredError.trimmed().isEmpty())
        {
            // Use a more descriptive label for errors
            m_outputText->append(tr("<span style='color: red; font-weight: bold;'>[STDERR] %1</span>").arg(filteredError.trimmed()));
        }
    }
}

void KernelRunner::updateProgress()
{
    if (m_isRunning)
    {
        // Simulate progress for long-running kernels
        static int progress = 0;
        progress = (progress + 5) % 100;
        emit progressUpdated(progress);
    }
}

void KernelRunner::refreshKernelList()
{
    loadKernelList();
}

void KernelRunner::cleanupProcess()
{
    // This slot is called via QTimer::singleShot to ensure proper event loop processing
    if (m_currentProcess && !m_currentProcess.isNull())
    {
        // Disconnect all signals first
        m_currentProcess->disconnect();
        
        // Terminate the process if it's still running
        if (m_currentProcess->state() != QProcess::NotRunning)
        {
            m_currentProcess->terminate();
            m_currentProcess->waitForFinished(1000);
            if (m_currentProcess->state() != QProcess::NotRunning)
            {
                m_currentProcess->kill();
                m_currentProcess->waitForFinished(500);
            }
        }
        
        // Direct deletion to avoid Qt's event system
        delete m_currentProcess.data();
        m_currentProcess.clear(); // Clear the QPointer
    }
}
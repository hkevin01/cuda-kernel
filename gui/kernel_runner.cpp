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
    if (m_currentProcess && m_currentProcess->state() == QProcess::Running)
    {
        m_currentProcess->terminate();
        m_currentProcess->waitForFinished(5000);
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
    
    // Set background color and improve readability
    m_outputText->setStyleSheet(
        "QTextEdit {"
        "    background-color: #f8f8f8;"
        "    border: 1px solid #cccccc;"
        "    padding: 5px;"
        "    line-height: 1.2;"
        "}"
    );
    
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

    // Define available kernels
    QStringList kernelNames = {
        "Vector Addition",
        "Matrix Multiplication",
        "Parallel Reduction",
        "2D Convolution",
        "Monte Carlo",
        "Advanced FFT",
        "Advanced Threading",
        "Dynamic Memory",
        "Warp Primitives",
        "3D FFT",
        "N-Body Simulation"};

    QStringList descriptions = {
        "Simple vector addition kernel demonstrating basic GPU operations",
        "Matrix multiplication using shared memory optimization",
        "Parallel reduction algorithm for finding sum/max/min",
        "2D convolution with configurable kernel size",
        "Monte Carlo simulation for numerical integration",
        "Advanced FFT implementation with multiple optimizations",
        "Demonstrates advanced thread cooperation patterns",
        "Dynamic memory allocation and management on GPU",
        "Warp-level primitives and synchronization",
        "3D FFT implementation for volumetric data",
        "N-body gravitational simulation with optimizations"};

    QStringList categories = {
        "Basic", "Basic", "Basic", "Basic", "Basic",
        "Advanced", "Advanced", "Advanced", "Advanced", "Advanced", "Advanced"}; // Map display names to actual executable names
    // These correspond to executables in build/bin/
    QMap<QString, QString> executableMap;
    executableMap["Vector Addition"] = "vector_addition";
    executableMap["Matrix Multiplication"] = "matrix_multiplication";
    executableMap["Parallel Reduction"] = "parallel_reduction";
    executableMap["2D Convolution"] = "convolution_2d";
    executableMap["Monte Carlo"] = "monte_carlo";
    executableMap["Advanced FFT"] = "advanced_fft";
    // executableMap["Advanced Threading"] = "advanced_threading";  // DISABLED: System crash
    executableMap["Dynamic Memory"] = "dynamic_memory";
    // executableMap["Warp Primitives"] = "warp_primitives";  // NOT BUILT: No executable
    executableMap["3D FFT"] = "advanced_fft";  // FIXED: Maps to existing advanced_fft executable
    executableMap["N-Body Simulation"] = "nbody_simulation";

    // Update: Most kernels are now built and available

    for (int i = 0; i < kernelNames.size(); ++i)
    {
        KernelInfo info;
        info.name = kernelNames[i];
        info.description = descriptions[i];
        info.category = categories[i];

        // Set executable name from mapping (empty if not available)
        info.executable = executableMap.value(kernelNames[i], "");

        // No hard-coded parameters - they're handled dynamically in runKernel()
        // info.parameters is now empty and will be set at runtime based on UI values

        m_kernels[info.name] = info;

        // Add to list with category prefix, and note if not available
        QString displayName = QString("[%1] %2").arg(info.category, info.name);
        if (info.executable.isEmpty())
        {
            displayName += " (Not Built)";
        }
        QListWidgetItem *item = new QListWidgetItem(displayName);
        item->setData(Qt::UserRole, info.name);
        m_kernelList->addItem(item);
    }

    // Select first kernel
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

    // Get executable path
    QString executable = getKernelExecutable(kernelName);
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

    // Create process
    m_currentProcess = new QProcess(this);

    // Set up process connections
    connect(m_currentProcess, QOverload<int, QProcess::ExitStatus>::of(&QProcess::finished),
            this, &KernelRunner::onProcessFinished);
    connect(m_currentProcess, &QProcess::errorOccurred,
            this, &KernelRunner::onProcessError);
    connect(m_currentProcess, &QProcess::readyReadStandardOutput,
            this, &KernelRunner::onProcessOutput);
    connect(m_currentProcess, &QProcess::readyReadStandardError,
            this, &KernelRunner::onProcessOutput);

    // Prepare arguments based on kernel type
    QStringList arguments;
    KernelInfo &info = m_kernels[kernelName];

    // Check if this kernel uses the new simplified argument format
    if (kernelName == "Vector Addition" ||
        kernelName == "Matrix Multiplication" ||
        kernelName == "Parallel Reduction" ||
        kernelName == "2D Convolution" ||
        kernelName == "Monte Carlo" ||
        // kernelName == "Advanced Threading" ||  // DISABLED: Causes system crash
        kernelName == "Warp Primitives" ||
        kernelName == "Advanced FFT" ||
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

QString KernelRunner::getKernelExecutable(const QString &kernelName)
{
    // Map kernel names to actual executable names
    // These correspond to executables in build/bin/
    QMap<QString, QString> executableMap;
    executableMap["Vector Addition"] = "vector_addition";
    executableMap["Matrix Multiplication"] = "matrix_multiplication";
    executableMap["Parallel Reduction"] = "parallel_reduction";
    executableMap["2D Convolution"] = "convolution_2d";
    executableMap["Monte Carlo"] = "monte_carlo";
    executableMap["Advanced FFT"] = "advanced_fft";
    // executableMap["Advanced Threading"] = "advanced_threading";  // DISABLED: System crash
    executableMap["Dynamic Memory"] = "dynamic_memory";
    // executableMap["Warp Primitives"] = "warp_primitives";  // NOT BUILT: No executable
    executableMap["3D FFT"] = "advanced_fft";  // FIXED: Maps to existing advanced_fft executable
    executableMap["N-Body Simulation"] = "nbody_simulation";

    QString executableName = executableMap.value(kernelName);
    if (executableName.isEmpty())
    {
        return QString();
    }

    // Look in the same directory as the GUI executable (build/bin)
    // and also check relative paths from there
    QStringList searchPaths = {
        QApplication::applicationDirPath() + "/" + executableName, // Same directory as GUI
        QApplication::applicationDirPath() + "/../bin/" + executableName,
        QApplication::applicationDirPath() + "/../build/bin/" + executableName,
        QApplication::applicationDirPath() + "/../../build/bin/" + executableName,
        QApplication::applicationDirPath() + "/../../../build/bin/" + executableName, // For build_gui/bin -> build/bin
        "./build/bin/" + executableName}; // Absolute fallback path

    for (const QString &executable : searchPaths)
    {
        QFileInfo fileInfo(executable);
        if (fileInfo.exists() && fileInfo.isExecutable())
        {
            return executable;
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

    QString params = tr("Parameters: --iterations <count> --size <elements> --platform <cuda|hip>");
    if (!info.parameters.isEmpty())
    {
        params += "\n" + tr("Additional: %1").arg(info.parameters.join(" "));
    }
    m_kernelParametersLabel->setText(params);
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

        // Enable/disable run button based on whether executable exists
        bool hasExecutable = !m_kernels[kernelName].executable.isEmpty();
        m_runButton->setEnabled(hasExecutable && !m_isRunning);

        if (!hasExecutable)
        {
            m_statusLabel->setText(tr("Kernel not built - executable not available"));
        }
        else
        {
            m_statusLabel->setText(tr("Ready"));
        }
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
    QString result = m_currentProcess->readAllStandardOutput() +
                     m_currentProcess->readAllStandardError();

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

    // Clean up process
    m_currentProcess->deleteLater();
    m_currentProcess = nullptr;

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

    if (m_currentProcess)
    {
        m_currentProcess->deleteLater();
        m_currentProcess = nullptr;
    }

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
        m_outputText->append(tr("<span style='color: red;'>%1</span>").arg(error));
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
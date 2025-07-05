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
        "Advanced", "Advanced", "Advanced", "Advanced", "Advanced", "Advanced"};

    for (int i = 0; i < kernelNames.size(); ++i)
    {
        KernelInfo info;
        info.name = kernelNames[i];
        info.description = descriptions[i];
        info.category = categories[i];

        // Set executable name based on platform
        QString baseName = kernelNames[i].toLower().replace(" ", "_");
        info.executable = QString("example_%1").arg(baseName);

        // Add parameters
        info.parameters << "--iterations" << "10";
        info.parameters << "--size" << "10000";

        m_kernels[info.name] = info;

        // Add to list with category prefix
        QString displayName = QString("[%1] %2").arg(info.category, info.name);
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

    // Prepare arguments
    QStringList arguments;
    arguments << "--iterations" << QString::number(m_iterationsSpinBox->value());
    arguments << "--size" << QString::number(m_dataSizeSpinBox->value());
    arguments << "--platform" << m_platformComboBox->currentText();

    // Add kernel-specific parameters
    KernelInfo &info = m_kernels[kernelName];
    arguments << info.parameters;

    // Start process
    m_outputText->append(tr("Starting %1...\n").arg(executable));
    m_outputText->append(tr("Arguments: %1\n\n").arg(arguments.join(" ")));

    m_currentProcess->start(executable, arguments);
}

QString KernelRunner::getKernelExecutable(const QString &kernelName)
{
    // Map kernel names to actual executable names
    QMap<QString, QString> executableMap;
    executableMap["Vector Addition"] = "01_vector_addition_hip";
    executableMap["Matrix Multiplication"] = "02_matrix_multiplication_hip";
    executableMap["Parallel Reduction"] = "03_parallel_reduction_hip";
    executableMap["2D Convolution"] = "04_convolution_2d_hip";
    executableMap["Monte Carlo"] = "05_monte_carlo_hip";
    executableMap["Advanced FFT"] = "06_advanced_fft_hip";
    executableMap["Advanced Threading"] = "07_advanced_threading_hip";
    executableMap["Dynamic Memory"] = "08_dynamic_memory_hip";
    executableMap["Warp Primitives"] = "09_warp_primitives_simplified_hip";
    executableMap["3D FFT"] = "10_advanced_fft_hip";
    executableMap["N-Body Simulation"] = "11_nbody_simulation_hip";

    QString executableName = executableMap.value(kernelName);
    if (executableName.isEmpty())
    {
        return QString();
    }

    // Look in multiple possible build directories
    QStringList buildDirs = {
        QApplication::applicationDirPath() + "/../build_hip",
        QApplication::applicationDirPath() + "/build_hip",
        QApplication::applicationDirPath() + "/../build",
        QApplication::applicationDirPath() + "/build"};

    for (const QString &buildDir : buildDirs)
    {
        QString executable = QString("%1/%2").arg(buildDir, executableName);
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
    // Parse performance metrics from output
    QTextStream stream(const_cast<QString *>(&output), QIODevice::ReadOnly);
    QString line;

    while (stream.readLineInto(&line))
    {
        if (line.contains("Time:", Qt::CaseInsensitive))
        {
            // Extract timing information
            m_outputText->append(tr("<b>%1</b>").arg(line));
        }
        else if (line.contains("Error:", Qt::CaseInsensitive))
        {
            // Highlight errors
            m_outputText->append(tr("<span style='color: red;'>%1</span>").arg(line));
        }
        else if (line.contains("Success:", Qt::CaseInsensitive))
        {
            // Highlight success messages
            m_outputText->append(tr("<span style='color: green;'>%1</span>").arg(line));
        }
        else
        {
            m_outputText->append(line);
        }
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
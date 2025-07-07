#include "example_tabs.h"
#include <QFile>
#include <QFileInfo>
#include <QDir>
#include <QMessageBox>
#include <QApplication>
#include <QTextStream>
#include <QScrollBar>
#include <QSyntaxHighlighter>
#include <QTextCharFormat>
#include <QFont>
#include <QRegularExpression>
#include <QDialog>
#include <QClipboard>

// C++/CUDA Syntax Highlighter
class CppSyntaxHighlighter : public QSyntaxHighlighter
{
    Q_OBJECT

public:
    explicit CppSyntaxHighlighter(QTextDocument *parent = nullptr);

protected:
    void highlightBlock(const QString &text) override;

private:
    struct HighlightingRule
    {
        QRegularExpression pattern;
        QTextCharFormat format;
    };
    QVector<HighlightingRule> highlightingRules;

    QRegularExpression commentStartExpression;
    QRegularExpression commentEndExpression;

    QTextCharFormat xmlElementFormat;
    QTextCharFormat xmlCommentFormat;
    QTextCharFormat xmlValueFormat;
    QTextCharFormat xmlCommentEndFormat;
};

CppSyntaxHighlighter::CppSyntaxHighlighter(QTextDocument *parent)
    : QSyntaxHighlighter(parent)
{
    HighlightingRule rule;

    // Keyword format
    QTextCharFormat keywordFormat;
    keywordFormat.setColor(QColor(86, 156, 214)); // Blue
    keywordFormat.setFontWeight(QFont::Bold);
    QStringList keywordPatterns;
    keywordPatterns << "\\bauto\\b" << "\\bbool\\b" << "\\bbreak\\b" << "\\bcase\\b"
                    << "\\bchar\\b" << "\\bclass\\b" << "\\bconst\\b" << "\\bcontinue\\b"
                    << "\\bdefault\\b" << "\\bdo\\b" << "\\bdouble\\b" << "\\belse\\b"
                    << "\\benum\\b" << "\\bexplicit\\b" << "\\bextern\\b" << "\\bfalse\\b"
                    << "\\bfloat\\b" << "\\bfor\\b" << "\\bfriend\\b" << "\\bif\\b"
                    << "\\binline\\b" << "\\bint\\b" << "\\blong\\b" << "\\bnamespace\\b"
                    << "\\bnew\\b" << "\\boperator\\b" << "\\bprivate\\b" << "\\bprotected\\b"
                    << "\\bpublic\\b" << "\\breturn\\b" << "\\bshort\\b" << "\\bsigned\\b"
                    << "\\bsizeof\\b" << "\\bstatic\\b" << "\\bstruct\\b" << "\\bswitch\\b"
                    << "\\btemplate\\b" << "\\bthis\\b" << "\\bthrow\\b" << "\\btrue\\b"
                    << "\\btry\\b" << "\\btypedef\\b" << "\\btypename\\b" << "\\bunion\\b"
                    << "\\bunsigned\\b" << "\\bvirtual\\b" << "\\bvoid\\b" << "\\bvolatile\\b"
                    << "\\bwhile\\b";

    // CUDA/HIP specific keywords
    keywordPatterns << "\\b__global__\\b" << "\\b__device__\\b" << "\\b__host__\\b"
                    << "\\b__shared__\\b" << "\\b__constant__\\b" << "\\b__managed__\\b"
                    << "\\b__restrict__\\b" << "\\bthreadIdx\\b" << "\\bblockIdx\\b"
                    << "\\bblockDim\\b" << "\\bgridDim\\b" << "\\b__syncthreads\\b"
                    << "\\b__syncwarp\\b" << "\\bhipMalloc\\b" << "\\bhipMemcpy\\b"
                    << "\\bhipFree\\b" << "\\bcudaMalloc\\b" << "\\bcudaMemcpy\\b"
                    << "\\bcudaFree\\b" << "\\b__shfl_down\\b" << "\\b__shfl_up\\b"
                    << "\\b__ballot\\b" << "\\b__any\\b" << "\\b__all\\b";

    foreach (const QString &pattern, keywordPatterns) {
        rule.pattern = QRegularExpression(pattern);
        rule.format = keywordFormat;
        highlightingRules.append(rule);
    }

    // Class format
    QTextCharFormat classFormat;
    classFormat.setFontWeight(QFont::Bold);
    classFormat.setColor(QColor(78, 201, 176)); // Teal
    rule.pattern = QRegularExpression(QStringLiteral("\\bQ[A-Za-z]+\\b"));
    rule.format = classFormat;
    highlightingRules.append(rule);

    // Function format
    QTextCharFormat functionFormat;
    functionFormat.setColor(QColor(220, 220, 170)); // Light yellow
    rule.pattern = QRegularExpression(QStringLiteral("\\b[A-Za-z0-9_]+(?=\\()"));
    rule.format = functionFormat;
    highlightingRules.append(rule);

    // String format
    QTextCharFormat quotationFormat;
    quotationFormat.setColor(QColor(206, 145, 120)); // Orange/brown
    rule.pattern = QRegularExpression(QStringLiteral("\".*\""));
    rule.format = quotationFormat;
    highlightingRules.append(rule);

    // Single line comment format
    QTextCharFormat singleLineCommentFormat;
    singleLineCommentFormat.setColor(QColor(106, 153, 85)); // Green
    rule.pattern = QRegularExpression(QStringLiteral("//[^\n]*"));
    rule.format = singleLineCommentFormat;
    highlightingRules.append(rule);

    // Multi-line comment format
    xmlCommentFormat.setColor(QColor(106, 153, 85)); // Green
    commentStartExpression = QRegularExpression(QStringLiteral("/\\*"));
    commentEndExpression = QRegularExpression(QStringLiteral("\\*/"));

    // Preprocessor format
    QTextCharFormat preprocessorFormat;
    preprocessorFormat.setColor(QColor(155, 155, 155)); // Gray
    rule.pattern = QRegularExpression(QStringLiteral("^\\s*#[^\n]*"));
    rule.format = preprocessorFormat;
    highlightingRules.append(rule);

    // Number format
    QTextCharFormat numberFormat;
    numberFormat.setColor(QColor(181, 206, 168)); // Light green
    rule.pattern = QRegularExpression(QStringLiteral("\\b[0-9]+\\.?[0-9]*[fF]?\\b"));
    rule.format = numberFormat;
    highlightingRules.append(rule);
}

void CppSyntaxHighlighter::highlightBlock(const QString &text)
{
    foreach (const HighlightingRule &rule, highlightingRules) {
        QRegularExpressionMatchIterator matchIterator = rule.pattern.globalMatch(text);
        while (matchIterator.hasNext()) {
            QRegularExpressionMatch match = matchIterator.next();
            setFormat(match.capturedStart(), match.capturedLength(), rule.format);
        }
    }

    // Handle multi-line comments
    setCurrentBlockState(0);

    QRegularExpressionMatch startMatch = commentStartExpression.match(text);
    int commentLength = 0;

    if (previousBlockState() != 1)
        commentLength = startMatch.capturedStart();
    else
        commentLength = 0;

    while (commentLength >= 0) {
        QRegularExpressionMatch endMatch = commentEndExpression.match(text, commentLength);
        int endIndex = endMatch.capturedStart();
        if (endIndex == -1) {
            setCurrentBlockState(1);
            commentLength = text.length() - commentLength;
        } else {
            commentLength = endIndex - commentLength + endMatch.capturedLength();
        }
        setFormat(commentLength, commentLength, xmlCommentFormat);
        commentLength = commentStartExpression.match(text, commentLength + commentLength).capturedStart();
    }
}

ExampleTab::ExampleTab(const QString &name, const QString &description, QWidget *parent)
    : QWidget(parent), m_name(name), m_description(description), m_process(nullptr), m_isRunning(false)
{
    setupUI();
}

ExampleTab::~ExampleTab()
{
    if (m_process && m_process->state() == QProcess::Running)
    {
        m_process->terminate();
        m_process->waitForFinished(5000);
    }
}

void ExampleTab::setupUI()
{
    QVBoxLayout *mainLayout = new QVBoxLayout(this);

    // Description section
    QGroupBox *descGroup = new QGroupBox(tr("Description"));
    QVBoxLayout *descLayout = new QVBoxLayout(descGroup);

    m_descriptionText = new QTextEdit();
    m_descriptionText->setReadOnly(true);
    m_descriptionText->setMaximumHeight(150);
    m_descriptionText->setHtml(m_description);
    descLayout->addWidget(m_descriptionText);

    mainLayout->addWidget(descGroup);

    // Control buttons
    QHBoxLayout *buttonLayout = new QHBoxLayout();
    m_runButton = new QPushButton(tr("Run Example"));
    m_sourceButton = new QPushButton(tr("View Source"));
    m_docButton = new QPushButton(tr("Documentation"));

    buttonLayout->addWidget(m_runButton);
    buttonLayout->addWidget(m_sourceButton);
    buttonLayout->addWidget(m_docButton);
    buttonLayout->addStretch();

    mainLayout->addLayout(buttonLayout);

    // Output section
    QGroupBox *outputGroup = new QGroupBox(tr("Output"));
    QVBoxLayout *outputLayout = new QVBoxLayout(outputGroup);

    m_outputText = new QTextEdit();
    m_outputText->setReadOnly(true);
    m_outputText->setMaximumHeight(200);
    outputLayout->addWidget(m_outputText);

    m_statusLabel = new QLabel(tr("Ready"));
    outputLayout->addWidget(m_statusLabel);

    mainLayout->addWidget(outputGroup);

    // Connect signals
    connect(m_runButton, &QPushButton::clicked, this, &ExampleTab::runExample);
    connect(m_sourceButton, &QPushButton::clicked, this, &ExampleTab::showSourceCode);
    connect(m_docButton, &QPushButton::clicked, this, &ExampleTab::showDocumentation);
}

void ExampleTab::setExecutablePath(const QString &path)
{
    m_executablePath = path;
}

void ExampleTab::setSourceCode(const QString &code)
{
    m_sourceCode = code;
}

void ExampleTab::setParameters(const QStringList &params)
{
    m_parameters = params;
}

void ExampleTab::runExample()
{
    if (m_isRunning)
    {
        QMessageBox::warning(this, tr("Example Running"),
                             tr("Example is already running. Please wait for it to complete."));
        return;
    }

    if (m_executablePath.isEmpty())
    {
        QMessageBox::critical(this, tr("Executable Not Found"),
                              tr("Could not find executable for example '%1'").arg(m_name));
        return;
    }

    QFileInfo fileInfo(m_executablePath);
    if (!fileInfo.exists() || !fileInfo.isExecutable())
    {
        QMessageBox::critical(this, tr("Executable Not Found"),
                              tr("Executable not found or not executable: %1").arg(m_executablePath));
        return;
    }

    m_isRunning = true;
    m_runButton->setEnabled(false);
    m_statusLabel->setText(tr("Running..."));
    m_outputText->clear();

    // Create process
    m_process = new QProcess(this);

    // Set up process connections
    connect(m_process, QOverload<int, QProcess::ExitStatus>::of(&QProcess::finished),
            this, &ExampleTab::onProcessFinished);
    connect(m_process, &QProcess::errorOccurred,
            this, &ExampleTab::onProcessError);
    connect(m_process, &QProcess::readyReadStandardOutput,
            this, &ExampleTab::onProcessOutput);
    connect(m_process, &QProcess::readyReadStandardError,
            this, &ExampleTab::onProcessOutput);

    // Start process
    updateOutput(tr("Starting %1...\n").arg(fileInfo.fileName()));
    m_process->start(m_executablePath, m_parameters);
}

void ExampleTab::showSourceCode()
{
    if (m_sourceCode.isEmpty())
    {
        QMessageBox::information(this, tr("No Source Code"),
                                 tr("Source code not available for this example."));
        return;
    }

    QDialog *dialog = new QDialog(this);
    dialog->setWindowTitle(tr("Source Code - %1").arg(m_name));
    dialog->resize(1000, 700);

    QVBoxLayout *layout = new QVBoxLayout(dialog);

    QTextEdit *sourceView = new QTextEdit();
    sourceView->setReadOnly(true);
    sourceView->setPlainText(m_sourceCode);
    
    // Set font and styling
    QFont font("Consolas", 11);
    if (!font.exactMatch()) {
        font.setFamily("Courier New");
    }
    sourceView->setFont(font);
    
    // Apply dark theme styling for better code visibility
    sourceView->setStyleSheet(
        "QTextEdit {"
        "    background-color: #1e1e1e;"
        "    color: #d4d4d4;"
        "    border: 1px solid #464647;"
        "    selection-background-color: #264f78;"
        "}"
    );

    // Apply syntax highlighting
    CppSyntaxHighlighter *highlighter = new CppSyntaxHighlighter(sourceView->document());

    layout->addWidget(sourceView);

    // Add button layout
    QHBoxLayout *buttonLayout = new QHBoxLayout();
    QPushButton *copyButton = new QPushButton(tr("Copy to Clipboard"));
    QPushButton *closeButton = new QPushButton(tr("Close"));
    
    buttonLayout->addWidget(copyButton);
    buttonLayout->addStretch();
    buttonLayout->addWidget(closeButton);
    
    layout->addLayout(buttonLayout);

    // Connect signals
    connect(copyButton, &QPushButton::clicked, [sourceView]() {
        QApplication::clipboard()->setText(sourceView->toPlainText());
    });
    connect(closeButton, &QPushButton::clicked, dialog, &QDialog::accept);

    dialog->exec();
    dialog->deleteLater();
}

void ExampleTab::showDocumentation()
{
    QString docText = tr("<h2>%1</h2>").arg(m_name);
    docText += m_description;
    docText += tr("<h3>Usage</h3>");
    docText += tr("<p>Click 'Run Example' to execute this kernel.</p>");
    docText += tr("<p>Click 'View Source' to see the implementation.</p>");

    QDialog *dialog = new QDialog(this);
    dialog->setWindowTitle(tr("Documentation - %1").arg(m_name));
    dialog->resize(600, 400);

    QVBoxLayout *layout = new QVBoxLayout(dialog);

    QTextEdit *docView = new QTextEdit();
    docView->setReadOnly(true);
    docView->setHtml(docText);

    layout->addWidget(docView);

    QPushButton *closeButton = new QPushButton(tr("Close"));
    connect(closeButton, &QPushButton::clicked, dialog, &QDialog::accept);
    layout->addWidget(closeButton);

    dialog->exec();
    dialog->deleteLater();
}

void ExampleTab::onProcessFinished(int exitCode, QProcess::ExitStatus exitStatus)
{
    m_isRunning = false;
    m_runButton->setEnabled(true);

    bool success = (exitCode == 0 && exitStatus == QProcess::NormalExit);
    QString result = m_process->readAllStandardOutput() +
                     m_process->readAllStandardError();

    if (success)
    {
        m_statusLabel->setText(tr("Completed successfully"));
        updateOutput(tr("\n<b>Example completed successfully!</b>"));
    }
    else
    {
        m_statusLabel->setText(tr("Failed"));
        updateOutput(tr("\n<b>Example failed with exit code %1</b>").arg(exitCode));
    }

    // Clean up process
    m_process->deleteLater();
    m_process = nullptr;

    // Emit signal
    emit exampleFinished(m_name, success, result);
}

void ExampleTab::onProcessError(QProcess::ProcessError error)
{
    m_isRunning = false;
    m_runButton->setEnabled(true);

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
    updateOutput(tr("\n<b>Error: %1</b>").arg(errorMsg), true);

    if (m_process)
    {
        m_process->deleteLater();
        m_process = nullptr;
    }

    emit exampleFinished(m_name, false, errorMsg);
}

void ExampleTab::onProcessOutput()
{
    if (!m_process)
        return;

    QString output = m_process->readAllStandardOutput();
    QString error = m_process->readAllStandardError();

    if (!output.isEmpty())
    {
        updateOutput(output);
    }

    if (!error.isEmpty())
    {
        updateOutput(error, true);
    }
}

void ExampleTab::updateOutput(const QString &text, bool isError)
{
    if (isError)
    {
        m_outputText->append(tr("<span style='color: red;'>%1</span>").arg(text));
    }
    else
    {
        m_outputText->append(text);
    }

    // Auto-scroll to bottom
    QScrollBar *scrollBar = m_outputText->verticalScrollBar();
    scrollBar->setValue(scrollBar->maximum());
}

// ExampleTabs implementation
ExampleTabs::ExampleTabs(QWidget *parent)
    : QWidget(parent)
{
    setupUI();
    loadExamples();
}

void ExampleTabs::setupUI()
{
    QVBoxLayout *mainLayout = new QVBoxLayout(this);

    m_tabWidget = new QTabWidget();
    mainLayout->addWidget(m_tabWidget);
}

void ExampleTabs::loadExamples()
{
    // Define all examples with detailed descriptions
    struct ExampleInfo
    {
        QString name;
        QString description;
        QString category;
        QString sourceFile;
    };

    QList<ExampleInfo> examples = {
        {"Vector Addition",
         R"(
<h3>Vector Addition Kernel</h3>
<p>This example demonstrates the fundamental concept of parallel vector addition on GPU. 
It shows how to:</p>
<ul>
<li>Allocate memory on both CPU and GPU</li>
<li>Transfer data between CPU and GPU</li>
<li>Launch a simple kernel with proper grid and block dimensions</li>
<li>Verify results and measure performance</li>
</ul>

<h4>Key Concepts:</h4>
<ul>
<li><b>Memory Management:</b> cudaMalloc, cudaMemcpy, cudaFree</li>
<li><b>Kernel Launch:</b> <<<grid, block>>> syntax</li>
<li><b>Thread Indexing:</b> threadIdx, blockIdx, blockDim</li>
<li><b>Synchronization:</b> cudaDeviceSynchronize</li>
</ul>

<h4>Performance Considerations:</h4>
<ul>
<li>Memory coalescing for optimal bandwidth</li>
<li>Proper grid and block sizing</li>
<li>Minimizing memory transfers</li>
</ul>
            )",
         "Basic",
         "src/01_vector_addition/vector_addition.cu"},
        {"Matrix Multiplication",
         R"(
<h3>Matrix Multiplication Kernel</h3>
<p>This example implements matrix multiplication using shared memory optimization. 
It demonstrates advanced GPU programming techniques:</p>
<ul>
<li>Shared memory usage for data reuse</li>
<li>2D thread block organization</li>
<li>Memory access pattern optimization</li>
<li>Performance comparison with CPU implementation</li>
</ul>

<h4>Key Concepts:</h4>
<ul>
<li><b>Shared Memory:</b> __shared__ keyword and synchronization</li>
<li><b>2D Threading:</b> 2D grid and block dimensions</li>
<li><b>Memory Tiling:</b> Breaking matrices into tiles</li>
<li><b>Bank Conflicts:</b> Avoiding shared memory bank conflicts</li>
</ul>

<h4>Optimization Techniques:</h4>
<ul>
<li>Tile-based computation for cache efficiency</li>
<li>Shared memory for data reuse</li>
<li>Memory coalescing for global memory access</li>
<li>Loop unrolling for instruction-level parallelism</li>
</ul>
            )",
         "Basic",
         "src/02_matrix_multiplication/matrix_mul.cu"},
        {"Parallel Reduction",
         R"(
<h3>Parallel Reduction Kernel</h3>
<p>This example implements parallel reduction algorithms for finding sum, maximum, 
or minimum values in an array. It shows efficient tree-based reduction:</p>
<ul>
<li>Multiple reduction algorithms (tree, warp-level, block-level)</li>
<li>Atomic operations for inter-block communication</li>
<li>Warp-level primitives for efficiency</li>
<li>Performance analysis of different approaches</li>
</ul>

<h4>Key Concepts:</h4>
<ul>
<li><b>Tree Reduction:</b> Log(n) steps for n elements</li>
<li><b>Warp Primitives:</b> __shfl_down, __reduce_add</li>
<li><b>Atomic Operations:</b> atomicAdd, atomicMax</li>
<li><b>Block-level Reduction:</b> Shared memory for intra-block reduction</li>
</ul>

<h4>Algorithm Variants:</h4>
<ul>
<li>Sequential addressing (no bank conflicts)</li>
<li>Interleaved addressing (potential bank conflicts)</li>
<li>Warp-level primitives (most efficient)</li>
<li>Multi-block reduction with atomic operations</li>
</ul>
            )",
         "Basic",
         "src/03_parallel_reduction/reduction.cu"},
        {"2D Convolution",
         R"(
<h3>2D Convolution Kernel</h3>
<p>This example implements 2D convolution with configurable kernel sizes. 
It demonstrates image processing on GPU:</p>
<ul>
<li>2D data layout and memory access patterns</li>
<li>Handling boundary conditions</li>
<li>Shared memory for kernel coefficients</li>
<li>Performance optimization for different kernel sizes</li>
</ul>

<h4>Key Concepts:</h4>
<ul>
<li><b>2D Memory Layout:</b> Row-major vs column-major access</li>
<li><b>Boundary Handling:</b> Clamp, wrap, or zero-padding</li>
<li><b>Shared Memory Usage:</b> Loading kernel coefficients</li>
<li><b>Memory Coalescing:</b> Optimal 2D memory access</li>
</ul>

<h4>Optimization Strategies:</h4>
<ul>
<li>Shared memory for frequently accessed data</li>
<li>Memory coalescing for global memory access</li>
<li>Loop unrolling for small kernel sizes</li>
<li>Specialized kernels for common kernel sizes</li>
</ul>
            )",
         "Basic",
         "src/04_convolution_2d/convolution.cu"},
        {"Monte Carlo",
         R"(
<h3>Monte Carlo Simulation</h3>
<p>This example implements Monte Carlo methods for numerical integration 
and random number generation on GPU:</p>
<ul>
<li>Random number generation using cuRAND</li>
<li>Monte Carlo integration for π calculation</li>
<li>Statistical analysis of results</li>
<li>Performance comparison with CPU implementation</li>
</ul>

<h4>Key Concepts:</h4>
<ul>
<li><b>Random Number Generation:</b> cuRAND library usage</li>
<li><b>Monte Carlo Methods:</b> Statistical sampling</li>
<li><b>Reduction:</b> Combining results from multiple threads</li>
<li><b>Precision:</b> Handling floating-point arithmetic</li>
</ul>

<h4>Applications:</h4>
<ul>
<li>Numerical integration</li>
<li>Option pricing in finance</li>
<li>Particle transport simulation</li>
<li>Statistical sampling</li>
</ul>
            )",
         "Basic",
         "src/05_monte_carlo/monte_carlo.cu"},
        {"Advanced FFT",
         R"(
<h3>Advanced FFT Implementation</h3>
<p>This example demonstrates advanced FFT implementation with multiple 
optimization techniques:</p>
<ul>
<li>Cooley-Tukey FFT algorithm implementation</li>
<li>Shared memory optimization</li>
<li>Multi-GPU FFT computation</li>
<li>Performance analysis and comparison</li>
</ul>

<h4>Key Concepts:</h4>
<ul>
<li><b>FFT Algorithm:</b> Cooley-Tukey butterfly operations</li>
<li><b>Twiddle Factors:</b> Pre-computed trigonometric values</li>
<li><b>Shared Memory:</b> Staging data for faster access</li>
<li><b>Multi-GPU:</b> Distributed FFT computation</li>
</ul>

<h4>Optimization Techniques:</h4>
<ul>
<li>Shared memory for intermediate results</li>
<li>Pre-computed twiddle factors</li>
<li>Memory coalescing for data access</li>
<li>Multi-GPU load balancing</li>
</ul>
            )",
         "Advanced",
         "src/06_advanced_fft/fft_kernels.hip"},
        {"Advanced Threading",
         R"(
<h3>Advanced Threading Patterns</h3>
<p>This example demonstrates advanced thread cooperation patterns 
and synchronization techniques:</p>
<ul>
<li>Producer-consumer patterns</li>
<li>Barrier synchronization</li>
<li>Dynamic parallelism</li>
<li>Warp-level programming</li>
</ul>

<h4>Key Concepts:</h4>
<ul>
<li><b>Thread Cooperation:</b> Shared memory and synchronization</li>
<li><b>Barriers:</b> __syncthreads() and cooperative groups</li>
<li><b>Dynamic Parallelism:</b> Kernel launching from device</li>
<li><b>Warp-level Primitives:</b> __shfl, __ballot, __any</li>
</ul>

<h4>Patterns Demonstrated:</h4>
<ul>
<li>Producer-consumer with shared memory</li>
<li>Multi-stage pipeline processing</li>
<li>Warp-level reduction and scan</li>
<li>Dynamic kernel launching</li>
</ul>
            )",
         "Advanced",
         "src/07_advanced_threading/advanced_threading_hip.hip"},
        {"Dynamic Memory",
         R"(
<h3>Dynamic Memory Management</h3>
<p>This example demonstrates dynamic memory allocation and management 
on GPU using HIP/CUDA:</p>
<ul>
<li>Dynamic memory allocation on device</li>
<li>Memory pools and caching</li>
<li>Fragmentation handling</li>
<li>Performance implications</li>
</ul>

<h4>Key Concepts:</h4>
<ul>
<li><b>Device Memory:</b> malloc/free on GPU</li>
<li><b>Memory Pools:</b> Efficient allocation strategies</li>
<li><b>Fragmentation:</b> Handling memory fragmentation</li>
<li><b>Performance:</b> Allocation overhead considerations</li>
</ul>

<h4>Memory Management Strategies:</h4>
<ul>
<li>Pre-allocated memory pools</li>
<li>Memory caching for reuse</li>
<li>Fragmentation-aware allocation</li>
<li>Memory compaction techniques</li>
</ul>
            )",
         "Advanced",
         "src/08_dynamic_memory/dynamic_memory_hip.hip"},
        {"Warp Primitives",
         R"(
<h3>Warp-Level Primitives</h3>
<p>This example demonstrates advanced warp-level programming 
and synchronization primitives:</p>
<ul>
<li>Warp shuffle operations</li>
<li>Warp vote functions</li>
<li>Warp-level reduction and scan</li>
<li>Performance optimization techniques</li>
</ul>

<h4>Key Concepts:</h4>
<ul>
<li><b>Warp Shuffle:</b> __shfl, __shfl_up, __shfl_down</li>
<li><b>Warp Vote:</b> __all, __any, __ballot</li>
<li><b>Warp Reduction:</b> Efficient intra-warp operations</li>
<li><b>Warp Synchronization:</b> Implicit warp-level sync</li>
</ul>

<h4>Applications:</h4>
<ul>
<li>Warp-level parallel reduction</li>
<li>Efficient sorting within warps</li>
<li>Warp-level prefix sum (scan)</li>
<li>Conditional execution optimization</li>
</ul>
            )",
         "Advanced",
         "src/09_warp_primitives/warp_primitives_hip.hip"},
        {"3D FFT",
         R"(
<h3>3D FFT Implementation</h3>
<p>This example implements 3D Fast Fourier Transform for 
volumetric data processing:</p>
<ul>
<li>3D FFT algorithm implementation</li>
<li>Memory layout optimization</li>
<li>Multi-dimensional data handling</li>
<li>Performance analysis for 3D data</li>
</ul>

<h4>Key Concepts:</h4>
<ul>
<li><b>3D Data Layout:</b> Memory organization for 3D arrays</li>
<li><b>3D FFT:</b> Three-dimensional transform</li>
<li><b>Memory Access:</b> Optimizing 3D memory access patterns</li>
<li><b>Transpose Operations:</b> Data rearrangement for FFT</li>
</ul>

<h4>Optimization Techniques:</h4>
<ul>
<li>3D memory coalescing</li>
<li>Shared memory for 3D tiles</li>
<li>Efficient transpose operations</li>
<li>Multi-GPU 3D FFT</li>
</ul>
            )",
         "Advanced",
         "src/10_advanced_fft/fft3d_hip.hip"},
        {"N-Body Simulation",
         R"(
<h3>N-Body Gravitational Simulation</h3>
<p>This example implements N-body gravitational simulation 
with multiple optimization strategies:</p>
<ul>
<li>Barnes-Hut tree algorithm</li>
<li>Spatial partitioning</li>
<li>Force calculation optimization</li>
<li>Multi-GPU load balancing</li>
</ul>

<h4>Key Concepts:</h4>
<ul>
<li><b>N-Body Physics:</b> Gravitational force calculations</li>
<li><b>Barnes-Hut Algorithm:</b> O(n log n) complexity</li>
<li><b>Spatial Partitioning:</b> Octree data structure</li>
<li><b>Force Approximation:</b> Multipole expansion</li>
</ul>

<h4>Optimization Strategies:</h4>
<ul>
<li>Octree construction on GPU</li>
<li>Force calculation vectorization</li>
<li>Memory access optimization</li>
<li>Multi-GPU domain decomposition</li>
</ul>
            )",
         "Advanced",
         "src/11_nbody_simulation/nbody_hip.hip"}};

    // Create tabs for each example
    for (const ExampleInfo &info : examples)
    {
        createExampleTab(info.name, info.description, info.category, info.sourceFile);
    }
}

void ExampleTabs::createExampleTab(const QString &name, const QString &description,
                                   const QString &category, const QString &sourceFile)
{
    ExampleTab *tab = new ExampleTab(name, description);

    // Set executable path
    QString baseName = name.toLower().replace(" ", "_");
    QString executable = QString("build/bin/%1").arg(baseName);
    tab->setExecutablePath(executable);

    // Try to load source code
    QFile sourceFileObj(sourceFile);
    if (sourceFileObj.open(QIODevice::ReadOnly | QIODevice::Text))
    {
        QTextStream in(&sourceFileObj);
        QString sourceCode = in.readAll();
        tab->setSourceCode(sourceCode);
        sourceFileObj.close();
    }

    // Add to tab widget with category prefix
    QString tabName = QString("[%1] %2").arg(category, name);
    m_tabWidget->addTab(tab, tabName);

    // Store reference
    m_exampleTabs[name] = tab;

    // Connect signals
    connect(tab, &ExampleTab::exampleFinished,
            this, &ExampleTabs::exampleFinished);
}

void ExampleTabs::runExample(const QString &exampleName)
{
    if (m_exampleTabs.contains(exampleName))
    {
        m_exampleTabs[exampleName]->runExample();
    }
}

void ExampleTabs::showExampleSource(const QString &exampleName)
{
    if (m_exampleTabs.contains(exampleName))
    {
        m_exampleTabs[exampleName]->showSourceCode();
    }
}

// Include the moc file for the syntax highlighter
#include "example_tabs.moc"
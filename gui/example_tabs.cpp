#include "example_tabs.h"
#include "example_loader.h"
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
    keywordFormat.setForeground(QColor(86, 156, 214)); // Blue
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
    classFormat.setForeground(QColor(78, 201, 176)); // Teal
    rule.pattern = QRegularExpression(QStringLiteral("\\bQ[A-Za-z]+\\b"));
    rule.format = classFormat;
    highlightingRules.append(rule);

    // Function format
    QTextCharFormat functionFormat;
    functionFormat.setForeground(QColor(220, 220, 170)); // Light yellow
    rule.pattern = QRegularExpression(QStringLiteral("\\b[A-Za-z0-9_]+(?=\\()"));
    rule.format = functionFormat;
    highlightingRules.append(rule);

    // String format
    QTextCharFormat quotationFormat;
    quotationFormat.setForeground(QColor(206, 145, 120)); // Orange/brown
    rule.pattern = QRegularExpression(QStringLiteral("\".*\""));
    rule.format = quotationFormat;
    highlightingRules.append(rule);

    // Single line comment format
    QTextCharFormat singleLineCommentFormat;
    singleLineCommentFormat.setForeground(QColor(106, 153, 85)); // Green
    rule.pattern = QRegularExpression(QStringLiteral("//[^\n]*"));
    rule.format = singleLineCommentFormat;
    highlightingRules.append(rule);

    // Multi-line comment format
    xmlCommentFormat.setForeground(QColor(106, 153, 85)); // Green
    commentStartExpression = QRegularExpression(QStringLiteral("/\\*"));
    commentEndExpression = QRegularExpression(QStringLiteral("\\*/"));

    // Preprocessor format
    QTextCharFormat preprocessorFormat;
    preprocessorFormat.setForeground(QColor(155, 155, 155)); // Gray
    rule.pattern = QRegularExpression(QStringLiteral("^\\s*#[^\n]*"));
    rule.format = preprocessorFormat;
    highlightingRules.append(rule);

    // Number format
    QTextCharFormat numberFormat;
    numberFormat.setForeground(QColor(181, 206, 168)); // Light green
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
    
    // Set dark background for better readability
    m_outputText->setStyleSheet(
        "QTextEdit {"
        "    background-color: #2b2b2b;"
        "    color: #ffffff;"
        "    font-family: 'Consolas', 'Monaco', 'Courier New', monospace;"
        "    font-size: 11px;"
        "    border: 1px solid #555;"
        "    border-radius: 4px;"
        "    padding: 8px;"
        "}"
    );
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
    new CppSyntaxHighlighter(sourceView->document());

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
    // Load examples from XML files
    QList<ExampleInfo> examples = ExampleLoader::loadExamples();
    
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
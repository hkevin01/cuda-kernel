#include "mainwindow.h"
#include "ui_mainwindow.h"

#include <QApplication>
#include <QMenuBar>
#include <QStatusBar>
#include <QMessageBox>
#include <QToolBar>
#include <QAction>
#include <QSettings>
#include <QCloseEvent>
#include <QTimer>
#include <QSystemTrayIcon>
#include <QMenu>
#include <QIcon>
#include <QStyle>
#include <QDir>
#include <QStandardPaths>

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent), m_settings(new QSettings(this)), m_systemTrayEnabled(false), m_statusTimer(new QTimer(this))
{
    setWindowTitle(tr("GPU Kernel Examples"));
    setMinimumSize(800, 600);
    resize(1200, 800);

    createActions();
    createMenus();
    createToolBars();
    createStatusBar();
    createSystemTray();
    createTabs();
    setupConnections();
    loadSettings();

    // Set up status message timer
    m_statusTimer->setSingleShot(true);
    connect(m_statusTimer, &QTimer::timeout, [this]()
            { m_statusLabel->clear(); });

    updateWindowTitle();
}

MainWindow::~MainWindow()
{
    saveSettings();
}

void MainWindow::setPlatform(const QString &platform)
{
    m_currentPlatform = platform;
    m_platformLabel->setText(tr("Platform: %1").arg(platform));
    updateWindowTitle();
}

void MainWindow::closeEvent(QCloseEvent *event)
{
    if (m_systemTrayEnabled && m_systemTrayIcon->isVisible())
    {
        QMessageBox::information(this, tr("GPU Kernel Examples"),
                                 tr("The application will keep running in the system tray. "
                                    "To terminate the program, choose <b>Quit</b> in the context menu "
                                    "of the system tray entry."));
        hide();
        event->ignore();
    }
    else
    {
        event->accept();
    }
}

void MainWindow::createActions()
{
    // File actions
    m_newAct = new QAction(tr("&New"), this);
    m_newAct->setShortcuts(QKeySequence::New);
    m_newAct->setStatusTip(tr("Create a new file"));
    connect(m_newAct, &QAction::triggered, this, [this]()
            { showStatusMessage(tr("New file action triggered")); });

    m_openAct = new QAction(tr("&Open..."), this);
    m_openAct->setShortcuts(QKeySequence::Open);
    m_openAct->setStatusTip(tr("Open an existing file"));
    connect(m_openAct, &QAction::triggered, this, [this]()
            { showStatusMessage(tr("Open file action triggered")); });

    m_saveAct = new QAction(tr("&Save"), this);
    m_saveAct->setShortcuts(QKeySequence::Save);
    m_saveAct->setStatusTip(tr("Save the document to disk"));
    connect(m_saveAct, &QAction::triggered, this, [this]()
            { showStatusMessage(tr("Save action triggered")); });

    m_saveAsAct = new QAction(tr("Save &As..."), this);
    m_saveAsAct->setShortcuts(QKeySequence::SaveAs);
    m_saveAsAct->setStatusTip(tr("Save the document under a new name"));
    connect(m_saveAsAct, &QAction::triggered, this, [this]()
            { showStatusMessage(tr("Save As action triggered")); });

    m_exitAct = new QAction(tr("E&xit"), this);
    m_exitAct->setShortcuts(QKeySequence::Quit);
    m_exitAct->setStatusTip(tr("Exit the application"));
    connect(m_exitAct, &QAction::triggered, this, &QWidget::close);

    // Edit actions
    m_cutAct = new QAction(tr("Cu&t"), this);
    m_cutAct->setShortcuts(QKeySequence::Cut);
    m_cutAct->setStatusTip(tr("Cut the current selection's contents to the clipboard"));
    connect(m_cutAct, &QAction::triggered, this, [this]()
            { showStatusMessage(tr("Cut action triggered")); });

    m_copyAct = new QAction(tr("&Copy"), this);
    m_copyAct->setShortcuts(QKeySequence::Copy);
    m_copyAct->setStatusTip(tr("Copy the current selection's contents to the clipboard"));
    connect(m_copyAct, &QAction::triggered, this, [this]()
            { showStatusMessage(tr("Copy action triggered")); });

    m_pasteAct = new QAction(tr("&Paste"), this);
    m_pasteAct->setShortcuts(QKeySequence::Paste);
    m_pasteAct->setStatusTip(tr("Paste the clipboard's contents into the current selection"));
    connect(m_pasteAct, &QAction::triggered, this, [this]()
            { showStatusMessage(tr("Paste action triggered")); });

    // Kernel actions
    m_runKernelAct = new QAction(tr("&Run Kernel"), this);
    m_runKernelAct->setStatusTip(tr("Run selected kernel"));
    connect(m_runKernelAct, &QAction::triggered, this, [this]()
            { m_kernelRunner->runSelectedKernel(); });

    m_runTestsAct = new QAction(tr("Run &Tests"), this);
    m_runTestsAct->setStatusTip(tr("Run all tests"));
    connect(m_runTestsAct, &QAction::triggered, this, [this]()
            { m_testRunner->runAllTests(); });

    m_showPerformanceAct = new QAction(tr("&Performance"), this);
    m_showPerformanceAct->setStatusTip(tr("Show performance metrics"));
    connect(m_showPerformanceAct, &QAction::triggered, this, [this]()
            { m_tabWidget->setCurrentWidget(m_performanceWidget); });

    // Help actions
    m_aboutAct = new QAction(tr("&About"), this);
    m_aboutAct->setStatusTip(tr("Show the application's About box"));
    connect(m_aboutAct, &QAction::triggered, this, &MainWindow::about);

    m_aboutQtAct = new QAction(tr("About &Qt"), this);
    m_aboutQtAct->setStatusTip(tr("Show the Qt library's About box"));
    connect(m_aboutQtAct, &QAction::triggered, this, &MainWindow::aboutQt);

    // System tray action
    m_toggleSystemTrayAct = new QAction(tr("&System Tray"), this);
    m_toggleSystemTrayAct->setCheckable(true);
    m_toggleSystemTrayAct->setStatusTip(tr("Toggle system tray icon"));
    connect(m_toggleSystemTrayAct, &QAction::triggered, this, &MainWindow::toggleSystemTray);
}

void MainWindow::createMenus()
{
    m_fileMenu = menuBar()->addMenu(tr("&File"));
    m_fileMenu->addAction(m_newAct);
    m_fileMenu->addAction(m_openAct);
    m_fileMenu->addAction(m_saveAct);
    m_fileMenu->addAction(m_saveAsAct);
    m_fileMenu->addSeparator();
    m_fileMenu->addAction(m_exitAct);

    m_editMenu = menuBar()->addMenu(tr("&Edit"));
    m_editMenu->addAction(m_cutAct);
    m_editMenu->addAction(m_copyAct);
    m_editMenu->addAction(m_pasteAct);

    m_viewMenu = menuBar()->addMenu(tr("&View"));
    m_viewMenu->addAction(m_showPerformanceAct);
    m_viewMenu->addSeparator();
    m_viewMenu->addAction(m_toggleSystemTrayAct);

    menuBar()->addSeparator();

    m_helpMenu = menuBar()->addMenu(tr("&Help"));
    m_helpMenu->addAction(m_aboutAct);
    m_helpMenu->addAction(m_aboutQtAct);
}

void MainWindow::createToolBars()
{
    m_fileToolBar = addToolBar(tr("File"));
    m_fileToolBar->setObjectName("fileToolBar");
    m_fileToolBar->addAction(m_newAct);
    m_fileToolBar->addAction(m_openAct);
    m_fileToolBar->addAction(m_saveAct);
    m_fileToolBar->addSeparator();
    m_fileToolBar->addAction(m_cutAct);
    m_fileToolBar->addAction(m_copyAct);
    m_fileToolBar->addAction(m_pasteAct);

    m_kernelToolBar = addToolBar(tr("Kernel"));
    m_kernelToolBar->setObjectName("kernelToolBar");
    m_kernelToolBar->addAction(m_runKernelAct);
    m_kernelToolBar->addAction(m_runTestsAct);
    m_kernelToolBar->addSeparator();
    m_kernelToolBar->addAction(m_showPerformanceAct);
}

void MainWindow::createStatusBar()
{
    m_statusLabel = new QLabel(tr("Ready"));
    statusBar()->addWidget(m_statusLabel);

    m_progressBar = new QProgressBar();
    m_progressBar->setVisible(false);
    statusBar()->addPermanentWidget(m_progressBar);

    m_platformLabel = new QLabel(tr("Platform: Unknown"));
    statusBar()->addPermanentWidget(m_platformLabel);
}

void MainWindow::createSystemTray()
{
    m_systemTrayIcon = new QSystemTrayIcon(this);
    m_systemTrayIcon->setIcon(style()->standardIcon(QStyle::SP_ComputerIcon));

    m_systemTrayMenu = new QMenu(this);
    m_systemTrayMenu->addAction(tr("&Show"), this, &QWidget::show);
    m_systemTrayMenu->addAction(tr("&Hide"), this, &QWidget::hide);
    m_systemTrayMenu->addSeparator();
    m_systemTrayMenu->addAction(m_exitAct);

    m_systemTrayIcon->setContextMenu(m_systemTrayMenu);
    connect(m_systemTrayIcon, &QSystemTrayIcon::activated,
            this, &MainWindow::systemTrayActivated);
}

void MainWindow::createTabs()
{
    m_tabWidget = new QTabWidget(this);
    setCentralWidget(m_tabWidget);

    m_kernelRunner = new KernelRunner(this);
    m_resultViewer = new ResultViewer(this);
    m_performanceWidget = new PerformanceWidget(this);
    m_exampleTabs = new ExampleTabs(this);

    m_tabWidget->addTab(m_exampleTabs, tr("Examples"));
    m_tabWidget->addTab(m_kernelRunner, tr("Kernel Runner"));
    m_tabWidget->addTab(m_resultViewer, tr("Results"));
    m_tabWidget->addTab(m_performanceWidget, tr("Performance"));
}

void MainWindow::setupConnections()
{
    // Connect kernel runner signals
    connect(m_kernelRunner, &KernelRunner::kernelFinished,
            this, &MainWindow::onKernelFinished);
    connect(m_kernelRunner, &KernelRunner::progressUpdated,
            this, &MainWindow::updateProgress);
    connect(m_testRunner, &TestRunner::progressUpdated,
            this, &MainWindow::updateProgress);

    // Connect performance widget signals
    connect(m_performanceWidget, &PerformanceWidget::dataUpdated,
            this, &MainWindow::onPerformanceDataUpdated);

    // Connect example tabs signals
    connect(m_exampleTabs, &ExampleTabs::exampleFinished,
            this, &MainWindow::onKernelFinished);
}

void MainWindow::loadSettings()
{
    m_settings->beginGroup("MainWindow");
    restoreGeometry(m_settings->value("geometry").toByteArray());
    restoreState(m_settings->value("windowState").toByteArray());
    m_systemTrayEnabled = m_settings->value("systemTrayEnabled", false).toBool();
    m_currentPlatform = m_settings->value("platform", "HIP").toString();
    m_settings->endGroup();

    m_toggleSystemTrayAct->setChecked(m_systemTrayEnabled);
    if (m_systemTrayEnabled)
    {
        m_systemTrayIcon->show();
    }
}

void MainWindow::saveSettings()
{
    m_settings->beginGroup("MainWindow");
    m_settings->setValue("geometry", saveGeometry());
    m_settings->setValue("windowState", saveState());
    m_settings->setValue("systemTrayEnabled", m_systemTrayEnabled);
    m_settings->setValue("platform", m_currentPlatform);
    m_settings->endGroup();
}

void MainWindow::updateWindowTitle()
{
    QString title = tr("GPU Kernel Examples");
    if (!m_currentPlatform.isEmpty())
    {
        title += tr(" - %1").arg(m_currentPlatform);
    }
    setWindowTitle(title);
}

void MainWindow::about()
{
    QMessageBox::about(this, tr("About GPU Kernel Examples"),
                       tr("<b>GPU Kernel Examples</b> is a comprehensive collection of "
                          "CUDA and HIP kernel examples with testing and performance analysis tools.<br><br>"
                          "Version: 1.0.0<br>"
                          "Platform: %1<br>"
                          "Built with Qt %2")
                           .arg(m_currentPlatform, QT_VERSION_STR));
}

void MainWindow::aboutQt()
{
    QMessageBox::aboutQt(this, tr("About Qt"));
}

void MainWindow::showStatusMessage(const QString &message)
{
    m_statusLabel->setText(message);
    m_statusTimer->start(3000); // Clear after 3 seconds
}

void MainWindow::updateProgress(int value)
{
    if (value < 0)
    {
        m_progressBar->setVisible(false);
    }
    else
    {
        m_progressBar->setVisible(true);
        m_progressBar->setValue(value);
    }
}

void MainWindow::onKernelFinished(const QString &kernelName, bool success, const QString &result)
{
    QString message = success ? tr("Kernel '%1' completed successfully").arg(kernelName) : tr("Kernel '%1' failed").arg(kernelName);

    showStatusMessage(message);
    m_resultViewer->addResult(kernelName, success, result);

    if (success)
    {
        m_systemTrayIcon->showMessage(tr("Kernel Completed"), message,
                                      QSystemTrayIcon::Information, 2000);
    }
    else
    {
        m_systemTrayIcon->showMessage(tr("Kernel Failed"), message,
                                      QSystemTrayIcon::Warning, 2000);
    }
}

void MainWindow::onPerformanceDataUpdated(const QVariantMap &data)
{
    showStatusMessage(tr("Performance data updated"));
}

void MainWindow::toggleSystemTray()
{
    m_systemTrayEnabled = m_toggleSystemTrayAct->isChecked();

    if (m_systemTrayEnabled)
    {
        m_systemTrayIcon->show();
        showStatusMessage(tr("System tray icon enabled"));
    }
    else
    {
        m_systemTrayIcon->hide();
        showStatusMessage(tr("System tray icon disabled"));
    }
}

void MainWindow::systemTrayActivated(QSystemTrayIcon::ActivationReason reason)
{
    if (reason == QSystemTrayIcon::DoubleClick)
    {
        if (isVisible())
        {
            hide();
        }
        else
        {
            show();
            raise();
            activateWindow();
        }
    }
}
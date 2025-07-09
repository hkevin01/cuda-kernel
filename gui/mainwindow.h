#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QTabWidget>
#include <QMenuBar>
#include <QStatusBar>
#include <QToolBar>
#include <QAction>
#include <QSettings>
#include <QTimer>
#include <QLabel>
#include <QProgressBar>
#include <QSystemTrayIcon>
#include <QMenu>

#include "kernel_runner.h"
#include "result_viewer.h"
#include "performance_widget.h"
#include "example_tabs.h"

QT_BEGIN_NAMESPACE
class QAction;
class QMenu;
class QToolBar;
class QTabWidget;
class QStatusBar;
class QProgressBar;
class QLabel;
class QSystemTrayIcon;
QT_END_NAMESPACE

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

    void setPlatform(const QString &platform);

protected:
    void closeEvent(QCloseEvent *event) override;

private slots:
    void about();
    void aboutQt();
    void showStatusMessage(const QString &message);
    void updateProgress(int value);
    void onKernelFinished(const QString &kernelName, bool success, const QString &result);
    void onPerformanceDataUpdated(const QVariantMap &data);
    void toggleSystemTray();
    void systemTrayActivated(QSystemTrayIcon::ActivationReason reason);

private:
    void createActions();
    void createMenus();
    void createToolBars();
    void createStatusBar();
    void createSystemTray();
    void createTabs();
    void setupConnections();
    void loadSettings();
    void saveSettings();
    void updateWindowTitle();

    // UI Components
    QTabWidget *m_tabWidget;
    KernelRunner *m_kernelRunner;
    ResultViewer *m_resultViewer;
    PerformanceWidget *m_performanceWidget;
    ExampleTabs *m_exampleTabs;

    // Menu and Toolbar
    QMenu *m_fileMenu;
    QMenu *m_editMenu;
    QMenu *m_viewMenu;
    QMenu *m_helpMenu;
    QToolBar *m_fileToolBar;
    QToolBar *m_kernelToolBar;

    // Actions
    QAction *m_newAct;
    QAction *m_openAct;
    QAction *m_saveAct;
    QAction *m_saveAsAct;
    QAction *m_exitAct;
    QAction *m_cutAct;
    QAction *m_copyAct;
    QAction *m_pasteAct;
    QAction *m_aboutAct;
    QAction *m_aboutQtAct;
    QAction *m_runKernelAct;
    QAction *m_runTestsAct;
    QAction *m_showPerformanceAct;
    QAction *m_toggleSystemTrayAct;

    // Status Bar
    QLabel *m_statusLabel;
    QProgressBar *m_progressBar;
    QLabel *m_platformLabel;

    // System Tray
    QSystemTrayIcon *m_systemTrayIcon;
    QMenu *m_systemTrayMenu;

    // Settings
    QSettings *m_settings;
    QString m_currentPlatform;
    bool m_systemTrayEnabled;

    // Timer for status messages
    QTimer *m_statusTimer;
};

#endif // MAINWINDOW_H
#include <QApplication>
#include <QStyleFactory>
#include <QDir>
#include <QStandardPaths>
#include <QMessageBox>
#include <QCommandLineParser>
#include <QCommandLineOption>
#include <QSplashScreen>
#include <QPixmap>
#include <QTimer>

#include "mainwindow.h"

int main(int argc, char *argv[])
{
    QApplication app(argc, argv);

    // Set application metadata
    app.setApplicationName("GPU Kernel Examples");
    app.setApplicationVersion("1.0.0");
    app.setOrganizationName("CUDA/HIP Kernel Project");
    app.setOrganizationDomain("gpu-kernel.example.com");

    // Set application style
    app.setStyle(QStyleFactory::create("Fusion"));

    // Create splash screen
    QPixmap splashPixmap(400, 300);
    splashPixmap.fill(Qt::darkBlue);

    QSplashScreen splash(splashPixmap);
    splash.show();

    // Process events to show splash screen
    app.processEvents();

    // Parse command line arguments
    QCommandLineParser parser;
    parser.setApplicationDescription("GPU Kernel Examples GUI");
    parser.addHelpOption();
    parser.addVersionOption();

    QCommandLineOption platformOption(QStringList() << "p" << "platform",
                                      "GPU platform to use (cuda/hip)", "platform", "hip");
    parser.addOption(platformOption);

    QCommandLineOption testModeOption(QStringList() << "t" << "test-mode",
                                      "Run in test mode");
    parser.addOption(testModeOption);

    parser.process(app);

    // Show splash message
    splash.showMessage("Initializing GPU platform...", Qt::AlignBottom | Qt::AlignCenter, Qt::white);
    app.processEvents();

    // Create main window
    MainWindow window;

    // Set platform
    QString platform = parser.value(platformOption);
    if (platform == "cuda")
    {
        window.setPlatform("CUDA");
    }
    else if (platform == "hip")
    {
        window.setPlatform("HIP");
    }

    // Show main window
    splash.showMessage("Loading examples...", Qt::AlignBottom | Qt::AlignCenter, Qt::white);
    app.processEvents();

    // Delay splash screen for a moment
    QTimer::singleShot(2000, &splash, &QSplashScreen::close);
    QTimer::singleShot(2000, &window, &MainWindow::show);

    // Run application
    int result = app.exec();

    return result;
}
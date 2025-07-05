#ifndef KERNELRUNNER_H
#define KERNELRUNNER_H

#include <QWidget>
#include <QListWidget>
#include <QPushButton>
#include <QTextEdit>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QGroupBox>
#include <QLabel>
#include <QSpinBox>
#include <QComboBox>
#include <QProgressBar>
#include <QProcess>
#include <QTimer>
#include <QMap>
#include <QString>
#include <QStringList>

class KernelRunner : public QWidget
{
    Q_OBJECT

public:
    explicit KernelRunner(QWidget *parent = nullptr);
    ~KernelRunner();

    void runSelectedKernel();
    void refreshKernelList();

signals:
    void kernelFinished(const QString &kernelName, bool success, const QString &result);
    void progressUpdated(int value);

private slots:
    void onRunButtonClicked();
    void onRefreshButtonClicked();
    void onKernelSelectionChanged();
    void onProcessFinished(int exitCode, QProcess::ExitStatus exitStatus);
    void onProcessError(QProcess::ProcessError error);
    void onProcessOutput();
    void updateProgress();

private:
    void setupUI();
    void loadKernelList();
    void runKernel(const QString &kernelName);
    void updateKernelInfo(const QString &kernelName);
    QString getKernelExecutable(const QString &kernelName);
    void parseKernelOutput(const QString &output);

    // UI Components
    QListWidget *m_kernelList;
    QTextEdit *m_outputText;
    QPushButton *m_runButton;
    QPushButton *m_refreshButton;
    QProgressBar *m_progressBar;
    QLabel *m_statusLabel;

    // Kernel configuration
    QSpinBox *m_iterationsSpinBox;
    QSpinBox *m_dataSizeSpinBox;
    QComboBox *m_platformComboBox;

    // Kernel information
    QLabel *m_kernelNameLabel;
    QLabel *m_kernelDescriptionLabel;
    QLabel *m_kernelParametersLabel;

    // Process management
    QProcess *m_currentProcess;
    QTimer *m_progressTimer;
    QString m_currentKernel;
    bool m_isRunning;

    // Kernel data
    struct KernelInfo
    {
        QString name;
        QString description;
        QString executable;
        QStringList parameters;
        QString category;
    };

    QMap<QString, KernelInfo> m_kernels;
    QStringList m_kernelCategories;
};

#endif // KERNELRUNNER_H
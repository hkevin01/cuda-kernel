#ifndef EXAMPLETABS_H
#define EXAMPLETABS_H

#include <QWidget>
#include <QTabWidget>
#include <QTextEdit>
#include <QPushButton>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QGroupBox>
#include <QLabel>
#include <QScrollArea>
#include <QProcess>
#include <QTimer>
#include <QMap>
#include <QString>

class ExampleTab : public QWidget
{
    Q_OBJECT

public:
    explicit ExampleTab(const QString &name, const QString &description, QWidget *parent = nullptr);
    ~ExampleTab();

    void setExecutablePath(const QString &path);
    void setSourceCode(const QString &code);
    void setParameters(const QStringList &params);

public slots:
    void runExample();
    void showSourceCode();
    void showDocumentation();

signals:
    void exampleFinished(const QString &name, bool success, const QString &result);

private slots:
    void onProcessFinished(int exitCode, QProcess::ExitStatus exitStatus);
    void onProcessError(QProcess::ProcessError error);
    void onProcessOutput();

private:
    void setupUI();
    void updateOutput(const QString &text, bool isError = false);

    QString m_name;
    QString m_description;
    QString m_executablePath;
    QString m_sourceCode;
    QStringList m_parameters;

    // UI Components
    QTextEdit *m_descriptionText;
    QTextEdit *m_outputText;
    QPushButton *m_runButton;
    QPushButton *m_sourceButton;
    QPushButton *m_docButton;
    QLabel *m_statusLabel;

    // Process management
    QProcess *m_process;
    bool m_isRunning;
};

class ExampleTabs : public QWidget
{
    Q_OBJECT

public:
    explicit ExampleTabs(QWidget *parent = nullptr);

public slots:
    void runExample(const QString &exampleName);
    void showExampleSource(const QString &exampleName);

signals:
    void exampleFinished(const QString &name, bool success, const QString &result);

private:
    void setupUI();
    void loadExamples();
    void createExampleTab(const QString &name, const QString &description,
                          const QString &category, const QString &sourceFile);

    QTabWidget *m_tabWidget;
    QMap<QString, ExampleTab *> m_exampleTabs;
};

#endif // EXAMPLETABS_H
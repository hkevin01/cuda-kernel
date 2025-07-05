#ifndef RESULTVIEWER_H
#define RESULTVIEWER_H

#include <QWidget>
#include <QTableWidget>
#include <QTextEdit>
#include <QPushButton>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QGroupBox>
#include <QLabel>
#include <QHeaderView>
#include <QDateTime>
#include <QMap>

class ResultViewer : public QWidget
{
    Q_OBJECT

public:
    explicit ResultViewer(QWidget *parent = nullptr);

public slots:
    void addResult(const QString &name, bool success, const QString &result);
    void clearResults();
    void exportResults();

private slots:
    void onResultSelected(int row, int column);
    void onClearButtonClicked();
    void onExportButtonClicked();

private:
    void setupUI();
    void updateStatistics();

    QTableWidget *m_resultsTable;
    QTextEdit *m_detailText;
    QPushButton *m_clearButton;
    QPushButton *m_exportButton;
    QLabel *m_statsLabel;

    struct ResultEntry
    {
        QString name;
        bool success;
        QString result;
        QDateTime timestamp;
    };

    QList<ResultEntry> m_results;
};

#endif // RESULTVIEWER_H
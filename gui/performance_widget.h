#ifndef PERFORMANCEWIDGET_H
#define PERFORMANCEWIDGET_H

#include <QWidget>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QGroupBox>
#include <QLabel>
#include <QPushButton>
#include <QTableWidget>
#include <QTimer>
#include <QMap>
#include <QString>

class PerformanceWidget : public QWidget
{
    Q_OBJECT

public:
    explicit PerformanceWidget(QWidget *parent = nullptr);

public slots:
    void updateData(const QVariantMap &data);
    void startMonitoring();
    void stopMonitoring();

signals:
    void dataUpdated(const QVariantMap &data);

private slots:
    void onStartButtonClicked();
    void onStopButtonClicked();
    void onClearButtonClicked();
    void onUpdateTimer();

private:
    void setupUI();

    // UI Components
    QPushButton *m_startButton;
    QPushButton *m_stopButton;
    QPushButton *m_clearButton;
    QTableWidget *m_performanceTable;
    QLabel *m_statusLabel;

    // Data
    QTimer *m_updateTimer;
    QMap<QString, QList<QPair<double, double>>> m_performanceData; // kernel -> [(time, throughput)]
    bool m_isMonitoring;
    int m_dataPointCount;
};

#endif // PERFORMANCEWIDGET_H
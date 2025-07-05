#include "performance_widget.h"
#include <QMessageBox>
#include <QHeaderView>

PerformanceWidget::PerformanceWidget(QWidget *parent)
    : QWidget(parent), m_isMonitoring(false), m_dataPointCount(0)
{
    setupUI();
}

void PerformanceWidget::setupUI()
{
    QVBoxLayout *mainLayout = new QVBoxLayout(this);

    // Control buttons
    QHBoxLayout *buttonLayout = new QHBoxLayout();
    m_startButton = new QPushButton(tr("Start Monitoring"));
    m_stopButton = new QPushButton(tr("Stop Monitoring"));
    m_clearButton = new QPushButton(tr("Clear Data"));

    m_stopButton->setEnabled(false);

    buttonLayout->addWidget(m_startButton);
    buttonLayout->addWidget(m_stopButton);
    buttonLayout->addWidget(m_clearButton);
    buttonLayout->addStretch();
    mainLayout->addLayout(buttonLayout);

    // Status label
    m_statusLabel = new QLabel(tr("Ready"));
    mainLayout->addWidget(m_statusLabel);

    // Performance table
    m_performanceTable = new QTableWidget();
    m_performanceTable->setColumnCount(3);
    m_performanceTable->setHorizontalHeaderLabels({tr("Kernel"), tr("Time (ms)"), tr("Throughput")});
    m_performanceTable->horizontalHeader()->setStretchLastSection(true);
    mainLayout->addWidget(m_performanceTable);

    // Timer for updates
    m_updateTimer = new QTimer(this);
    m_updateTimer->setInterval(1000); // 1 second

    // Connect signals
    connect(m_startButton, &QPushButton::clicked, this, &PerformanceWidget::onStartButtonClicked);
    connect(m_stopButton, &QPushButton::clicked, this, &PerformanceWidget::onStopButtonClicked);
    connect(m_clearButton, &QPushButton::clicked, this, &PerformanceWidget::onClearButtonClicked);
    connect(m_updateTimer, &QTimer::timeout, this, &PerformanceWidget::onUpdateTimer);
}

void PerformanceWidget::updateData(const QVariantMap &data)
{
    // Simple data update - just add to table
    QString kernelName = data.value("kernel", "Unknown").toString();
    double time = data.value("time", 0.0).toDouble();
    double throughput = data.value("throughput", 0.0).toDouble();

    // Add to table
    int row = m_performanceTable->rowCount();
    m_performanceTable->insertRow(row);
    m_performanceTable->setItem(row, 0, new QTableWidgetItem(kernelName));
    m_performanceTable->setItem(row, 1, new QTableWidgetItem(QString::number(time, 'f', 2)));
    m_performanceTable->setItem(row, 2, new QTableWidgetItem(QString::number(throughput, 'f', 2)));

    // Store data
    m_performanceData[kernelName].append({time, throughput});
    m_dataPointCount++;

    // Emit signal
    emit dataUpdated(data);
}

void PerformanceWidget::startMonitoring()
{
    m_isMonitoring = true;
    m_startButton->setEnabled(false);
    m_stopButton->setEnabled(true);
    m_statusLabel->setText(tr("Monitoring..."));
    m_updateTimer->start();
}

void PerformanceWidget::stopMonitoring()
{
    m_isMonitoring = false;
    m_startButton->setEnabled(true);
    m_stopButton->setEnabled(false);
    m_statusLabel->setText(tr("Stopped"));
    m_updateTimer->stop();
}

void PerformanceWidget::onStartButtonClicked()
{
    startMonitoring();
}

void PerformanceWidget::onStopButtonClicked()
{
    stopMonitoring();
}

void PerformanceWidget::onClearButtonClicked()
{
    m_performanceData.clear();
    m_performanceTable->setRowCount(0);
    m_dataPointCount = 0;
    m_statusLabel->setText(tr("Data cleared"));
}

void PerformanceWidget::onUpdateTimer()
{
    if (!m_isMonitoring)
        return;

    // Simulate some data for demonstration
    QVariantMap data;
    data["kernel"] = QString("Kernel_%1").arg(m_dataPointCount % 5);
    data["time"] = 10.0 + (rand() % 20);          // Random time between 10-30ms
    data["throughput"] = 1000.0 + (rand() % 500); // Random throughput

    updateData(data);
}
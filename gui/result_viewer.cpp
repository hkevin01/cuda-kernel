#include "result_viewer.h"
#include <QHeaderView>
#include <QDateTime>
#include <QFileDialog>
#include <QMessageBox>
#include <QTextStream>
#include <QApplication>
#include <QStandardPaths>

ResultViewer::ResultViewer(QWidget *parent)
    : QWidget(parent)
{
    setupUI();
}

void ResultViewer::setupUI()
{
    QVBoxLayout *mainLayout = new QVBoxLayout(this);

    // Top section with table and buttons
    QHBoxLayout *topLayout = new QHBoxLayout();

    // Results table
    QGroupBox *tableGroup = new QGroupBox(tr("Results"));
    QVBoxLayout *tableLayout = new QVBoxLayout(tableGroup);

    m_resultsTable = new QTableWidget();
    m_resultsTable->setColumnCount(4);
    m_resultsTable->setHorizontalHeaderLabels({tr("Name"), tr("Status"), tr("Time"), tr("Duration")});
    m_resultsTable->horizontalHeader()->setStretchLastSection(true);
    m_resultsTable->setSelectionBehavior(QAbstractItemView::SelectRows);
    m_resultsTable->setAlternatingRowColors(true);
    tableLayout->addWidget(m_resultsTable);

    // Buttons
    QHBoxLayout *buttonLayout = new QHBoxLayout();
    m_clearButton = new QPushButton(tr("Clear All"));
    m_exportButton = new QPushButton(tr("Export"));
    buttonLayout->addWidget(m_clearButton);
    buttonLayout->addWidget(m_exportButton);
    buttonLayout->addStretch();
    tableLayout->addLayout(buttonLayout);

    topLayout->addWidget(tableGroup);

    // Right side - Details and statistics
    QVBoxLayout *rightLayout = new QVBoxLayout();

    // Statistics
    QGroupBox *statsGroup = new QGroupBox(tr("Statistics"));
    QVBoxLayout *statsLayout = new QVBoxLayout(statsGroup);
    m_statsLabel = new QLabel(tr("No results yet"));
    statsLayout->addWidget(m_statsLabel);
    rightLayout->addWidget(statsGroup);

    // Details
    QGroupBox *detailGroup = new QGroupBox(tr("Details"));
    QVBoxLayout *detailLayout = new QVBoxLayout(detailGroup);
    m_detailText = new QTextEdit();
    m_detailText->setReadOnly(true);
    detailLayout->addWidget(m_detailText);
    rightLayout->addWidget(detailGroup);

    topLayout->addLayout(rightLayout);
    mainLayout->addLayout(topLayout);

    // Connect signals
    connect(m_clearButton, &QPushButton::clicked, this, &ResultViewer::onClearButtonClicked);
    connect(m_exportButton, &QPushButton::clicked, this, &ResultViewer::onExportButtonClicked);
    connect(m_resultsTable, &QTableWidget::cellClicked, this, &ResultViewer::onResultSelected);
}

void ResultViewer::addResult(const QString &name, bool success, const QString &result)
{
    ResultEntry entry;
    entry.name = name;
    entry.success = success;
    entry.result = result;
    entry.timestamp = QDateTime::currentDateTime();

    m_results.append(entry);

    // Add to table
    int row = m_resultsTable->rowCount();
    m_resultsTable->insertRow(row);

    m_resultsTable->setItem(row, 0, new QTableWidgetItem(name));

    QTableWidgetItem *statusItem = new QTableWidgetItem(success ? tr("Success") : tr("Failed"));
    statusItem->setBackground(success ? QColor(200, 255, 200) : QColor(255, 200, 200));
    m_resultsTable->setItem(row, 1, statusItem);

    m_resultsTable->setItem(row, 2, new QTableWidgetItem(entry.timestamp.toString("hh:mm:ss")));
    m_resultsTable->setItem(row, 3, new QTableWidgetItem("--")); // Duration placeholder

    // Auto-select the new result
    m_resultsTable->selectRow(row);
    onResultSelected(row, 0);

    updateStatistics();
}

void ResultViewer::clearResults()
{
    m_results.clear();
    m_resultsTable->setRowCount(0);
    m_detailText->clear();
    updateStatistics();
}

void ResultViewer::exportResults()
{
    QString fileName = QFileDialog::getSaveFileName(this, tr("Export Results"),
                                                    QStandardPaths::writableLocation(QStandardPaths::DocumentsLocation) + "/gpu_results.csv",
                                                    tr("CSV Files (*.csv);;Text Files (*.txt)"));

    if (fileName.isEmpty())
        return;

    QFile file(fileName);
    if (!file.open(QIODevice::WriteOnly | QIODevice::Text))
    {
        QMessageBox::warning(this, tr("Export Failed"), tr("Could not open file for writing."));
        return;
    }

    QTextStream out(&file);

    // Write header
    out << "Name,Status,Timestamp,Result\n";

    // Write data
    for (const ResultEntry &entry : m_results)
    {
        out << QString("\"%1\",%2,\"%3\",\"%4\"\n")
                   .arg(entry.name)
                   .arg(entry.success ? "Success" : "Failed")
                   .arg(entry.timestamp.toString("yyyy-MM-dd hh:mm:ss"))
                   .arg(QString(entry.result).replace("\"", "\"\"")); // Escape quotes
    }

    file.close();
    QMessageBox::information(this, tr("Export Complete"),
                             tr("Results exported to %1").arg(fileName));
}

void ResultViewer::onResultSelected(int row, int column)
{
    if (row >= 0 && row < m_results.size())
    {
        const ResultEntry &entry = m_results[row];

        QString detailText = tr("<h3>%1</h3>").arg(entry.name);
        detailText += tr("<p><b>Status:</b> %1</p>").arg(entry.success ? tr("Success") : tr("Failed"));
        detailText += tr("<p><b>Timestamp:</b> %1</p>").arg(entry.timestamp.toString("yyyy-MM-dd hh:mm:ss"));
        detailText += tr("<p><b>Result:</b></p>");
        detailText += tr("<pre>%1</pre>").arg(entry.result);

        m_detailText->setHtml(detailText);
    }
}

void ResultViewer::onClearButtonClicked()
{
    if (QMessageBox::question(this, tr("Clear Results"),
                              tr("Are you sure you want to clear all results?"),
                              QMessageBox::Yes | QMessageBox::No) == QMessageBox::Yes)
    {
        clearResults();
    }
}

void ResultViewer::onExportButtonClicked()
{
    if (m_results.isEmpty())
    {
        QMessageBox::information(this, tr("No Results"), tr("No results to export."));
        return;
    }
    exportResults();
}

void ResultViewer::updateStatistics()
{
    if (m_results.isEmpty())
    {
        m_statsLabel->setText(tr("No results yet"));
        return;
    }

    int total = m_results.size();
    int successful = 0;

    for (const ResultEntry &entry : m_results)
    {
        if (entry.success)
            successful++;
    }

    double successRate = (double)successful / total * 100.0;

    QString stats = tr("Total: %1 | Success: %2 | Failed: %3 | Success Rate: %4%")
                        .arg(total)
                        .arg(successful)
                        .arg(total - successful)
                        .arg(successRate, 0, 'f', 1);

    m_statsLabel->setText(stats);
}
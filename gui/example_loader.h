#ifndef EXAMPLE_LOADER_H
#define EXAMPLE_LOADER_H

#include <QString>
#include <QList>
#include <QXmlStreamReader>
#include <QFile>
#include <QTextStream>
#include <QApplication>
#include <QDir>

struct ExampleInfo
{
    QString name;
    QString description;
    QString category;
    QString sourceFile;
};

class ExampleLoader
{
public:
    static QList<ExampleInfo> loadExamples();
    
private:
    static QString getDataPath();
    static ExampleInfo parseExampleXml(const QString& filename);
    static QString formatHtmlFromXml(QXmlStreamReader& xml);
};

#endif // EXAMPLE_LOADER_H

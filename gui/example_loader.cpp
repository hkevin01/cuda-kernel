#include "example_loader.h"
#include <QDebug>
#include <QXmlStreamReader>

QString ExampleLoader::getDataPath()
{
    // Get the application directory and look for data folder
    QString appDir = QApplication::applicationDirPath();
    QString dataPath = appDir + "/../data/examples";
    
    QDir dir(dataPath);
    if (!dir.exists()) {
        // Try alternative paths
        dataPath = appDir + "/data/examples";
        dir.setPath(dataPath);
        if (!dir.exists()) {
            dataPath = "./data/examples";
            dir.setPath(dataPath);
            if (!dir.exists()) {
                qWarning() << "Could not find data/examples directory. Searched:" 
                          << appDir + "/../data/examples"
                          << appDir + "/data/examples"
                          << "./data/examples";
                return "";
            }
        }
    }
    
    return dataPath;
}

QList<ExampleInfo> ExampleLoader::loadExamples()
{
    QList<ExampleInfo> examples;
    QString dataPath = getDataPath();
    
    if (dataPath.isEmpty()) {
        qWarning() << "Data path not found, falling back to hardcoded examples";
        // Return hardcoded fallback examples if XML loading fails
        examples << ExampleInfo{"Vector Addition", 
                               "<h3>Vector Addition Kernel</h3><p>Basic parallel vector addition example.</p>", 
                               "Basic", 
                               "src/01_vector_addition/vector_addition.cu"};
        examples << ExampleInfo{"Warp Primitives", 
                               "<h3>Warp-Level Primitives</h3><p>Advanced warp-level programming techniques.</p>", 
                               "Advanced", 
                               "src/09_warp_primitives/warp_primitives_hip.hip"};
        return examples;
    }
    
    // Read the examples list file
    QString listFilePath = dataPath + "/examples_list.xml";
    QFile listFile(listFilePath);
    if (!listFile.open(QIODevice::ReadOnly | QIODevice::Text)) {
        qWarning() << "Could not open examples list file:" << listFilePath;
        return examples;
    }
    
    QXmlStreamReader listXml(&listFile);
    while (!listXml.atEnd() && !listXml.hasError()) {
        QXmlStreamReader::TokenType token = listXml.readNext();
        
        if (token == QXmlStreamReader::StartElement) {
            if (listXml.name().toString() == "example") {
                QString filename = listXml.attributes().value("file").toString();
                if (!filename.isEmpty()) {
                    ExampleInfo example = parseExampleXml(dataPath + "/" + filename);
                    if (!example.name.isEmpty()) {
                        examples.append(example);
                    }
                }
            }
        }
    }
    
    if (listXml.hasError()) {
        qWarning() << "XML parsing error in examples list:" << listXml.errorString();
    }
    
    return examples;
}

ExampleInfo ExampleLoader::parseExampleXml(const QString& filename)
{
    ExampleInfo example;
    
    QFile file(filename);
    if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) {
        qWarning() << "Could not open example file:" << filename;
        return example;
    }
    
    QXmlStreamReader xml(&file);
    
    while (!xml.atEnd() && !xml.hasError()) {
        QXmlStreamReader::TokenType token = xml.readNext();
        
        if (token == QXmlStreamReader::StartElement) {
            if (xml.name().toString() == "name") {
                example.name = xml.readElementText();
            } else if (xml.name().toString() == "category") {
                example.category = xml.readElementText();
            } else if (xml.name().toString() == "sourceFile") {
                example.sourceFile = xml.readElementText();
            } else if (xml.name().toString() == "description") {
                example.description = formatHtmlFromXml(xml);
            }
        }
    }
    
    if (xml.hasError()) {
        qWarning() << "XML parsing error in" << filename << ":" << xml.errorString();
    }
    
    return example;
}

QString ExampleLoader::formatHtmlFromXml(QXmlStreamReader& xml)
{
    QString html;
    QString currentElement;
    int depth = 0;
    
    while (!xml.atEnd() && !xml.hasError()) {
        QXmlStreamReader::TokenType token = xml.readNext();
        
        if (token == QXmlStreamReader::StartElement) {
            QString elementName = xml.name().toString();
            depth++;
            
            if (elementName == "title") {
                html += "<h3>";
                html += xml.readElementText();
                html += "</h3>\n";
                depth--;
            } else if (elementName == "analogy") {
                html += "<p>";
                html += xml.readElementText();
                html += "</p>\n";
                depth--;
            } else if (elementName == "overview") {
                html += "<p>";
                html += xml.readElementText();
                html += "</p>\n";
                depth--;
            } else if (elementName == "features") {
                html += "<ul>\n";
                while (xml.readNextStartElement()) {
                    if (xml.name().toString() == "feature") {
                        html += "<li>" + xml.readElementText() + "</li>\n";
                    }
                }
                html += "</ul>\n";
                depth--;
            } else if (elementName == "concepts") {
                html += "<h4>Key Concepts:</h4>\n<ul>\n";
                while (xml.readNextStartElement()) {
                    if (xml.name().toString() == "concept") {
                        QString title, description;
                        while (xml.readNextStartElement()) {
                            if (xml.name().toString() == "title") {
                                title = xml.readElementText();
                            } else if (xml.name().toString() == "description") {
                                description = xml.readElementText();
                            } else {
                                xml.skipCurrentElement();
                            }
                        }
                        html += "<li><b>" + title + ":</b> " + description + "</li>\n";
                    }
                }
                html += "</ul>\n";
                depth--;
            } else if (elementName == "applications") {
                html += "<h4>🚀 Real-World Applications:</h4>\n<ul>\n";
                while (xml.readNextStartElement()) {
                    if (xml.name().toString() == "application") {
                        QString title, description;
                        while (xml.readNextStartElement()) {
                            if (xml.name().toString() == "title") {
                                title = xml.readElementText();
                            } else if (xml.name().toString() == "description") {
                                description = xml.readElementText();
                            } else {
                                xml.skipCurrentElement();
                            }
                        }
                        html += "<li><b>" + title + ":</b> " + description + "</li>\n";
                    }
                }
                html += "</ul>\n";
                depth--;
            } else if (elementName == "optimizations") {
                html += "<h4>Optimization Techniques:</h4>\n<ul>\n";
                while (xml.readNextStartElement()) {
                    if (xml.name().toString() == "optimization") {
                        html += "<li>" + xml.readElementText() + "</li>\n";
                    }
                }
                html += "</ul>\n";
                depth--;
            } else if (elementName == "patterns") {
                html += "<h4>Patterns Demonstrated:</h4>\n<ul>\n";
                while (xml.readNextStartElement()) {
                    if (xml.name().toString() == "pattern") {
                        html += "<li>" + xml.readElementText() + "</li>\n";
                    }
                }
                html += "</ul>\n";
                depth--;
            } else if (elementName == "algorithms") {
                html += "<h4>Algorithm Variants:</h4>\n<ul>\n";
                while (xml.readNextStartElement()) {
                    if (xml.name().toString() == "algorithm") {
                        html += "<li>" + xml.readElementText() + "</li>\n";
                    }
                }
                html += "</ul>\n";
                depth--;
            } else if (elementName == "performance") {
                html += "<h4>Performance Considerations:</h4>\n<ul>\n";
                while (xml.readNextStartElement()) {
                    if (xml.name().toString() == "consideration") {
                        html += "<li>" + xml.readElementText() + "</li>\n";
                    }
                }
                html += "</ul>\n";
                depth--;
            } else if (elementName == "importance") {
                html += "<h4>⚡ Why This Matters:</h4>\n";
                while (xml.readNextStartElement()) {
                    if (xml.name().toString() == "why") {
                        html += "<p>" + xml.readElementText() + "</p>\n";
                    } else if (xml.name().toString() == "performance") {
                        html += "<h4>📊 Performance Impact:</h4>\n";
                        html += "<p>" + xml.readElementText() + "</p>\n";
                    } else {
                        xml.skipCurrentElement();
                    }
                }
                depth--;
            }
        } else if (token == QXmlStreamReader::EndElement) {
            depth--;
            if (xml.name().toString() == "description" && depth == 0) {
                break;
            }
        }
    }
    
    return html;
}

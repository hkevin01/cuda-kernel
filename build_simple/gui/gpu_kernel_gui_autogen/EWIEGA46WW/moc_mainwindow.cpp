/****************************************************************************
** Meta object code from reading C++ file 'mainwindow.h'
**
** Created by: The Qt Meta Object Compiler version 68 (Qt 6.4.2)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include <memory>
#include "../../../../gui/mainwindow.h"
#include <QtGui/qtextcursor.h>
#include <QtCore/qmetatype.h>
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'mainwindow.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 68
#error "This file was generated using the moc from 6.4.2. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

#ifndef Q_CONSTINIT
#define Q_CONSTINIT
#endif

QT_BEGIN_MOC_NAMESPACE
QT_WARNING_PUSH
QT_WARNING_DISABLE_DEPRECATED
namespace {
struct qt_meta_stringdata_MainWindow_t {
    uint offsetsAndSizes[40];
    char stringdata0[11];
    char stringdata1[6];
    char stringdata2[1];
    char stringdata3[8];
    char stringdata4[18];
    char stringdata5[8];
    char stringdata6[15];
    char stringdata7[6];
    char stringdata8[17];
    char stringdata9[11];
    char stringdata10[8];
    char stringdata11[7];
    char stringdata12[15];
    char stringdata13[9];
    char stringdata14[25];
    char stringdata15[5];
    char stringdata16[17];
    char stringdata17[20];
    char stringdata18[34];
    char stringdata19[7];
};
#define QT_MOC_LITERAL(ofs, len) \
    uint(sizeof(qt_meta_stringdata_MainWindow_t::offsetsAndSizes) + ofs), len 
Q_CONSTINIT static const qt_meta_stringdata_MainWindow_t qt_meta_stringdata_MainWindow = {
    {
        QT_MOC_LITERAL(0, 10),  // "MainWindow"
        QT_MOC_LITERAL(11, 5),  // "about"
        QT_MOC_LITERAL(17, 0),  // ""
        QT_MOC_LITERAL(18, 7),  // "aboutQt"
        QT_MOC_LITERAL(26, 17),  // "showStatusMessage"
        QT_MOC_LITERAL(44, 7),  // "message"
        QT_MOC_LITERAL(52, 14),  // "updateProgress"
        QT_MOC_LITERAL(67, 5),  // "value"
        QT_MOC_LITERAL(73, 16),  // "onKernelFinished"
        QT_MOC_LITERAL(90, 10),  // "kernelName"
        QT_MOC_LITERAL(101, 7),  // "success"
        QT_MOC_LITERAL(109, 6),  // "result"
        QT_MOC_LITERAL(116, 14),  // "onTestFinished"
        QT_MOC_LITERAL(131, 8),  // "testName"
        QT_MOC_LITERAL(140, 24),  // "onPerformanceDataUpdated"
        QT_MOC_LITERAL(165, 4),  // "data"
        QT_MOC_LITERAL(170, 16),  // "toggleSystemTray"
        QT_MOC_LITERAL(187, 19),  // "systemTrayActivated"
        QT_MOC_LITERAL(207, 33),  // "QSystemTrayIcon::ActivationRe..."
        QT_MOC_LITERAL(241, 6)   // "reason"
    },
    "MainWindow",
    "about",
    "",
    "aboutQt",
    "showStatusMessage",
    "message",
    "updateProgress",
    "value",
    "onKernelFinished",
    "kernelName",
    "success",
    "result",
    "onTestFinished",
    "testName",
    "onPerformanceDataUpdated",
    "data",
    "toggleSystemTray",
    "systemTrayActivated",
    "QSystemTrayIcon::ActivationReason",
    "reason"
};
#undef QT_MOC_LITERAL
} // unnamed namespace

Q_CONSTINIT static const uint qt_meta_data_MainWindow[] = {

 // content:
      10,       // revision
       0,       // classname
       0,    0, // classinfo
       9,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       0,       // signalCount

 // slots: name, argc, parameters, tag, flags, initial metatype offsets
       1,    0,   68,    2, 0x08,    1 /* Private */,
       3,    0,   69,    2, 0x08,    2 /* Private */,
       4,    1,   70,    2, 0x08,    3 /* Private */,
       6,    1,   73,    2, 0x08,    5 /* Private */,
       8,    3,   76,    2, 0x08,    7 /* Private */,
      12,    3,   83,    2, 0x08,   11 /* Private */,
      14,    1,   90,    2, 0x08,   15 /* Private */,
      16,    0,   93,    2, 0x08,   17 /* Private */,
      17,    1,   94,    2, 0x08,   18 /* Private */,

 // slots: parameters
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void, QMetaType::QString,    5,
    QMetaType::Void, QMetaType::Int,    7,
    QMetaType::Void, QMetaType::QString, QMetaType::Bool, QMetaType::QString,    9,   10,   11,
    QMetaType::Void, QMetaType::QString, QMetaType::Bool, QMetaType::QString,   13,   10,   11,
    QMetaType::Void, QMetaType::QVariantMap,   15,
    QMetaType::Void,
    QMetaType::Void, 0x80000000 | 18,   19,

       0        // eod
};

Q_CONSTINIT const QMetaObject MainWindow::staticMetaObject = { {
    QMetaObject::SuperData::link<QMainWindow::staticMetaObject>(),
    qt_meta_stringdata_MainWindow.offsetsAndSizes,
    qt_meta_data_MainWindow,
    qt_static_metacall,
    nullptr,
    qt_incomplete_metaTypeArray<qt_meta_stringdata_MainWindow_t,
        // Q_OBJECT / Q_GADGET
        QtPrivate::TypeAndForceComplete<MainWindow, std::true_type>,
        // method 'about'
        QtPrivate::TypeAndForceComplete<void, std::false_type>,
        // method 'aboutQt'
        QtPrivate::TypeAndForceComplete<void, std::false_type>,
        // method 'showStatusMessage'
        QtPrivate::TypeAndForceComplete<void, std::false_type>,
        QtPrivate::TypeAndForceComplete<const QString &, std::false_type>,
        // method 'updateProgress'
        QtPrivate::TypeAndForceComplete<void, std::false_type>,
        QtPrivate::TypeAndForceComplete<int, std::false_type>,
        // method 'onKernelFinished'
        QtPrivate::TypeAndForceComplete<void, std::false_type>,
        QtPrivate::TypeAndForceComplete<const QString &, std::false_type>,
        QtPrivate::TypeAndForceComplete<bool, std::false_type>,
        QtPrivate::TypeAndForceComplete<const QString &, std::false_type>,
        // method 'onTestFinished'
        QtPrivate::TypeAndForceComplete<void, std::false_type>,
        QtPrivate::TypeAndForceComplete<const QString &, std::false_type>,
        QtPrivate::TypeAndForceComplete<bool, std::false_type>,
        QtPrivate::TypeAndForceComplete<const QString &, std::false_type>,
        // method 'onPerformanceDataUpdated'
        QtPrivate::TypeAndForceComplete<void, std::false_type>,
        QtPrivate::TypeAndForceComplete<const QVariantMap &, std::false_type>,
        // method 'toggleSystemTray'
        QtPrivate::TypeAndForceComplete<void, std::false_type>,
        // method 'systemTrayActivated'
        QtPrivate::TypeAndForceComplete<void, std::false_type>,
        QtPrivate::TypeAndForceComplete<QSystemTrayIcon::ActivationReason, std::false_type>
    >,
    nullptr
} };

void MainWindow::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        auto *_t = static_cast<MainWindow *>(_o);
        (void)_t;
        switch (_id) {
        case 0: _t->about(); break;
        case 1: _t->aboutQt(); break;
        case 2: _t->showStatusMessage((*reinterpret_cast< std::add_pointer_t<QString>>(_a[1]))); break;
        case 3: _t->updateProgress((*reinterpret_cast< std::add_pointer_t<int>>(_a[1]))); break;
        case 4: _t->onKernelFinished((*reinterpret_cast< std::add_pointer_t<QString>>(_a[1])),(*reinterpret_cast< std::add_pointer_t<bool>>(_a[2])),(*reinterpret_cast< std::add_pointer_t<QString>>(_a[3]))); break;
        case 5: _t->onTestFinished((*reinterpret_cast< std::add_pointer_t<QString>>(_a[1])),(*reinterpret_cast< std::add_pointer_t<bool>>(_a[2])),(*reinterpret_cast< std::add_pointer_t<QString>>(_a[3]))); break;
        case 6: _t->onPerformanceDataUpdated((*reinterpret_cast< std::add_pointer_t<QVariantMap>>(_a[1]))); break;
        case 7: _t->toggleSystemTray(); break;
        case 8: _t->systemTrayActivated((*reinterpret_cast< std::add_pointer_t<QSystemTrayIcon::ActivationReason>>(_a[1]))); break;
        default: ;
        }
    }
}

const QMetaObject *MainWindow::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->dynamicMetaObject() : &staticMetaObject;
}

void *MainWindow::qt_metacast(const char *_clname)
{
    if (!_clname) return nullptr;
    if (!strcmp(_clname, qt_meta_stringdata_MainWindow.stringdata0))
        return static_cast<void*>(this);
    return QMainWindow::qt_metacast(_clname);
}

int MainWindow::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QMainWindow::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        if (_id < 9)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 9;
    } else if (_c == QMetaObject::RegisterMethodArgumentMetaType) {
        if (_id < 9)
            *reinterpret_cast<QMetaType *>(_a[0]) = QMetaType();
        _id -= 9;
    }
    return _id;
}
QT_WARNING_POP
QT_END_MOC_NAMESPACE

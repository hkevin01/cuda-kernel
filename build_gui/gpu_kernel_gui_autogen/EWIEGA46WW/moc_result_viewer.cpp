/****************************************************************************
** Meta object code from reading C++ file 'result_viewer.h'
**
** Created by: The Qt Meta Object Compiler version 68 (Qt 6.4.2)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include <memory>
#include "../../../gui/result_viewer.h"
#include <QtGui/qtextcursor.h>
#include <QtCore/qmetatype.h>
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'result_viewer.h' doesn't include <QObject>."
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
struct qt_meta_stringdata_ResultViewer_t {
    uint offsetsAndSizes[26];
    char stringdata0[13];
    char stringdata1[10];
    char stringdata2[1];
    char stringdata3[5];
    char stringdata4[8];
    char stringdata5[7];
    char stringdata6[13];
    char stringdata7[14];
    char stringdata8[17];
    char stringdata9[4];
    char stringdata10[7];
    char stringdata11[21];
    char stringdata12[22];
};
#define QT_MOC_LITERAL(ofs, len) \
    uint(sizeof(qt_meta_stringdata_ResultViewer_t::offsetsAndSizes) + ofs), len 
Q_CONSTINIT static const qt_meta_stringdata_ResultViewer_t qt_meta_stringdata_ResultViewer = {
    {
        QT_MOC_LITERAL(0, 12),  // "ResultViewer"
        QT_MOC_LITERAL(13, 9),  // "addResult"
        QT_MOC_LITERAL(23, 0),  // ""
        QT_MOC_LITERAL(24, 4),  // "name"
        QT_MOC_LITERAL(29, 7),  // "success"
        QT_MOC_LITERAL(37, 6),  // "result"
        QT_MOC_LITERAL(44, 12),  // "clearResults"
        QT_MOC_LITERAL(57, 13),  // "exportResults"
        QT_MOC_LITERAL(71, 16),  // "onResultSelected"
        QT_MOC_LITERAL(88, 3),  // "row"
        QT_MOC_LITERAL(92, 6),  // "column"
        QT_MOC_LITERAL(99, 20),  // "onClearButtonClicked"
        QT_MOC_LITERAL(120, 21)   // "onExportButtonClicked"
    },
    "ResultViewer",
    "addResult",
    "",
    "name",
    "success",
    "result",
    "clearResults",
    "exportResults",
    "onResultSelected",
    "row",
    "column",
    "onClearButtonClicked",
    "onExportButtonClicked"
};
#undef QT_MOC_LITERAL
} // unnamed namespace

Q_CONSTINIT static const uint qt_meta_data_ResultViewer[] = {

 // content:
      10,       // revision
       0,       // classname
       0,    0, // classinfo
       6,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       0,       // signalCount

 // slots: name, argc, parameters, tag, flags, initial metatype offsets
       1,    3,   50,    2, 0x0a,    1 /* Public */,
       6,    0,   57,    2, 0x0a,    5 /* Public */,
       7,    0,   58,    2, 0x0a,    6 /* Public */,
       8,    2,   59,    2, 0x08,    7 /* Private */,
      11,    0,   64,    2, 0x08,   10 /* Private */,
      12,    0,   65,    2, 0x08,   11 /* Private */,

 // slots: parameters
    QMetaType::Void, QMetaType::QString, QMetaType::Bool, QMetaType::QString,    3,    4,    5,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void, QMetaType::Int, QMetaType::Int,    9,   10,
    QMetaType::Void,
    QMetaType::Void,

       0        // eod
};

Q_CONSTINIT const QMetaObject ResultViewer::staticMetaObject = { {
    QMetaObject::SuperData::link<QWidget::staticMetaObject>(),
    qt_meta_stringdata_ResultViewer.offsetsAndSizes,
    qt_meta_data_ResultViewer,
    qt_static_metacall,
    nullptr,
    qt_incomplete_metaTypeArray<qt_meta_stringdata_ResultViewer_t,
        // Q_OBJECT / Q_GADGET
        QtPrivate::TypeAndForceComplete<ResultViewer, std::true_type>,
        // method 'addResult'
        QtPrivate::TypeAndForceComplete<void, std::false_type>,
        QtPrivate::TypeAndForceComplete<const QString &, std::false_type>,
        QtPrivate::TypeAndForceComplete<bool, std::false_type>,
        QtPrivate::TypeAndForceComplete<const QString &, std::false_type>,
        // method 'clearResults'
        QtPrivate::TypeAndForceComplete<void, std::false_type>,
        // method 'exportResults'
        QtPrivate::TypeAndForceComplete<void, std::false_type>,
        // method 'onResultSelected'
        QtPrivate::TypeAndForceComplete<void, std::false_type>,
        QtPrivate::TypeAndForceComplete<int, std::false_type>,
        QtPrivate::TypeAndForceComplete<int, std::false_type>,
        // method 'onClearButtonClicked'
        QtPrivate::TypeAndForceComplete<void, std::false_type>,
        // method 'onExportButtonClicked'
        QtPrivate::TypeAndForceComplete<void, std::false_type>
    >,
    nullptr
} };

void ResultViewer::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        auto *_t = static_cast<ResultViewer *>(_o);
        (void)_t;
        switch (_id) {
        case 0: _t->addResult((*reinterpret_cast< std::add_pointer_t<QString>>(_a[1])),(*reinterpret_cast< std::add_pointer_t<bool>>(_a[2])),(*reinterpret_cast< std::add_pointer_t<QString>>(_a[3]))); break;
        case 1: _t->clearResults(); break;
        case 2: _t->exportResults(); break;
        case 3: _t->onResultSelected((*reinterpret_cast< std::add_pointer_t<int>>(_a[1])),(*reinterpret_cast< std::add_pointer_t<int>>(_a[2]))); break;
        case 4: _t->onClearButtonClicked(); break;
        case 5: _t->onExportButtonClicked(); break;
        default: ;
        }
    }
}

const QMetaObject *ResultViewer::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->dynamicMetaObject() : &staticMetaObject;
}

void *ResultViewer::qt_metacast(const char *_clname)
{
    if (!_clname) return nullptr;
    if (!strcmp(_clname, qt_meta_stringdata_ResultViewer.stringdata0))
        return static_cast<void*>(this);
    return QWidget::qt_metacast(_clname);
}

int ResultViewer::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QWidget::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        if (_id < 6)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 6;
    } else if (_c == QMetaObject::RegisterMethodArgumentMetaType) {
        if (_id < 6)
            *reinterpret_cast<QMetaType *>(_a[0]) = QMetaType();
        _id -= 6;
    }
    return _id;
}
QT_WARNING_POP
QT_END_MOC_NAMESPACE

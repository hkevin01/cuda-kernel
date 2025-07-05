/****************************************************************************
** Meta object code from reading C++ file 'performance_widget.h'
**
** Created by: The Qt Meta Object Compiler version 68 (Qt 6.4.2)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include <memory>
#include "../../../../gui/performance_widget.h"
#include <QtGui/qtextcursor.h>
#include <QtCore/qmetatype.h>
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'performance_widget.h' doesn't include <QObject>."
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
struct qt_meta_stringdata_PerformanceWidget_t {
    uint offsetsAndSizes[22];
    char stringdata0[18];
    char stringdata1[12];
    char stringdata2[1];
    char stringdata3[5];
    char stringdata4[11];
    char stringdata5[16];
    char stringdata6[15];
    char stringdata7[21];
    char stringdata8[20];
    char stringdata9[21];
    char stringdata10[14];
};
#define QT_MOC_LITERAL(ofs, len) \
    uint(sizeof(qt_meta_stringdata_PerformanceWidget_t::offsetsAndSizes) + ofs), len 
Q_CONSTINIT static const qt_meta_stringdata_PerformanceWidget_t qt_meta_stringdata_PerformanceWidget = {
    {
        QT_MOC_LITERAL(0, 17),  // "PerformanceWidget"
        QT_MOC_LITERAL(18, 11),  // "dataUpdated"
        QT_MOC_LITERAL(30, 0),  // ""
        QT_MOC_LITERAL(31, 4),  // "data"
        QT_MOC_LITERAL(36, 10),  // "updateData"
        QT_MOC_LITERAL(47, 15),  // "startMonitoring"
        QT_MOC_LITERAL(63, 14),  // "stopMonitoring"
        QT_MOC_LITERAL(78, 20),  // "onStartButtonClicked"
        QT_MOC_LITERAL(99, 19),  // "onStopButtonClicked"
        QT_MOC_LITERAL(119, 20),  // "onClearButtonClicked"
        QT_MOC_LITERAL(140, 13)   // "onUpdateTimer"
    },
    "PerformanceWidget",
    "dataUpdated",
    "",
    "data",
    "updateData",
    "startMonitoring",
    "stopMonitoring",
    "onStartButtonClicked",
    "onStopButtonClicked",
    "onClearButtonClicked",
    "onUpdateTimer"
};
#undef QT_MOC_LITERAL
} // unnamed namespace

Q_CONSTINIT static const uint qt_meta_data_PerformanceWidget[] = {

 // content:
      10,       // revision
       0,       // classname
       0,    0, // classinfo
       8,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       1,       // signalCount

 // signals: name, argc, parameters, tag, flags, initial metatype offsets
       1,    1,   62,    2, 0x06,    1 /* Public */,

 // slots: name, argc, parameters, tag, flags, initial metatype offsets
       4,    1,   65,    2, 0x0a,    3 /* Public */,
       5,    0,   68,    2, 0x0a,    5 /* Public */,
       6,    0,   69,    2, 0x0a,    6 /* Public */,
       7,    0,   70,    2, 0x08,    7 /* Private */,
       8,    0,   71,    2, 0x08,    8 /* Private */,
       9,    0,   72,    2, 0x08,    9 /* Private */,
      10,    0,   73,    2, 0x08,   10 /* Private */,

 // signals: parameters
    QMetaType::Void, QMetaType::QVariantMap,    3,

 // slots: parameters
    QMetaType::Void, QMetaType::QVariantMap,    3,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,

       0        // eod
};

Q_CONSTINIT const QMetaObject PerformanceWidget::staticMetaObject = { {
    QMetaObject::SuperData::link<QWidget::staticMetaObject>(),
    qt_meta_stringdata_PerformanceWidget.offsetsAndSizes,
    qt_meta_data_PerformanceWidget,
    qt_static_metacall,
    nullptr,
    qt_incomplete_metaTypeArray<qt_meta_stringdata_PerformanceWidget_t,
        // Q_OBJECT / Q_GADGET
        QtPrivate::TypeAndForceComplete<PerformanceWidget, std::true_type>,
        // method 'dataUpdated'
        QtPrivate::TypeAndForceComplete<void, std::false_type>,
        QtPrivate::TypeAndForceComplete<const QVariantMap &, std::false_type>,
        // method 'updateData'
        QtPrivate::TypeAndForceComplete<void, std::false_type>,
        QtPrivate::TypeAndForceComplete<const QVariantMap &, std::false_type>,
        // method 'startMonitoring'
        QtPrivate::TypeAndForceComplete<void, std::false_type>,
        // method 'stopMonitoring'
        QtPrivate::TypeAndForceComplete<void, std::false_type>,
        // method 'onStartButtonClicked'
        QtPrivate::TypeAndForceComplete<void, std::false_type>,
        // method 'onStopButtonClicked'
        QtPrivate::TypeAndForceComplete<void, std::false_type>,
        // method 'onClearButtonClicked'
        QtPrivate::TypeAndForceComplete<void, std::false_type>,
        // method 'onUpdateTimer'
        QtPrivate::TypeAndForceComplete<void, std::false_type>
    >,
    nullptr
} };

void PerformanceWidget::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        auto *_t = static_cast<PerformanceWidget *>(_o);
        (void)_t;
        switch (_id) {
        case 0: _t->dataUpdated((*reinterpret_cast< std::add_pointer_t<QVariantMap>>(_a[1]))); break;
        case 1: _t->updateData((*reinterpret_cast< std::add_pointer_t<QVariantMap>>(_a[1]))); break;
        case 2: _t->startMonitoring(); break;
        case 3: _t->stopMonitoring(); break;
        case 4: _t->onStartButtonClicked(); break;
        case 5: _t->onStopButtonClicked(); break;
        case 6: _t->onClearButtonClicked(); break;
        case 7: _t->onUpdateTimer(); break;
        default: ;
        }
    } else if (_c == QMetaObject::IndexOfMethod) {
        int *result = reinterpret_cast<int *>(_a[0]);
        {
            using _t = void (PerformanceWidget::*)(const QVariantMap & );
            if (_t _q_method = &PerformanceWidget::dataUpdated; *reinterpret_cast<_t *>(_a[1]) == _q_method) {
                *result = 0;
                return;
            }
        }
    }
}

const QMetaObject *PerformanceWidget::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->dynamicMetaObject() : &staticMetaObject;
}

void *PerformanceWidget::qt_metacast(const char *_clname)
{
    if (!_clname) return nullptr;
    if (!strcmp(_clname, qt_meta_stringdata_PerformanceWidget.stringdata0))
        return static_cast<void*>(this);
    return QWidget::qt_metacast(_clname);
}

int PerformanceWidget::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QWidget::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        if (_id < 8)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 8;
    } else if (_c == QMetaObject::RegisterMethodArgumentMetaType) {
        if (_id < 8)
            *reinterpret_cast<QMetaType *>(_a[0]) = QMetaType();
        _id -= 8;
    }
    return _id;
}

// SIGNAL 0
void PerformanceWidget::dataUpdated(const QVariantMap & _t1)
{
    void *_a[] = { nullptr, const_cast<void*>(reinterpret_cast<const void*>(std::addressof(_t1))) };
    QMetaObject::activate(this, &staticMetaObject, 0, _a);
}
QT_WARNING_POP
QT_END_MOC_NAMESPACE

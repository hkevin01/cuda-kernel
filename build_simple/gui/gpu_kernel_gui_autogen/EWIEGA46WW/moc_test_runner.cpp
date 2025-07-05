/****************************************************************************
** Meta object code from reading C++ file 'test_runner.h'
**
** Created by: The Qt Meta Object Compiler version 68 (Qt 6.4.2)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include <memory>
#include "../../../../gui/test_runner.h"
#include <QtGui/qtextcursor.h>
#include <QtCore/qmetatype.h>
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'test_runner.h' doesn't include <QObject>."
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
struct qt_meta_stringdata_TestRunner_t {
    uint offsetsAndSizes[44];
    char stringdata0[11];
    char stringdata1[13];
    char stringdata2[1];
    char stringdata3[9];
    char stringdata4[8];
    char stringdata5[7];
    char stringdata6[16];
    char stringdata7[6];
    char stringdata8[22];
    char stringdata9[27];
    char stringdata10[20];
    char stringdata11[23];
    char stringdata12[18];
    char stringdata13[9];
    char stringdata14[21];
    char stringdata15[11];
    char stringdata16[15];
    char stringdata17[23];
    char stringdata18[6];
    char stringdata19[16];
    char stringdata20[15];
    char stringdata21[12];
};
#define QT_MOC_LITERAL(ofs, len) \
    uint(sizeof(qt_meta_stringdata_TestRunner_t::offsetsAndSizes) + ofs), len 
Q_CONSTINIT static const qt_meta_stringdata_TestRunner_t qt_meta_stringdata_TestRunner = {
    {
        QT_MOC_LITERAL(0, 10),  // "TestRunner"
        QT_MOC_LITERAL(11, 12),  // "testFinished"
        QT_MOC_LITERAL(24, 0),  // ""
        QT_MOC_LITERAL(25, 8),  // "testName"
        QT_MOC_LITERAL(34, 7),  // "success"
        QT_MOC_LITERAL(42, 6),  // "result"
        QT_MOC_LITERAL(49, 15),  // "progressUpdated"
        QT_MOC_LITERAL(65, 5),  // "value"
        QT_MOC_LITERAL(71, 21),  // "onRunAllButtonClicked"
        QT_MOC_LITERAL(93, 26),  // "onRunSelectedButtonClicked"
        QT_MOC_LITERAL(120, 19),  // "onStopButtonClicked"
        QT_MOC_LITERAL(140, 22),  // "onTestSelectionChanged"
        QT_MOC_LITERAL(163, 17),  // "onProcessFinished"
        QT_MOC_LITERAL(181, 8),  // "exitCode"
        QT_MOC_LITERAL(190, 20),  // "QProcess::ExitStatus"
        QT_MOC_LITERAL(211, 10),  // "exitStatus"
        QT_MOC_LITERAL(222, 14),  // "onProcessError"
        QT_MOC_LITERAL(237, 22),  // "QProcess::ProcessError"
        QT_MOC_LITERAL(260, 5),  // "error"
        QT_MOC_LITERAL(266, 15),  // "onProcessOutput"
        QT_MOC_LITERAL(282, 14),  // "updateProgress"
        QT_MOC_LITERAL(297, 11)   // "runNextTest"
    },
    "TestRunner",
    "testFinished",
    "",
    "testName",
    "success",
    "result",
    "progressUpdated",
    "value",
    "onRunAllButtonClicked",
    "onRunSelectedButtonClicked",
    "onStopButtonClicked",
    "onTestSelectionChanged",
    "onProcessFinished",
    "exitCode",
    "QProcess::ExitStatus",
    "exitStatus",
    "onProcessError",
    "QProcess::ProcessError",
    "error",
    "onProcessOutput",
    "updateProgress",
    "runNextTest"
};
#undef QT_MOC_LITERAL
} // unnamed namespace

Q_CONSTINIT static const uint qt_meta_data_TestRunner[] = {

 // content:
      10,       // revision
       0,       // classname
       0,    0, // classinfo
      11,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       2,       // signalCount

 // signals: name, argc, parameters, tag, flags, initial metatype offsets
       1,    3,   80,    2, 0x06,    1 /* Public */,
       6,    1,   87,    2, 0x06,    5 /* Public */,

 // slots: name, argc, parameters, tag, flags, initial metatype offsets
       8,    0,   90,    2, 0x08,    7 /* Private */,
       9,    0,   91,    2, 0x08,    8 /* Private */,
      10,    0,   92,    2, 0x08,    9 /* Private */,
      11,    0,   93,    2, 0x08,   10 /* Private */,
      12,    2,   94,    2, 0x08,   11 /* Private */,
      16,    1,   99,    2, 0x08,   14 /* Private */,
      19,    0,  102,    2, 0x08,   16 /* Private */,
      20,    0,  103,    2, 0x08,   17 /* Private */,
      21,    0,  104,    2, 0x08,   18 /* Private */,

 // signals: parameters
    QMetaType::Void, QMetaType::QString, QMetaType::Bool, QMetaType::QString,    3,    4,    5,
    QMetaType::Void, QMetaType::Int,    7,

 // slots: parameters
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void, QMetaType::Int, 0x80000000 | 14,   13,   15,
    QMetaType::Void, 0x80000000 | 17,   18,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,

       0        // eod
};

Q_CONSTINIT const QMetaObject TestRunner::staticMetaObject = { {
    QMetaObject::SuperData::link<QWidget::staticMetaObject>(),
    qt_meta_stringdata_TestRunner.offsetsAndSizes,
    qt_meta_data_TestRunner,
    qt_static_metacall,
    nullptr,
    qt_incomplete_metaTypeArray<qt_meta_stringdata_TestRunner_t,
        // Q_OBJECT / Q_GADGET
        QtPrivate::TypeAndForceComplete<TestRunner, std::true_type>,
        // method 'testFinished'
        QtPrivate::TypeAndForceComplete<void, std::false_type>,
        QtPrivate::TypeAndForceComplete<const QString &, std::false_type>,
        QtPrivate::TypeAndForceComplete<bool, std::false_type>,
        QtPrivate::TypeAndForceComplete<const QString &, std::false_type>,
        // method 'progressUpdated'
        QtPrivate::TypeAndForceComplete<void, std::false_type>,
        QtPrivate::TypeAndForceComplete<int, std::false_type>,
        // method 'onRunAllButtonClicked'
        QtPrivate::TypeAndForceComplete<void, std::false_type>,
        // method 'onRunSelectedButtonClicked'
        QtPrivate::TypeAndForceComplete<void, std::false_type>,
        // method 'onStopButtonClicked'
        QtPrivate::TypeAndForceComplete<void, std::false_type>,
        // method 'onTestSelectionChanged'
        QtPrivate::TypeAndForceComplete<void, std::false_type>,
        // method 'onProcessFinished'
        QtPrivate::TypeAndForceComplete<void, std::false_type>,
        QtPrivate::TypeAndForceComplete<int, std::false_type>,
        QtPrivate::TypeAndForceComplete<QProcess::ExitStatus, std::false_type>,
        // method 'onProcessError'
        QtPrivate::TypeAndForceComplete<void, std::false_type>,
        QtPrivate::TypeAndForceComplete<QProcess::ProcessError, std::false_type>,
        // method 'onProcessOutput'
        QtPrivate::TypeAndForceComplete<void, std::false_type>,
        // method 'updateProgress'
        QtPrivate::TypeAndForceComplete<void, std::false_type>,
        // method 'runNextTest'
        QtPrivate::TypeAndForceComplete<void, std::false_type>
    >,
    nullptr
} };

void TestRunner::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        auto *_t = static_cast<TestRunner *>(_o);
        (void)_t;
        switch (_id) {
        case 0: _t->testFinished((*reinterpret_cast< std::add_pointer_t<QString>>(_a[1])),(*reinterpret_cast< std::add_pointer_t<bool>>(_a[2])),(*reinterpret_cast< std::add_pointer_t<QString>>(_a[3]))); break;
        case 1: _t->progressUpdated((*reinterpret_cast< std::add_pointer_t<int>>(_a[1]))); break;
        case 2: _t->onRunAllButtonClicked(); break;
        case 3: _t->onRunSelectedButtonClicked(); break;
        case 4: _t->onStopButtonClicked(); break;
        case 5: _t->onTestSelectionChanged(); break;
        case 6: _t->onProcessFinished((*reinterpret_cast< std::add_pointer_t<int>>(_a[1])),(*reinterpret_cast< std::add_pointer_t<QProcess::ExitStatus>>(_a[2]))); break;
        case 7: _t->onProcessError((*reinterpret_cast< std::add_pointer_t<QProcess::ProcessError>>(_a[1]))); break;
        case 8: _t->onProcessOutput(); break;
        case 9: _t->updateProgress(); break;
        case 10: _t->runNextTest(); break;
        default: ;
        }
    } else if (_c == QMetaObject::IndexOfMethod) {
        int *result = reinterpret_cast<int *>(_a[0]);
        {
            using _t = void (TestRunner::*)(const QString & , bool , const QString & );
            if (_t _q_method = &TestRunner::testFinished; *reinterpret_cast<_t *>(_a[1]) == _q_method) {
                *result = 0;
                return;
            }
        }
        {
            using _t = void (TestRunner::*)(int );
            if (_t _q_method = &TestRunner::progressUpdated; *reinterpret_cast<_t *>(_a[1]) == _q_method) {
                *result = 1;
                return;
            }
        }
    }
}

const QMetaObject *TestRunner::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->dynamicMetaObject() : &staticMetaObject;
}

void *TestRunner::qt_metacast(const char *_clname)
{
    if (!_clname) return nullptr;
    if (!strcmp(_clname, qt_meta_stringdata_TestRunner.stringdata0))
        return static_cast<void*>(this);
    return QWidget::qt_metacast(_clname);
}

int TestRunner::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QWidget::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        if (_id < 11)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 11;
    } else if (_c == QMetaObject::RegisterMethodArgumentMetaType) {
        if (_id < 11)
            *reinterpret_cast<QMetaType *>(_a[0]) = QMetaType();
        _id -= 11;
    }
    return _id;
}

// SIGNAL 0
void TestRunner::testFinished(const QString & _t1, bool _t2, const QString & _t3)
{
    void *_a[] = { nullptr, const_cast<void*>(reinterpret_cast<const void*>(std::addressof(_t1))), const_cast<void*>(reinterpret_cast<const void*>(std::addressof(_t2))), const_cast<void*>(reinterpret_cast<const void*>(std::addressof(_t3))) };
    QMetaObject::activate(this, &staticMetaObject, 0, _a);
}

// SIGNAL 1
void TestRunner::progressUpdated(int _t1)
{
    void *_a[] = { nullptr, const_cast<void*>(reinterpret_cast<const void*>(std::addressof(_t1))) };
    QMetaObject::activate(this, &staticMetaObject, 1, _a);
}
QT_WARNING_POP
QT_END_MOC_NAMESPACE

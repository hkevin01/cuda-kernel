/****************************************************************************
** Meta object code from reading C++ file 'kernel_runner.h'
**
** Created by: The Qt Meta Object Compiler version 68 (Qt 6.4.2)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include <memory>
#include "../../../../gui/kernel_runner.h"
#include <QtGui/qtextcursor.h>
#include <QtCore/qmetatype.h>
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'kernel_runner.h' doesn't include <QObject>."
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
struct qt_meta_stringdata_KernelRunner_t {
    uint offsetsAndSizes[40];
    char stringdata0[13];
    char stringdata1[15];
    char stringdata2[1];
    char stringdata3[11];
    char stringdata4[8];
    char stringdata5[7];
    char stringdata6[16];
    char stringdata7[6];
    char stringdata8[19];
    char stringdata9[23];
    char stringdata10[25];
    char stringdata11[18];
    char stringdata12[9];
    char stringdata13[21];
    char stringdata14[11];
    char stringdata15[15];
    char stringdata16[23];
    char stringdata17[6];
    char stringdata18[16];
    char stringdata19[15];
};
#define QT_MOC_LITERAL(ofs, len) \
    uint(sizeof(qt_meta_stringdata_KernelRunner_t::offsetsAndSizes) + ofs), len 
Q_CONSTINIT static const qt_meta_stringdata_KernelRunner_t qt_meta_stringdata_KernelRunner = {
    {
        QT_MOC_LITERAL(0, 12),  // "KernelRunner"
        QT_MOC_LITERAL(13, 14),  // "kernelFinished"
        QT_MOC_LITERAL(28, 0),  // ""
        QT_MOC_LITERAL(29, 10),  // "kernelName"
        QT_MOC_LITERAL(40, 7),  // "success"
        QT_MOC_LITERAL(48, 6),  // "result"
        QT_MOC_LITERAL(55, 15),  // "progressUpdated"
        QT_MOC_LITERAL(71, 5),  // "value"
        QT_MOC_LITERAL(77, 18),  // "onRunButtonClicked"
        QT_MOC_LITERAL(96, 22),  // "onRefreshButtonClicked"
        QT_MOC_LITERAL(119, 24),  // "onKernelSelectionChanged"
        QT_MOC_LITERAL(144, 17),  // "onProcessFinished"
        QT_MOC_LITERAL(162, 8),  // "exitCode"
        QT_MOC_LITERAL(171, 20),  // "QProcess::ExitStatus"
        QT_MOC_LITERAL(192, 10),  // "exitStatus"
        QT_MOC_LITERAL(203, 14),  // "onProcessError"
        QT_MOC_LITERAL(218, 22),  // "QProcess::ProcessError"
        QT_MOC_LITERAL(241, 5),  // "error"
        QT_MOC_LITERAL(247, 15),  // "onProcessOutput"
        QT_MOC_LITERAL(263, 14)   // "updateProgress"
    },
    "KernelRunner",
    "kernelFinished",
    "",
    "kernelName",
    "success",
    "result",
    "progressUpdated",
    "value",
    "onRunButtonClicked",
    "onRefreshButtonClicked",
    "onKernelSelectionChanged",
    "onProcessFinished",
    "exitCode",
    "QProcess::ExitStatus",
    "exitStatus",
    "onProcessError",
    "QProcess::ProcessError",
    "error",
    "onProcessOutput",
    "updateProgress"
};
#undef QT_MOC_LITERAL
} // unnamed namespace

Q_CONSTINIT static const uint qt_meta_data_KernelRunner[] = {

 // content:
      10,       // revision
       0,       // classname
       0,    0, // classinfo
       9,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       2,       // signalCount

 // signals: name, argc, parameters, tag, flags, initial metatype offsets
       1,    3,   68,    2, 0x06,    1 /* Public */,
       6,    1,   75,    2, 0x06,    5 /* Public */,

 // slots: name, argc, parameters, tag, flags, initial metatype offsets
       8,    0,   78,    2, 0x08,    7 /* Private */,
       9,    0,   79,    2, 0x08,    8 /* Private */,
      10,    0,   80,    2, 0x08,    9 /* Private */,
      11,    2,   81,    2, 0x08,   10 /* Private */,
      15,    1,   86,    2, 0x08,   13 /* Private */,
      18,    0,   89,    2, 0x08,   15 /* Private */,
      19,    0,   90,    2, 0x08,   16 /* Private */,

 // signals: parameters
    QMetaType::Void, QMetaType::QString, QMetaType::Bool, QMetaType::QString,    3,    4,    5,
    QMetaType::Void, QMetaType::Int,    7,

 // slots: parameters
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void, QMetaType::Int, 0x80000000 | 13,   12,   14,
    QMetaType::Void, 0x80000000 | 16,   17,
    QMetaType::Void,
    QMetaType::Void,

       0        // eod
};

Q_CONSTINIT const QMetaObject KernelRunner::staticMetaObject = { {
    QMetaObject::SuperData::link<QWidget::staticMetaObject>(),
    qt_meta_stringdata_KernelRunner.offsetsAndSizes,
    qt_meta_data_KernelRunner,
    qt_static_metacall,
    nullptr,
    qt_incomplete_metaTypeArray<qt_meta_stringdata_KernelRunner_t,
        // Q_OBJECT / Q_GADGET
        QtPrivate::TypeAndForceComplete<KernelRunner, std::true_type>,
        // method 'kernelFinished'
        QtPrivate::TypeAndForceComplete<void, std::false_type>,
        QtPrivate::TypeAndForceComplete<const QString &, std::false_type>,
        QtPrivate::TypeAndForceComplete<bool, std::false_type>,
        QtPrivate::TypeAndForceComplete<const QString &, std::false_type>,
        // method 'progressUpdated'
        QtPrivate::TypeAndForceComplete<void, std::false_type>,
        QtPrivate::TypeAndForceComplete<int, std::false_type>,
        // method 'onRunButtonClicked'
        QtPrivate::TypeAndForceComplete<void, std::false_type>,
        // method 'onRefreshButtonClicked'
        QtPrivate::TypeAndForceComplete<void, std::false_type>,
        // method 'onKernelSelectionChanged'
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
        QtPrivate::TypeAndForceComplete<void, std::false_type>
    >,
    nullptr
} };

void KernelRunner::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        auto *_t = static_cast<KernelRunner *>(_o);
        (void)_t;
        switch (_id) {
        case 0: _t->kernelFinished((*reinterpret_cast< std::add_pointer_t<QString>>(_a[1])),(*reinterpret_cast< std::add_pointer_t<bool>>(_a[2])),(*reinterpret_cast< std::add_pointer_t<QString>>(_a[3]))); break;
        case 1: _t->progressUpdated((*reinterpret_cast< std::add_pointer_t<int>>(_a[1]))); break;
        case 2: _t->onRunButtonClicked(); break;
        case 3: _t->onRefreshButtonClicked(); break;
        case 4: _t->onKernelSelectionChanged(); break;
        case 5: _t->onProcessFinished((*reinterpret_cast< std::add_pointer_t<int>>(_a[1])),(*reinterpret_cast< std::add_pointer_t<QProcess::ExitStatus>>(_a[2]))); break;
        case 6: _t->onProcessError((*reinterpret_cast< std::add_pointer_t<QProcess::ProcessError>>(_a[1]))); break;
        case 7: _t->onProcessOutput(); break;
        case 8: _t->updateProgress(); break;
        default: ;
        }
    } else if (_c == QMetaObject::IndexOfMethod) {
        int *result = reinterpret_cast<int *>(_a[0]);
        {
            using _t = void (KernelRunner::*)(const QString & , bool , const QString & );
            if (_t _q_method = &KernelRunner::kernelFinished; *reinterpret_cast<_t *>(_a[1]) == _q_method) {
                *result = 0;
                return;
            }
        }
        {
            using _t = void (KernelRunner::*)(int );
            if (_t _q_method = &KernelRunner::progressUpdated; *reinterpret_cast<_t *>(_a[1]) == _q_method) {
                *result = 1;
                return;
            }
        }
    }
}

const QMetaObject *KernelRunner::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->dynamicMetaObject() : &staticMetaObject;
}

void *KernelRunner::qt_metacast(const char *_clname)
{
    if (!_clname) return nullptr;
    if (!strcmp(_clname, qt_meta_stringdata_KernelRunner.stringdata0))
        return static_cast<void*>(this);
    return QWidget::qt_metacast(_clname);
}

int KernelRunner::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QWidget::qt_metacall(_c, _id, _a);
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

// SIGNAL 0
void KernelRunner::kernelFinished(const QString & _t1, bool _t2, const QString & _t3)
{
    void *_a[] = { nullptr, const_cast<void*>(reinterpret_cast<const void*>(std::addressof(_t1))), const_cast<void*>(reinterpret_cast<const void*>(std::addressof(_t2))), const_cast<void*>(reinterpret_cast<const void*>(std::addressof(_t3))) };
    QMetaObject::activate(this, &staticMetaObject, 0, _a);
}

// SIGNAL 1
void KernelRunner::progressUpdated(int _t1)
{
    void *_a[] = { nullptr, const_cast<void*>(reinterpret_cast<const void*>(std::addressof(_t1))) };
    QMetaObject::activate(this, &staticMetaObject, 1, _a);
}
QT_WARNING_POP
QT_END_MOC_NAMESPACE

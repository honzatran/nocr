#include "mainwindow.h"
#include <QApplication>
#include <QMessageBox>

#include <opencv2/core/core.hpp>
#include <QString>
#include <QDebug>

#include <string>

void fatalErrorMsg(const QString &msg)
{
    QMessageBox error_box(QMessageBox::Critical, "Fatal error",
                          "NOCRGui has encountered an error and cannot continue to work.\n"
                             "Please press OK button to quit." );
    error_box.setDetailedText(msg);
    error_box.exec();
}

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    MainWindow w;
    try
    {
        w.initializeApp();
    }
    catch (std::exception &ex)
    {
        std::string msg = ex.what();
        fatalErrorMsg(QString::fromStdString(msg));
        return 1;
    }
    w.show();

    return a.exec();
}

#ifndef NEWWORDSDIALOG_H
#define NEWWORDSDIALOG_H

#include <QDialog>
#include <QRegExpValidator>
#include <QStringList>
#include <QKeyEvent>
#include <QSizePolicy>

namespace Ui {
class NewWordsDialog;
}

class NewWordsDialog : public QDialog
{
    Q_OBJECT
    
public:
    explicit NewWordsDialog(QWidget *parent = 0);
    ~NewWordsDialog();
    
private:
    Ui::NewWordsDialog *ui;
    QStringList string_list_;

private slots:
    void acceptNewWords();
signals:
    void newWordsAccepted(QStringList new_words);


protected:
    void keyPressEvent(QKeyEvent *event);



};

#endif // NEWWORDSDIALOG_H

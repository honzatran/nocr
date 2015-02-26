#include "newwordsdialog.h"
#include "ui_newwordsdialog.h"

#include <QDebug>

NewWordsDialog::NewWordsDialog(QWidget *parent) :
    QDialog(parent),
    ui(new Ui::NewWordsDialog)
{
    ui->setupUi(this);
    QRegExp  regexp("^(\\w+\\s+)*\\w+$");
//    QRegExp  regexp("^\\d+$");
    QValidator *validator = new QRegExpValidator(regexp,this);

    ui->word_input_->setValidator(validator);
    connect(ui->cancel_button_, &QPushButton::clicked, this, &NewWordsDialog::reject);
    connect(ui->Ok_button, &QPushButton::clicked,this, &NewWordsDialog::acceptNewWords);
    connect(ui->word_input_, &QLineEdit::returnPressed, ui->Ok_button, &QPushButton::click );
    setWindowTitle("Add new words to dictionary");
    ui->gridWidget->setSizePolicy(QSizePolicy::Fixed,QSizePolicy::Expanding);
    setSizePolicy(QSizePolicy::Fixed, QSizePolicy::Expanding);

}

NewWordsDialog::~NewWordsDialog()
{
    delete ui;
}

void NewWordsDialog::acceptNewWords()
{
    QString text = ui->word_input_->text().trimmed();
    if ( text.isEmpty() )
    {
        return;
    }
    QStringList words = text.split(QRegExp("\\s+"));

    done(0);
    emit newWordsAccepted(words);
    ui->word_input_->clear();
}


void NewWordsDialog::keyPressEvent(QKeyEvent *event)
{
    int key_code = event->key();
    if ( key_code == Qt::Key_Return || key_code == Qt::Key_Enter )
    {
        return;
    }
}

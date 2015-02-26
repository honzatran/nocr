#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QAction>
#include <memory>
#include <QFileDialog>
#include <QGraphicsView>
#include <QTextEdit>
#include <QMessageBox>
#include <QThread>
#include <QListWidget>

#include <nocrlib/dictionary.h>
#include <nocrlib/word_generator.h>



#include "imageworker.h"
#include "translationrecord.h"
#include "newwordsdialog.h"

namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT
    
public:
    explicit MainWindow(QWidget *parent = 0);
    ~MainWindow();

    void initializeApp();
    
private:
    void setUp();


    void createMenu();
    void setUpDictMenu( QMenu *dict_menu );


    QActionGroup *action_group_;

    QGraphicsView *image_viewer_;
    QTextEdit *output_display_;
    QMessageBox *message_box_;
    QVector<TranslationRecord> records_;
    Ui::MainWindow *ui_;

    ImageWorker image_worker_;
    Dictionary dictionary_;
    QThread *worker_thread_;

    QFileDialog* file_dialog_;
    QWidget* createCentralLayout();


    QPushButton *next_, *prev_;
    QLabel *current_image_name_;
    NewWordsDialog dialog;

    void setCurrentRecord( TranslationRecord &record );
    void showImage( QGraphicsScene * scene );
    int current_item_;

    void showMsgBox(const QString &new_msg);
    QString msg = "reading text from image";
    const QString title = "NOCR GUI";
    const QString k_dict = "Dictionary";
    const QString dict_type = "Dictionary files(*.dict)";
    const QString k_new_word_add = "Adding new words to the dictionary ";
    const QString k_new_word_load = "Loading new words from file ";
    const QString k_new_dict_load = "Loading new dictionary from file ";
    const QString k_save_dict = "Saving dictionary to ";
    const QString k_no_image = "No image loaded";

    void setUpWorker();

protected:
    void resizeEvent(QResizeEvent *resize_event) override;


private slots:
    void openMenu();
    void readImages( const QStringList &files );
    void handleResult();
    void changeFile(const QString &image);

    void movePrev();
    void moveNext();

    void initialize();

    void exportXml();
    void fatalErrorMsg( const QString &msg );
signals:
    void startReadingMultiple( const QStringList &files );

    void loadNewDict( const QString &dict_file );
    void loadWords( const QString &dict_file );
    void saveDictionary( const QString &dict_file );
};

#endif // MAINWINDOW_H

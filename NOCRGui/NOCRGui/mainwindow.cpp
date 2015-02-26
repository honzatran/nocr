#include "mainwindow.h"
#include "ui_mainwindow.h"
#include "convertoropencv.h"

#include <iostream>
#include <string>


#include <QDebug>
#include <QPixmap>
#include <QGraphicsPixmapItem>
#include <QLabel>
#include <QHBoxLayout>
#include <QDockWidget>
#include <QPushButton>
#include <QDialogButtonBox>
#include <QTextEdit>
#include <QInputDialog>
#include <QApplication>

#include "xmlcreator.h"

#include <opencv2/core/core.hpp>

#include <nocrlib/utilities.h>
#include <nocrlib/ocr.h>


using namespace std;

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui_(new Ui::MainWindow)
{
    ui_->setupUi(this);
}

MainWindow::~MainWindow()
{
    delete ui_;
    worker_thread_->quit();
}

void MainWindow::initializeApp()
{

    action_group_ = new QActionGroup(this);
    createMenu();
    worker_thread_ = new QThread(this);
    setUp();
    worker_thread_->start();
    connect(worker_thread_, &QThread::finished, worker_thread_, &QThread::deleteLater);
    output_display_->setReadOnly(true);
    setWindowTitle(title);
}

void MainWindow::setUp()
{
    file_dialog_ = new QFileDialog(this,"Open image",
                                        QDir::homePath(), tr("Images (*.jpg *.png *.bmp)"));
    file_dialog_->setViewMode(QFileDialog::Detail);
    file_dialog_->setFileMode(QFileDialog::ExistingFiles);
    connect(file_dialog_, &QFileDialog::filesSelected, this, &MainWindow::readImages);
    connect( this, &MainWindow::startReadingMultiple, &image_worker_, &ImageWorker::processImages );
    connect(&image_worker_, &ImageWorker::readingDone, this, &MainWindow::handleResult );

    connect(&dialog, &NewWordsDialog::newWordsAccepted, [this] (const QStringList &list)
    {
        showMsgBox(k_new_word_add);
        image_worker_.addNewWords(list);
    });

    connect(this, &MainWindow::loadNewDict, &image_worker_, &ImageWorker::loadNewDict);
    connect(this, &MainWindow::loadWords, &image_worker_, &ImageWorker::addWordsFile);
    connect(this, &MainWindow::saveDictionary, &image_worker_, &ImageWorker::saveDictToFile );

    connect(&image_worker_, &ImageWorker::operationDone, [this] ()
    {
        message_box_->done(5);
    });


    QWidget *central_widget = createCentralLayout();
    setCentralWidget(central_widget);
    setUpWorker();



    message_box_->setStandardButtons(QMessageBox::NoButton);
    initialize();
}

void MainWindow::createMenu()
{
    QAction *open_action = new QAction(tr("Open image"),this);
    connect(open_action,&QAction::triggered, this, &MainWindow::openMenu);

    QAction* clear = new QAction(tr("Clear results"),this);
    connect( clear, &QAction::triggered, [this] ()
        { records_.clear(); initialize(); } );

    QAction* xml = new QAction(tr("Export to XML"), this);
    connect( xml, &QAction::triggered, this, &MainWindow::exportXml );

    QAction* exit = new QAction(tr("Exit"), this);
    connect( exit, &QAction::triggered, this, &MainWindow::close );

    action_group_->addAction(clear);
    action_group_->addAction(xml);

    const QString k_main_menu = "Main Menu";

    //=================== dictionary set up=================
    auto main_menu = menuBar()->addMenu(k_main_menu);
    main_menu->addAction(open_action);
    auto dict_menu = main_menu->addMenu(k_dict);
    setUpDictMenu(dict_menu);
    main_menu->addSeparator();
    main_menu->addAction(clear);
    main_menu->addAction(xml);
    main_menu->addSeparator();
    main_menu->addAction(exit);

}

void MainWindow::setUpDictMenu(QMenu *dict_menu)
{
    QAction *load_new_dict = new QAction(tr("Load new dictionary"),this);
    QAction *load_words_file = new QAction(tr("Load new words from file"), this);
    QAction *add_words = new QAction(tr("Add words to dictionary"),this);
    QAction *save_dict = new QAction(tr("Save current dictionary"), this);


    connect( load_new_dict, &QAction::triggered, [this] ()
    {
        QString dict_file = QFileDialog::getOpenFileName(this,
                        "Open new dictionary file", QDir::homePath(),
                                                        dict_type);
        if ( dict_file.isEmpty() )
        {
            return;
        }

        showMsgBox( k_new_dict_load + dict_file );
        emit loadNewDict(dict_file);
    });

    connect( load_words_file, &QAction::triggered, [this] ()
    {
        QString dict_file = QFileDialog::getOpenFileName(this,
                        "Open file with new words", QDir::homePath(),
                                                         dict_type );

        if ( dict_file.isEmpty() )
        {
            return;
        }

        showMsgBox(k_new_word_load + dict_file );
        emit loadWords(dict_file);
    });

    connect( save_dict, &QAction::triggered, [this] ()
    {
        QString init_file = QDir::home().absoluteFilePath("untitled.dict");
        QString save_file = QFileDialog::getSaveFileName(this, "Save Dictionary",
                                                          init_file, dict_type );
        if ( save_file.isEmpty() )
        {
            return;
        }

        showMsgBox(k_save_dict + save_file );
        emit saveDictionary(save_file);
    });

    connect( add_words, &QAction::triggered, &dialog, &NewWordsDialog::exec );

    dict_menu->addAction(add_words);
    dict_menu->addAction(load_new_dict);
    dict_menu->addAction(load_words_file);
    dict_menu->addAction(save_dict);
}

void MainWindow::openMenu()
{
    file_dialog_->setVisible(true);
}

void MainWindow::readImages(const QStringList &files)
{
    emit startReadingMultiple(files);
    message_box_->exec();
}


void MainWindow::handleResult()
{
    QVector<TranslationRecord> results = image_worker_.getDetectedWords();
    image_worker_.clearDetection();

    if ( !action_group_->isEnabled() )
    {
        action_group_->setEnabled(true);
    }

    records_ += results;
    message_box_->done(5);
    current_item_ = records_.size() - 1;
    setCurrentRecord(records_[current_item_]);
    if (  records_.size() > 1 )
    {
        prev_->setEnabled(true);
    }
    next_->setEnabled(false);
}

void MainWindow::changeFile(const QString &image)
{
    message_box_->setText(msg + ' ' +  image);
}

QWidget *MainWindow::createCentralLayout()
{
    next_ = new QPushButton( "Next image", this );
    connect( next_,&QPushButton::clicked, this, &MainWindow::moveNext );

    prev_ = new QPushButton( "Previous image", this );
    connect( prev_ ,&QPushButton::clicked, this, &MainWindow::movePrev );

    image_viewer_ = new QGraphicsView(this);
    output_display_ = new QTextEdit(this);
    output_display_->setSizePolicy(QSizePolicy::Minimum,QSizePolicy::MinimumExpanding);
    next_->setSizePolicy(QSizePolicy::Minimum,QSizePolicy::Fixed);
    prev_->setSizePolicy(QSizePolicy::Minimum,QSizePolicy::Fixed);

    QGridLayout *grid_layout = new QGridLayout();


    QLabel *detected_text = new QLabel("Recognized words:",this);
    detected_text->setSizePolicy(QSizePolicy::Fixed, QSizePolicy::Fixed);
    current_image_name_ = new QLabel(k_no_image,this);
    current_image_name_->setSizePolicy(QSizePolicy::Fixed, QSizePolicy::Fixed);

    grid_layout->addWidget(current_image_name_,0,10,1,1);
    grid_layout->addWidget(prev_,1,10,1,1);
    grid_layout->addWidget(next_,2,10,1,1);
    grid_layout->addWidget(detected_text,3,10,1,1);
    grid_layout->addWidget(output_display_,4,10,6,1);
    grid_layout->addWidget(image_viewer_,0,0,10,10);


    QWidget *output = new QWidget(this);
    output->setLayout(grid_layout);

    message_box_ = new QMessageBox(this);


    return output;
}

void MainWindow::setCurrentRecord(TranslationRecord &record)
{
    QGraphicsScene *scene = record.getQGScene();
    QStringList words = record.getDetectedText();
    output_display_->clear();
    current_image_name_->setText(record.getFileName());

    for ( const QString &w : words )
    {
        output_display_->append(w);
    }

    showImage(scene);
}

void MainWindow::showImage(QGraphicsScene *scene)
{
    QRectF scene_rect = scene->itemsBoundingRect();
    image_viewer_->setScene( scene );
    image_viewer_->fitInView( scene_rect, Qt::KeepAspectRatio);
}

void MainWindow::showMsgBox(const QString &new_msg)
{
    message_box_->setText(new_msg);
    message_box_->open();
}

void MainWindow::setUpWorker()
{
    const std::string dict = "conf/dict";
    // const std::string boost_ER1Phase = "conf/boostGeom.conf";
    const std::string boost_ER1Phase = "conf/boost_er1stage.conf";
    // const std::string svm_ER2Phase = "conf/svmERGeom.conf";
    const std::string svm_ER2Phase = "conf/svm_er2stage.conf";
    const std::string svm_merge = "conf/svmMerge.conf";
    const std::string ocr_conf = "conf/iksvm.conf";

    image_worker_.loadConfiguration( dict, boost_ER1Phase, svm_ER2Phase,
                                     svm_merge);
    std::unique_ptr<AbstractOCR> ocr( new MyOCR(ocr_conf) );
    image_worker_.loadOcr(std::move(ocr));
    image_worker_.moveToThread(worker_thread_);

    connect(&image_worker_, &ImageWorker::newImage, this, &MainWindow::changeFile );
}

void MainWindow::movePrev()
{
    if ( current_item_ > 0 )
    {
        if ( current_item_ == records_.size() - 1 )
        {
            next_->setEnabled(true);
        }

        current_item_--;
        setCurrentRecord(records_[current_item_]);

        if ( current_item_ == 0 )
        {
            prev_->setEnabled(false);
        }
    }

}

void MainWindow::moveNext()
{
    int records_last = records_.size() - 1;
    if ( current_item_ < records_last )
    {
        if ( current_item_ == 0 )
        {
            prev_->setEnabled(true);
        }

        current_item_++;
        setCurrentRecord(records_[current_item_]);

        if ( current_item_ == records_last )
        {
            next_->setEnabled(false);
        }
    }
}

void MainWindow::initialize()
{
    prev_->setEnabled(false);
    next_->setEnabled(false);

    current_item_ = -1;
    current_image_name_->setText(k_no_image);
    output_display_->clear();

    action_group_->setEnabled(false);
}

void MainWindow::exportXml()
{

    QDir init_file = QDir::homePath();
    QString save_file = QFileDialog::getSaveFileName(this, tr("Save File"),
                                                     init_file.absoluteFilePath("untitled.xml")
                                                     ,tr("XML files(*.xml") );
    if ( save_file.isEmpty() )
    {
        return;
    }

    QFile xml_file( save_file );
    xml_file.open(QIODevice::WriteOnly| QIODevice::Text );
    QTextStream out( &xml_file );

    XmlCreator xml_creator;
    xml_creator.createDomTree(records_);
    out << "<?xml version=\"1.0\"?>\n";
    out << xml_creator.toString();
    xml_file.close();
}

void MainWindow::fatalErrorMsg(const QString &msg)
{
    QMessageBox error_box(QMessageBox::Critical, tr("Fatal error"),
                          tr("%1 has encountered an error and cannot continue to work.\n"
                             "Please press OK button to quit.").arg(qApp->applicationName()),
                          QMessageBox::Ok,this);

//    connect(&error_box, QMessageBox::buttonClicked, this, QMainWindow::close );
    error_box.setDetailedText(msg);
    error_box.exec();
    qApp->exit(1);

}

void MainWindow::resizeEvent(QResizeEvent *resize_event)
{
    if ( image_viewer_->scene() )
    {
        QRectF bounds = image_viewer_->scene()->itemsBoundingRect();
        image_viewer_->fitInView(bounds, Qt::KeepAspectRatio );
        image_viewer_->centerOn(0,0);
    }

}


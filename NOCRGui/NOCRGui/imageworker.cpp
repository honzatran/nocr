#include "imageworker.h"
#include <nocrlib/word_generator.h>
#include <iostream>
#include <QThread>
#include <QDebug>
#include <QFile>
#include <vector>
#include <string>

ImageWorker::ImageWorker(QObject *parent) :
    QObject(parent)
{
}

using namespace std;

void ImageWorker::loadConfiguration(const std::string &dict, const std::string &er_first_stage,
                                    const std::string &er_second_stage, const std::string &merge_conf)
{
    text_recognition_.loadConfiguration(er_first_stage,er_second_stage,merge_conf);
    dictionary_.loadWords(dict);
}

void ImageWorker::loadOcr(std::unique_ptr<AbstractOCR> ocr)
{
    // force move semantic
    ocr_ = std::move(ocr);
    // pointer ocr isn't valid anymore
    text_recognition_.loadOcr(ocr_.get());
}

cv::Mat ImageWorker::getImage()
{
    return image_;
}

std::vector<TranslatedWord> ImageWorker::getWords()
{
    return words_;
}

QVector<TranslationRecord> ImageWorker::getDetectedWords()
{
    return detected_words_;
}

void ImageWorker::clearDetection()
{
    detected_words_.clear();
}



void ImageWorker::addNewWords(const QStringList &new_words)
{
//    emit newOperation("Adding new words to dictionary");
    for ( const QString &w: new_words )
    {
        dictionary_.addWord(w.toStdString());
    }
    emit operationDone();
    //    dictionary_.print();
}

void ImageWorker::addWordsFile(const QString &dict_file)
{
    dictionary_.loadWords(dict_file.toStdString());
    emit operationDone();
}


void ImageWorker::processImages(const QStringList &images)
{
    for ( auto it = images.begin(); it != images.end(); ++it )
    {
        emit newImage(*it);
        cv::Mat image = cv::imread( it->toStdString(), CV_LOAD_IMAGE_COLOR );
        words_ = text_recognition_.recognize(image, dictionary_);
        image_ = image;
        detected_words_.push_back( TranslationRecord( *it, words_ ) );
    }
    emit readingDone();
}


void ImageWorker::loadNewDict(const QString &dict_file)
{
    dictionary_.clearDictionary();
    dictionary_.loadWords(dict_file.toStdString());
    emit operationDone();
}

void ImageWorker::saveDictToFile(const QString &dict_file)
{
    QFile dict(dict_file);
//    emit newOperation("Saving to dictionary file " + dict_file );
    dict.open(QIODevice::WriteOnly| QIODevice::Text );
    QTextStream dict_stream( &dict);

    vector<string> words = dictionary_.getAllWords();
    for ( const string &w: words )
    {
        dict_stream << QString::fromStdString(w) << "\n";
    }
    emit operationDone();
}

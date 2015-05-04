#ifndef IMAGEWORKER_H
#define IMAGEWORKER_H

#include <QObject>
#include <string>
#include <QVector>
#include <QString>
#include <QStringList>

#include <nocrlib/segment.h>
#include <nocrlib/dictionary.h>
#include <nocrlib/ocr.h>
#include <nocrlib/word_generator.h>
#include <nocrlib/text_recognition.h>

#include <opencv2/core/core.hpp>

#include "translationrecord.h"


class ImageWorker : public QObject
{
    Q_OBJECT
public:
    explicit ImageWorker(QObject *parent = 0);

    void loadConfiguration( const std::string &dict,
        const std::string & er_first_stage, const std::string &er_second_stage,
                            const std::string & merge_confs );

    void loadOcr(std::unique_ptr<AbstractOCR> ocr);
    cv::Mat getImage();
    std::vector<TranslatedWord> getWords();

    QVector<TranslationRecord> getDetectedWords();
    void clearDetection();

signals:
    void readingDone();
    void newImage(const QString &file);

//    void newOperation( QString text );
    void operationDone();
    
public slots:
    void processImages( const QStringList &images );
    void addNewWords( const QStringList &new_words);
    void addWordsFile( const QString &dict_file );
    void loadNewDict( const QString &dict_file );
    void saveDictToFile( const QString &dict_file );


private:
    TextRecognition<ERTextDetection, AbstractOCR> text_recognition_;
    cv::Mat image_;
    std::vector<TranslatedWord> words_;
    QVector<TranslationRecord> detected_words_;
    Dictionary dictionary_;
    std::unique_ptr<AbstractOCR> ocr_;
};

#endif // IMAGEWORKER_H

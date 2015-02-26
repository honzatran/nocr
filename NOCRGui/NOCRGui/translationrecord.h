#ifndef TRANSLATIONRECORD_H
#define TRANSLATIONRECORD_H

#include <nocrlib/word_generator.h>

#include <QPixmap>
#include <QGraphicsScene>
#include <QString>
#include <QStringList>
#include <QSharedPointer>

class TranslationRecord
{
public:

    TranslationRecord()
    {
        file_name_ = "";
    }

    TranslationRecord( const QString &file_name, const std::vector<TranslatedWord> &words )
        : file_name_(file_name), words_(words) { }

    QGraphicsScene* getQGScene();
    QStringList getDetectedText();
    QString getFileName() const { return file_name_; }

    std::vector<TranslatedWord> getWords() const { return words_; }


private:
    QString file_name_;
    std::vector<TranslatedWord> words_;

    QSharedPointer<QGraphicsScene> scene_;

    QGraphicsScene * loadScene();

};
#endif // TRANSLATIONRECORD_H

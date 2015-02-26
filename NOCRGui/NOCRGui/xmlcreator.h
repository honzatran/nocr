#ifndef XMLCREATOR_H
#define XMLCREATOR_H


#include <QDomDocument>
#include "translationrecord.h"
#include <nocrlib/word_generator.h>

class XmlCreator
{
public:
    XmlCreator();
    void createDomTree(const QVector<TranslationRecord> &records );
    QString toString() const { return doc_.toString(); }
private:
    QDomDocument doc_;
    const QString k_root_name = "detected-text";
    const QString k_image_tag = "image";
    const QString k_path_tag = "path-to-image";
    const QString k_word_tag = "word";
    const QString k_text_tag = "text";
    const QString k_bbox_tag = "bounding-box";


    QDomElement createImageElement( const TranslationRecord &rec );
    QDomElement createWordElement( const TranslatedWord &word );


};

#endif // XMLCREATOR_H

#include "xmlcreator.h"
#include <QDebug>

#include <nocrlib/word_generator.h>
#include <opencv2/core/core.hpp>

XmlCreator::XmlCreator()
{

}

void XmlCreator::createDomTree(const QVector<TranslationRecord> &records)
{
    qDebug() << "start";
    doc_.clear();
    qDebug() << "creating";

    QDomElement root = doc_.createElement(k_root_name);
    doc_.appendChild(root);
    for( const auto &rec: records )
    {
        QDomElement image = createImageElement( rec );
        root.appendChild(image);
    }
    qDebug() << "done";
}

QDomElement XmlCreator::createImageElement(const TranslationRecord &rec)
{
    QDomElement image = doc_.createElement(k_image_tag);
    QDomElement path = doc_.createElement(k_path_tag);
    path.appendChild( doc_.createTextNode(rec.getFileName()) );
    auto words = rec.getWords();

    for ( const auto &w : words )
    {
        image.appendChild( createWordElement(w) );
    }

    image.appendChild(path);
    return image;
}

QDomElement XmlCreator::createWordElement(const TranslatedWord &word)
{
    QDomElement word_elem = doc_.createElement(k_word_tag);

    QDomElement text = doc_.createElement(k_text_tag);
    QString translation = QString::fromStdString(word.translation_);
    text.appendChild(doc_.createTextNode(translation));
    word_elem.appendChild(text);

    QDomElement bounding_box = doc_.createElement(k_bbox_tag);
    cv::Rect rect = word.visual_information_.getRectangle();
    bounding_box.setAttribute("x", rect.x);
    bounding_box.setAttribute("y", rect.y);
    bounding_box.setAttribute("width", rect.width);
    bounding_box.setAttribute("height", rect.height);
    word_elem.appendChild(bounding_box);

    return word_elem;
}

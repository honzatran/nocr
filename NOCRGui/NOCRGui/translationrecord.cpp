#include "translationrecord.h"
#include <QDebug>

#define SIZE 1024

QGraphicsScene* TranslationRecord::getQGScene()
{
    if ( scene_.isNull() )
    {
        scene_ = QSharedPointer<QGraphicsScene>(loadScene());
    }
    return scene_.data();
}

QStringList TranslationRecord::getDetectedText()
{
    QStringList output;
    for ( const auto &w : words_ )
    {
        output += QString::fromStdString(w.translation_);
    }
    return output;
}

QGraphicsScene* TranslationRecord::loadScene()
{
    QPixmap pixmap( file_name_ );
    QGraphicsScene *scene = new QGraphicsScene();
    scene->addPixmap(pixmap);

    double scale = 1;
    if (scene->width() < SIZE && scene->height() < SIZE)
    {
        bool aspect = scene->width() < scene->height();
        scale = aspect ? (double)(scene->height()/SIZE) : (double)(scene->width()/SIZE);
        qDebug() << scale;
    }

    QPen pen(Qt::red, 3, Qt::SolidLine, Qt::SquareCap, Qt::BevelJoin);

    for ( auto &w: words_ )
    {
        cv::Rect rect = w.visual_information_.getRectangle();
        scene->addRect(scale * rect.x, scale * rect.y,
                scale * rect.width, scale * rect.height, pen );

        for (const auto & l : w.visual_information_.getLetters())
        {
            cv::Rect l_rect = l.getRectangle();

            scene->addRect(scale * l_rect.x, scale * l_rect.y,
                    scale * l_rect.width, scale * l_rect.height, pen );
        }
    }

    return scene;
}


#ifndef CONVERTOROPENCV_H
#define CONVERTOROPENCV_H

#include <opencv2/core/core.hpp>
#include <QPixmap>

class ConvertorOpenCv
{
public:
    ConvertorOpenCv();

    cv::Mat3b convert( const QPixmap &pixmap );
    QPixmap convert( const cv::Mat3b &mat );


};

#endif // CONVERTOROPENCV_H

#include "convertoropencv.h"

ConvertorOpenCv::ConvertorOpenCv()
{
}

cv::Mat3b ConvertorOpenCv::convert(const QPixmap &pixmap)
{
    QImage bitmap = pixmap.toImage();
    cv::Mat3b output( bitmap.height(), bitmap.width(), CV_8UC3 );
    int width = bitmap.width();
    int height = bitmap.height();

    for ( int y = 0; y < height; ++y )
    {
      cv::Vec3b *row = output[y];
      for ( int x = 0; x < width; ++x )
      {
        QRgb pixel = bitmap.pixel(x,y);
        row[x] = cv::Vec3b( qBlue(pixel), qGreen(pixel), qRed(pixel) );
      }
    }

    return output;
}

QPixmap ConvertorOpenCv::convert(const cv::Mat3b &mat)
{
    QImage dest(mat.cols, mat.rows, QImage::Format_ARGB32);
    for (int y = 0; y < mat.rows; ++y) {
        const cv::Vec3b *matrow = mat[y];
        QRgb *destrow = (QRgb*)dest.scanLine(y);
        for (int x = 0; x < mat.cols; ++x) {
            destrow[x] = qRgba(matrow[x][2], matrow[x][1], matrow[x][0], 255);
        }
    }
    return QPixmap::fromImage(dest);
}



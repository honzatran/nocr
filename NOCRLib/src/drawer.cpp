/* 
 * Tran Tuan Hiep
 * Implementation of methods and classes declared in drawer.h
 * 
 * Compiler: g++ 4.8.3
 */
#include "../include/nocrlib/drawer.h"

RectangleDrawer::RectangleDrawer()
    : color_(cv::Scalar(0, 0, 255 ))
{
}

void RectangleDrawer::init( const cv::Mat &image )
{
    if ( image.type() == CV_8UC3 )
    {
        canvas_ = image.clone();
    }
    else
    {
        cv::cvtColor( image, canvas_, CV_GRAY2BGR );
    }
}

void RectangleDrawer::draw( const Component &c )
{
    drawRectangle( c.rectangle() );
}

void RectangleDrawer::draw( const Letter &l )
{
    drawRectangle( l.getRectangle() );
}

void RectangleDrawer::draw( const Word &w )
{
    drawRectangle( w.getRectangle() );
}

void RectangleDrawer::drawRectangle( const cv::Rect &rect)
{
    cv::rectangle( canvas_, rect, color_ );
}

void RectangleDrawer::setColor(const cv::Scalar & color) 
{
    color_ = color;
}

//=================
void BinaryDrawer::init( const cv::Mat &image )
{
    canvas_ = cv::Mat( image.rows, image.cols, CV_8UC1, cv::Scalar( background_) );
}

void BinaryDrawer::draw( const Component &c )
{
    auto point = c.getPoints();
    std::for_each( point.begin(), point.end(), 
            [this] (const cv::Point &p) { canvas_.at<uchar>( p.y, p.x ) = foreground_; } );
}

void BinaryDrawer::draw( const Letter &l )
{
    auto point = l.getPoints();
    std::for_each( point.begin(), point.end(), 
            [this] (const cv::Point &p) { canvas_.at<uchar>( p.y, p.x ) = foreground_; } );
}

void BinaryDrawer::draw( const Word &w )
{
   auto letters = w.getLetters(); 
   for( const Letter &l : letters )
   {
       draw(l);
   }
}

/* 
 * Tran Tuan Hiep
 * Implementation of methods and classes declared in component.h
 *
 * Compiler: g++ 4.8.3, 
 */

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

//#include <opencv2/highgui/highgui.cpp>

#include <vector>
#include <cmath>
#include <limits>
#include <iostream>
#include <chrono>
#include <ostream>
#include <queue>
#include <algorithm>

#include "../include/nocrlib/component.h"

using namespace std;
using namespace cv;


Component::Component() 
{
    init();
}

void Component::addPoint( const cv::Point &point)  
{
    updateSize(point);
    points_.push_back(point); 
}



void Component::updateSize(const cv::Point &point) 
{
    sumX_ += point.x;
    sumY_ += point.y;

    if ( point.x < left_ ) left_ = point.x;
    else if ( point.x > right_ ) right_ = point.x;

    if ( point.y < upper_ ) upper_ = point.y;
    else if ( point.y > lower_ ) lower_ = point.y;
}


bool Component::contain( const Component &c ) const 
{
    auto crect = c.rectangle();
    auto rect = rectangle();
    return ( crect == ( crect & rect ) && rect !=crect );
}

double Component::getDiagonal() const  
{
    int width = getWidth();
    int height = getHeight();
    return std::sqrt( width * width + height * height );
}




Point Component::indexTransposition( const Point &point ) const
{
    return cv::Point( point.x - left_, point.y - upper_);
}


Mat Component::cutComponentFromImage( const Mat& image, bool zeroPadding ) const
{
    cv::Mat cuttedImage;
    cuttedImage = Mat( image, Range( upper_, lower_ + 1), Range( left_, right_ + 1 ) ); 
   
    if ( !zeroPadding ) 
    {
        return cuttedImage;
    }
    else 
    {
        Mat cuttedImageWithZeroPadding;
        cv::copyMakeBorder( cuttedImage, cuttedImageWithZeroPadding, 20,20,20,20,0 );
        return cuttedImageWithZeroPadding;
    }
}

Mat Component::createBinaryMat(uchar component_value, uchar background_value ) const 
{
    cv::Mat output( getHeight(), getWidth(), CV_8UC1, Scalar(background_value) );
    for( const auto &p:points_ ) 
    {
        auto transPoint = indexTransposition( p );
        output.at<uchar>( transPoint.y, transPoint.x ) 
            = component_value;
    }
    
    return output;
}

// ===========================================================
// function computing information from component 
//

static bool isInsideBounds( const cv::Point &p, const cv::Size &size );

std::vector<cv::Point> getPerimeterPoints
                       ( Component &c, const cv::Size &bounds )
{
    typedef Neighbourhood<connectivity::eightpass> Neighbourhood;
    cv::Mat binary = c.getBinaryMat();
    vector<Point> output;
    Point offset( c.getLeft() - 1, c.getUpper() - 1 );

    for ( auto it = binary.begin<uchar>(); it != binary.end<uchar>(); ++it )
    {
        if ( *it == 0 )
        {
            Point p = it.pos();
            for ( int i = 0; i < 8; ++i )
            {
                Point neighbour = Neighbourhood::getNeighbour( i, p );
                if ( isInsideBounds( neighbour, binary.size() ) && 
                        binary.at<uchar>(neighbour.y,neighbour.x) > 0 )
                {
                    Point transposed_p = p + offset;
                    if ( isInsideBounds( transposed_p, bounds ) )
                    {
                        output.push_back( transposed_p );
                    }
                }
            }
        }
    }

    return output; 
}

static bool isInsideBounds( const cv::Point &p, const cv::Size &bounds )
{
    return ( p.x >= 0 && p.y >= 0 && p.x < bounds.width && p.y < bounds.height );
}


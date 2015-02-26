/* 
 * Tran Tuan Hiep
 * Implementation of methods and classes declared in direction_histogram.h
 *
 * Compiler: g++ 4.8.3
 */

#include "../include/nocrlib/direction_histogram.h"
#include "../include/nocrlib/utilities.h"

using namespace std;

vector<float> DirectionHistogram::compute( Component &c )
{
    return computeHistogram( c.getBinaryMat() );
}

vector<float> DirectionHistogram::computeHistogram( const cv::Mat &image )
{
    cv::Mat resized, edges, grad_x, grad_y;
    cv::resize( image, resized, cv::Size(image_size,image_size) );
    cv::Canny( resized, edges, 50, 150, 3 );
    cv::Sobel( resized, grad_x, CV_16S, 1, 0, 3 ); 
    cv::Sobel( resized, grad_y, CV_16S, 0, 1, 3 );
    
    std::vector<float> concanated_histogram;

    for ( int i = 0; i < 4; ++i )
    {
        for( int j = 0; j < 4; ++j )
        {
            int x = j * cell_size;
            int y = i * cell_size;
            cv::Rect crop_rect( x, y, cell_size, cell_size );
            std::vector<float> histogram = extract( edges(crop_rect), grad_x(crop_rect), grad_y(crop_rect) );
            concanated_histogram.insert( concanated_histogram.end(), histogram.begin(), histogram.end() );
        }
    }

    return concanated_histogram;
}

vector<float> DirectionHistogram::extract( const cv::Mat &edges, 
        const cv::Mat &grad_x, const cv::Mat &grad_y )
{
    std::vector<float> histogram( 8, 0 );
    // 0 , 45, 90, 135, 180, 225, 270, 315, 360;
    auto begin = edges.begin<uchar>();
    auto end = edges.end<uchar>();
    for ( auto it = begin; it != end; ++it ) 
    {
        if ( *it == 0 )
        {
            continue;
        }

        cv::Point edge = it.pos();
        float direction = cv::fastAtan2( grad_y.at<short>(edge.y, edge.x), 
                grad_x.at<short>(edge.y,edge.x) );
        int i = std::floor(direction/45);
        histogram[i] += 1;
    }
    return histogram;
}



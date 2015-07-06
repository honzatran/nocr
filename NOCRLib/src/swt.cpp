/* 
 * Tran Tuan Hiep
 * Implementation of methods and classes declared in swt.h
 *
 * Compiler: g++ 4.8.3
 */

#include "../include/nocrlib/swt.h"
#include "../include/nocrlib/utilities.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <map>
#include <vector>
#include <set>
#include <chrono>
#include <utility>
#include <deque>
#include <stack>

using namespace cv;
using namespace std;


cv::Mat SwtTransform::operator() ( const cv::Mat &gray_image, bool zero_padding ) 
{
    if ( zero_padding )
    {
        input_ = gray_image; 
    }
    else 
    {
        copyMakeBorder( gray_image, input_, 1,1,1,1, BORDER_CONSTANT, 0);
    }
    reached_positions_ = helper::getAccesibilityMaskWithNegativeBorder( input_ );  

    return getStrokeWidthTransformation();
}


cv::Mat SwtTransform::getStrokeWidthTransformation()
{
    cv::Mat_<float> distanceImage = getDistanceTransform();
    vector<cv::Point> foregroundPixels = getForegroundPixels( distanceImage );
    prepareForTransformation( distanceImage , foregroundPixels ); 
    cv::Mat tmp = transform( distanceImage ); 
    // distances_points_.clear();
    point_neighbours_.clear();
    return tmp;
}


cv::Mat_<float> SwtTransform::getDistanceTransform() 
{
    Mat_<float> dist;
    distanceTransform( input_ , dist, CV_DIST_L2, CV_DIST_MASK_PRECISE );
    return dist; 
}

vector<Point> SwtTransform::getForegroundPixels( const Mat &image )
{
    auto begin = image.begin<float>();
    auto end = image.end<float>();
   
    vector<Point> foreground_pixels;
    for ( auto it = begin; it != end; ++it ) 
    {
        if ( *it > 0 )
        {
            foreground_pixels.push_back( it.pos() );
        }
    }

    return foreground_pixels; 
}

vector<float> SwtTransform::roundValues( const cv::Mat &matrix ) 
{
    vector<float> output( matrix.rows * matrix.cols , 0); 
    auto begin = matrix.begin<float>();
    auto end = matrix.end<float>();
   
    for ( auto it = begin; it != end; ++it ) 
    {
        int key = getCodeOfPosition( it.pos() );
        float val = *it;
        if ( val != 0 ) 
        {
            output[key] = std::round( val );
        }
    }

    return output; 
}

void SwtTransform::prepareForTransformation( const Mat_<float> &dist_image, const vector<Point> &foreground_pixels )
{
    point_neighbours_.reserve( foreground_pixels.size() );
    for ( const Point &p : foreground_pixels ) 
    {
        float distValue = dist_image.at<float>( p.y, p.x ); 
        lookUp( p, distValue, dist_image ); 
        makeRecordDistValuePoint( distValue, p ); 
    }
}

void SwtTransform::lookUp( const Point &position, float val, const Mat_<float> &dist_mat )
{
    typedef Neighbourhood<connectivity::eightpass> nc;
    
    //for ( const auto &neighbourPosition: neighbours )
    VecPoint tmp;
    tmp.reserve(8);
    for ( int i = 0; i < 8; ++i )
    {
        Point neighbour = nc::getNeighbour(i,position);
        float neighbour_val = dist_mat.at<float>( neighbour.y, neighbour.x );
        if ( neighbour_val != 0 &&  neighbour_val < val )
        {
            tmp.push_back( neighbour );
        }
    }

    if ( !tmp.empty() ) 
    {
        int key = getCodeOfPosition( position );
        auto new_pair = pair<int, VecPoint>( key, tmp );
        point_neighbours_.insert( new_pair );
    }
}


void SwtTransform::makeRecordDistValuePoint
            ( const float &dist, const Point &position )
{
    ValuePosition tmp( dist, position );
    distances_points_.push( tmp );
}

Mat SwtTransform::transform( const Mat &roundDist )
{
    (void)(roundDist);
    
    Mat StrokeWidth = Mat(input_.rows, input_.cols, CV_32FC1, Scalar(0));
    while( !distances_points_.empty() ) 
    {
        ValuePosition top = distances_points_.top();
        changeVal( top.getPosition(), StrokeWidth, top.getVal() );
        distances_points_.pop();
    }

    return StrokeWidth;
}

// dopsat jeste nejak poresit tu kdy hodnota 0
void SwtTransform::changeVal( const cv::Point &p, cv::Mat &strokeWidth, float stroke )
{
    if ( isReached(p) ) 
    {
        return;
    }

    std::stack<Point> stack;
    stack.push( p );
    // std::deque<Point> neighbours(1,p);
    changeToReached(p);

    while( !stack.empty() ) 
    {
        Point top = stack.top();
        stack.pop();

        auto tmp = getLookUp( top );
        strokeWidth.at<float>( top.y, top.x ) = stroke;
        if ( tmp == point_neighbours_.end() ) 
            continue;

        for ( const Point &t : tmp->second )
        {
            if ( !isReached(t) )
            {
                changeToReached(t);
                stack.push(t);
            }
        }
    }

}


auto SwtTransform::getLookUp( const cv::Point &p )
    -> LookUpDataStructure::iterator
{
    int key = getCodeOfPosition( p );
    return point_neighbours_.find( key );
}

void SwtTransform::show( const cv::Mat &distances )
{
    cv::Mat tmp, tmp1;
    cv::normalize( distances, tmp, 0, 1, NORM_MINMAX );
    tmp.convertTo( tmp1, CV_8UC1, 255, 0);
    gui::showImage( tmp1, "swt transform" );
}

//================stroke width transform=================== 
//

std::vector<float> SwtRatio::compute( Component &c )
{
    cv::Mat binary = c.getBinaryMat();
    cv::Mat swtImage = swt_( binary, true );
    // swt_.show( swtImage );
    // auto swtEnd = std::chrono::steady_clock::now();
    // std::cout << "computing of stroke width takes "<< tc( start,swtEnd ).count() << "ms" << std::endl;
    //
    float mean, std_dev;

    std::tie(mean, std_dev) = getMeanStdDev<float>(c.getPoints(), swtImage, c.getLeftUpperCorner() - cv::Point(1,1));

    std::size_t k = c.size();

    return { std_dev/mean * (1 + 1/(4*k)) };
}

std::vector<float> SwtMean::compute( Component &c )
{
    std::vector<float> output;
    float f = computeSwtMean(c);

    cout  << f << endl;
    return { f };
}

float SwtMean::computeSwtMean( Component &c )
{
    cv::Mat binary = c.getBinaryMat();
    cv::Mat swtImage = swt_( binary, true );

    float sum = 0;
    std::for_each( swtImage.begin<float>(), swtImage.end<float>(),
            [&sum] ( float f ) { sum += f; });

    return sum/c.size();
}







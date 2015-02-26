#include "../include/nocrlib/features.h"
#include "../include/nocrlib/iooper.h"
#include "../include/nocrlib/component.h"
#include "../include/nocrlib/assert.h"

#include <opencv2/core/core.hpp>

#include <vector>
#include <iostream>


using namespace std;
using namespace cv;

/* 
 * Tran Tuan Hiep
 * Implementation of methods and classes declared in features.h
 *
 * Compiler: g++ 4.8.3
 */

BackgroundMergeRule::BackgroundMergeRule( const cv::Mat &bitmap )
    : bitmap_(bitmap)
{
    // accessedBorderPoints_ = std::vector<bool>( bitmap_.cols * bitmap_.rows , false );
    // bordePoints_ = std::vector< Point >();
}

std::vector<float> BackgroundMergeRule::compute( Component &c )
{
    cv::Mat binary = c.getBinaryMat();
    // ComponentFinder< BackgroundMergeRule ,connectivity::fourpass > backgrounExtractor( binary );
    // Component background = backgrounExtractor.findComp( cv::Point(0,0) );

    cv::Mat copy = binary.clone();
    cv::floodFill( copy, cv::Point(0,0), cv::Scalar(255), 0 );
    int tmp = binary.rows * binary.cols - cv::countNonZero(copy);

    // int holeArea = ( binary.rows * binary.cols - c_ptr->size() - background.size() );
    std::vector<float> output = 
    {
        (float)tmp/c.size() 
    };
    return output;

}

/*
 * std::vector<float> BackgroundMergeRule::compute( const ct::ptrComp &c_ptr )
 * {
 *     return compute(*c_ptr);
 *     
 *      * cv::Mat binary = c_ptr->getBinaryMat();
 * 
 *      * cv::Mat copy = binary.clone();
 *      * cv::floodFill( copy, cv::Point(0,0), cv::Scalar(255), 0 );
 *      * int tmp = binary.rows * binary.cols - cv::countNonZero(copy);
 * 
 *      * // int holeArea = ( binary.rows * binary.cols - c_ptr->size() - background.size() );
 *      * std::vector<float> output = 
 *      * {
 *      *     (float)tmp/c_ptr->size() 
 *      * };
 *      * return output;
 *      
 * }
 */

bool BackgroundMergeRule::canBeMerged( Point pointOfComponent, Point outsidePoint )
{
    uchar outsidePointVal = bitmap_.at<uchar>( outsidePoint.y, outsidePoint.x );

    // int key = outsidePoint.y * bitmap_.cols + outsidePoint.x;
    // if ( ( outsidePointVal !=0 ) && !accessedBorderPoints_[key] )
    // {
    //     accessedBorderPoints_[key] = true;        
    //     bordePoints_.push_back(outsidePoint);
    //     return false;
    // }

    return outsidePointVal == 0; 
}

bool BackgroundMergeRule::isStartPointOfComponent( Point p )
{
    return bitmap_.at<uchar>( p.y,p.x ) == 0;
}

//======================== FeatureFinder===========================

/*
 * 1 2 3 
 * 0 p 4
 * 7 6 5
 *
 */

std::vector<float> CompositeFeatureExtractor::compute( Component &c )
{
    std::vector<float> output;
    for ( AbstractFeatureExtractor* f: features_ )
    {
        auto tmp = f->compute(c);
        output.insert( output.end(), tmp.begin(), tmp.end() );
    }
    return output;
}



const float QuadScanner::c = 1/sqrt(2);

//======================== ConvexAreaFinder========================

std::vector<float> ConvexHullAreaFinder::compute( Component &c )
{ 
    cv::Mat binary = c.getBinaryMat();
    cv::Mat tmp = binary.clone();
    vector< vector<Point> > poly;
    vector< Vec4i > hierarchy;
    cv::findContours( tmp, poly, hierarchy, RETR_TREE, CHAIN_APPROX_NONE, Point(0,0) );
    vector<cv::Point> convex_hull ;
    cv::convexHull( poly[0], convex_hull );
    size_t convex_area = cv::contourArea( convex_hull );


    float convexity = (float)c.size()/convex_area;

    vector<float> output = 
    {
        convexity 
    };
    return output;
}

size_t ConvexHullAreaFinder::getConvexHullArea
        (const std::vector<cv::Point> &perimeterPoints)
{
    vector<Point> convexHullPoints;
    cv::convexHull( perimeterPoints , convexHullPoints );
    return cv::contourArea( convexHullPoints ); 
}


//==============================Euler number Computation===============
// implementation of gray algorithm for Euler Number  of binary image
// Component (see PRATT,Digital Image Processing 594)
// imagine a square with 4 fields 
// row 1 :  0 | 1
// row 2 :  3 | 2
// use 255 instead of 1 for better visualization

// setting up group members 
//
//
std::vector<float> QuadScanner::compute( Component &c )
{
    std::vector<float> output;
    scan( c.getBinaryMat() );
    output.push_back( (float)std::sqrt( c.size() )/getPerimeterLength() );
    // cout << 1 - getEulerNumber() << endl;
    int hole_number = 1 - getEulerNumber();
    output.push_back( hole_number ); 
    init();
    return output;
}

void QuadScanner::scan( const cv::Mat &image ) 
{
    for ( int i = 0; i < image.rows - 1; ++i ) 
    {
        for ( int j = 0; j < image.cols -1; ++j )
        {
            cv::Mat tmp = image( cv::Rect( j, i, 2, 2) );
            updateCounts( tmp );
        }
    }
}


void QuadScanner::updateCounts( const cv::Mat &pattern )
{
    int nonZeroNumber = getNumberOfNonZeroElement( pattern );

    switch ( nonZeroNumber ) 
    {
        case 1: ++q1Count_; break;
        case 2: 
                isDiagonal( pattern ) ? ++q2DCount_ : ++q2Count_ ;
                break;
        case 3: ++q3Count_; break;
        default: break;
    }
}


int QuadScanner::getNumberOfNonZeroElement( const cv::Mat &pattern )
{
    int sum = 0;
    auto begin = pattern.begin<uchar>();
    auto end = pattern.end<uchar>();
    for ( auto it = begin; it != end; ++it )
    {
        if (  (*it) > 0 )
        {
            ++sum;
        }
    }
    return sum;
}


bool QuadScanner::isDiagonal( const cv::Mat &pattern )
{
    bool a = pattern.at<uchar>(0,0) > 0;
    bool b = pattern.at<uchar>(0,1) > 0;

    bool c = pattern.at<uchar>(1,0) > 0;
    bool d = pattern.at<uchar>(1,1) > 0;
    return ( ( a && d ) || (  c && b ) );
}

void QuadScanner::init() 
{
    q1Count_ = 0;
    q2Count_ = 0;
    q2DCount_ = 0;
    q3Count_ = 0;
}


//=================compute hole area============================
//
//

std::vector<float> HorizontalCrossing::compute( Component &c )
{
    cv::Mat binary = c.getBinaryMat();
    int height = c.getHeight();
    int crossingA = getNumberOfCrossingAt( 1+height/6, binary ); 
    int crossingB = getNumberOfCrossingAt( 1+height/2, binary );
    int crossingC = getNumberOfCrossingAt( 1+height*5/6, binary );
    int med_crossing = statistic<int>::median( crossingA, crossingB, crossingC );
    vector<float> output = { (float)med_crossing }; 
    return output;

}

int HorizontalCrossing::getNumberOfCrossingAt( const int &rowIndex, const cv::Mat &image )
{
    cv::Mat row = image.row( rowIndex );
    int horizontalCrossing = 0;
    for ( auto it = row.begin<uchar>() + 1 ; it != row.end<uchar>() -1 ; ++it ) 
    {
        if ( *it > 0 ) 
        {
            auto tmp = it - 1;
            if ( *tmp == 0 )
            {
                ++ horizontalCrossing;
            }

            tmp = it + 1;
            if ( *tmp == 0 )
            {
                ++ horizontalCrossing;
            }

        }
    }

    return horizontalCrossing;
}



std::vector<float> InflectionPoints::compute( Component &c )
{
    std::vector<float> output;
    cv::Mat binary = c.getBinaryMat();

    // double epsilon = c_ptr->size() < 200 ? 0.2:(double)std::min( binary.rows, binary.cols )/17;

    cv::Mat tmp = binary.clone();
    vector< vector<Point> > poly;
    vector< Vec4i > hierarchy;

    cv::findContours( tmp, poly, hierarchy, RETR_TREE, CHAIN_APPROX_NONE, Point(0,0) );

    double tmp_min = (double)std::min( binary.rows,binary.cols );
    double epsilon = tmp_min/17;
    // int number_inflections = computeNumberOfInflections( poly[0], epsilon );
    int number_inflections = computeNumberOfInflections( poly[0], epsilon );
    size_t convex_hull_area = computeHullArea( poly[0] );


    output.push_back( ( float ) c.size()/convex_hull_area );
    output.push_back( number_inflections );
    return output;
}

int InflectionPoints::computeNumberOfInflections( const std::vector<cv::Point> &points, double epsilon )
{
    vector<Point> approxCurve;
    cv::approxPolyDP( points, approxCurve, epsilon, true );

    int inflectionPoints = 0;
    bool wasConvex = false;

    auto it = approxCurve.begin();
    auto prev_it = approxCurve.end() - 1;
    for ( ; it != approxCurve.end(); ++it ) 
    {
        auto next_it = it + 1;
        if ( next_it == approxCurve.end() )
        {
            next_it = approxCurve.begin(); 
        }

        cv::Point prev_vector = *prev_it - *it;
        cv::Point next_vector = *next_it - *it;

        double angle_prev = atan2( prev_vector.y, prev_vector.x );
        double angle_next = atan2( next_vector.y, next_vector.x );
        if ( angle_prev < 0 )
        {
            angle_prev = 2 * CV_PI + angle_prev;
        }
        
        if ( angle_next < 0 )
        {
            angle_next = 2 * CV_PI + angle_next;
        }

        
        double angle = angle_next - angle_prev;
        
        if ( angle < 0 )
        {
            angle = 2 * CV_PI + angle; 
        }

        if ( it != approxCurve.begin() )
        {
            if ( (wasConvex && angle > CV_PI ) || ( !wasConvex && angle <= CV_PI ) )
            {
                ++inflectionPoints;
            }
        }


        wasConvex = angle <= CV_PI ; 
        prev_it = it;
    }

    // cout<< inflectionPoints << endl;
    return inflectionPoints;
}

size_t InflectionPoints::computeHullArea( const std::vector<cv::Point> &points )
{
    std::vector<cv::Point> convex_hull;
    cv::convexHull( points, convex_hull );
    return cv::contourArea( convex_hull );
}

std::vector<float> AspectRatioRotRect::compute( Component &c )
{
    std::vector<float> output;
    cv::RotatedRect rot_rect = c.getMinAreaRect();
    output.push_back( (float)rot_rect.size.width/rot_rect.size.height );
    return output;
}

PerimeterValuesMean::PerimeterValuesMean( const cv::Mat &image )
{
    vector<Mat> bgr_mat;
    if ( image.type() == CV_8UC3 )
    {
        // image has BGR format
        cv::split( image, bgr_mat );
        cv::Mat gray_image;
        cv::cvtColor( image, gray_image, CV_BGR2GRAY ); 
        bgr_mat.push_back(gray_image);
    }
    else if ( image.type() == CV_8UC1 )
    {
        // image has grayscale format
        cv::Mat bgr_image;
        cv::cvtColor( image, bgr_image, CV_GRAY2BGR );
        cv::split( bgr_image, bgr_mat );
        bgr_mat.push_back( image );
    }
    else
    {
        NOCR_ASSERT(false, "wrong input image format"); 
    }

    cv::merge( bgr_mat, value_mat_ );
    bounds_ = image.size();
}

void PerimeterValuesMean::setValuesMat( const cv::Mat4b &values_mat )
{
    value_mat_ = values_mat;
}

std::vector<float> PerimeterValuesMean::compute( Component &c )
{
    vector<Point> perimeter_points = getPerimeterPoints( c, bounds_ );
    Vec4i sums( 0, 0, 0, 0);
    for ( Point p: perimeter_points )
    {
        sums += value_mat_.at<Vec4b>( p.y, p.x );
    }

    int size = perimeter_points.size(); 
    std::vector<float> output;
    for ( int i = 0; i < 4; ++i )
    {
        output.push_back( (float) sums[i]/size );
    }
    return output;
}


/*
 * vector<Point> PerimeterValuesMean::getPerimeterPoints(Component &c) 
 * {
 *     cv::Mat binary = c.getBinaryMat();
 *     vector<Point> output;
 *     Point offset( c.getLeft() - 1, c.getUpper() - 1 );
 * 
 *     for ( auto it = binary.begin<uchar>(); it != binary.end<uchar>(); ++it )
 *     {
 *         if ( *it == 0 )
 *         {
 *             Point p = it.pos();
 *             for ( int i = 0; i < 8; ++i )
 *             {
 *                 Point neighbour = neigbourhood::getNeighbour( i, p );
 *                 if ( isInBounds( neighbour, binary.rows, binary.cols) && 
 *                         binary.at<uchar>(neighbour.y,neighbour.x) > 0 )
 *                 {
 *                     Point transposed_p = p + offset;
 *                     if ( isInBounds( transposed_p, value_mat_.rows, value_mat_.cols ) )
 *                     {
 *                         output.push_back( transposed_p );
 *                     }
 *                 }
 *             }
 *         }
 *     }
 * 
 *     return output; 
 * }
 */

void HogExtractor::setShortDescriptor()
{
    // 64,64
    hog_.winSize = cv::Size(64, 64);
    //32,32
    hog_.blockSize = cv::Size(32,32);
    // hog_.blockSize = cv::Size(16,16);
    // 32,32
    // hog_.blockStride = cv::Size(8, 8);
    hog_.blockStride = cv::Size(32,32);
    // 16,16
    // hog_.cellSize = cv::Size(8,8);
    hog_.cellSize = cv::Size(16,16);
}

void HogExtractor::setLongDescriptor()
{
    hog_.winSize = cv::Size(64, 64);
    //32,32
    hog_.blockSize = cv::Size(16,16);
    // 32,32
    hog_.blockStride = cv::Size(8, 8);
    // 16,16
    hog_.cellSize = cv::Size(8,8);
}



std::vector<float> HogExtractor::compute( Component &c )
{
    // cout << hog_.getDescriptorSize() << endl;
    std::vector<float> hog_values;
    cv::Mat resized_image( 64, 64, CV_8UC1 );
    cv::resize( c.getBinaryMat(), resized_image, resized_image.size() );
    // gui::showImage( resized_image, "64*128" );
    hog_.compute( resized_image, hog_values );
    // std::cout << hogValues.size() << std::endl;
    return hog_values;
}

SiftExtractor::SiftExtractor()
{
    const int key_num = 22;
    sift_ = cv::SIFT( key_num, 3, 0.04, 10, 1.6 ); 
    descriptor_length_ = key_num * 128;
}

std::vector<float> SiftExtractor::compute( Component &c )
{
    cv::Mat descriptors = getKeyPointsDescription( c.getBinaryMat() );

    int size = std::min( descriptor_length_, descriptors.size().area() );
    vector<float> output( descriptor_length_, 0 );

    uchar *descriptors_data = descriptors.data;
    for ( int i = 0; i < size; ++i )
    {
        output[i] = descriptors_data[i]/255.f; 
    }

    return output;
}

cv::Mat SiftExtractor::getKeyPointsDescription( const cv::Mat &image )
{
    vector<cv::KeyPoint> key_points; 
    cv::Mat descriptors;
    sift_( image, cv::Mat(), key_points, descriptors );
    return descriptors;
}


/**
 * @file swt.h
 * @brief Implementation of Stroke Width Transform algorithm by Chen 
 * @author Tran Tuan Hiep
 * @version 
 * @date 2014-09-14
 */


#ifndef NOCRLIB_SWT_H
#define NOCRLIB_SWT_H


#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <map>
#include <vector>
#include <unordered_map>
#include <queue>
#include <iostream>
#include <memory>

#include "utilities.h"
#include "segment.h"
#include "abstract_feature.h"
#include "feature_traits.h"

/// @cond
// class storing position in image
// ad its value
class ValuePosition
{
    public:
        ValuePosition(const float &val, const cv::Point &position)
            : val_(val), position_(position) { }

        ~ValuePosition( ) { }

        float getVal() const { return val_; }
        cv::Point getPosition() const { return position_; }

        friend bool operator< ( const ValuePosition &a, const ValuePosition &b )
        {
            return a.val_ < b.val_;
        }
    private:
        float val_;
        cv::Point position_;
};
/// @endcond



/**
 * @brief stroke width transform
 */
class SwtTransform
{
    public:
        typedef std::vector< cv::Point > VecPoint;
        typedef std::unordered_map< int, VecPoint > LookUpDataStructure; 

        SwtTransform() = default;

        /**
         * @brief perform stroke width transform of image
         *
         * @param gray_image must be binarized
         * @param zero_padding if true additional border will be added to the 
         * picture
         *
         * @return stroke width transform of image 
         */
        cv::Mat operator() ( const cv::Mat &gray_image, bool zero_padding = false );

        static void show( const cv::Mat &distances );
    private:
        cv::Mat input_;

        LookUpDataStructure point_neighbours_;
        std::priority_queue< ValuePosition > distances_points_;
        std::vector< bool > reached_positions_;

        cv::Mat getStrokeWidthTransformation(); 
        

        int getCodeOfPosition( const cv::Point &p ) 
        {
            return p.y * input_.cols + p.x; 
        }

        cv::Point getPositionFromCode( int code )
        {
            int y = code / input_.cols;
            int x = code % input_.cols;
            return cv::Point( x,y );
        }

        cv::Mat_<float> getDistanceTransform();

        std::vector<float> roundValues( const cv::Mat &matrix );

        std::vector<cv::Point> getForegroundPixels( const cv::Mat &image );

        void prepareForTransformation
            ( const cv::Mat_<float> &roundedDistances, const std::vector<cv::Point> &foreground_pixels );

        void lookUp( const cv::Point &position, float val, const cv::Mat_<float> &dist_image );

        float getValFromPosition( const cv::Point &position , const cv::Mat &roundedDist )
        {
            return roundedDist.at<float>(position.y, position.x);
        }

        void makeRecordDistValuePoint( const float &dist, const cv::Point &position );

        cv::Mat transform( const cv::Mat &roundDist );

        void changeVal( const cv::Point &p, cv::Mat &strokeWidth, float stroke );

        bool isReached( const cv::Point &p )
        {
            int key = getCodeOfPosition(p);
            return reached_positions_[key];
        }

        bool isReached( int key ) 
        {
            return reached_positions_[key];
        }

        void changeToReached( const cv::Point &p )
        {
            int key = getCodeOfPosition(p);
            reached_positions_[key] = true;
        }

        void changeToReached( int key ) 
        {
            reached_positions_[key] = true;
        }

        LookUpDataStructure::iterator getLookUp( const cv::Point &p );
};

// segmentation of letter using swt
//

class SwtMergerRule
{
    public:
        SwtMergerRule(const cv::Mat &bitmap): bitmap_(bitmap) { }
        ~SwtMergerRule() { }

        bool isStartPointOfComponent(const cv::Point &startPoint ) 
        {
            return getValueFromBitmap( startPoint ) != 0;
        }

        bool canBeMerged( const cv::Point &pointOfComponent, const cv::Point potentialPoint )
        {
            float val = getValueFromBitmap( pointOfComponent )/ getValueFromBitmap( potentialPoint );
            return ( val >= 0.5f && val <= 2.f );
        }

    private:
        cv::Mat bitmap_;

        float getValueFromBitmap( const cv::Point &p ) 
        {
            return bitmap_.at<float>(p.y, p.x);
        }
};


/**
 * @brief Computes ratio between variance and mean of SWT of component.
 * Using Cheng algorithm to compute SWT.
 */
class SwtRatio : public AbstractFeatureExtractor
{
    public:
        std::vector<float> compute( Component &c ) override;
        float getSwtMean() const { return swt_mean_; }
    private:
        SwtTransform swt_;
        float swt_mean_;
};

/**
 * @brief Computes mean of SWT of a component.
 * Using Cheng algorithm to compute SWT.
 */
class SwtMean : public AbstractFeatureExtractor 
{
    public:
       std::vector<float> compute( Component &c ) override; 
       float computeSwtMean( Component &c );
    private:
       SwtTransform swt_;
};

template <typename T>
std::pair<float, float> getMeanStdDev(const std::vector<cv::Point> & points,
        const cv::Mat & img, cv::Point offset = cv::Point(0,0))
{
    float sum, sum_sqr;
    sum = sum_sqr = 0;

    for (cv::Point p : points)
    {
        cv::Point tmp = p - offset;
        T val = img.at<T>(tmp.y, tmp.x);

        sum += val;
        sum_sqr += val * val;
    }

    std::size_t k = points.size();

    float mean = sum/k;
    float variance = (sum_sqr + k * mean * mean - 2 * mean * sum)/(k - 1);

    return std::make_pair(mean, std::sqrt(variance));
}


#endif

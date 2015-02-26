/**
 * @file features.h
 * @brief contain declaration of AbstractFeatureExtractor, class 
 * that computes features from given components
 * and a few derived classes
 * @author Tran Tuan Hiep
 * @version 1.0
 * @date 2014-06-04
 *
 * Contains declaration of abstract base clase AbstractFeatureExtractor, 
 * which extracts features from input
 * components using method compute and its derived classed 
 * such as class for computing number of 
 * inflection points, euler number etc. We also provide class 
 * CompositeFeatureExtractor, which enables 
 * combining different derived class of AbstractFeatureExtractor.
 */
#ifndef NOCRLIB_FEATURES_H
#define NOCRLIB_FEATURES_H

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/ml/ml.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/features2d.hpp>


#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <algorithm>

#include "iooper.h"
#include "component.h"
#include "utilities.h"
#include "drawer.h"
#include "abstract_feature.h"






/**
 * @brief AspectRation computes aspect ratio of component
 */
class AspectRatio : public AbstractFeatureExtractor
{
    public:
        std::vector<float> compute( Component &c )
        {
            std::vector<float> output = 
            {
                (float) c.getWidth()/c.getHeight()
            };
            return output;
        }
};


/**
 * @brief ConvexHullAreaFinder computes ratio between component 
 * area and its convex hull area
 */
class ConvexHullAreaFinder : public AbstractFeatureExtractor
{
    public:
        typedef std::vector<cv::Point> vecPoint;
        std::vector<float> compute( Component &c ) override;
    private:
        size_t getConvexHullArea( const std::vector<cv::Point> &points );
};


/**
 * @brief Quad Scanner computes number of holes 
 * and compactness of component using algorithm from Gray 
 * and formulas from Gray and 
 */
class QuadScanner : public AbstractFeatureExtractor 
{
    public:

        static const float c;
        QuadScanner()
            : q1Count_(0), q3Count_(0), q2DCount_(0), q2Count_(0) { }

        std::vector<float> compute( Component &c ) override;


        /**
         * @brief computes quad counts
         *
         * @param image
         *
         * Methods computes quad counts in component binary image
         */
        void scan( const cv::Mat &image );

        /**
         * @brief based on quad counts computes euler number 
         *
         * @return euler number of component
         *
         * Method computes euler number based on formula from Gray
         */
        int getEulerNumber() const 
        {
            return ( q1Count_ - q3Count_ + 2 * q2DCount_ )/4;
        }

        /**
         * @brief based on quad counts computes perimeter length
         *
         * @return length of component perimeter
         *
         * Method computes perimeter length based on formula in ...
         */
        float getPerimeterLength() const 
        {
            return q2Count_ + c * ( q1Count_ + 2 * q2DCount_ + q3Count_ );
        } 
    private:
        int q1Count_;
        int q3Count_;
        int q2DCount_;
        int q2Count_;

        void updateCounts( const cv::Mat &pattern );
        int getNumberOfNonZeroElement( const cv::Mat &pattern );
        bool isDiagonal( const cv::Mat &pattern );
        void init();
};


/**
 * @brief HorizontalCrossing computes number horizontal crossing,
 * crossing between components pixel and background pixel 
 */
class HorizontalCrossing : public AbstractFeatureExtractor
{
    public:
        std::vector<float> compute( Component &c ) override;
        /**
         * @brief return number of crossing in image row number \p row
         *
         * @param row specifies number of row
         * @param image binary image of component
         *
         * @return number of horizontal crossing in specific row
         */
        int getNumberOfCrossingAt( const int &row, const cv::Mat &image );
    private:
};

/**
 * @brief Class computes ratio between component hole area and its area.
 */
class BackgroundMergeRule : public AbstractFeatureExtractor
{
    public:
        BackgroundMergeRule() = default;
        BackgroundMergeRule( const cv::Mat &bitmap );
        ~BackgroundMergeRule() { }

        std::vector<float> compute( Component &c ) override;

        bool canBeMerged( cv::Point pointOfComponent, cv::Point outsidePoint );
        bool isStartPointOfComponent( cv::Point p );

    private:
        cv::Mat bitmap_;
};




/**
 * @brief Computes number of inflection points of a component.
 */
class InflectionPoints : public AbstractFeatureExtractor 
{
    public:
        std::vector<float> compute( Component &c ) override;
    private:
        size_t computeHullArea( const std::vector<cv::Point> &points );
        int computeNumberOfInflections( const std::vector<cv::Point> &points, double eps );
        
};

/**
 * @brief Compute aspect ratio of component bounding box.
 */
class AspectRatioRotRect : public AbstractFeatureExtractor 
{
    public:
        std::vector<float> compute( Component &c ) override;
    private:
};


/**
 * @brief PerimeterValuesMean computes mean of values of component 
 * perimeter points on any channel of input image
 */
class PerimeterValuesMean : public AbstractFeatureExtractor 
{
    public:
        PerimeterValuesMean() = default;
        PerimeterValuesMean( const cv::Mat &image );

        /**
         * @brief set new image with 4 channels R, G, B, and grayscale 
         *
         * @param values_mat cv::Mat with 4 channels R,G,B and grayscale
         */
        void setValuesMat( const cv::Mat4b &values_mat );

        /**
         * @brief return current 4 channels RGB, Grayscale image
         *
         * @return 4 channels RGB, Grayscale image
         */
        cv::Mat4b getValuesMat() const { return value_mat_; }

        std::vector<float> compute( Component &c ) override;

    private:
        cv::Mat4b value_mat_;
        cv::Size bounds_;
        typedef Neighbourhood<connectivity::eightpass> neigbourhood;

        // std::vector<cv::Point> getPerimeterPoints(Component &c_ptr);
        bool isInBounds( const cv::Point &p, int rows, int cols )
        {
            return p.x >= 0 && p.x < cols && p.y >= 0 && p.y < rows; 
        }
};



/**
 * @brief Implementation of composite design pattern for AbstractFeatureExtractor
 */
class CompositeFeatureExtractor : public AbstractFeatureExtractor
{
    public:
        CompositeFeatureExtractor() = default;
        CompositeFeatureExtractor( std::vector<AbstractFeatureExtractor*> features )
            : features_(features) { }

        ~CompositeFeatureExtractor() 
        {
            for ( AbstractFeatureExtractor* f: features_ )
            {
                if ( f ) delete f;
            }
        }

        std::vector<float> compute( Component &c ) override;
        void addFeatureExtractor( AbstractFeatureExtractor *newFeature )
        {
            features_.push_back( newFeature );
        }

        
    private:
        std::vector<AbstractFeatureExtractor*> features_;
};

/**
 * @brief Computes hog descriptor from binary image of component.
 */
class HogExtractor : public AbstractFeatureExtractor 
{
    public:
        HogExtractor() 
        {
            setShortDescriptor();
        }

        std::vector<float> compute( Component &c ) override;
        void setShortDescriptor();
        void setLongDescriptor();
    private:
        cv::HOGDescriptor hog_;
};


/**
 * @brief Computes sift descriptor from binary image of component.
 */
class SiftExtractor: public AbstractFeatureExtractor
{
    public:
        SiftExtractor();
        std::vector<float> compute( Component &c ) override;
        cv::Mat getKeyPointsDescription(const cv::Mat &image );
        
    private:
        int descriptor_length_;
        cv::SIFT sift_;
        const static int image_size = 48; 
};







#endif

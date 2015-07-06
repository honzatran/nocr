#ifndef NOCRLIB_OPENCV_MSER_H
#define NOCRLIB_OPENCV_MSER_H


#include "component.h"
#include "segment.h"
#include "ocr.h"
#include "structures.h"
#include "features.h"
#include "classifier_wrap.h"
#include "extremal_region.h"


#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>

#include <memory>
#include <vector>

class CvMSERDetection 
{
    public:
        typedef std::shared_ptr<Component> CompPtr;
        typedef std::shared_ptr< LibSVM<feature::ERGeom1> > SvmPointer;

        CvMSERDetection() = default;
        ~CvMSERDetection() { }

        std::vector<CompPtr> getLetters( const cv::Mat &image);
        
        void loadConfiguration( const std::string &config_file );

    private:
        // std::shared_ptr< LibSVM<feature::ERGeom1> > svm_;
        std::shared_ptr< ScalingLibSVM<feature::ERGeom1> > svm_;
        
        std::vector<CompPtr> getCompPtr( const cv::Mat &image );
};

/**
 * @brief ValuesMean computes mean of values of components pixel 
 * on any channel of input image
 */
class ValuesMean 
{
    public:
        ValuesMean() = default;
        /**
         * @brief Constructor initialize input image
         *
         * @param image input image
         */
        ValuesMean( const cv::Mat &image );

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

        std::vector<float> compute( Component &c );

        /**
         * @brief computes color means of points in the Value Mat
         *
         * @param points coordinates in Value Mat
         *
         * @return opencv vector of means
         */
        cv::Vec4f scanMean( const std::vector<cv::Point> &points ) const;
    private:
        cv::Mat4b value_mat_;

        void computeValuesMat( const cv::Mat &gray, const cv::Mat3b &bgr_mat );
};

class MserVisualComputer
{
    public:
        MserVisualComputer() = default;

        void setImage( const cv::Mat &image );
        ImageLetterInfo convert( 
                const std::shared_ptr<Component> &c_ptr) const;
    private:
        ValuesMean values_mean_;
        cv::Size bounds_;
};

template <> 
class SegmentationPolicy<CvMSERDetection>
{
    public:
        typedef std::shared_ptr<Component> MethodOutput;

        static const bool k_perform_nm_suppresion = true;


        static std::vector<MethodOutput> extract
            ( CvMSERDetection * mser_extraction,
              const cv::Mat &image )
        {
            return mser_extraction->getLetters(image);
        }

        static TranslationInfo translate( 
                AbstractOCR *ocr,
                const MethodOutput &c_ptr )
        {
            std::vector<double> probabilities;
            char c = ocr->translate( c_ptr, probabilities );
            return TranslationInfo( c, probabilities );
        }

        static bool haveSignificantOverlap( 
                const MethodOutput &a, 
                const MethodOutput &b )
        {
            cv::Rect a_rect = a->rectangle();
            cv::Rect b_rect = b->rectangle();
            cv::Rect intersection = a_rect & b_rect;

            if ( intersection.area() == 0 )
            {
                return false;
            }

            double overlap_ratio_a = 
                        (double) intersection.area()/a_rect.area(); 
            double overlap_ratio_b = 
                        (double) intersection.area()/b_rect.area(); 

            return (overlap_ratio_a > k_epsilon) 
                    && (overlap_ratio_b > k_epsilon);
        }

        
        static Letter convert( 
                const MethodOutput &c_ptr, 
                const TranslationInfo &translation )
        {
            return Letter( c_ptr, translation );
        }


    private:
        const static double k_epsilon;
};



#endif /* opencv_mser.h */

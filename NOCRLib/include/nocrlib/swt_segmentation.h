#ifndef NOCRLIB_SWT_SEGMENTATION_H
#define NOCRLIB_SWT_SEGMENTATION_H

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <memory>
#include <vector>

#include "component.h"
#include "classifier_wrap.h"
#include "feature_traits.h"
#include "swt.h"
#include "abstract_feature.h"
#include "structures.h"
#include "opencv_mser.h"

class SwtLetterSegmentation
{
    public:
        typedef std::shared_ptr<Component> CompPtr;

        SwtLetterSegmentation();

        std::vector< CompPtr> segmentFromImage( const cv::Mat &image );

        void setMinArea( std::size_t min_area )
        {
            min_area_ = min_area;
        }

        void setMaxArea( std::size_t max_area )
        {
            max_area_ = max_area;
        }

        void loadConfiguration(const std::string & classifer_configuration);

    private:
        std::size_t min_area_, max_area_;

        const int rect_size = 32;
        SwtTransform swt_;
        std::unique_ptr< AbstractFeatureExtractor> geom_feature_;
        

        std::shared_ptr< LibSVM<feature::SwtGeom1> > svm_;

        cv::Mat localBinarization( const cv::Mat &image );

        std::vector< CompPtr > 
            getLetterStorages( const cv::Mat &binary );

};

class SwtStorageConvertor
{
    public:
        void setImage( const cv::Mat &image );

        ImageLetterInfo convert
            ( const LetterStorage<float> & l_storage ) const;
    private:
        ValuesMean values_mean_;
        cv::Size bounds_;
        
};


template <> 
class SegmentationPolicy<SwtLetterSegmentation>
{
    public:
        typedef std::shared_ptr<Component> MethodOutput;

        static const bool k_perform_nm_suppresion = false;

        static std::vector<MethodOutput> extract
            ( SwtLetterSegmentation * swt_segmentation,
              const cv::Mat &image )
        {
            return swt_segmentation->segmentFromImage(image);
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
                const MethodOutput &l_storage,
                const TranslationInfo &translation )
        {
            return Letter( l_storage, translation );
        }


    private:
        const static double k_epsilon;
};

struct SwtGeom1Factory : public AbstractFeatureFactory
{
    FeaturePtr createFeatureExtractor() const 
    {
        CompositeFeatureExtractor* composite = new CompositeFeatureExtractor();
        composite->addFeatureExtractor( new AspectRatio() );
        composite->addFeatureExtractor( new QuadScanner() );
        composite->addFeatureExtractor( new HorizontalCrossing() );

        composite->addFeatureExtractor( new BackgroundMergeRule() );
        composite->addFeatureExtractor( new InflectionPoints() );
        composite->addFeatureExtractor( new SwtRatio() );
        return FeaturePtr( composite ); 
    }
};

template <>
struct FeatureTraits<feature::SwtGeom1>
{
    static const int features_length = 8;
    typedef SwtGeom1Factory FactoryType;
};




#endif /* swt_segmentation.h */

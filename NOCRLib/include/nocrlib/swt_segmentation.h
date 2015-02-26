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

        std::vector< LetterStorage<float> > segmentFromImage( const cv::Mat &image );

        void setMinArea( size_t min_area )
        {
            min_area_ = min_area;
        }

        void setMaxArea( size_t max_area )
        {
            max_area_ = max_area;
        }

    private:
        size_t min_area_, max_area_;

        const int rect_size = 32;
        SwtTransform swt_;
        std::unique_ptr< AbstractFeatureExtractor> geom_feature_;
        

        std::unique_ptr< LibSVM<feature::SwtGeom1> > svm_;

        cv::Mat localBinarization( const cv::Mat &image );

        std::vector< LetterStorage<float> > 
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
        typedef LetterStorage<float> MethodOutput;
        typedef SwtStorageConvertor VisualConvertor;

        static const bool k_perform_nm_suppresion = false;


        static void initialize( VisualConvertor &visual_convertor, 
                const cv::Mat &image )
        {
            visual_convertor.setImage( image );
        }


        static std::vector<MethodOutput> extract
            ( const std::unique_ptr<SwtLetterSegmentation> &swt_segmentation,
              const cv::Mat &image )
        {
            return swt_segmentation->segmentFromImage(image);
        }

        static TranslationInfo translate( 
                AbstractOCR *ocr,
                const MethodOutput &l_storage )
        {
            std::vector<double> probabilities;
            char c = ocr->translate( l_storage.c_ptr_, probabilities );
            return TranslationInfo( c, probabilities );
        }

        static bool haveSignificantOverlap( 
                const MethodOutput &a, 
                const MethodOutput &b )
        {
            cv::Rect a_rect = a.c_ptr_->rectangle();
            cv::Rect b_rect = b.c_ptr_->rectangle();
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
                const VisualConvertor &visual_convertor,
                const MethodOutput &l_storage,
                const TranslationInfo &translation )
        {
            ImageLetterInfo visual = visual_convertor.convert( l_storage );
            return Letter( l_storage.c_ptr_, visual, translation );
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

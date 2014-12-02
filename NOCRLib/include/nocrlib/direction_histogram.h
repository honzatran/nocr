/**
 * @file direction_histogram.h
 * @brief class computing the direction histogram 
 * proposed by Gomez
 * @author Tran Tuan Hiep
 * @version 
 * @date 2014-09-16
 */

#ifndef NOCRLIB_DIRECTION_HISTOGRAM_H
#define NOCRLIB_DIRECTION_HISTOGRAM_H

#include "features.h"
#include "component.h"
#include "feature_traits.h"
#include "abstract_feature_factory.h"

#include <opencv2/core/core.hpp>

#include <vector>

/**
 * @brief computes direction histogram proposed by Gomez
 */
class DirectionHistogram : public AbstractFeatureExtractor
{
    public:
        DirectionHistogram() = default;

        std::vector<float> compute( Component &c ) override;

        /**
         * @brief compute direction histogram from image
         *
         * @param image input image
         *
         * @return direction histogram 
         */
        std::vector<float> computeHistogram( const cv::Mat &image );
    private:
        const static int cell_size = 32;
        const static int image_size = 128;

        std::vector<float> extract( const cv::Mat &edges, const cv::Mat &grad_x, const cv::Mat &grad_y );
};


/**
 * @brief creates return composite computing direction histogram
 */
struct DirectionHistogramFactory : public AbstractFeatureFactory
{
    DirectionHistogramFactory() = default;

    FeaturePtr createFeatureExtractor() const 
    {
        return FeaturePtr( new DirectionHistogram() );
    }
};

/// @cond
template<>
struct FeatureTraits<feature::DirectionHist>
{
    static const int features_length = 128;
    typedef DirectionHistogramFactory FactoryType;
};
/// @endcond

#endif /* direction_histogram.h */



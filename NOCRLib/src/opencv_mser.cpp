#include "../include/nocrlib/opencv_mser.h"
#include "../include/nocrlib/component.h"
#include "../include/nocrlib/swt.h"
#include "../include/nocrlib/assert.h"

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>

const double SegmentationPolicy<CvMSERDetection>::k_epsilon = 0.7;



void CvMSERDetection::loadConfiguration( const std::string &config_file )
{
    if ( svm_ == nullptr )
    {
        // svm_ = std::make_shared< LibSVM<feature::ERGeom1> >();
        svm_ = std::make_shared< ScalingLibSVM<feature::ERGeom1> >();
    }

    // svm_->loadConfiguration( config_file, "scaling_er2stage.conf" );
    svm_->loadConfiguration( config_file );
}



auto CvMSERDetection::getLetters( const cv::Mat &image ) 
    -> vector<CompPtr>
{
    vector<CompPtr> comp_ptrs;

    if ( image.type() == CV_8UC3 )
    {
        cv::Mat gray_image;
        cv::cvtColor( image, gray_image, CV_BGR2GRAY );
        comp_ptrs = getCompPtr( gray_image );
    }
    else 
    {
        comp_ptrs = getCompPtr( image );
    }

    FeatureTraits<feature::ERGeom1>::FactoryType factory;
    auto features_extractor = factory.createFeatureExtractor();

    std::vector<CompPtr> output;
    for ( const auto &c_ptr : comp_ptrs )
    {
        auto descriptor  = features_extractor->compute( c_ptr );
        if ( svm_->predict( descriptor) == 1 )
        {
            output.push_back(c_ptr);
        }
    }

    return output;
}

auto CvMSERDetection::getCompPtr( const cv::Mat &image )
    -> std::vector<CompPtr>
{
    int tmp = 0.00007 * image.cols * image.rows;

    double min_area_ratio, max_area_ratio;
    std::tie(min_area_ratio, max_area_ratio) = ErLimitSize::getErSizeLimits(image.size());

    std::size_t area = image.size().area();

    int min_size = std::max<int>(20, min_area_ratio * area); 
    int max_size = max_area_ratio * area;

    cv::MSER extractor( 3, min_size, max_size, 0.1, 0.1 );
    std::vector<std::vector<cv::Point> > msers;
    
    extractor( image,  msers );
    vector<CompPtr> components;
    components.reserve( msers.size() );
    for ( const auto &vec: msers ) 
    {
        CompPtr pointer( new Component(vec) ); 
        components.push_back( pointer ); 
    }
    return components;
}


ValuesMean::ValuesMean( const cv::Mat &image )
{
    if ( image.channels() == 3 )
    {
        cv::Mat gray;
        cv::cvtColor( image, gray, CV_BGR2GRAY ); 
        computeValuesMat( gray, image );
    }
    else if ( image.channels() == 1 )
    {
        cv::Mat bgr;
        cv::cvtColor( image, bgr, CV_GRAY2BGR );
        computeValuesMat( image, bgr );
    }
}

void ValuesMean::computeValuesMat( const cv::Mat &gray, const cv::Mat3b &bgr )
{
    vector<cv::Mat> splitted_channels;
    cv::split( bgr, splitted_channels );
    splitted_channels.push_back(gray);

    cv::merge( splitted_channels, value_mat_ );
}




void ValuesMean::setValuesMat( const cv::Mat4b &values_mat )
{
    value_mat_ = values_mat;
}

std::vector<float> ValuesMean::compute( Component &c )
{
    cv::Vec4f means = scanMean( c.getPoints() );

    vector<float> output; output.reserve(4);
    for ( int i = 0; i < 4; ++i )
    {
        output.push_back( (float) means[i] );
    }
    return output;
}

// std::vector<float> ValuesMean::scanMean
cv::Vec4f ValuesMean::scanMean
        ( const std::vector<cv::Point> &points ) const
{
    cv::Vec4f means( 0, 0, 0, 0 );
    for( const cv::Point &p: points )
    {
        means += value_mat_.at<cv::Vec4b>( p.y, p.x );
    }

    /*
     * for ( int i = 0; i < 4; ++i )
     * {
     *     means[i] /= (float) points.size();
     * }
     */
    means /= (float) points.size();

    return means;
}

void MserVisualComputer::setImage( const cv::Mat &image )
{
    values_mean_ = ValuesMean(image);
    bounds_ = image.size();
}

ImageLetterInfo MserVisualComputer::convert( 
        const std::shared_ptr<Component> &c_ptr) const 
{
    cv::Vec4f component_means = values_mean_.scanMean(
            c_ptr->getPoints() );

    cv::Vec4f perim_means = values_mean_.scanMean(
            getPerimeterPoints( *c_ptr, bounds_ ) );

    SwtMean swt_mean_extractor;
    float swt_mean = swt_mean_extractor.computeSwtMean(*c_ptr);

    return ImageLetterInfo( component_means, perim_means, swt_mean );
}

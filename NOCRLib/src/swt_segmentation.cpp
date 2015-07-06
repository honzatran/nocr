
#include "../include/nocrlib/swt_segmentation.h"
#include "../include/nocrlib/utilities.h"
#include "../include/nocrlib/extremal_region.h"

SwtLetterSegmentation::SwtLetterSegmentation()
{
    min_area_ = 100;
    max_area_ = 50000;
    svm_ = nullptr;
    
    FeatureTraits<feature::ERGeom1>::FactoryType factory;

    geom_feature_ = factory.createFeatureExtractor();

}

void SwtLetterSegmentation::loadConfiguration(const std::string & classifer_configuration)
{
    if (!svm_)
    {
        svm_ = create<LibSVM, feature::SwtGeom1>();
    }

    svm_->loadConfiguration(classifer_configuration);
}

auto SwtLetterSegmentation::segmentFromImage( const cv::Mat &image )
    -> std::vector< CompPtr >
{
    cv::Mat grayscale;

    if (image.type() == CV_8UC3)
    {
        cv::cvtColor(image, grayscale, CV_BGR2GRAY);
    }
    else
    {
        grayscale = image;
    }

    cv::Mat binary = localBinarization(grayscale);

    std::vector< CompPtr > 
            comp_ptrs = getLetterStorages(binary);

    cv::bitwise_not(binary, binary);
    auto tmp = getLetterStorages( binary );

    comp_ptrs.insert( comp_ptrs.end(), tmp.begin(), tmp.end() );

    cout << comp_ptrs.size() << endl;
    return comp_ptrs;
}

cv::Mat SwtLetterSegmentation::localBinarization( const cv::Mat &image )
{
    int rows_pad = rect_size - (image.rows % rect_size);
    int cols_pad = rect_size - (image.cols % rect_size);

    cv::Mat binary;
    cv::copyMakeBorder( image, binary, 0, rows_pad, 0, cols_pad, cv::BORDER_CONSTANT, 0);

    for ( int i = 0; i < binary.rows; i += rect_size ) 
    {
        for ( int j = 0; j < binary.cols; j += rect_size )
        {
            cv::Mat cropped( binary, 
                    cv::Rect( j, i, rect_size, rect_size ) );

            cv::threshold( cropped, cropped, 0, 255, cv::THRESH_OTSU | cv::THRESH_BINARY );
        }
    }

    return binary;
}

auto SwtLetterSegmentation::getLetterStorages( const cv::Mat &binary )
    -> std::vector< CompPtr >
{
    cv::Mat swt_image = swt_(binary);


    ComponentFinder< SwtMergerRule, 
        connectivity::eightpass > component_finder( swt_image, SwtMergerRule( swt_image ) );

    double min_area_ratio, max_area_ratio;
    std::tie(min_area_ratio, max_area_ratio) = ErLimitSize::getErSizeLimits(binary.size());

    std::size_t area = binary.size().area();

    min_area_ = std::max<std::size_t>(20, min_area_ratio * area); 
    max_area_ = max_area_ratio * area;


    auto comps = component_finder.findAllComponents();
    std::vector< CompPtr > comp_ptrs;
    for ( auto &c : comps )
    {
        if ( c.size() > min_area_ &&  c.size() < max_area_ )
        {
            auto descriptor = geom_feature_->compute(c);
            float mean, std_dev;
            std::tie(mean, std_dev) = getMeanStdDev<float>(c.getPoints(), swt_image);
            std::size_t k = c.size();

            descriptor.push_back(std_dev/mean * (1 + 1/(4*k)));

            // gui::showImage(c.getBinaryMat(), "comp");

            if ( svm_->predict( descriptor ) != 0)
            {
                comp_ptrs.push_back( 
                            std::make_shared<Component>(c));
            }
        }
    }

    return comp_ptrs;
}

void SwtStorageConvertor::setImage( const cv::Mat &image )
{
    values_mean_ = ValuesMean(image);
    bounds_ = image.size();
}

ImageLetterInfo SwtStorageConvertor::convert
        ( const LetterStorage<float> & l_storage ) const
{
    auto c_ptr = l_storage.c_ptr_;

    cv::Vec4f component_means = values_mean_.scanMean(
            c_ptr->getPoints() );

    cv::Vec4f perim_means = values_mean_.scanMean(
            getPerimeterPoints( *c_ptr, bounds_ ) );

    return ImageLetterInfo( component_means, perim_means, l_storage.stat_ );
}


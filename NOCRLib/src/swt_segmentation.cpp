
#include "../include/nocrlib/swt_segmentation.h"
#include "../include/nocrlib/utilities.h"

SwtLetterSegmentation::SwtLetterSegmentation()
{
    min_area_ = 100;
    max_area_ = 50000;
    svm_ = nullptr;
    geom_feature_ = nullptr;
}

auto SwtLetterSegmentation::segmentFromImage( const cv::Mat &image )
    -> std::vector< LetterStorage<float> >
{
    cv::Mat binary = localBinarization(image);

    std::vector< LetterStorage<float> > 
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
    -> std::vector< LetterStorage<float> >
{
    cv::Mat swt_image = swt_(binary);

    ComponentFinder< SwtMergerRule, 
        connectivity::eightpass > component_finder( swt_image, SwtMergerRule( swt_image ) );

    auto comps = component_finder.findAllComponents();
    std::vector< LetterStorage<float> > comp_ptrs;
    for ( auto &c : comps )
    {
        if ( c.size() > min_area_ &&  c.size() < max_area_ )
        {
            auto descriptor = geom_feature_->compute( c );
            cv::Scalar_<float> swt_mean, swt_stddev;
            cv::meanStdDev( swt_image, swt_mean, swt_stddev, c.getPoints() );
            descriptor.push_back( swt_mean[0]/swt_stddev[0] );

            if ( svm_->predict( descriptor ) )
            {
                comp_ptrs.push_back( 
                        LetterStorage<float>( 
                            std::make_shared<Component>( std::move(c) ),
                            swt_mean[0]) );
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

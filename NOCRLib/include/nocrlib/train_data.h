/**
 * @file train_data.h
 * @brief file containing class for computing descriptors
 * from training data used in training, saving computed descriptors
 * to output file and loading descriptors from file.
 * @author Tran Tuan Hiep
 * @version 
 * @date 2014-09-14
 */


#ifndef NOCRLIB_TRAIN_DATA_H
#define NOCRLIB_TRAIN_DATA_H

#include "abstract_feature.h"
#include "iooper.h"
#include "feature_traits.h"

#include <memory>
#include <vector>
#include <string>
#include <algorithm>

/**
 * @brief enum for component extraction in image 
 * to perform training on features extracted from components
 */
enum class extraction { BlackAndWhite, BWMaxComponent };


/**
 * @brief policy class, that specialize the extraction
 *
 * Policy design pattern usage. We must alway specialize this class by 
 * template parameter e to use extraction e and implement method 
 * static std::vector<Component> extract( const cv::Mat &image ) 
 *
 * @tparam E extraction type
 */
template <extraction E> 
class TrainExtractionPolicy
{
    /*
     * static std::vector<Component> extract( const cv::Mat &image );
     */
};

/// @cond
template <> 
class TrainExtractionPolicy<extraction::BlackAndWhite>
{
    public:
        static std::vector<Component> extract( const cv::Mat &image ) 
        {
            ComponentFinder<ComponentMergeRule, connectivity::eightpass> cf( image );
            return cf.findAllComponents();
        }

};

template<>
class TrainExtractionPolicy<extraction::BWMaxComponent>
{
    public:
        static std::vector<Component> extract( 
                const cv::Mat &image ) 
        {
            ComponentFinder<ComponentMergeRule, connectivity::eightpass> cf( image );
            std::vector<Component> comp = cf.findAllComponents();

            auto comp_it = std::max_element(comp.begin(), comp.end(), [] (const Component & a, const Component & b) 
                    {
                        return a.size() < b.size();
                    });

            return { *comp_it };
        }
            
};

/// @endcond

/**
 * @brief compute features from list of training samples and
 * save to output file.
 *
 * @tparam F specifies what kind of composite we use
 * @tparam E specifies what kind of extraction we use
 */
template < feature F, extraction E = extraction::BlackAndWhite >
struct TrainDataCreator
{
    static const int k_min_size = 70;
    std::unique_ptr<AbstractFeatureExtractor> feature_;


    /**
     * @brief constructor
     */
    TrainDataCreator() 
    {
        typename FeatureTraits<F>::FactoryType factory;
        feature_ = factory.createFeatureExtractor();
    }

    /**
     * @brief constructor
     *
     * @param factory factory to create specific composite
     */
    TrainDataCreator( const typename FeatureTraits<F>::FactoryType &factory )
    {
        feature_ = factory.createFeatureExtractor();
    }
    
    /**
     * @brief computes descriptor from training samples 
     * and saves thme to \p output file.
     *
     * @param list_training_samples list of training samples
     * @param output path to output file
     */
    void loadAndProcessSamples( const std::string &list_training_samples, const std::string &output )
    {
        loader ld;
        std::vector<fileInfo> fileList = ld.getFileList( list_training_samples ); 
        std::ofstream ofs(output);
        OutputWriter out( &ofs );

        for ( const auto &info: fileList )
        {
            cv::Mat image = cv::imread( info.getPathToFile(), CV_LOAD_IMAGE_GRAYSCALE );
            if ( image.empty() )
            {
                std::cout << info.getPathToFile() << std::endl;
                // TODO
                // throw exception
                continue;
            }

            auto comps = TrainExtractionPolicy<E>::extract(image);
            for( auto &c : comps )
            {
                if ( c.size() > k_min_size ) 
                {
                    std::vector<float> sample = feature_->compute( c );
                    out.write( sample, info.getLabel() );
                }
            }
        }
    }
};

/**
 * @brief Class loads and prepare data for classifier training from file
 */
class TrainDataLoader
{
    public:
        /**
         * @brief constructor
         *
         * @param features_length length of one training sample
         */
        TrainDataLoader( int features_length )
            : features_length_( features_length ) { }


        /**
         * @brief reads training data from \p data_file and 
         * loads training descriptors them to \p train_data 
         * and labels of their classes to \p labels
         *
         * @param data_file file with training points
         * @param train_data matrix where methods loads samples
         * @param labels matrix where methods stores labels of samples
         */
        void prepareDataForTraining
            ( const std::string &data_file, cv::Mat &train_data, cv::Mat &labels );
    private:
        int features_length_;
};
#endif /* train_data_creator.h */

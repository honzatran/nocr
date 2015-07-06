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
#include "direction_histogram.h"

#include <memory>
#include <vector>
#include <string>
#include <algorithm>
#include <limits>

#define TRAIN_DATA_DEBUG 0

/**
 * @brief enum for component extraction in image 
 * to perform training on features extracted from components
 */
enum class extraction { BlackAndWhite, BWMaxComponent, OneComponent };


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

template <>
class TrainExtractionPolicy<extraction::OneComponent>
{
    public:
        static std::vector<Component> extract(const cv::Mat & image)
        {
            std::vector<cv::Point> points;

            for (auto it = image.begin<uchar>(); it != image.end<uchar>(); ++it)
            {
                if (*it > 200)
                {
                    points.push_back(it.pos());
                }
            }

            return { Component(points) };
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
#if TRAIN_DATA_DEBUG
            if (comp.size() > 1)
            {
                for (auto & c : comp)
                {
                    gui::showImage(c.getBinaryMat(), "comp");
                }

                auto whole_comp = TrainExtractionPolicy<extraction::OneComponent>::extract(image);

                gui::showImage(whole_comp.front().getBinaryMat(), "whole component");
            }
#endif

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
        std::ofstream ofs(output);
        std::ifstream ifs(list_training_samples);
        
        if (!ofs.is_open())
        {
            // throw exception
        }

        if (!ifs.is_open()) 
        {
            //throw exception
        }


        ofs.precision(std::numeric_limits<float>::digits10);

        std::string buffer;

        while (std::getline(ifs, buffer))
        {
            std::string file_path;
            int label;

            std::size_t pos = buffer.find_last_of(':');
            file_path = buffer.substr(0, pos);
            label = std::stoi(buffer.substr(pos + 1));

            cv::Mat image = cv::imread( file_path, CV_LOAD_IMAGE_GRAYSCALE );
#if TRAIN_DATA_DEBUG
            std::cout << file_path << std::endl;
#endif
            if ( image.empty() )
            {
                // std::cout << info.getPathToFile() << std::endl;
                // TODO
                // throw exception
                continue;
            }

            auto comps = TrainExtractionPolicy<E>::extract(image);
            for( auto &c : comps )
            {
                std::vector<float> sample = feature_->compute( c );
                for (std::size_t i = 0; i < sample.size(); ++i)
                {
                    ofs << sample[i] << ' ';
                }

                ofs << label << std::endl;
            }
        }

        ofs.close();
        ifs.close();
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

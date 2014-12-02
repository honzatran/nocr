/**
 * @file bag_of_words.h
 * @brief header implementation of bag of words(keypoints) approach
 * @author Tran Tuan Hiep
 * @version 1.0
 * @date 2014-09-12
 */


#ifndef NOCRLIB_BAG_OF_WORDS_H
#define NOCRLIB_BAG_OF_WORDS_H

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>

#include "key_point_extractor.h"
#include "abstract_feature.h"
#include "abstract_feature_factory.h"
#include "component.h"

#include <string>


/**
 * @brief produces visual vocabulary using K-Means clustering on given 
 * descriptors
 *
 * BoWWrap is a wrap of OpenCV implementation cv:BOWKMeansTrainer. 
 */
class BoWWrap
{
    public:
        typedef std::unique_ptr< IKeyPointDescriptor >  KeyPointDescPtr;

        /**
         * @brief constuctor
         *
         * @param extractor unique pointer to instance of derived class from IKeyPointDescriptor
         * @param vocabulary_size size of visual vocabulary
         */
        BoWWrap( KeyPointDescPtr extractor, int vocabulary_size );

        /**
         * @brief add descriptors to the list
         *
         * @param train_data new descriptors
         *
         * One row of \p train_data is one descriptor.
         */
        void addDescriptors( const cv::Mat &train_data );

        /**
         * @brief add descriptors to the list, from file \p train_data
         *
         * @param train_file path to the file with descriptors
         */
        void addDescriptors( const std::string &train_file );

        /**
         * @brief compute descriptors from file in file list and add to the list 
         * of descritors
         *
         * @param file_list
         */
        void computeAndAddDescriptors( const std::string &file_list );

        /**
         * @brief run kmeans algorithm on list of added descriptors
         *
         * @return cv::Mat, where each row is one word in visual vocabulary
         */
        cv::Mat getClusters();

        /**
         * @brief run kmeans algorithm on added descritors and save to
         * the file \p output
         *
         * @param output path to output file
         */
        void saveClusters(const std::string &output);

    private:
        cv::BOWKMeansTrainer bow_kmeans_ = cv::BOWKMeansTrainer(0);
        KeyPointDescPtr extractor_;
        int vocabulary_size_;
};

/**
 * @brief computes bow descriptor based on key point descriptor
 * and visual vocabulary.
 */
class BoWDesc : public AbstractFeatureExtractor
{
    public:
        typedef std::unique_ptr< IKeyPointDescriptor >  KeyPointDescPtr;

        /**
         * @brief constructor
         */
        BoWDesc() = default;

        /**
         * @brief constructor
         *
         * @param vocabulary_file file with visual vocabulary
         * @param key_points_desc unique pointer to instance of derived class from IKeyPointDescriptor
         *
         * Loads visual vocabulary and keypoint descriptor, which are necessery to
         * compute BoW descriptor. Keypoint descriptor is used to extract features, on which
         * we compute BoW descriptor using the visual vocabulary.
         */
        BoWDesc( const std::string &vocabulary_file, KeyPointDescPtr key_points_desc);
    
        /**
         * @brief initiaze BoWDesc
         *
         * @param vocabulary_file file with visual vocabulary
         * @param key_points_desc unique pointer to instance of derived class from IKeyPointDescriptor
         *
         * Loads visual vocabulary and keypoint descriptor, which are necessery to
         * compute BoW descriptor. Keypoint descriptor is used to extract features, on which
         * we compute BoW descriptor using the visual vocabulary.
         */
        void initialize(const std::string &vocabulary_file, KeyPointDescPtr key_points_desc);


        /**
         * @brief compute BoW descriptor for image
         *
         * @param image input image, CV_8UC1 format is expected
         *
         * @return BoW descriptor 
         *
         * BoW descriptors are computed directly from the image, without any
         * specific manipulation with the image.
         */
        std::vector<float> getDescriptor( const cv::Mat &image );

        /**
         * @brief 
         *
         * @param c
         *
         * @return 
         *  
         *  Compute BoW descriptor from binary image of \p c.
         */
        std::vector<float> compute( Component &c ) override;


        /**
         * @brief create training BoW descriptors for classifier training
         *
         * @param file_list list of files used for training 
         * @param output output file 
         *
         * Methods compute BoW descriptors for every file in file list and 
         * then save the descriptors to the output file. There is no specific 
         * manipulation with images in the file list.
         */
        void createTrainingData( const std::string &file_list, const std::string &output );

    private:
        cv::FlannBasedMatcher flann_;
        KeyPointDescPtr key_points_desc_;
        int dictionary_size_;
};

/**
 * @brief creates composite of features for BoW
 *
 * @tparam D type of key point descriptor
 * It's necessary D to be a derived class of IKeyPointDescriptor.
 */
template <typename D> 
struct BoWFactory: AbstractFeatureFactory
{
    BoWFactory( const std::string &vocabulary_file )
        : vocabulary_file_( vocabulary_file )
    {

    } 

    FeaturePtr createFeatureExtractor() const 
    {
        std::unique_ptr<IKeyPointDescriptor> key_point_desc ( new D() );
        return FeaturePtr( new BoWDesc( vocabulary_file_, std::move(key_point_desc) ) );
    }

    std::unique_ptr<BoWDesc> createKeypointExtractor() const 
    {
        std::unique_ptr<IKeyPointDescriptor> key_points_desc( new D() );
        return std::unique_ptr<BoWDesc>( 
                new BoWDesc(vocabulary_file_, std::move(key_points_desc)) );
    }

    std::string vocabulary_file_;
};






#endif/* bag_of_words.h */

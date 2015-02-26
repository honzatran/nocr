/**
 * @file key_point_extractor.h
 * @brief interface for key point extraction
 * @author Tran Tuan Hiep
 * @version 1.0
 * @date 2014-09-16
 */

#ifndef NOCRLIB_KEY_POINT_EXTRACTOR_H
#define NOCRLIB_KEY_POINT_EXTRACTOR_H

#include <string>
#include <memory>
#include <opencv2/core/core.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <random>

/*
 * extern "C" 
 * {
 *     #include <vl/generic.h>
 *     #include <vl/dsift.h>
 * }
 */

#include "features.h"

/**
 * @brief Interface for key point descriptors
 */
class IKeyPointDescriptor 
{
    public:
        /**
         * @brief extract descriptors from image 
         *
         * @param image input image
         *
         * @return list of desriptors
         *
         * Each row of output matrix is one descriptor extracted from one keypoint.
         */
        virtual cv::Mat_<float> getKeyPointsDescription(const cv::Mat &image) = 0;

        /**
         * @brief loads descriptors from \p file
         *
         * @param file path to file
         *
         * @return loaded descriptors
         */
        virtual cv::Mat loadFromFile( const std::string &file ) = 0;
};


/**
 * @brief sift key descriptor from lowe
 */
class SiftKeyPointDescriptor: public IKeyPointDescriptor
{
    public:
        const static size_t descriptor_length_ = 128;

        /**
         * @brief constructor
         */
        SiftKeyPointDescriptor();
        cv::Mat_<float> getKeyPointsDescription(const cv::Mat &image) override;
        cv::Mat loadFromFile( const std::string &file );
    private:
        cv::SIFT sift_;
};

class Random;

/**
 * @brief extract keypoint descriptors from list of training images in \p file_list
 */
class KeyPointDescriptorUtility
{
    public:
        typedef std::unique_ptr<IKeyPointDescriptor> KeyPointDescPtr; 
        /**
         * @brief extract all descriptors from files in \p file_list
         *
         * @param ptr unique pointer to IKeyPointDescriptor
         * @param file_list list of training samples
         * @param output_file output file
         */
        void extractDescriptors( const KeyPointDescPtr &ptr, const std::string &file_list, const std::string &output_file );
        /**
         * @brief extract random number of descriptors from files in \p file_list
         *
         * @param ptr unique pointer to IKeyPointDescriptor
         * @param file_list list of training samples
         * @param output_file output file
         * @param number_random number of random samples
         */
        void extractDescriptorsRandom( const KeyPointDescPtr &ptr, const std::string &file_list, const std::string &output_file,
                int number_random );
    private:
        std::vector<int> getRandom( int max, int number_random ); 

};

#endif /* key_point_extractor */



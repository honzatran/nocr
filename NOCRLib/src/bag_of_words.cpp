/* 
 * Tran Tuan Hiep
 * Implementation of methods and classes declared in bag_of_words.h
 *
 * Compiler: g++ 4.8.3, 
 */

#include "../include/nocrlib/bag_of_words.h"
#include "../include/nocrlib/iooper.h"
#include "../include/nocrlib/utilities.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <vector>
#include <string>
#include <iostream>
#include <fstream>

using namespace std;

BoWWrap::BoWWrap( KeyPointDescPtr extractor, int vocabulary_size )
    : extractor_( std::move(extractor) ), vocabulary_size_(vocabulary_size) 
{
    const cv::TermCriteria term_criteria( cv::TermCriteria::EPS | cv::TermCriteria::COUNT, 1e5, 1e-3 );
    const int attemps = 5;
    const int flags = cv::KMEANS_PP_CENTERS;
    bow_kmeans_ = cv::BOWKMeansTrainer( vocabulary_size, term_criteria, attemps, flags );
}


void BoWWrap::computeAndAddDescriptors( const std::string &file_list )
{
    loader ld;
    vector<string> files = ld.getFileContent(file_list);
    for ( const std::string &file_name : files )
    {
        cv::Mat image = cv::imread( file_name, CV_LOAD_IMAGE_GRAYSCALE );
        cv::Mat_<float> key_points_desc = extractor_->getKeyPointsDescription( image );
        if ( key_points_desc.empty() )
        {
            continue;
        }

        addDescriptors( key_points_desc );
    }
}

void BoWWrap::addDescriptors( const std::string &train_file )
{
    cv::Mat train_data = extractor_->loadFromFile( train_file );
    addDescriptors( train_data );
}


void BoWWrap::addDescriptors( const cv::Mat &train_data )
{
    bow_kmeans_.add( train_data );
}

cv::Mat BoWWrap::getClusters() 
{
    return bow_kmeans_.cluster();
}

void BoWWrap::saveClusters( const std::string &output )
{
    cv::Mat clusters = getClusters();
    std::ofstream oss(output);
    OutputWriter writer(&oss);
    for ( int i = 0; i < clusters.rows; ++i )
    {
        cv::Mat curr_row = clusters.row(i);
        std::vector<float> cluster( curr_row.begin<float>(), curr_row.end<float>() );
        writer.writeln(cluster);
    }

}

// ==============================BowDesc====================
BoWDesc::BoWDesc( const std::string &dictionary_file, KeyPointDescPtr key_points_desc )
    : key_points_desc_(std::move( key_points_desc ))
{
    cv::Mat dictionary = key_points_desc_->loadFromFile( dictionary_file );
    flann_.add( vector<cv::Mat>(1,dictionary) );
    dictionary_size_ = dictionary.rows;
}

void BoWDesc::initialize( const std::string &vocabulary_file, KeyPointDescPtr key_points_desc )
{
    key_points_desc_ = std::move( key_points_desc );

    cv::Mat dictionary = key_points_desc_->loadFromFile( vocabulary_file );
    flann_.add( vector<cv::Mat>(1,dictionary) );
    dictionary_size_ = dictionary.rows;
}

void BoWDesc::createTrainingData( const std::string &file_list, const std::string &output )
{
    // loader ld;
    // auto train_images_path = ld.getFileList(file_list);
    // std::ofstream oss(output);
    // OutputWriter out(&oss);
    //
    // for ( const auto &info: train_images_path )
    // {
    //     cv::Mat image = cv::imread( info.getPathToFile(), CV_LOAD_IMAGE_GRAYSCALE );
    //     if ( image.empty() )
    //     {
    //         // std::cout << info.getPathToFile() << std::endl;
    //         // TODO
    //         // throw exception
    //         continue;
    //     }
    //     // auto samples = featUtil::extract( image );
    //     auto sample = getDescriptor(image);
    //     out.write( sample, info.getLabel() );
    // }
}



std::vector<float> BoWDesc::getDescriptor( const cv::Mat &image )
{
    cv::Mat key_points_descriptor = key_points_desc_->getKeyPointsDescription(image);
    vector<cv::DMatch> matches;
    flann_.match( key_points_descriptor, matches );
    std::vector<float> output( dictionary_size_, 0 );
    // float magnitude = 0;

    for ( const auto &match : matches )
    {
        int cluster_index = match.trainIdx;
        // int old_val = output[cluster_index];
        output[cluster_index] += 1;
        // magnitude += 2 * old_val + 1; 
    }
    // magnitude = std::sqrt( magnitude );
    //
    // if ( magnitude == 0 )
    // {
    //     return output;
    // }
    //
    // for ( float &f : output )
    // {
    //     f /= magnitude;
    // }

    return output;
}

std::vector<float> BoWDesc::compute( Component &c )
{
    return getDescriptor( c.getBinaryMat() );
}

    







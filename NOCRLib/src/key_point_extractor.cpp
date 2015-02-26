/* 
 * Tran Tuan Hiep
 * Implementation of methods and classes declared in key_point_extractor.h
 *
 * Compiler: g++ 4.8.3
 */
#include <string>
#include <vector>
#include <iostream>
#include <random>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "key_point_extractor.h"
#include "iooper.h"
#include "exception.h"


using namespace std;


SiftKeyPointDescriptor::SiftKeyPointDescriptor()
{
    sift_ = cv::SIFT( 0, 3, 0.04, 10, 1.6 ); 
}


auto SiftKeyPointDescriptor::getKeyPointsDescription( const cv::Mat &image )
    -> cv::Mat_<float>
{
    vector<cv::KeyPoint> key_points; 
    cv::Mat_<float> descriptors;
    sift_( image, cv::Mat(), key_points, descriptors );
    return descriptors;
}


cv::Mat SiftKeyPointDescriptor::loadFromFile( const std::string &file )
{
    loader ld;
    auto matrix = ld.loadDataToFloatMatrix( file );
    int rows = matrix.size();
    int cols = descriptor_length_;
    cv::Mat desc( rows, cols, CV_32FC1 );
    for ( int i = 0; i < rows; ++i )
    {
        if ( matrix[i].size() != descriptor_length_ ) 
        {
            throw BadFileFormatting( file + " line" + to_string(i) + "#elems != " + to_string(descriptor_length_) ); 
        }

        for ( int j = 0; j < cols; ++j )         
        {
            desc.at<float>(i,j) = matrix[i][j];
        }
    }
    return desc; 
}

//================================================
void KeyPointDescriptorUtility::extractDescriptors( const KeyPointDescPtr &ptr, 
        const std::string &file_list, const std::string &output_file )
{
    loader ld;
    vector<string> files = ld.getFileContent(file_list);
    OutputWriter output( output_file );
    for ( const std::string &file_name : files )
    {
        cv::Mat image = cv::imread( file_name, CV_LOAD_IMAGE_GRAYSCALE );
        cv::Mat_<float> key_points_desc = ptr->getKeyPointsDescription( image );
        for ( int i = 0; i < key_points_desc.rows; ++i )
        {
            cv::Mat_<float> curr_row = key_points_desc.row(i);
            vector<float> desc( curr_row.begin(), curr_row.end() ); 
            output.writeln(desc);
        }
    }
}

void KeyPointDescriptorUtility::extractDescriptorsRandom( const KeyPointDescPtr &ptr, 
        const std::string &file_list, const std::string &output_file, int number_random )
{
    loader ld;
    vector<string> files = ld.getFileContent(file_list);
    OutputWriter output( output_file );
    for ( const std::string &file_name : files )
    {
        cv::Mat image = cv::imread( file_name, CV_LOAD_IMAGE_GRAYSCALE );
        cv::Mat_<float> key_points_desc = ptr->getKeyPointsDescription( image );
        vector<int> generatedKeypoints = getRandom( key_points_desc.rows - 1, number_random );
        for ( int i : generatedKeypoints )
        {
            cv::Mat_<float> curr_row = key_points_desc.row(i);
            vector<float> desc( curr_row.begin(), curr_row.end() ); 
            output.writeln( desc );
        }
    }
}

vector<int> KeyPointDescriptorUtility::getRandom( int max, int number_random ) 
{
    if ( number_random > max )
    {
        vector<int> output( max, 0 );
        for ( int i = 0; i < max; ++i )
        {
            output[i] = i;
        }
        return output;  
    }

    std::random_device rdev{};
    std::default_random_engine generator(rdev());
    std::uniform_int_distribution<int> distribution(0, max);

    vector<int> output( number_random , 0);
    vector<bool> mask( max, false );
    for ( int i = 0; i < number_random; ++i )
    {
        int val = distribution(generator);
        while( mask[val] )
        {
            val = distribution(generator); 
        }
        mask[val] = true;
        output[i] = val;
    }

    return output;
}


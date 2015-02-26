/* 
 * Tran Tuan Hiep
 * Implementation of methods and classes declared in train_data.h
 *
 * Compiler: g++ 4.8.3
 */

#include "../include/nocrlib/train_data.h"

void TrainDataLoader::prepareDataForTraining
        ( const std::string &data_file, cv::Mat &train_data, cv::Mat &labels )
{
    loader ld;
    std::vector< std::vector<float> > data = ld.loadDataToFloatMatrix( data_file );
    // sort samples by their labels
    std::sort( data.begin(), data.end(), 
            [this] ( const std::vector<float> &a, const std::vector<float> &b )
            {
                return a[features_length_] < b[features_length_];
            });
    
    train_data = cv::Mat( data.size(), features_length_, CV_32FC1 );
    labels = cv::Mat( data.size(), 1, CV_32FC1 );
    for( size_t i = 0; i < data.size(); ++i )
    {
        for( int j = 0; j < features_length_; ++j )
        {
            train_data.at<float>(i,j) = data[i][j];
        }
        labels.at<float>(i,0) = data[i][features_length_];
    }
}		

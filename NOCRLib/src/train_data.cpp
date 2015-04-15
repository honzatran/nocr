/* 
 * Tran Tuan Hiep
 * Implementation of methods and classes declared in train_data.h
 *
 * Compiler: g++ 4.8.3
 */

#include "../include/nocrlib/train_data.h"
#include "../include/nocrlib/assert.h"

#include <fstream>




void TrainDataLoader::prepareDataForTraining
        ( const std::string &data_file, cv::Mat &train_data, cv::Mat &labels )
{
    loader ld;
    std::vector< std::vector<float> > data = ld.loadDataToFloatMatrix( data_file );
    std::ifstream ifs(data_file);
    if (!ifs.is_open())
    {
        //throw exception
    }

    train_data = cv::Mat(0, features_length_, CV_32FC1);
    labels = cv::Mat(0, 1, CV_32FC1);
    string line;
    cv::Mat train_sample(1 , features_length_, CV_32FC1);
    std::size_t count = 0;
    float tmp;
    while (std::getline(ifs, line))
    {
        ++count;
        std::stringstream ss(line);
        for (std::size_t i = 0; i < features_length_; ++i)
        {
            ss >> tmp;
            train_sample.at<float>(0, i) = tmp;
        }

        train_data.push_back(train_sample);
        ss >> tmp;
        labels.push_back(tmp);
        NOCR_ASSERT(tmp == 0 || tmp == 1, "wrong tmp value " + std::to_string(tmp) + " count:" + std::to_string(count));
    }


    // sort samples by their labels
    // std::sort( data.begin(), data.end(), 
    //         [this] ( const std::vector<float> &a, const std::vector<float> &b )
    //         {
    //             return a[features_length_] < b[features_length_];
    //         });
}		

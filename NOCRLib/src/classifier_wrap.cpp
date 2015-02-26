/* 
 * Tran Tuan Hiep
 * Implementation of methods and classes declared in classifier_wrap.h
 *
 * Compiler: g++ 4.8.3, 
 */

#include "../include/nocrlib/classifier_wrap.h"

#include <libsvm/svm.h>
#include <liblinear/linear.h>

#include <opencv2/core/core.hpp>

#include <vector>
#include <memory>
#include <string>
#include <functional>
#include <algorithm>
#include <sstream>
#include <fstream>

using namespace std;

svm_model* LibSVMTrainBridge::train( const cv::Mat &train_data, const cv::Mat &labels, 
        svm_parameter *params )
{
    svm_problem problem = constructProblem( train_data, labels ); 
    std::cout << "start training" << std::endl;
    
    cout << problem.l << endl;
    svm_model *model = svm_train( &problem , params );

    std::cout << "delete done" << std::endl;
    delete[] problem.y;
    for ( int i = 0; i < train_data.rows; ++i )
    {
        delete[] problem.x[i];
    }
    delete[] problem.x;

    return model; 
}

svm_problem LibSVMTrainBridge::constructProblem
    ( const cv::Mat &train_data, const cv::Mat &labels ) const
{
    svm_problem problem;
    problem.l = train_data.rows;
    problem.y = new double[train_data.rows]; 
    problem.x = new svm_node*[train_data.rows];
    for ( int i = 0; i < train_data.rows; ++i )
    {
        problem.y[i] = labels.at<float>(i,0);
        problem.x[i] = new svm_node[train_data.cols+1];
        for ( int j = 0; j < train_data.cols; ++j )
        {
            problem.x[i][j] = { j, train_data.at<float>(i,j) }; 
        }

        problem.x[i][train_data.cols] = { -1 , 25 };
    }

    return problem;
}

svm_node* LibSVMTrainBridge::constructSample( const std::vector<float> &data ) const
{
    svm_node *sample = new svm_node[data.size()+1];
    for ( size_t i = 0; i < data.size(); ++i )
    {
        sample[i].index = i;
        sample[i].value = data[i];
    }
    sample[data.size()].index = -1;
    return sample;
}

// =================================================================

model* LibLINEARTrainBridge::trainModel( const cv::Mat &train_data, const cv::Mat &labels, 
        parameter *params)
{
    problem linear_problem = constructProblem( train_data, labels ); 
    std::cout << "start training" << std::endl;
    
    cout << linear_problem.l << endl;
    model *linear_model = train( &linear_problem , params );

    std::cout << "delete done" << std::endl;
    delete[] linear_problem.y;
    for ( int i = 0; i < train_data.rows; ++i )
    {
        delete[] linear_problem.x[i];
    }
    delete[] linear_problem.x;

    return linear_model;
}

problem LibLINEARTrainBridge::constructProblem
    ( const cv::Mat &train_data, const cv::Mat &labels ) const
{
    problem linear_problem;
    linear_problem.l = train_data.rows;
    linear_problem.n = train_data.cols;
    linear_problem.bias = -1;
    linear_problem.y = new double[train_data.rows]; 
    linear_problem.x = new feature_node*[train_data.rows];
    for ( int i = 0; i < train_data.rows; ++i )
    {
        linear_problem.y[i] = labels.at<float>(i,0);
        linear_problem.x[i] = new feature_node[train_data.cols+1];
        for ( int j = 0; j < train_data.cols; ++j )
        {
            linear_problem.x[i][j] = { j + 1, train_data.at<float>(i,j) }; 
        }
        linear_problem.x[i][train_data.cols] = { -1 , 25 };
    }

    return linear_problem;
}

feature_node* LibLINEARTrainBridge::constructSample( const std::vector<float> &data ) const
{
    feature_node *sample = new feature_node[data.size()+1];
    for ( size_t i = 0; i < data.size(); ++i )
    {
        sample[i].index = i+1;
        sample[i].value = data[i];
    }
    sample[data.size()].index = -1;
    return sample;
}

// void SVM::train( const std::string &data_file, svm_parameter *param )
// {
//     cv::Mat train_data, labels;
//
//     TrainDataLoader train_data_loader( length_ ); 
//     train_data_loader.prepareDataForTraining( data_file, train_data, labels );
//
//     svm_ = bridge_.train( train_data, labels, param );
//     number_of_classes_ = svm_get_nr_class( svm_ );
// }
//
// void SVM::saveConfiguration( const std::string &conf_file )
// {
//     int result = svm_save_model( conf_file.c_str(), svm_ );
//     //TODO
// }
//
// void SVM::loadConfiguration( const std::string &conf_file )
// {
//     svm_ = svm_load_model( conf_file.c_str() );
//     //TODO
//     if ( svm_ == nullptr )
//     {
//         throw FileNotFoundException(conf_file + ", libsvm configuration not found");
//     }
//     number_of_classes_ = svm_get_nr_class( svm_ );
// }
//
//
// float SVM::predict(const std::vector<float> &data ) const
// {
//     NOCR_ASSERT( svm_ != nullptr , "no configuration loaded yet" );
//
//     svm_node *nodes = bridge_.constructSample( data );
//     float out = svm_predict( svm_ , nodes );
//     delete[] nodes;
//     return out;
// }
//
// double SVM::predictProbabilities(const std::vector<float> &data, 
//                             std::vector<double> &probabilities ) const  
// {
//     NOCR_ASSERT( svm_ != nullptr , "no configuration loaded yet" );
//
//
//     svm_node *nodes = bridge_.constructSample( data );
//     probabilities.resize( number_of_classes_ );
//     double out = svm_predict_probability( svm_ , nodes, 
//                                         probabilities.data() ); 
//     delete[] nodes;
//     return out;
// }

FeatureScaler::FeatureScaler(float min, float max, float scaled_min, float scaled_max)
    :min_(min), scaled_min_(scaled_min)
{
    interval_length_ = max - min;
    scaled_interval_length_ = scaled_max - scaled_min;
}

float FeatureScaler::scale(float val) const 
{
    return scaled_min_ + (val-min_)/interval_length_ * scaled_interval_length_;
}

std::vector<float> 
    DataScaling::scale(const std::vector<float> &descriptor) const
{
    vector<float> scaled_descriptor(descriptor.size());
    for ( size_t i = 0; i < descriptor.size(); ++i )
    {
        scaled_descriptor[i] = 
                    scalers_[i].scale(descriptor[i]);
    }
    
    return scaled_descriptor;
}

void DataScaling::setUp( const cv::Mat &train_data )
{
    for ( int i = 0; i < train_data.cols; ++i )
    {
        cv::Mat curr_row = train_data.col(i);
        double min, max;
        cv::minMaxLoc(curr_row, &min, &max);
        FeatureScaler scaler( min, max);

        auto begin = curr_row.begin<float>();
        auto end = curr_row.end<float>();

        for ( auto it = begin; it != end; ++it )
        {
            *it = scaler.scale(*it);
        }

        scalers_.push_back(scaler);
    }
}

void DataScaling::saveScaling( const std::string &scaling_file )
{
    std::ofstream ofs(scaling_file);
    if ( !ofs.is_open() )
    {
    }

    for ( const auto s : scalers_ )
    {
        ofs << s.min_ << " " << s.min_ + s.interval_length_ << " " 
            << s.scaled_min_ << " " << s.scaled_min_ + s.scaled_interval_length_ << endl;
    }
    ofs.close();
}

void DataScaling::loadScaling( const std::string &scaling_file )
{
    std::ifstream ifs(scaling_file);
    if ( !ifs.is_open() )
    {
    }

    string line;
    while( std::getline(ifs, line) )
    {
        std::stringstream ss( line );
        float min, max, scaled_min, scaled_max;

        ss >> min; 
        ss >> max;
        ss >> scaled_min;
        ss >> scaled_max;
        ss.flush();

        scalers_.push_back( FeatureScaler(min, max, scaled_min, scaled_max) ) ;
    }
}



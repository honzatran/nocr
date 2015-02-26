/* 
 * Tran Tuan Hiep
 * Implementation of methods and classes declared in knn_ocr.h
 *
 * Compiler: g++ 4.8.3
 */

#include "../include/nocrlib/knn_ocr.h"
#include "../include/nocrlib/features.h"

#include <iostream>

using namespace std;


const std::string KNNOcr::alpha_ = "oi23456789abcdefghjkmnpqrstuvwxyzABDEFGHKLMNQRTY";


void KNNOcr::loadTrainData( const std::string &train_data_file )
{
    LoadTrainData<feature::DirectionHist>::load( train_data_file, train_data_, labels_ );
    index_ = std::unique_ptr<IndexType>(new IndexType( train_data_, cvflann::KMeansIndexParams() ));
}


char KNNOcr::translate( Component &c, std::vector<double> &probabilities )
{
    std::vector<float> desc = dir_hist_->compute( c );
    cv::Mat tmp( desc );
    cv::Mat mat_desc;   
    cv::transpose( tmp, mat_desc );

    // cv::Mat result, neighbour_responses, dist;
    // float label = knn_.find_nearest( mat_desc, k, result, neighbour_responses, dist );


    cv::Mat tmp_ind( 1, k, CV_32SC1 ); 
    cv::Mat tmp_dist( 1, k, CV_32FC1 ); 

    cv::Mat neighbour_responses( 1, k, CV_32FC1 );
    index_->knnSearch( mat_desc, tmp_ind, tmp_dist, k, cvflann::SearchParams() );
    int i = 0;
    for ( auto it = tmp_ind.begin<int>(); it != tmp_ind.end<int>(); ++it )
    {
        neighbour_responses.at<float>(0,i) = labels_.at<float>(0,*it);
        ++i;
    }
    probabilities = computeProbabilities( neighbour_responses, tmp_dist );

    int max_idx = 0;
    for ( size_t i = 1; i < probabilities.size(); ++i )
    {
        if ( probabilities[i] > probabilities[max_idx] )
        {
            max_idx = i;
        }
    }

    return alpha_[max_idx];
}

vector<double> KNNOcr::computeProbabilities( const cv::Mat &neighbour_responses, const cv::Mat &dist )
{
    std::vector<float> r(k);
    double dist_min;
    cv::minMaxLoc( dist, &dist_min );

    double sum = 0;
    for ( size_t i = 0; i < r.size(); ++i )
    {
        r[i] = dist_min/dist.at<float>(0,i);
        sum += r[i];
    }
    double probability = 1/sum;
         
    std::multimap<float, size_t> occurences = getClassOccurences( neighbour_responses );

    std::vector<double> probabilities( num_of_classes_, 0 );
    for ( int i = 0; i < num_of_classes_; ++i )
    {
        auto itlow = occurences.lower_bound(i);
        auto ithigh = occurences.upper_bound(i);
        for ( auto it = itlow; it != ithigh; ++it )
        {
            probabilities[i] += r[it->second] * probability;
        }
    }

    return probabilities;
}



std::multimap<float,size_t> KNNOcr::getClassOccurences( const cv::Mat &neighbour_responses )
{
    std::multimap<float,size_t> output;
    for ( auto it = neighbour_responses.begin<float>(); 
            it != neighbour_responses.end<float>(); ++it )
    {
        int index = it.pos().x;
        output.insert( std::make_pair(*it, index) );
    }
    return output;
}





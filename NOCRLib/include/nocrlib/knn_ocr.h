/**
 * @file knn_ocr.h
 * @brief ocr proposed by gomez using kNN and direction 
 * histogram
 * @author Tran Tuan Hiep
 * @version 1.0
 * @date 2014-09-16
 */


#ifndef NOCRLIB_KNN_OCR_H
#define NOCRLIB_KNN_OCR_H

#include "ocr.h"
#include "classifier_wrap.h"
#include "component.h"
#include "direction_histogram.h"
#include "iksvm.h"

#include <string>
#include <vector>
#include <map>

#include <opencv2/core/core.hpp>
#include <opencv2/ml/ml.hpp>
#include <opencv2/flann/flann.hpp>

/**
 * @brief ocr algorithm proposed by Gomez
 *
 * This class isn't copyable and copy-assignable.
 */
class KNNOcr : public AbstractOCR
{
    public:
        /**
         * @brief constructor
         */
        KNNOcr()
        {
            DirectionHistogramFactory factory;
            dir_hist_ = factory.createFeatureExtractor();
        }

        /**
         * @brief forrbiden copy constructor
         *
         * @param other
         */
        KNNOcr( const KNNOcr &other ) = delete; 

        /**
         * @brief forrbiden copy-assignable constructor
         *
         * @param other
         */
        KNNOcr& operator=(const KNNOcr &other) = delete; 

        /**
         * @brief loads configuration for knn 
         *
         * @param train_data_file path to file with training data
         */
        void loadTrainData( const std::string &train_data_file );
        char translate( Component &c, std::vector<double> &probabilities ) override;

    private:
        typedef cv::flann::GenericIndex< cv::flann::ChiSquareDistance<float> > IndexType;
        cv::Mat train_data_;
        cv::Mat labels_;

        const static int k = 30;
        const static int num_of_classes_ = 48; 
        const static std::string alpha_;
        // HogExtractor hog_;
        std::unique_ptr<AbstractFeatureExtractor> dir_hist_;
        // cv::flann::GenericIndex< cv::flann::ChiSquareDistance<double> > index_; 
        std::unique_ptr<IndexType> index_;

        std::vector<double> computeProbabilities(const cv::Mat &neighbour_responses, const cv::Mat &distances);
        std::multimap<float, size_t> getClassOccurences(const cv::Mat &neighbour_responses);
        

};


#endif /* knn_ocr_h */

/**
 * @file letter_equiv.h 
 * @brief letter word equivalence
 * @author Tran Tuan Hiep
 * @version 
 * @date 2014-09-16
 */


#ifndef NOCRLIB_LETTER_EQUIV_H
#define NOCRLIB_LETTER_EQUIV_H

#include "classifier_wrap.h"
#include "component.h"
#include "utilities.h"
#include "structures.h"
#include "features.h"
#include "extremal_region.h"

#include <vector>
#include <map>
#include <stack>
#include <tuple>
#include <cmath>
#include <memory>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>




template <>
struct FeatureTraits<feature::MergeLetter>
{
    static const int features_length = 10;
    typedef LetterMergingFactory FactoryType;
};


/**
 * @brief equivalence between letters if they can be in one word.
 *
 * Theoretic background of letter word equivalence is described 
 * in programming documentation.
 */
class LetterWordEquiv
{
    public:

        /**
         * @brief default constructor, doesn't initiaze anything
         */
        LetterWordEquiv() 
            : svm_(nullptr)
        {
        }

        /**
         * @brief constructor
         *
         * @param train_configurator configuration file for svm
         */
        LetterWordEquiv( const std::string &train_configurator )
        {
            svm_ = create<LibSVM, feature::MergeLetter>();
            svm_->loadConfiguration( train_configurator );
        }

        void loadConfiguration( const std::string &train_configurator )
        {
            if (svm_ == nullptr)
            {
                svm_ = create<LibSVM, feature::MergeLetter>();
            }

            svm_->loadConfiguration( train_configurator );
        }

        /**
         * @brief determine if two letters are in letter equivalence
         *
         * @param a
         * @param b
         *
         * @return true are equivalent, else false
         */
        bool areEquivalent( const Letter &a, const Letter &b ) const 
        {
            std::vector<float> tmp = getFeatures(a,b);
            cv::Mat features( tmp ); 
            return svm_->predict(features) == 1;
        }

        /**
         * @brief compute probability, that letters a,b are in equivalence
         *
         * @param a
         * @param b
         *
         * @return probabality a is equivalent to b
         */
        double computeProbability( const Letter &a, const Letter &b ) const
        {
            std::vector<double> tmp;
            std::vector<float> features = getFeatures( a,b );
            svm_->predictProbabilities( features, tmp );
            return tmp[1];
        }

        /**
         * @brief get feature vector from letters to determine if 
         * they are equivalent
         *
         * @param a
         * @param b
         *
         * @return descriptor used in for computing equivalence 
         */
        std::vector<float> getFeatures( const Letter &a, const Letter &b ) const
        {
            ImageLetterInfo a_info = a.getImageLetterInfo();
            ImageLetterInfo b_info = b.getImageLetterInfo();
            std::vector<float> features;

            features.push_back( getRatio(a.getHeight(), b.getHeight()) );
            for ( int i = 0; i < 4; ++i ) 
            {
                features.push_back( getRatio(a_info.object_mean_[i], b_info.object_mean_[i] ) );
            }
            features.push_back( getRatio(a_info.swt_mean_, b_info.swt_mean_ ) );

            for ( int i = 0; i < 4; ++i ) 
            {
                features.push_back( getRatio(a_info.border_mean_[i], b_info.border_mean_[i] ) );
            }
            
            return features; 
        }

        static std::vector<float> getFeatures( const std::vector<float> &a_features, const std::vector<float> &b_features )
        {
            std::vector<float> output_features( a_features.size() - 1, 0 );
            output_features[0] = getRatio( a_features[0], b_features[0] );
            // output_features[1] = std::abs( a_features[1] - b_features[1] ) / 360; 

            for ( size_t i = 2; i < a_features.size(); ++i )
            {
                output_features[i-1] = getRatio( a_features[i], b_features[i] );
            }

            return output_features;
        }

    private:
        // supportVectorMachine<feature::MergeLetter> svm_;
        std::shared_ptr< LibSVM<feature::MergeLetter> > svm_;
        // Boost<feature::MergeLetter> boosting_;

        static float getRatio( float a, float b )
        {
            return std::min(a,b)/std::max(a,b);
        }

};





#endif /*mergeLetter.h*/


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
#include "word_deformation.h"

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
    static const int features_length = 5;
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
        bool areEquivalent(const LetterEquivInfo & a, const LetterEquivInfo & b) const; 

        /**
         * @brief get feature vector from letters to determine if 
         * they are equivalent
         *
         * @param a
         * @param b
         *
         * @return descriptor used in for computing equivalence 
         */
        static std::vector<float> getDescriptor( 
                const LetterEquivInfo & a,
                const LetterEquivInfo & b);

    private:

        // supportVectorMachine<feature::MergeLetter> svm_;
        std::shared_ptr< LibSVM<feature::MergeLetter> > svm_;
        std::vector<LetterEquivInfo> letter_equiv_infos_;
        // Boost<feature::MergeLetter> boosting_;

        static float getRatio( float a, float b )
        {
            return std::min(a,b)/std::max(a,b);
        }

        std::vector<float> getLetterDescriptor(const Letter & a);

};





#endif /*mergeLetter.h*/


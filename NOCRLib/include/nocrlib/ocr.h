

/**
 * @file ocr.h
 * @brief declaration of base class for OCR
 * @author Tran Tuan Hiep
 * @version 1.0
 * @date 2014-09-12
 */


#ifndef NOCRLIB_OCR_H
#define NOCRLIB_OCR_H

#include <opencv2/core/core.hpp>

#include <vector> 
#include <iostream>
#include <string>
#include <algorithm>
#include <iostream>

#include "component.h"
#include "classifier_wrap.h"
#include "features.h"
#include "exception.h"
#include "iksvm.h"
#include "abstract_feature_factory.h"
#include "abstract_ocr.h"


// const static std::string alpha;

const std::string alpha = "oi23456789abcdefghjkmnpqrstuvwxyzABDEFGHKLMNQRTY";


/**
 * @brief Creates HoG descriptor
 */
struct HogFactory : public AbstractFeatureFactory
{
    HogFactory() = default;

    FeaturePtr createFeatureExtractor() const 
    {
        return FeaturePtr( new HogExtractor() );
    }
};

/**
 * @brief Creates longer HoG Descriptor
 */
struct HogLongFactory : public AbstractFeatureFactory
{
    FeaturePtr createFeatureExtractor() const 
    {
        HogExtractor *hog_extractor = new HogExtractor();
        hog_extractor->setLongDescriptor();
        return FeaturePtr( hog_extractor );
    }
};


template <>
struct FeatureTraits<feature::hogOcr>
{
    static const int features_length = 144;
    // static const int features_length = 1764;
    typedef HogFactory FactoryType;
};


template <>
struct FeatureTraits<feature::hogLongOcr>
{
    static const int features_length = 1764;
    typedef HogLongFactory FactoryType;
};

/**
 * @brief OCR using HOG combined with SVM with fast intersection kernel 
 * for charakter recognition
 */
class MyOCR : public AbstractOCR
{
    public:
        /**
         * @brief constructor, that initializes SVM
         *
         * @param conf_file path to config file for SVM
         * 
         * Constructor initializing SVM with fast intersection kernel 
         * using @p conf_file
         */
        MyOCR( const std::string &conf_file )
        {
            iksvm_.load(conf_file);
            HogFactory factory;
            hog_ = factory.createFeatureExtractor();
        }
        
        ~MyOCR() 
        { 
        }

        char translate( Component &c, std::vector<double> &probabilities ) override;
    private:
        IKSVM iksvm_;
        std::unique_ptr<AbstractFeatureExtractor> hog_;
};



#endif /* ocr.h */

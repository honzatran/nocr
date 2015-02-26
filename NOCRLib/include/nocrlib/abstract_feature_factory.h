#ifndef NOCRLIB_ABSTRACT_FEATURE_FACTORY_H
#define NOCRLIB_ABSTRACT_FEATURE_FACTORY_H

#include "features.h"

/**
 * @brief Abstract base class for factories, that creates composite of features
 */
class AbstractFeatureFactory
{
    public:
        typedef std::unique_ptr<AbstractFeatureExtractor> FeaturePtr;
        virtual ~AbstractFeatureFactory() { } 
        /**
         * @brief creates composite derived from AbstractFeatureFactory
         *
         * @return unique pointer from STL of given Composite
         */
        virtual FeaturePtr createFeatureExtractor() const = 0;
};



#endif /* abstract_feature_factory.h */

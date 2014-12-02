
#ifndef NOCRLIB_ABSTART_FEATURE_H
#define NOCRLIB_ABSTART_FEATURE_H

#include <memory>

#include "component.h"

/**
 * @brief Abstract base class for extracting features from Component
 */
class AbstractFeatureExtractor 
{
    public:
        typedef std::shared_ptr<Component> CompPtr;

        virtual ~AbstractFeatureExtractor() { }

        /**
         * @brief computes features from \p c_ptr
         *
         * @param c_ptr pointer to component
         *
         * @return vector<float> computed features from c_ptr
         *
         * Computes features from component to which c_ptr points.
         */
        std::vector<float> compute( const CompPtr &c_ptr ) 
        { 
            return compute(*c_ptr); 
        }

        /**
         * @brief computes features from \p c
         *
         * @param c component we compute features from
         *
         * @return vector<float> computed features from component
         *
         * Computes features from \p c.
         */
        virtual std::vector<float> compute( Component &c ) = 0;
    private:
};

#endif /* abstract_feature.h */

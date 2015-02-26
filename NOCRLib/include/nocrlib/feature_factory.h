// =====================================================================================
//
//       Filename:  feature_factory.h
//
//    Description:  
//
//        Version:  1.0
//        Created:  07/10/2014 10:52:35 PM
//       Revision:  none
//       Compiler:  g++
//
//         Author:  Tran Tuan Hiep (), honza.tran@gmail.com
//   Organization:  
//
// =====================================================================================
//
//
/**
 * @file feature_factory.h
 * @brief this file contains factory classes, that encapsulates 
 * creating of different composites.
 * @author Tran Tuan Hiep
 * @version 1.0
 * @date 2014-09-13
 */

#ifndef NOCRLIB_FEATURE_FACTORY_H
#define NOCRLIB_FEATURE_FACTORY_H

#include "features.h"
#include "abstract_feature_factory.h"

/**
 * @brief Factory, that constructs composite for first 
 * stage of extremal region algorithm
 */
struct ERFilter1StageFactory : public AbstractFeatureFactory
{
    FeaturePtr createFeatureExtractor() const 
    {
        CompositeFeatureExtractor* composite = new CompositeFeatureExtractor() ;
        composite->addFeatureExtractor( new AspectRatio() );
        composite->addFeatureExtractor( new QuadScanner() );
        composite->addFeatureExtractor( new HorizontalCrossing() );

        return FeaturePtr( composite ); 
    }
};

/**
 * @brief Factory, that constructs composite for second 
 * stage of extremal region algorithm
 */
struct ERFilter2StageFactory : public AbstractFeatureFactory
{

    /**
     * @brief creates composite, that contains only features
     * used in second stage of extramal region but not in the first one
     *
     * @return unique pointer from STL to specific composite
     */
    AbstractFeatureExtractor* getOnly2StageFeatureExtractor() const 
    {
        CompositeFeatureExtractor* composite = new CompositeFeatureExtractor();

        composite->addFeatureExtractor( new BackgroundMergeRule() );
        composite->addFeatureExtractor( new InflectionPoints() );
        return composite;
    }


    FeaturePtr createFeatureExtractor() const 
    {
        CompositeFeatureExtractor* composite = new CompositeFeatureExtractor();
        composite->addFeatureExtractor( new AspectRatio() );
        composite->addFeatureExtractor( new QuadScanner() );
        composite->addFeatureExtractor( new HorizontalCrossing() );

        composite->addFeatureExtractor( new BackgroundMergeRule() );
        composite->addFeatureExtractor( new InflectionPoints() );
        return FeaturePtr( composite ); 
    }
};


/*
 * struct DSiftFactory : public AbstractFeatureFactory
 * {
 *     FeaturePtr createFeatureExtractor() const 
 *     {
 *         return FeaturePtr( new DSiftKeyPointDescriptor() );
 *     }
 * };
 */

/**
 * @brief Creates Sift descriptor
 */
struct SiftFactory : public AbstractFeatureFactory
{
    FeaturePtr createFeatureExtractor() const 
    {
        return FeaturePtr( new SiftExtractor() );
    }
};


#endif /* feature_factory.h */


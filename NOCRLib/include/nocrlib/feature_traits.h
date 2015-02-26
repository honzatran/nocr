/**
 * @file feature_traits.h
 * @brief traits that identifies specific 
 * composite of features
 * @author Tran Tuan Hiep
 * @version 1.0
 * @date 2014-09-16
 */



#ifndef NOCRLIB_FEATURE_TRAITS_H
#define NOCRLIB_FEATURE_TRAITS_H

#include "feature_factory.h"
#include "key_point_extractor.h"
#include "bag_of_words.h"

/**
 * @brief enum for feature type, each member represents one composite
 */
enum class feature 
{
    geom, hogOcr, hogLongOcr ,momentsOcr, ERGeom, ERGeom1, MergeLetter, DSift, Sift, 
    SiftBow, DirectionHist, DSiftBoW, SwtGeom1
};

/**
 * @brief FeatureTraits contains the length of descriptor and type of factory
 * for creating composite for the descriptor.
 *
 * @tparam F
 */
template <feature F> 
struct FeatureTraits;

/// @cond

template<> 
struct FeatureTraits<feature::Sift>
{
    static const int features_length = 22 * 128; 
    typedef SiftFactory FactoryType;
};

template<>
struct FeatureTraits<feature::SiftBow>
{
    static const int features_length = 245;
    typedef BoWFactory<SiftKeyPointDescriptor> FactoryType;
};

/// @uncond

#endif /* feature_traits.h */




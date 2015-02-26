/* 
 * Tran Tuan Hiep
 * Implementation of methods and classes declared in segment.h
 *
 * Compiler: g++ 4.8.3
 */

#include "../include/nocrlib/segment.h" 
#include "../include/nocrlib/utilities.h"

#include "../include/nocrlib/component.h"
#include "../include/nocrlib/swt.h"
#include "../include/nocrlib/drawer.h"
#include "../include/nocrlib/structures.h"
#include "../include/nocrlib/direction_histogram.h"
#include "../include/nocrlib/component_tree_builder.h"
#include "../include/nocrlib/component_tree_node.h"
#include "../include/nocrlib/extremal_region.h"

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <valarray>
#include <iostream>
#include <cmath>
#include <unordered_map>
#include <map>
#include <limits>
#include <algorithm>
#include <chrono>
#include <sstream>

using namespace cv;
using namespace std;

const double SegmentationPolicy<ERTextDetection>::k_epsilon = 0.7;

MaskCreator::MaskCreator( size_t size )
{
    mask_.resize( size, true );
}

void MaskCreator::update( const TranslationInfo &a, size_t i, 
        const TranslationInfo &b, size_t j )
{
    double a_confidence = a.getConfidence();
    double b_confidence = b.getConfidence();

    if ( a_confidence < b_confidence )
    {
        mask_[i] = false;
    }
    else 
    {
        mask_[j] = false;
    }
}




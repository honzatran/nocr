/* 
 * Tran Tuan Hiep
 * Implementation of methods and classes declared in letter_equiv.h
 *
 * Compiler: g++ 4.8.3
 */

#include "../include/nocrlib/structures.h"
#include "../include/nocrlib/letter_equiv.h"

#include <opencv2/core/core.hpp>

#include <vector>
#include <set>
#include <cmath>
#include <algorithm>
#include <utility>
#include <tuple>



using namespace std;
using namespace cv;

std::vector<float> LetterWordEquiv::getDescriptor( const LetterEquivInfo & a, const LetterEquivInfo & b)
{
    cv::Vec3f color_diff = a.color_medians - b.color_medians;
    color_diff /= 256;

    vector<float> desc(5);
    desc[0] = std::abs(color_diff[0]);
    desc[1] = std::abs(color_diff[1]);
    desc[2] = std::abs(color_diff[2]);
    desc[3] = getRatio(a.swt_median, b.swt_median);
    desc[4] = getRatio(a.height, b.height);

    return desc;
}

bool LetterWordEquiv::areEquivalent(const LetterEquivInfo & a, const LetterEquivInfo & b) const
{
    std::vector<float> desc = getDescriptor(a, b);
    return svm_->predict(desc) == 1;
}



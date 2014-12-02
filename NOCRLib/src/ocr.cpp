/* 
 * Tran Tuan Hiep
 * Implementation of methods and classes declared in specified header
 *
 * Compiler: g++ 4.8.3
 */

#include <string>
#include <vector>

#include <opencv2/core/core.hpp>

#include "../include/nocrlib/ocr.h"
#include "../include/nocrlib/component.h"
#include "../include/nocrlib/features.h"
#include "../include/nocrlib/feature_factory.h"


using namespace std;


char MyOCR::translate( Component &c, std::vector<double> &probabilities )
{
    vector<float> features;
    features = hog_->compute( c );
    // float index_letter = svm_.predictProbabilities( features, probabilities ); 

    vector<double> tmp( features.begin(), features.end() );
    double index_letter; 
    std::tie(index_letter,probabilities) = iksvm_.predictProbability( tmp );

    return alpha[index_letter];
}


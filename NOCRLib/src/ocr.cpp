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

std::vector<char> MyOCR::translate( const std::vector<std::shared_ptr<Component> > & comp_ptrs, 
        std::vector<double> & probabilities)
{
    vector<double> features;
    features.reserve(comp_ptrs.size() * FeatureTraits<feature::hogOcr>::features_length);

    for (const auto & c_ptr : comp_ptrs)
    {
        auto tmp = hog_->compute(c_ptr);
        features.insert(features.end(), tmp.begin(), tmp.end());
    }

    std::vector<double> labels;
    std::tie(labels, probabilities) = iksvm_.predictProbabilityMultiple(features);

    std::vector<char> characters;
    characters.reserve(labels.size());

    for (int l : labels)
    {
        characters.push_back(alpha[l]);
    }

    return characters;
}

std::vector<char> MyOCR::translate( std::vector<Component> & components, 
        std::vector<double> & probabilities)
{
    vector<double> features;
    features.reserve(components.size() * FeatureTraits<feature::hogOcr>::features_length);

    for (auto & c : components)
    {
        auto tmp = hog_->compute(c);
        features.insert(features.end(), tmp.begin(), tmp.end());
    }

    std::vector<double> labels;
    std::tie(labels, probabilities) = iksvm_.predictProbabilityMultiple(features);

    std::vector<char> characters;
    characters.reserve(labels.size());

    for (int l : labels)
    {
        characters.push_back(alpha[l]);
    }

    return characters;
}

char HogRBFOcr::translate(Component & c, std::vector<double> & probabilities)
{
    vector<float> features = hog_->compute( c );
    int index_letter = svm_->predictProbabilities(features, probabilities);

    return alpha[index_letter];
}

char DirHistRBFOcr::translate(Component & c, std::vector<double> & probabilities)
{
    vector<float> features = dir_hist_.compute( c );
    int index_letter = svm_->predictProbabilities(features, probabilities);

    return alpha[index_letter];
}



#include "../include/nocrlib/word_deformation.h"

std::vector<float> 
WordDeformation::getDescriptor(const Word & word)
{
    cv::Vec3f color_variation = getColorInformation(word.getLetters());
    std::vector<cv::Point> centroids = word.getCentroids();

    float angle = getAngle(centroids);
    float dist_coef_variation = getDistCoefVariation(centroids);
    float height_variation = getHeightCoefVariation(word.getLetters());

    return { color_variation[0], color_variation[1], color_variation[2],
        angle, dist_coef_variation, height_variation };
}

std::vector<float>
WordDeformation::getDescriptor(const std::vector<Component> & components)
{
    std::vector<cv::Point> centroids;

    for (const auto & c : components)
    {
        centroids.push_back(c.centroid());
    }


    cv::Vec3f color_variation = getColorInformation(components);
    float angle = getAngle(centroids);
    float dist_coef_variation = getDistCoefVariation(centroids);
    float height_variation = getHeightCoefVariation(components);

    return { color_variation[0], color_variation[1], color_variation[2],
        angle, dist_coef_variation, height_variation };
}

float 
WordDeformation::getAngle(const std::vector<cv::Point> & centroids)
{
    if (centroids.size() > 2)
    {
        cv::Point p1 =  centroids[0];
        cv::Point p2 =  centroids[1];
        cv::Point p3 =  centroids[2];

        cv::Point v1 = p1 - p2;
        cv::Point v2 = p3 - p2;

        return std::acos(v1.ddot(v2)/(cv::norm(v1)*cv::norm(v2)));
    }
    else
    {
        return CV_PI; 
    }
}

float 
WordDeformation::getDistCoefVariation(const std::vector<cv::Point> & centroids)
{
    float sum_dist = 0;
    float sum_sqr_dist = 0;

    std::size_t k = centroids.size() - 1;
    for (std::size_t i = 1; i < centroids.size(); ++i) 
    {
        float dist = cv::norm(centroids[i-1] - centroids[i]);

        sum_dist += dist;
        sum_sqr_dist += dist * dist;
    }

    float mean = sum_dist/k;
    float variance = (sum_sqr_dist + k * mean * mean - 2 * mean * sum_dist)/(k-1);
    // cout << mean << " " << variance << endl;
    
    return (1.f + 1.f/(4*k))*std::sqrt(variance)/mean;
}

///=====================Deformation Cost =====================
DeformationCostEvaluator::DeformationCostEvaluator()
{
    classifier_ = nullptr;
}

void DeformationCostEvaluator::loadConfiguration(const std::string & conf_file)
{
    if (!classifier_)
    {
        classifier_ = create<Boost, feature::DCDescriptor>();
    }

    classifier_->loadConfiguration(conf_file);
    classifier_->setReturningSum(true);
}


double DeformationCostEvaluator::getCost(
        const std::vector<float> & descriptor)
{
    float sum = classifier_->predict(descriptor);
    return 1 - 1/(1 + std::exp( -2 * sum ) );
}




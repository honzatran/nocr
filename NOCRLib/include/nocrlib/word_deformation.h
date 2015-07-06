#ifndef NOCRLIB_WORD_DEFORMATION_H
#define NOCRLIB_WORD_DEFORMATION_H 

#include "structures.h"
#include "classifier_wrap.h"
#include <algorithm>

struct LetterEquivInfo
{
    cv::Vec3f color_medians;
    float swt_median, height;
};


class WordDeformation
{
public:
    WordDeformation() = default;

    std::vector<float> getDescriptor(const Word &  w);
    std::vector<float> getDescriptor(const std::vector<Component> & components);

    void setImage(const cv::Mat & image)
    {
        image_ = image;
    }

    template <typename OBJECT>
    cv::Vec3f getColorMedians(const OBJECT & obj)
    {
        vector<std::uint8_t> b_val, g_val, r_val;
        auto points = obj.getPoints();
        b_val.reserve(points.size());
        g_val.reserve(points.size());
        r_val.reserve(points.size());
        
        for (auto p : points) 
        {
            cv::Vec3b pixel_val = image_.at<cv::Vec3b>(p.y, p.x);

            b_val.push_back(pixel_val[0]);
            g_val.push_back(pixel_val[1]);
            r_val.push_back(pixel_val[2]);
        }

        int half = points.size()/2;

        std::nth_element(b_val.begin(), b_val.begin() + half, b_val.end());
        std::nth_element(g_val.begin(), g_val.begin() + half, g_val.end());
        std::nth_element(r_val.begin(), r_val.begin() + half, r_val.end());

        return cv::Vec3f(b_val[half], g_val[half], r_val[half]);
    }

    template <typename OBJECT>
    LetterEquivInfo computeEquivDescriptor(const OBJECT & obj)
    {
        cv::Vec3f color_medians = getColorMedians(obj);

        SwtTransform swt;
        cv::Mat swt_transformation = swt(obj.getBinaryMat(), false);

        auto points = obj.getPoints();
        vector<float> values;
        values.reserve(points.size());

        for (auto it = swt_transformation.begin<float>(); it != swt_transformation.end<float>(); ++it)
        {
            if (*it > 0)
            {
                values.push_back(*it);
            }
        }

        auto swt_median_it = values.begin() + values.size() / 2;

        std::nth_element(values.begin(), swt_median_it, values.end());
        float height = obj.getHeight();

        return { color_medians, *swt_median_it, height };
    }


private:
    cv::Mat image_;

    template <typename OBJECT>
    cv::Vec3f getColorInformation(const std::vector<OBJECT> & data)
    {
        cv::Vec3f sums(0, 0, 0);
        cv::Vec3f sums_square(0, 0, 0);

        for (const OBJECT & l : data)
        {
            cv::Vec3f medians = getColorMedians(l);
            sums += medians;
            sums_square += medians.mul(medians);

            // auto points = l.getPoints();
            // k += points.size();
            // for (const auto & p : points)
            // {
            //     cv::Vec3i pixel_val =  image_.at<cv::Vec3b>(p.y, p.x);
            //     sums += pixel_val;
            //
            //     sums_square += pixel_val.mul(pixel_val);
            // }
        }

        std::size_t k = data.size();
        cv::Vec3f means = sums * (1./k);
        cv::Vec3f variance;

        // variance[0] = k * means[0] * means[0] - 2 * means[0] * sums[0];
        // variance[1] = k * means[1] * means[1] - 2 * means[1] * sums[1];
        // variance[2] = k * means[2] * means[2] - 2 * means[2] * sums[2];
        variance = (int)k * means.mul(means) - 2 * means.mul(sums);

        variance += sums_square;
        variance *= (double)1/(k - 1);

        cv::Vec3f coef_variation;
        coef_variation[0] = std::sqrt(variance[0])/means[0];
        coef_variation[1] = std::sqrt(variance[1])/means[1];
        coef_variation[2] = std::sqrt(variance[2])/means[2];

        double fact = 1. + 1./(4 * k);

        return coef_variation * fact;
    }


    float getAngle(const std::vector<cv::Point> & centroids);
    float getDistCoefVariation(const std::vector<cv::Point> & centroids);

    template <typename OBJECT>
    float getHeightCoefVariation(const std::vector<OBJECT> & data)
    {
        float sum_height = 0;
        float sum_sqr_height = 0;

        std::size_t k = data.size();
        for (std::size_t i = 0; i < k; ++i) 
        {
            float height = data[i].getHeight();

            sum_height += height;
            sum_sqr_height += height * height;
        }


        float mean = sum_height/k;
        float variance = (sum_sqr_height + k * mean * mean - 2 * mean * sum_height)/(k-1);

        return (1.f + 1.f/(4*k))*std::sqrt(variance)/mean;
    }


    /* data */
};

class DeformationCostEvaluator
{
public:
    DeformationCostEvaluator();

    double getCost(const std::vector<float> & descriptor);

    void loadConfiguration(const std::string & conf_file);
private:
    std::shared_ptr<Boost<feature::DCDescriptor> > classifier_;
    /* data */
};


#endif /* word_deformation.h */

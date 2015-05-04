

#include "segmentation_test.hpp"
#include <nocrlib/drawer.h>

#define PRINT_SIZE 0
#define BOOST 0

void 
LetterSegmentTesting::updateScores(const std::string & image_name,
        const std::vector<Component> & components)
{
    auto gt_it = ground_truth_.find(image_name);

    if (gt_it == ground_truth_.end())
    {
        return;
    }

    vector<cv::Rect> gt_rectangles = gt_it->second;
    vector<bool> detected(gt_rectangles.size(), false);

    for (const Component & c : components)
    {
        cv::Rect c_rect = c.rectangle();

        for (size_t i = 0; i < gt_rectangles.size(); ++i)
        {
            if (areMatching(c_rect, gt_rectangles[i]))
            {
                if (!detected[i])
                {
                    ++true_positives_;
                    detected[i] = true;
                }
                else
                {
                    --number_results_;
                }
                break;
            }
        }
    }

    number_ground_truth_ += gt_rectangles.size();
    number_results_ += components.size();
}

double LetterSegmentTesting::getPrecision() const  
{
    return (double) true_positives_/number_results_;
}

double LetterSegmentTesting::getRecall() const 
{
    return (double) true_positives_/number_ground_truth_;
}

void LetterSegmentTesting::makeRecord( std::ostream &oss ) const 
{
    oss << "Text recognition session:" << endl;

    oss << "Precision:" << getPrecision() 
        << " (#true positive/#detected words)" << endl;

    oss << "Recall:" << getRecall() 
        << " (#true positive/#ground truth)" << endl;

    oss << "true positive count: " << true_positives_ << endl; 
    oss << "ground truth count: " << number_ground_truth_ << endl;
    oss << "results count: " << number_results_ << endl;
}

void
LetterSegmentTesting::setImageName(const std::string & image_name, const cv::Mat & img)
{
    curr_image_it_ = ground_truth_.find(image_name);

    if (curr_image_it_ == ground_truth_.end())
    {
        std::cout << image_name << std::endl;
    }
    
    curr_img_ = cv::Mat(img.rows, img.cols, CV_8UC3, cv::Scalar(0, 0, 0));
    auto rects = curr_image_it_->second;

    std::for_each(rects.begin(), rects.end(), [this]
            (const cv::Rect & rect)
            {
                cv::rectangle(curr_img_, rect, cv::Scalar(0, 255, 0), 1);
            });

    detected_.resize(curr_image_it_->second.size());
    std::fill(detected_.begin(), detected_.end(), false);
    
    number_ground_truth_ += detected_.size();
}


void 
LetterSegmentTesting::operator() (const ERRegion & err)
{
    cv::Rect err_rect = err.getRectangle() - cv::Point(1,1);

    std::vector<cv::Rect> gt_rects = curr_image_it_->second;

    for (std::size_t i = 0; i < gt_rects.size(); ++i)
    {
        if (areMatching(err_rect, gt_rects[i]))
        {
            if (!detected_[i])
            {
                ++true_positives_;
                detected_[i] = true;
                draw(err.toComponent());
#if PRINT_SIZE
                std::cout << curr_image_it_->first << " " << err.getSize() << std::endl;
#endif
            }
            else
            {
                --number_results_;
            }
            break;
        }
    }

    number_results_++;
}

void
LetterSegmentTesting::draw(const Component & c)
{
    auto points = c.getPoints();
    for (auto p : points)
    {
        auto pixel_val = curr_img_.at<cv::Vec3b>(p.y, p.x);
        if (pixel_val == cv::Vec3b(0, 0, 0) || pixel_val == cv::Vec3b(0,0, 255))
        {
            curr_img_.at<cv::Vec3b>(p.y, p.x) = cv::Vec3b(255, 255, 255);
        }
    }
    cv::rectangle(curr_img_, c.rectangle(), cv::Scalar(255, 0, 0));
}

SecondStageTesting::SecondStageTesting()
{
    ERFilter2StageFactory factory;
    extractor_ = std::unique_ptr<AbstractFeatureExtractor>(
            factory.getOnly2StageFeatureExtractor());
}

void
SecondStageTesting::setSvmConfiguration(const std::string & svm_conf)
{
#if BOOST
    boost_.loadConfiguration(svm_conf);
#else
    svm_.loadConfiguration(svm_conf);
#endif
}

void 
SecondStageTesting::operator() (const ERRegion & err)
{
    std::vector<float> desc  = err.getFeatures();
    Component c = err.toComponent();
    std::vector<float> additional_2stage_desc = extractor_->compute(c);

    desc.insert(desc.end(), additional_2stage_desc.begin(), 
            additional_2stage_desc.end());

#if BOOST
    bool positive_class = boost_.predict(desc) == 1;
#else
    bool positive_class = svm_.predict(desc) == 1;
#endif

    if (positive_class)
    {
        number_results_++;
    }

    cv::Rect err_rect = err.getRectangle() - cv::Point(1,1);

    std::vector<cv::Rect> gt_rects = curr_image_it_->second;

    for (std::size_t i = 0; i < gt_rects.size(); ++i)
    {
        if (areMatching(err_rect, gt_rects[i]))
        {
            if (positive_class)
            {
                if (!detected_[i])
                {
                    ++true_positives_;
                    detected_[i] = true;
                    draw(err.toComponent());
                }
                else
                {
                    --number_results_;
                }
            }
            else
            {
                drawError(err.toComponent());
            }
            break;
        }
    }
}

void
SecondStageTesting::drawError(const Component & c)
{
    auto points = c.getPoints();
    for (auto p : points)
    {
        if (curr_img_.at<cv::Vec3b>(p.y, p.x) == cv::Vec3b(0, 0, 0))
        {
            curr_img_.at<cv::Vec3b>(p.y, p.x) = cv::Vec3b(0, 0, 255);
        }
    }
}

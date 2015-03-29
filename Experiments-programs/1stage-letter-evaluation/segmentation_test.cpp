

#include "segmentation_test.hpp"

void 
LetterSegmentTesting::loadGroundTruthXML(const std::string & xml_gt_file)
{
    std::ifstream ifs; 
    ifs.open(xml_gt_file);

    pugi::xml_document doc;
    pugi::xml_parse_result result = doc.load( ifs );

    pugi::xml_node root = doc.first_child();

    for (auto img_node : root.children())
    {
        string name = img_node.child("image-name").child_value();
        vector<cv::Rect> rectangles;

        for (auto rect_node : img_node.children("rectangle"))
        {
            int x = rect_node.attribute("x").as_int();
            int y = rect_node.attribute("y").as_int();
            int width = rect_node.attribute("width").as_int();
            int height = rect_node.attribute("height").as_int();
            rectangles.push_back(cv::Rect( x, y, width, height ));
        }

        ground_truth_.insert(make_pair(name, rectangles));
    }
}

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

bool 
LetterSegmentTesting::areMatching(
        const cv::Rect & c_rect, 
        const cv::Rect & gt_rect)
{
    cv::Rect intersection = c_rect & gt_rect;

    if (intersection.area() == 0)
    {
        return false;
    }

    bool gt_condition = (double)intersection.area()/gt_rect.area() > 0.8;
    bool detected_condition = (double)intersection.area()/c_rect.area() > 0.4;

    return gt_condition && detected_condition;
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
LetterSegmentTesting::setImageName(const std::string & image_name)
{
    curr_image_it_ = ground_truth_.find(image_name);
    if (curr_image_it_ == ground_truth_.end())
    {
        std::cout << image_name << std::endl;
    }
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

SecondStageTesting::SecondStageTesting()
{
    ERFilter2StageFactory factory;
    extractor_ = std::unique_ptr<AbstractFeatureExtractor>(
            factory.getOnly2StageFeatureExtractor());
}

void
SecondStageTesting::setSvmConfiguration(const std::string & svm_conf)
{
    svm_.loadConfiguration(svm_conf);
}

void 
SecondStageTesting::operator() (const ERRegion & err)
{
    std::vector<float> desc  = err.getFeatures();
    Component c = err.toComponent();
    std::vector<float> additional_2stage_desc = extractor_->compute(c);

    desc.insert(desc.end(), additional_2stage_desc.begin(), 
            additional_2stage_desc.end());

    if (svm_.predict(desc) == 1)
    {
        number_results_++;
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
                }
                else
                {
                    --number_results_;
                }
                break;
            }
        }
    }
}

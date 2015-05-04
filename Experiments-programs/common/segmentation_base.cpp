
#include "segmentation_base.hpp"

using namespace std;

void 
LetterSegmentBase::loadGroundTruthXML(const std::string & xml_gt_file)
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

bool 
LetterSegmentBase::areMatching(
        const cv::Rect & c_rect, 
        const cv::Rect & gt_rect)
{
    cv::Rect intersection = c_rect & gt_rect;

    if (intersection.area() == 0)
    {
        return false;
    }

    bool gt_condition = (double)intersection.area()/gt_rect.area() > 0.70;
    bool detected_condition = (double)intersection.area()/c_rect.area() > 0.4;

    return gt_condition && detected_condition;
}

void 
LetterSegmentBase::notifyResize(const std::string & name, double scale)
{
    auto it = ground_truth_.find(name);
    std::for_each(it->second.begin(), it->second.end(), 
            [scale] (cv::Rect & rect)
            {
                cv::Point tl_rect = rect.tl();
                cv::Point br_rect = rect.br();

                cv::Rect scaled_rect = cv::Rect(scale * tl_rect,scale * br_rect);
                rect = scaled_rect;
            });
}

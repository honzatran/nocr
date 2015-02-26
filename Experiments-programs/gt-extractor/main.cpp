

#include <iostream>
#include <map>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <nocrlib/utilities.h>
#include <pugi/pugixml.hpp>


using namespace std;

struct Record
{
    cv::Vec3b color;
    cv::Rect rect;
};


vector<cv::Rect> scanImage(const cv::Mat & image)
{
    const cv::Vec3b white(255, 255, 255);
    vector<Record> detections;

    for (auto it = image.begin<cv::Vec3b>(); it != image.end<cv::Vec3b>(); ++it)
    {
        cv::Vec3b pix_val = *it;
        if (pix_val != white)
        {
            auto rect_it = std::find_if( detections.begin(), detections.end(),
                    [pix_val] (const Record & rec) -> bool { return pix_val == rec.color; });

            if (rect_it != detections.end())
            {
                rect_it->rect |= cv::Rect(it.pos(), cv::Size(1, 1));
            }
            else
            {
                detections.push_back(
                        { pix_val, cv::Rect(it.pos(), cv::Size(1, 1)) });
            }
        }
    }

    vector<cv::Rect> output;
    for (auto it = detections.begin(); it != detections.end(); ++it)
    {
        output.push_back(it->rect);
    }

    return output;
}


int main(int argc, char ** argv)
{
    pugi::xml_document doc;
    pugi::xml_node images = doc.append_child();

    images.set_name("GT-segmantation");

    string line;
    while(getline(cin, line))
    {
        cv::Mat image = cv::imread(line, CV_LOAD_IMAGE_COLOR);
        vector<cv::Rect> rectangles = scanImage(image);

        pugi::xml_node image_node = images.append_child("image");

        string file_name = getFileName(line);

        pugi::xml_node image_name = image_node.append_child("image-name");
        
        image_name.append_child(pugi::node_pcdata).set_value(file_name.c_str());

        for (const auto &r : rectangles)
        {
            pugi::xml_node rect_node = image_node.append_child("rectangle");
            rect_node.append_attribute("x") = r.x;
            rect_node.append_attribute("y") = r.y;
            rect_node.append_attribute("width") = r.width;
            rect_node.append_attribute("height") = r.height;

            // cv::rectangle(image, r, cv::Scalar(0, 0, 255), 2);
        }

        // gui::showImage(image,  "detected rectangles");
    }

    doc.print(std::cout);

    return 0;
}

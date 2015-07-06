
#ifndef _SEGMENTATION_BASE_HPP
#define _SEGMENTATION_BASE_HPP

#include <iostream>
#include <string>
#include <memory>
#include <vector>
#include <fstream>
#include <sstream>

#include <pugi/pugixml.hpp>
#include <opencv2/core/core.hpp>
#include <nocrlib/exception.h>

class LetterSegmentBase
{
    public:
        void loadGroundTruthXML(const std::string & xml_gt_file);
        bool areMatching(const cv::Rect & c_rect, const cv::Rect & gt_rect);

        void notifyResize(const std::string & name, double scale);
    protected:
        std::map<std::string, std::vector<cv::Rect> > ground_truth_;
};

struct ParseError
{
};

inline std::pair<std::string, std::string> parse(const std::string & line)
{
    std::size_t pos = line.find_last_of(':');

    if (pos == std::string::npos)
    {
        throw NocrException<ParseError>("line has no seperator :");
    }

    std::string path = line.substr(0, pos);
    std::string gt_file_name = line.substr(pos + 1);

    return std::make_pair(path, gt_file_name);
}
#endif

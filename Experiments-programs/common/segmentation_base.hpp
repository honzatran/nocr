
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

class LetterSegmentBase
{
    public:
        void loadGroundTruthXML(const std::string & xml_gt_file);
        bool areMatching(const cv::Rect & c_rect, const cv::Rect & gt_rect);

        void notifyResize(const std::string & name, double scale);
    protected:
        std::map<std::string, std::vector<cv::Rect> > ground_truth_;
};

inline std::pair<std::string, std::string> parse(const std::string & line)
{
    std::stringstream ss(line);
    std::string path, gt_file_name;
    std::getline(ss, path, ':');
    std::getline(ss, gt_file_name, ':');

    return std::make_pair(path, gt_file_name);
}
#endif

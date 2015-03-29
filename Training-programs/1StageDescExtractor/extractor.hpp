
#ifndef _EXTRACTOR_H
#define _EXTRACTOR_H


#include <string>
#include <vector>
#include <map>
#include <algorithm>

#include <nocrlib/extremal_region.h>
#include <nocrlib/component_tree_builder.h>
#include <nocrlib/component.h>
#include <nocrlib/utilities.h>


class AbstractOutput
{
public:
    virtual ~AbstractOutput() { }
    virtual void save(ERRegion & err, 
            const std::string & file_name, char c) = 0;
};

class TruePositiveExtractor 
{
public:
    TruePositiveExtractor() : positive_(true) { }

    void loadXml(const std::string & xml_file);

    void operator() (ERRegion & err);

    void setDetectPositive(bool positive)
    {
        positive_ = positive;
    }

    void setOutput(AbstractOutput * output)
    {
        output_ = output;
    }

    void setFileImage(const std::string & file_name);

private:
    struct LetterRecord
    {
        cv::Vec3b color;
        cv::Rect rect;
        cv::Point center;

        char letter;
    };

    bool positive_;

    std::map<std::string, std::vector<LetterRecord> > gt_records_;
    decltype(gt_records_.begin()) current_it_;
    AbstractOutput * output_;

    bool areMatching(const cv::Rect & c_rect, const cv::Rect & gt_rect);
};

#endif

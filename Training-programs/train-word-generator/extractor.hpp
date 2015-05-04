
#ifndef _EXTRACTOR_H
#define _EXTRACTOR_H


#include <string>
#include <vector>
#include <map>
#include <algorithm>
#include <ostream>
#include <random>

#include <nocrlib/extremal_region.h>
#include <nocrlib/component_tree_builder.h>
#include <nocrlib/component.h>
#include <nocrlib/utilities.h>

#include <pugi/pugixml.hpp>

class Extractor 
{
public:
    Extractor() : positive_(true) 
    { 
        os_ = &std::cout;
    }

    void loadXml(const std::string & xml_file);
    void setUpErTree(const std::string & er1_conf_file, const std::string & er2_conf_file);

    void findImage(const std::string & file_name);
private:
    struct LetterRecord
    {
        cv::Vec3b color;
        cv::Rect rect;
        cv::Point center;

        char letter;
    };

    struct WordRecord
    {
        std::string text;
        std::vector<LetterRecord> letters;

        WordRecord(const std::string & _text, const std::vector<LetterRecord> & _letters)
            : text(_text), letters(_letters) { }

        WordRecord() = default;
    };
        

    ERTree er_tree_;
    std::ostream * os_;
    bool positive_;
    std::map<std::string, pugi::xml_node> gt_records_;

    pugi::xml_document doc_;

    std::vector<WordRecord> getWords(const pugi::xml_node & gt_node);
    std::vector<Component> extractWord(const WordRecord & word_record, const cv::Mat & gt_image);

    cv::Vec3f getColorInformation(const WordRecord & word_record, 
            const std::vector<Component> & comp, const cv::Mat & image);

    float getAngle(cv::Point p1, cv::Point p2, cv::Point p3);
    float coefVarianceDist(const WordRecord & word_record);

    cv::Rect getRectangle(const std::vector<LetterRecord> & letters);
    cv::Rect getRectangle(const std::vector<Component> & components);

    std::vector<Component> randomNegative(const std::vector<Component> & components, 
            const std::vector<cv::Rect> & rectangles, std::size_t length,
            std::vector<bool> & mask);

    std::mt19937 generator_;
    std::uniform_int_distribution<std::size_t> dist_;

    long id = 0;
};

#endif

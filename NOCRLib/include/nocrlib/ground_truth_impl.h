
#ifndef NOCRLIB_GROUND_TRUTH_IMPL_H
#define NOCRLIB_GROUND_TRUTH_IMPL_H

#include "testing.h"
#include <map>
#include <string>
#include <pugi/pugixml.hpp>

class XmlGroundTruth : public GroundTruthInterface
{
    public:
        void loadData( const std::string &gt_xml_file );

        void storeGroundTruth
            ( std::map<std::string, ImageGroundTruth> &ground_truth) override;

    private:
        std::map< std::string, ImageGroundTruth > loaded_data_;
        void updateGroundTruth( const pugi::xml_node &img_node );

        std::string getAlphaNum( const std::string & str );
};

class TextGroundTruth : public GroundTruthInterface
{
    public:
        void loadData( const std::string &file, const std::string &img_name );

        void storeGroundTruth
            ( std::map<std::string, ImageGroundTruth> &ground_truth) override;
    private: 
        std::map< std::string, ImageGroundTruth > loaded_data_;
        void updateGroundTruth( ImageGroundTruth &img_ground_truth,
                const std::string & line );
};

struct LetterSegmentGT
{
    cv::Rect rectangle;
    cv::Vec3b color;
    char letter;
};







#endif /* ground_truth_impl.h */

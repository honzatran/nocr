

#ifndef NOCRLIB_STREET_VIEW_SCENE_H
#define NOCRLIB_STREET_VIEW_SCENE_H

#include "testing.h"
#include "dictionary.h"

#include <string>

#include <pugi/pugixml.hpp>

struct StreetViewRecord
{
    std::string file_name_;
    std::vector<std::string> dictionary;
};

class StreetViewGroundTruth : public GroundTruthInterface
{
    public:
        StreetViewGroundTruth() = default;

        void loadFromFile( const std::string &svt_gt_file );

        void storeGroundTruth( 
                std::map<std::string, ImageGroundTruth> &ground_truth) override;

        std::vector<StreetViewRecord> getSamples() const;
    private:
        std::map<std::string, ImageGroundTruth> ground_truth_;
        std::vector<StreetViewRecord> street_view_records_;

        void processXml(const pugi::xml_node &root_xml);
        ImageGroundTruth createGroundTruth( const pugi::xml_node & gt_nodes );
        StreetViewRecord createStreetViewRecord( const pugi::xml_node & img_node);
};

class ICDARGroundTruth : public GroundTruthInterface
{
    public:
        ICDARGroundTruth() = default;

        void loadFromFile( const std::string &icdar_gt_file );
        void storeGroundTruth( 
                std::map<std::string, ImageGroundTruth> &ground_truth) override;

        std::vector<std::string> getFilesPath() const
        {
            return files_;
        }


    private:
        std::map<std::string, ImageGroundTruth> ground_truth_;
        std::vector<std::string> files_;
        ImageGroundTruth createGroundTruth( const pugi::xml_node & gt_nodes );
};

#endif /* street_view_scene.h */

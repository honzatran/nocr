#include "../include/nocrlib/street_view_scene.h"

#include <fstream>
#include <iostream>
#include <vector>
#include <sstream>

using namespace std;

void StreetViewGroundTruth::loadFromFile(
        const std::string &svt_gt_file )
{
    std::ifstream ifs( svt_gt_file );
    pugi::xml_document doc;
    pugi::xml_parse_result result = doc.load( ifs );

    if ( !result )
    {
        cout << "chyba" << endl;
        return;
        //TODO throw vyjimka
    }

    pugi::xml_node root_xml = doc.first_child();
    processXml(root_xml);

    ifs.close();
}

void StreetViewGroundTruth::processXml( const pugi::xml_node &root_xml )
{
    for ( pugi::xml_node img : root_xml.children("image") )
    {
        string tmp = img.child("imageName").child_value();
        string img_name = getFileName(tmp);

        street_view_records_.push_back(
                createStreetViewRecord( img ));
       
        cout << img_name << endl;
        auto gt_nodes = img.child("taggedRectangles");
        ImageGroundTruth img_gt = createGroundTruth( gt_nodes );

        ground_truth_.insert(
                std::make_pair(img_name, img_gt) );
    }

}

ImageGroundTruth StreetViewGroundTruth::createGroundTruth( 
        const pugi::xml_node &gt_root_node )
{
    ImageGroundTruth img_gt;

    for ( const auto & word_node : 
            gt_root_node.children("taggedRectangle"))
    {
        string text = word_node.child("tag").child_value();
        int x = word_node.attribute("x").as_int();
        int y = word_node.attribute("y").as_int();
        int width = word_node.attribute("width").as_int();
        int height = word_node.attribute("height").as_int();

        img_gt.addGroundTruth( text, cv::Rect(x, y, width, height) );
    }

    return img_gt;
}

StreetViewRecord StreetViewGroundTruth::createStreetViewRecord(
        const pugi::xml_node &img_node)
{
    string img_name = img_node.child("imageName").child_value();
    cout << img_name << endl;

    string raw_lex = img_node.child("lex").child_value();

    std::stringstream ss(raw_lex);
    vector<string> lex;
    string word;

    while( std::getline(ss, word, ',') )
    {
        lex.push_back(word);
    }

    return { img_name, lex };
}

std::vector<StreetViewRecord> StreetViewGroundTruth::getSamples() const
{
    return street_view_records_;
}


void StreetViewGroundTruth::storeGroundTruth( 
        std::map<std::string, ImageGroundTruth> &output_ground_truth) 
{
    output_ground_truth.insert(
            ground_truth_.begin(), 
            ground_truth_.end());
}


// ==========================ICDAR 2003=======================================

void ICDARGroundTruth::loadFromFile(
        const std::string &icdar_gt_file )
{
    std::ifstream ifs( icdar_gt_file );
    pugi::xml_document doc;
    pugi::xml_parse_result result = doc.load( ifs );

    if ( !result )
    {
        cout << "chyba" << endl;
        return;
    }

    pugi::xml_node root_xml = doc.first_child();
    for ( pugi::xml_node img : root_xml.children("image") )
    {
        string img_path = img.child("imageName").child_value();
        string img_name = getFileName(img_path);
        files_.push_back(img_path);
       
        cout << img_path << endl;
        auto gt_nodes = img.child("taggedRectangles");
        ImageGroundTruth img_gt = createGroundTruth( gt_nodes );
        
        ground_truth_.insert(
                std::make_pair(img_name, img_gt) );
    }

    ifs.close();
}

ImageGroundTruth ICDARGroundTruth::createGroundTruth( 
        const pugi::xml_node &gt_root_node )
{
    ImageGroundTruth img_gt;

    for ( const auto & word_node : 
            gt_root_node.children("taggedRectangle"))
    {
        string text = word_node.child("tag").child_value();
        int x = word_node.attribute("x").as_int();
        int y = word_node.attribute("y").as_int();
        int width = word_node.attribute("width").as_int();
        int height = word_node.attribute("height").as_int();

        img_gt.addGroundTruth( text, cv::Rect(x, y, width, height) );
    }

    return img_gt;
}

void ICDARGroundTruth::storeGroundTruth(
        std::map<std::string, ImageGroundTruth> &ground_truth) 
{
    ground_truth.insert( ground_truth_.begin(), ground_truth_.end());
}


#include "../include/nocrlib/ground_truth_impl.h"

#include <fstream>
#include <sstream>
#include <iostream>
#include <string>
#include <locale>
#include <algorithm>

#include <pugi/pugixml.hpp>

using namespace std;

void XmlGroundTruth::loadData( const std::string &gt_xml_file )
{
    std::ifstream ifs; 
    ifs.open( gt_xml_file );

    pugi::xml_document doc;
    pugi::xml_parse_result result = doc.load( ifs );

    if ( !result )
    {
        cout << "chyba" << endl;
        return;
    }

    pugi::xml_node ground_truth = doc.first_child();
    for ( auto img = ground_truth.first_child(); img; img = img.next_sibling() )
    {
        updateGroundTruth(img);
    }

    ifs.close();
}

void XmlGroundTruth::updateGroundTruth(const pugi::xml_node & img_node)
{
    ImageGroundTruth img_ground_truth;
    string img_name = img_node.child("name").child_value();

    for ( const pugi::xml_node & word_node : img_node.children("word") )
    {
        string text = getAlphaNum(word_node.child_value());
        int x = word_node.attribute("x").as_int();
        int y = word_node.attribute("y").as_int();
        int width = word_node.attribute("width").as_int();
        int height = word_node.attribute("height").as_int();
        cv::Rect word_bbox = cv::Rect( x, y, width, height );

        img_ground_truth.addGroundTruth( text, word_bbox );
    }
    
    loaded_data_.insert( std::make_pair( img_name, img_ground_truth ) );
}

std::string XmlGroundTruth::getAlphaNum(const std::string & str)
{
    string output;
    output.reserve(str.size());

    std::copy_if(str.begin(), str.end(), std::back_inserter(output), [] (char c) 
            { return isalnum(c); });

    return output;
}



void XmlGroundTruth::storeGroundTruth
        ( std::map<std::string, ImageGroundTruth> &ground_truth) 
{
    ground_truth.insert( loaded_data_.begin(), loaded_data_.end() );
}

void TextGroundTruth::loadData( const std::string &data_file, 
        const std::string &img_name )
{
    ifstream ifs;
    ifs.open( data_file );
    std::string line;
    ImageGroundTruth ground_truth;

    while( std::getline(ifs, line) )
    {
        updateGroundTruth( ground_truth, line );
    }

    loaded_data_.insert(
            std::make_pair( img_name, ground_truth ) );

    ifs.close();
}

void TextGroundTruth::updateGroundTruth
    ( ImageGroundTruth &image_ground_truth,
      const std::string &line )
{
    int x, y, max_x, max_y;; 
    string text;

    stringstream ss(line);
    ss >> x; 
    ss >> y;
    ss >> max_x; 
    ss >> max_y;
    ss >> text;

    image_ground_truth.addGroundTruth(text, cv::Rect(x, y, max_x - x + 1, max_y - y + 1));
}

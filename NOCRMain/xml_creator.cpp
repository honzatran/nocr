// =====================================================================================
//
//       Filename:  xml_creator.cpp
//
//    Description:
//
//        Version:  1.0
//        Created:  08/31/2014 12:20:41 AM
//       Revision:  none
//       Compiler:  g++
//
//         Author:  Tran Tuan Hiep (), honza.tran@gmail.com
//   Organization:  
//
// =====================================================================================


#include "xml_creator.h"

XmlCreator::XmlCreator()
{
    root_ = doc_.append_child( k_root_tag.c_str() );
}

void XmlCreator::save(std::ostream &oss)
{
    doc_.save( oss );
}

void XmlCreator::makeRecord( const std::string &file_path, 
        const std::vector<TranslatedWord> &words )
{
    pugi::xml_node image = root_.append_child( k_image_tag.c_str() );
    pugi::xml_node path = image.append_child( k_path_tag.c_str() );
    path.append_child( pugi::node_pcdata ).text().set( file_path.c_str() );

    for ( const auto &w : words )
    {
        addWordTag( image, w );
    }
}

void XmlCreator::addWordTag( pugi::xml_node &image_node, const TranslatedWord &w )
{
    pugi::xml_node word_node = image_node.append_child( k_word_tag.c_str() );

    auto text_node = word_node.append_child( k_text_tag.c_str() );
    text_node.append_child(pugi::node_pcdata).text().set( w.translation_.c_str() );

    auto bbox_node = word_node.append_child( k_bbox_tag.c_str() );
    cv::Rect rect = w.visual_information_.getRectangle();
    bbox_node.append_attribute( "x" ) = rect.x;
    bbox_node.append_attribute( "y" ) = rect.y;
    bbox_node.append_attribute( "width" ) = rect.width;
    bbox_node.append_attribute( "height" ) = rect.height;
    
}





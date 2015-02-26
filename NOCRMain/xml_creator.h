// =====================================================================================
//
//       Filename:  xml_creator.h
//
//    Description:  
//
//        Version:  1.0
//        Created:  08/29/2014 01:17:14 AM
//       Revision:  none
//       Compiler:  g++
//
//         Author:  Tran Tuan Hiep (), honza.tran@gmail.com
//   Organization:  
//
// =====================================================================================

#ifndef _XML_CREATOR_H
#define _XML_CREATOR_H

#include <nocrlib/word_generator.h>
#include <pugi/pugixml.hpp>
#include <ostream>
#include "recorder_interface.h"

class XmlCreator : public RecorderInterface
{
    public:
        XmlCreator();

        void makeRecord( const std::string &file_path, const std::vector<TranslatedWord> &words );
        void save( std::ostream &oss );
        void addWordTag( pugi::xml_node &image_node, const TranslatedWord &w );
    private:
        pugi::xml_document doc_;
        pugi::xml_node root_;

        const std::string k_root_tag = "text-detection";
        const std::string k_image_tag = "image";
        const std::string k_path_tag = "path-to-image";
        const std::string k_word_tag = "word";
        const std::string k_bbox_tag = "bounding-box";
        const std::string k_text_tag = "text";



};


#endif /* xml_creator.h */

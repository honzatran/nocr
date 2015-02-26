// =====================================================================================
//
//       Filename:  recorder_interface.h
//
//    Description:  
//
//        Version:  1.0
//        Created:  08/31/2014 10:36:12 PM
//       Revision:  none
//       Compiler:  g++
//
//         Author:  Tran Tuan Hiep (), honza.tran@gmail.com
//   Organization:  
//
// =====================================================================================

#ifndef _RECORDER_INTERFACE_H
#define _RECORDER_INTERFACE_H

#include <vector>
#include <nocrlib/word_generator.h>
#include <ostream>
#include <string>

class RecorderInterface
{
    public:
        virtual ~RecorderInterface() { }
        virtual void makeRecord( const std::string &file_path, const std::vector<TranslatedWord> &words ) = 0;
        virtual void save( std::ostream &oss ) = 0;
};


#endif /* recorder_interface.h */

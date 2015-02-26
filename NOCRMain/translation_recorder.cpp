// =====================================================================================
//
//       Filename:  translation_recorder.cpp
//
//    Description:  
//
//        Version:  1.0
//        Created:  08/31/2014 11:01:57 PM
//       Revision:  none
//       Compiler:  g++
//
//         Author:  Tran Tuan Hiep (), honza.tran@gmail.com
//   Organization:  
//
// =====================================================================================
//

#include "translation_recorder.h"

using namespace std;

#include <iostream>

void TranslationRecorder::makeRecord( const std::string &file_path,
        const std::vector<TranslatedWord> &words )
{
    ss_ << file_path << endl;
    for ( const auto &w : words )
    {
        ss_ << w << endl;
    }
    ss_ << seperator << endl;
}

void TranslationRecorder::save( std::ostream &oss )
{
    oss << ss_.str();
}



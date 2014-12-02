// =====================================================================================
//
//       Filename:  translation_recorder.h
//
//    Description:  
//
//        Version:  1.0
//        Created:  08/31/2014 10:57:52 PM
//       Revision:  none
//       Compiler:  g++
//
//         Author:  Tran Tuan Hiep (), honza.tran@gmail.com
//   Organization:  
//
// =====================================================================================

#ifndef _TRANSLATION_RECORDER_H
#define _TRANSLATION_RECORDER_H

#include "recorder_interface.h"
#include <sstream>
#include <nocrlib/word_generator.h>

class TranslationRecorder : public RecorderInterface
{
    public:
        ~TranslationRecorder() { }
        void makeRecord( const std::string &file_path, 
                const std::vector<TranslatedWord> &words ) override;
        void save( std::ostream &oss ) override;
    private:
        std::stringstream ss_;
        const std::string seperator = "================================================";
};

#endif /* translation_recorder.h */


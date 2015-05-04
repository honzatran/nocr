/* 
 * Tran Tuan Hiep
 * Implementation of methods and classes declared in train_data.h
 *
 * Compiler: g++ 4.8.3
 */

#include "../include/nocrlib/text_recognition.h"
#include "../include/nocrlib/word_generator.h"
#include "../include/nocrlib/drawer.h"
#include "../include/nocrlib/exception.h"
#include "../include/nocrlib/knn_ocr.h"

#include <string>
#include <fstream>
#include <chrono>

#include <opencv2/core/core.hpp>


using namespace std;



/*
 * std::vector<Letter> TextRecognition::getRemainingLetters( const WordGenerator &generator,
 *         const std::vector<Letter> &letters )
 * {
 *     vector<Letter> remaining_letters;
 *     vector<bool> mask_used_letters = generator.getUsedLetters();
 *     for ( int i = 0; i < mask_used_letters.size(); ++i )
 *     {
 *          if (!mask_used_letters[i] )
 *          {
 *              remaining_letters.push_back( letters[i] );
 *          }
 *     }
 *     return remaining_letters;
 * }
 */


/*
 * void TextRecognition::addWord( const std::string &word )
 * {
 *     dict_.addWord(word);
 * }
 * 
 * void TextRecognition::loadDictionary( const std::string &file )
 * {
 *     dict_.loadWords(file);
 * }
 * 
 * void TextRecognition::loadDictionary( const std::string &file )
 * {
 *     dict_.loadWords(file);
 * }
 */


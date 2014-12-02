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

void TextRecognition::loadConfiguration( // const std::string &dictionary_file, 
        const std::string &er1_conf_file, 
        const std::string &er2_conf_file, 
        const std::string &merge_conf_file )
{
    er_text_detection_ = std::unique_ptr<ERTextDetection>
                ( new ERTextDetection( er1_conf_file, er2_conf_file) );

    segmentation_.loadMethod( er_text_detection_.get() );

    word_equivalence_.loadConfiguration( merge_conf_file );
}

std::vector<TranslatedWord> TextRecognition::recognize( 
        cv::Mat &image, 
        const Dictionary &dictionary )
{
    if ( image.rows < SIZE && image.cols < SIZE )
    {
        image = resizer_.resizeKeepAspectRatio(image);
    }

    auto letters = segmentation_.segment( image );

    if ( show_letters_ )
    {
        showLetters( letters, image );
    }

    WordGenerator generator;

    generator.initHorizontalDetection( letters, word_equivalence_ );
    vector<TranslatedWord> words = generator.process( dictionary ); 

    auto remaining_letters = generator.getRemainingLetters();
    
    generator.initVerticalDetection(remaining_letters, word_equivalence_ );
    auto vertical_words = generator.process( dictionary );

    words.insert( words.end(), vertical_words.begin(), vertical_words.end() );

    if ( show_words_ )
    {
        showWords( words, image );
    }

    return words;
}


std::vector<TranslatedWord> TextRecognition::recognize
    ( const std::string &image_path, const Dictionary &dictionary )
{
    cv::Mat input_image = loadImage( image_path );
    return recognize( input_image, dictionary );
}

cv::Mat TextRecognition::loadImage( const std::string &image_path )
{
    cv::Mat image = cv::imread( image_path, CV_LOAD_IMAGE_COLOR );
    if ( image.empty() )
    {
        throw FileNotFoundException( "image at path " + 
                image_path + " doesn't exist" );
    }
    return image;
}

void TextRecognition::showLetters( const std::vector<Letter> &letters, const cv::Mat &image )
{
    std::unique_ptr<DrawerInterface> drawer( new BinaryDrawer() );
    drawer->init(image);
    for ( const auto &l : letters )
    {
        drawer->draw(l);
    }
    gui::showImage( drawer->getImage(), "detected letters" );
}

void TextRecognition::showWords( const std::vector<TranslatedWord> &words, const cv::Mat &image )
{
    std::unique_ptr<DrawerInterface> drawer( new RectangleDrawer() );
    drawer->init(image);
    for ( const auto &w: words )
    {
        drawer->draw( w.visual_information_ );
        auto letters = w.visual_information_.getLetters();
        for ( const auto &l :letters ) 
        {
            drawer->draw(l);
        }
    }
    gui::showImage( drawer->getImage(), "detected words" );
}

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


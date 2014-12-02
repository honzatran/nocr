/**
 * @file text_recognition.h
 * @brief text_recognition header contains class
 * for text recognition in images with dictionary
 * using the method proposed in my bachelor thesis
 * @author Tran Tuan Hiep
 * @version 1.0
 * @date 2014-06-21
 */



#ifndef NOCRLIB_TEXT_RECOGNITION_H
#define NOCRLIB_TEXT_RECOGNITION_H

#include <string>

#include "segment.h"
#include "ocr.h"
#include "dictionary.h"
#include "letter_equiv.h"
#include "word_generator.h"
#include "extremal_region.h"

#define SIZE 1024

/**
 * @brief class for text recognition and extraction in images with dictionary 
 * using extremal region approach combined with dictionary. 
 *
 * Can also display dettected letters and words.
 */
class TextRecognition
{
    public:
        /**
         * @brief default constructor
         */
        TextRecognition() 
            : er_text_detection_(nullptr),
              show_letters_(false), show_words_(false)
        {
            resizer_.setSize(SIZE);
        }

        /**
         * @brief forbidden copy constructor 
         *
         * @param other
         */
        TextRecognition( const TextRecognition &other ) = delete;
        /**
         * @brief forbidden assigment constructor
         *
         * @param other
         *
         * @return 
         */
        TextRecognition& operator=( const TextRecognition &other ) = delete;

        /**
         * @brief recognize and extracts words from image with given dictionary
         * using the algorithm described in my bachelor thesis
         *
         * @param image input, must be in BGR format
         * @param dict dictionary, containing words that can be recognized in image
         *
         * @return vector of instances of class TranslatedWord containing visual and text information
         * about detected words.
         */
        std::vector<TranslatedWord> recognize( cv::Mat &image, const Dictionary &dictionary );

        /**
         * @brief loads image from \p image_path and recognize and extracts words with given dictionary
         * using the algorithm described in my bachelor thesis
         *
         * @param image_path path to input, must be in BGR format
         * @param dict dictionary, containing words that can be recognized in image
         *
         * @throws FileNotFoundException if image does't exist
         *
         * @return vector of instances of class TranslatedWord containing visual and text information
         * about detected words.
         */
        std::vector<TranslatedWord> recognize( const std::string &image_path, const Dictionary &dictionary );
                

        /**
         * @brief load configuration files necceserry for the algorithm
         *
         * @param er1_conf_file configuration file for 1 stage of ER
         * @param er2_conf_file configuration file for 2 stage of ER
         * @param merge_conf_file configuration file for letter word equivalence
         */
        void loadConfiguration( const std::string &er1_conf_file, 
                const std::string &er2_conf_file, const std::string &merge_conf_file );

        /**
         * @brief loads ocr to be used for nonmax suppresion
         *
         * @param ocr
         */
        void loadOcr( AbstractOCR * ocr )
        {
            segmentation_.loadOcr( ocr );
        }

        /**
         * @brief enable/disable showing extracted letters
         *
         * @param show_letters true enable, false disable
         */
        void setShowingLetters( bool show_letters )
        {
            show_letters_ = show_letters; 
        }

        /**
         * @brief enable/disable showing extracted words 
         *
         * @param show_words true enable, false disable
         */
        void setShowingWords( bool show_words )
        {
            show_words_ = show_words;
        }
 

    private:
        Segment<ERTextDetection> segmentation_; 
        LetterWordEquiv word_equivalence_;
        std::unique_ptr<ERTextDetection> er_text_detection_;

        Resizer resizer_;

        cv::Mat loadImage( const std::string &image_path );

        bool show_letters_, show_words_;
        void showLetters( const std::vector<Letter> &letters, const cv::Mat &image );
        void showWords( const std::vector<TranslatedWord> &words, const cv::Mat &image );
};


template <typename T> 
std::vector<TranslatedWord> recognizeWords( Segment<T> &segmentation, 
        const LetterWordEquiv &equiv,
        const Dictionary &dictionary, 
        const cv::Mat &image )
{

    auto letters = segmentation.segment( image );

    WordGenerator generator;

    generator.initHorizontalDetection( letters, equiv );
    vector<TranslatedWord> words = generator.process( dictionary ); 

    auto remaining_letters = generator.getRemainingLetters();
    
    // generator.initVerticalDetection( remaining_letters, equiv );
    // auto vertical_words = generator.process( dictionary );
    //
    // words.insert( words.end(), vertical_words.begin(), vertical_words.end() );

    return words;
}
#endif /* TextRecognition.h */

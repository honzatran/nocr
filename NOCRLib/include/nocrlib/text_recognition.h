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
template <typename EXTRACTION = ERTextDetection, typename OCR = AbstractOCR>
class TextRecognition
{
    public:
        /**
         * @brief default constructor
         */
        TextRecognition() 
            : show_letters_(false), show_words_(false),
              extraction_allocated_(false), extraction_(nullptr)
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

        ~TextRecognition() 
        {
            if (extraction_allocated_)
                delete extraction_;
        }

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
                

        template <typename ... ARGS>
        void constructExtractionMethod(ARGS && ... args)
        {
            if (extraction_allocated_)
                 delete extraction_;

            extraction_allocated_ = true;
            extraction_ = new EXTRACTION(std::forward<ARGS>(args)...);
            segmentation_.loadMethod( extraction_);
        }

        /**
         * @brief loads ocr to be used for nonmax suppresion
         *
         * @param ocr
         */
        void loadOcr( OCR * ocr )
        {
            segmentation_.loadOcr( ocr );
        }

        void loadExtraction(EXTRACTION * extraction )
        {
            if (extraction_allocated_)
                 delete extraction_;

            extraction_allocated_ = false;
            extraction_ = extraction;
            segmentation_.loadMethod(extraction_);
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

        void loadEquivConfiguration(const std::string & equiv_config)
        {
            word_equivalence_.loadConfiguration(equiv_config);
        }
 
    private:
        Segment<EXTRACTION, OCR> segmentation_; 
        LetterWordEquiv word_equivalence_;

        bool extraction_allocated_;

        EXTRACTION * extraction_;

        Resizer resizer_;

        cv::Mat loadImage( const std::string &image_path );

        bool show_letters_, show_words_;
        void showLetters( const std::vector<Letter> &letters, const cv::Mat &image );
        void showWords( const std::vector<TranslatedWord> &words, const cv::Mat &image );
};

// template <typename EXTRACTION, typename OCR>
// void TextRecognition<EXTRACTION, OCR>::loadConfiguration( // const std::string &dictionary_file, 
//         const std::string &er1_conf_file, 
//         const std::string &er2_conf_file, 
//         const std::string &merge_conf_file )
// {
//     er_text_detection_ = std::unique_ptr<EXTRACTION>
//                 ( new ERTextDetection( er1_conf_file, er2_conf_file) );
//
//     segmentation_.loadMethod( er_text_detection_.get() );
//
//     word_equivalence_.loadConfiguration( merge_conf_file);
// }


template <typename EXTRACTION, typename OCR>
std::vector<TranslatedWord> TextRecognition<EXTRACTION, OCR>::recognize( 
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

    generator.initHorizontalDetection( letters, word_equivalence_, image );
    vector<TranslatedWord> words = generator.process( dictionary ); 

    // auto remaining_letters = generator.getRemainingLetters();
    //
    // generator.initVerticalDetection(remaining_letters, word_equivalence_ );
    // auto vertical_words = generator.process( dictionary );
    //
    // words.insert( words.end(), vertical_words.begin(), vertical_words.end() );

    if ( show_words_ )
    {
        showWords( words, image );
    }

    return words;
}


template <typename EXTRACTION, typename OCR>
std::vector<TranslatedWord> TextRecognition<EXTRACTION, OCR>::recognize
    ( const std::string &image_path, const Dictionary &dictionary )
{
    cv::Mat input_image = loadImage( image_path );
    return recognize( input_image, dictionary );
}


template <typename EXTRACTION, typename OCR>
cv::Mat TextRecognition<EXTRACTION, OCR>::loadImage( const std::string &image_path )
{
    cv::Mat image = cv::imread( image_path, CV_LOAD_IMAGE_COLOR );
    if ( image.empty() )
    {
        throw FileNotFoundException( "image at path " + 
                image_path + " doesn't exist" );
    }
    return image;
}


template <typename EXTRACTION, typename OCR>
void TextRecognition<EXTRACTION, OCR>::showLetters( const std::vector<Letter> &letters, const cv::Mat &image )
{
    std::unique_ptr<DrawerInterface> drawer( new BinaryDrawer() );
    drawer->init(image);
    for ( const auto &l : letters )
    {
        drawer->draw(l);
    }

    std::unique_ptr<DrawerInterface> rect_drawer( new RectangleDrawer() );
    rect_drawer->init(drawer->getImage());

    for ( const auto &l : letters )
    {
        rect_drawer->draw(l);
    }

    gui::showImage( rect_drawer->getImage(), "detected letters" );
}

template <typename EXTRACTION, typename OCR>
void TextRecognition<EXTRACTION, OCR>::showWords( 
        const std::vector<TranslatedWord> &words, 
        const cv::Mat &image )
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




template <typename T, typename OCR> 
std::vector<TranslatedWord> recognizeWords( Segment<T, OCR> &segmentation, 
        const LetterWordEquiv &equiv,
        const Dictionary &dictionary, 
        const cv::Mat &image )
{
    auto letters = segmentation.segment( image );

    WordGenerator generator;

    generator.initHorizontalDetection( letters, equiv, image );
    vector<TranslatedWord> words = generator.process( dictionary ); 

    // auto remaining_letters = generator.getRemainingLetters();
    
    // generator.initVerticalDetection( remaining_letters, equiv );
    // auto vertical_words = generator.process( dictionary );
    //
    // words.insert( words.end(), vertical_words.begin(), vertical_words.end() );

    return words;
}

template <typename T>
struct SegmentOCRPolicy<MyOCR, LetterStorage<T> > 
{
    static std::vector<TranslationInfo> translate(MyOCR * ocr, const std::vector<LetterStorage<T> > & letter_candidates)
    {
        std::vector<TranslationInfo> translations;
        std::vector<std::shared_ptr<Component> > components;
        components.reserve(translations.size());
        for (auto & storage : letter_candidates) 
        {
            components.push_back(storage.c_ptr_);
        }

        translations.reserve( letter_candidates.size() );
        std::vector<double> probabilities;

        // for ( const auto &object : letter_candidates )
        // {
        //     translations.push_back( SegmentationPolicy<ERTextDetection>::translate(ocr, object) );
        // }


        auto characters = ocr->translate(components, probabilities);
        int nr_class = ocr->getNumberOfClasses();

        for (std::size_t i = 0; i < characters.size(); ++i) 
        {
            auto  it = probabilities.begin() + i * nr_class;
            vector<double> tmp(it, it + nr_class);
            translations.emplace_back(characters[i], tmp);
        }

        return translations;
    }
};

template <typename T>
struct SegmentOCRPolicy<AbstractOCR, LetterStorage<T> > 
{
    static std::vector<TranslationInfo> translate(AbstractOCR * ocr, const std::vector<LetterStorage<T> > & letter_candidates)
    {
        std::vector<TranslationInfo> translations;
        std::vector<std::shared_ptr<Component> > components;
        components.reserve(translations.size());
        for (auto & storage : letter_candidates) 
        {
            components.push_back(storage.c_ptr_);
        }

        translations.reserve( letter_candidates.size() );
        std::vector<double> probabilities;

        for ( const auto &object : letter_candidates )
        {
            translations.push_back( SegmentationPolicy<ERTextDetection>::translate(ocr, object) );
        }


        return translations;
    }
};

template <>
struct SegmentOCRPolicy<MyOCR, std::shared_ptr<Component> > 
{
    static std::vector<TranslationInfo> translate(MyOCR * ocr, const std::vector<std::shared_ptr<Component> > & letter_candidates)
    {
        std::vector<TranslationInfo> translations;
        translations.reserve( letter_candidates.size() );
        std::vector<double> probabilities;

        auto characters = ocr->translate(letter_candidates, probabilities);
        int nr_class = ocr->getNumberOfClasses();

        for (std::size_t i = 0; i < characters.size(); ++i) 
        {
            auto  it = probabilities.begin() + i * nr_class;
            vector<double> tmp(it, it + nr_class);
            translations.emplace_back(characters[i], tmp);
        }

        return translations;
    }
};


#endif /* TextRecognition.h */

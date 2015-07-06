
/**
 * @file structures.h
 * @brief declaration of classes Letter, Word, LetterStorage, TranslatedWord and etc.
 * @author Tran Tuan Hiep
 * @version 1.0
 * @date 2014-09-20
 */


#ifndef NOCRLIB_STRUCTURES_H
#define NOCRLIB_STRUCTURES_H

#include "component.h"

#include <utility>
#include <vector>
#include <memory>
#include <istream>
#include <ostream>
#include <sstream>
#include <string>
#include <unordered_map>


/**
 * @brief class containing some class S and shared pointer to component
 *
 * @tparam S storage type
 */
template <typename S> struct LetterStorage
{
    typedef std::shared_ptr<Component> CompPtr;

    LetterStorage( const CompPtr &c_ptr, S stat ) 
        : c_ptr_(c_ptr), stat_(stat) 
    { 
    }

    CompPtr c_ptr_;
    S stat_;
};

/**
 * @brief used for computing letter word equivalence
 */
struct ImageLetterInfo 
{
    ImageLetterInfo( const cv::Vec4f &object_mean, const cv::Vec4f &border_mean, float swt_mean ) 
        : object_mean_( object_mean ), border_mean_( border_mean ), swt_mean_( swt_mean )
    {

    }


    cv::Vec4f object_mean_;
    cv::Vec4f border_mean_;
    float swt_mean_;
};

/**
 * @brief lexicographical information for letter candidate r
 */
class TranslationInfo 
{
    public:
        TranslationInfo() = default;

        /**
         * @brief constructor
         *
         * @param translation char translation
         * @param probabilities probabilities for every character in alphabet
         */
        TranslationInfo( char translation, const std::vector<double> &probabilities )
            : translation_( translation ), probabilities_( probabilities )
        {
            confidence_ = getProbability( translation_ );
        }

        /**
         * @brief return probability for \p c
         *
         * @param c charakter
         *
         * @return \f$p( c | r)\f$
         */
        double getProbability( char c ) const;
        /**
         * @brief return character with maximal probability
         *
         * @return character with maximal probability
         */
        char getTranslation() const { return translation_; }


        /**
         * @brief confidece is maximal probability for object
         *
         * @return confidence of candidate
         */
        double getConfidence() const { return confidence_; }

        static bool haveSameLabels(char a, char b);
        static std::vector<int> getLabels(const std::string & str);
    private:
        char translation_;
        std::vector<double> probabilities_;
        double confidence_;

        const static std::unordered_map<char, int> alpha_label_; 
};


/**
 * @brief letter and candidates for letters
 */
class Letter 
{
    public:
        typedef std::shared_ptr<Component> CompPtr;
        
        /**
         * @brief constructor
         */
        Letter() = default;
         
        /**
         * @brief constructor
         *
         * @param comp_ptr shared pointer to component information 
         * @param info color information 
         * @param translation lexicographical information
         */
        Letter( const CompPtr &comp_ptr,
                const TranslationInfo &translation)
                : comp_ptr_(comp_ptr), features_loaded_(false),
                translation_( translation )
        { 
        }

        ~Letter() { }

        /**
         * @brief return x coordiante of left border
         *
         * @return x coordiante of left border
         */
        int getLeftBorder() const { return comp_ptr_->getLeft(); }


        /**
         * @brief return x coordiante of right border
         *
         * @return x coordiante of right border
         */
        int getRightBorder() const { return comp_ptr_->getRight(); }

        /**
         * @brief return y coordiante of upper border
         *
         * @return y coordiante of upper border
         */
        int getUpperBorder() const { return comp_ptr_->getUpper(); }

        /**
         * @brief return y coordiante of lower border
         *
         * @return y coordiante of lower border
         */
        int getLowerBorder() const { return comp_ptr_->getLower(); }

        /**
         * @brief returns height of letter
         *
         * @return height 
         */
        int getHeight() const { return comp_ptr_->getHeight(); }

        /**
         * @brief returns width of letter
         *
         * @return width
         */
        int getWidth() const { return comp_ptr_->getWidth(); }

        double getDiagonal() const { return comp_ptr_->getDiagonal(); }

        /**
         * @brief returns coordinates of left upper corner of bounding box
         *
         * @return left upper corner
         */
        cv::Point getLeftUpperCorner() const 
        {
            return cv::Point( getLeftBorder(), getUpperBorder() );
        }

        /**
         * @brief returns coordinates of right lower corner of bounding box
         *
         * @return right lower corner
         */
        cv::Point getRightLowerCorner() const 
        {
            return cv::Point( getRightBorder(), getLowerBorder() );
        }

        /**
         * @brief returns centroid of letters pixels
         *
         * @return centroid of letters pixels
         */
        cv::Point2d getCentroid() const 
        {
            return comp_ptr_->centroid(); 
        }

        /**
         * @brief returns all letters pixels
         *
         * @return letters pixels
         */
        std::vector<cv::Point> getPoints() const { return comp_ptr_->getPoints(); }

        /**
         * @brief cut letter from image
         *
         * @param image
         *
         * @return cropped letter from image
         */
        cv::Mat cutFromImage(const cv::Mat &image) const 
        { 
            return comp_ptr_->cutComponentFromImage(image, false); 
        }

        /**
         * @brief return binary image of letter
         *
         * @return binary image of letter
         */
        cv::Mat getBinaryMat() const 
        {
            return comp_ptr_->getBinaryMat();
        }

        /**
         * @brief returns bounding box of letter
         *
         * @return bounding box of letter
         */
        cv::Rect getRectangle() const { return comp_ptr_->rectangle(); }

        /**
         * @brief returns pointer to letter component
         *
         * @return pointer to letter component
         */
        CompPtr getPtrComp() const { return comp_ptr_; }

        /**
         * @brief returns character of letter 
         *
         * @return character of letter
         */
        char getTranslation() const 
        {
            return translation_.getTranslation();
        }

        /**
         * @brief returns probability of letter being character c
         *
         * @param c character
         *
         * @return probability of letter being character c
         */
        double getProbability( char c ) const
        {
            return translation_.getProbability(c);
        }

        /**
         * @brief returns confidence of letter
         *
         * @return confidence of letter
         */
        double getConfidence() const 
        {
            return translation_.getConfidence();
        }


    private:
        CompPtr comp_ptr_;

        bool features_loaded_;
        std::vector<float> features_;
        TranslationInfo translation_;

        friend class Word;
};


/**
 * @brief compute from class LetterStorage<S> class ImageLetterInfo
 *
 * @tparam S storage type
 */
template <typename S> 
class StatInfoConvertor 
{
    public:
        /**
         * @brief convert class S to ImageLetterInfo
         *
         * @param storage
         *
         * @return computed ImageLetterInfo
         */
        ImageLetterInfo convert( const LetterStorage<S> &storage ) const = delete;
};


/**
 * @brief Visual and geometric information about detected word in 
 * bitmap
 */
class Word
{ 
    public:
        typedef std::shared_ptr<Component> CompPtr;

        /**
         * @brief constructor
         *
         * @param l first letter of word
         */
        Word( const Letter &l ) 
        {
            rect_ = l.getRectangle();
            letters_.push_back(l);
        }

        /**
         * @brief constructor
         *
         * @param rect bounding box of word
         */
        Word( const cv::Rect &rect )
            : rect_(rect)
        {
        }

        ~Word() { }

        /**
         * @brief add letter to word
         *
         * @param l new letter
         */
        void addLetter(const Letter &l)
        {
            letters_.push_back( l );
            rect_ |= l.getRectangle();
        }

        /**
         * @brief cut word bounding box from image
         *
         * @param image input image
         *
         * @return cropped word from image
         */
        cv::Mat cutFromImage(const cv::Mat &image) const 
        {
            cv::Mat output = image( rect_ );
            return output;
        }

        /**
         * @brief get all letters
         *
         * @return letters in word
         */
        std::vector<Letter> getLetters() const { return letters_; }

        /**
         * @brief get centroids of all letters in word
         *
         * @return 
         */
        std::vector<cv::Point> getCentroids() const;

        /**
         * @brief get number of letters in word
         *
         * @return number of letters
         */
        size_t getSize() const { return letters_.size(); }

        /**
         * @brief return bounding box of word
         *
         * @return bounding box of word
         */
        cv::Rect getRectangle() const 
        {
            return rect_;
        }
        
        friend std::ostream& operator<<( std::ostream &oss, const Word &w )
        {
            oss << w.rect_.tl().x << ':' << w.rect_.tl().y  << ':'
                << w.rect_.size().width << ':' << w.rect_.size().height;

            return oss;
        }

    private:
        std::vector<Letter> letters_;
        cv::Rect rect_;
};


/**
 * @brief TranslatedWord is an output class for WordGenerator, it contains lexicographical information 
 * and visual and geometric information
 */
struct TranslatedWord
{
    /**
     * @brief cons
     *
     * @param visual_information contains geometric and visual information of the detected word
     * @param translation lexicographical information about the word, its translation from 
     * the vocabulary
     */
    TranslatedWord( const Word &visual_information, const std::string &translation )
        : visual_information_( visual_information ), translation_( translation )
    {

    }

    Word visual_information_;
    std::string translation_;

    friend std::ostream& operator<<( std::ostream &oss, const TranslatedWord &word )
    {
        oss << word.translation_ << ':' << word.visual_information_;
        return oss;
    }
};


#endif


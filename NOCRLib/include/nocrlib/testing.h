/**
 * @file testing.h
 * @brief declarations of classes for evaluation of
 * statistics about text recognition in image
 * @author Tran Tuan Hiep
 * @version 1.0
 * @date 2014-10-11
 */


#ifndef NOCRLIB_TESTING_H
#define NOCRLIB_TESTING_H

#ifndef _MSC_VER
#define NOEXCEPT noexcept
#else
#define NOEXCEPT
#endif

#include <memory>
#include <vector>
#include <map>
#include <string>
#include <exception>
#include <ostream>


#include <opencv2/core/core.hpp>

#include "structures.h"


class TruePositiveInterface;

/**
 * @brief contain information about ground truth of detected words 
 * 
 */
class ImageGroundTruth
{
    public:
        /**
         * @brief add new ground truth
         *
         * @param word text of ground truth
         * @param bound_box bounding box of ground truth
         */
        void addGroundTruth( const std::string &word, 
                        const cv::Rect &bound_box );

        /**
         * @brief returns rectangle to the ground truth of 
         * word p, if word isn't ground truth empty rectangle is 
         * returned
         *
         * @param word new word
         *
         * @return rectangle of word in image, empty rectangle 
         * if words isn't on image
         */
        cv::Rect getRectangle( const std::string &word );

        /**
         * @brief returns number of ground truth on image
         *
         * @return number of ground truth on image
         */
        unsigned int getCount() const { return records_.size(); }

        void resize(double scale);

        size_t getLetterCount() const;

        bool containLetter(const Letter &l) const;

        bool isLetterComponent(const Component & c) const;

        bool containWord(
                const TranslatedWord & word, 
                TruePositiveInterface * tp_ptr);

    private:
        std::multimap< std::string, cv::Rect > records_; 
        decltype(records_)::const_iterator 
            findContainingRecord(const cv::Rect &rect) const;
};

/**
 * @brief interface for class, that stores ground truth
 */
class GroundTruthInterface 
{
    public:
        /**
         * @brief add stored ground truth to the container 
         * \p ground_truth
         *
         * @param ground_truth
         *
         * @return 
         */
        virtual void storeGroundTruth
            ( std::map<std::string, ImageGroundTruth> &ground_truth) = 0;

};




class TruePositiveInterface
{
    public:
        virtual bool isTruePositive 
            ( const cv::Rect &detected_bbox, 
              const cv::Rect &gt_bbox ) = 0;
};

class TruePositiveTest : public TruePositiveInterface 
{
    public:
        bool isTruePositive( 
                const cv::Rect &detected_bbox,
                const cv::Rect &gt_bbox ) override;
};

/**
 * @brief exception thrown when we have detected text from image,
 * and evaluate statistics without providing ground truth
 */
class TestingException : public std::exception
{
    public:
        TestingException( const std::string &image_name )
            : image_name_( image_name )
        {

        }

        virtual const char* what() const NOEXCEPT override
        {
            std::string final_msg = "TestingException: for" + 
                image_name_ + " ground truth not found";

            return final_msg.c_str();
        }

    private:
        std::string image_name_;
};

/**
 * @brief class evaluating statistics about recognition 
 * of words in image
 */
class Testing
{
    public:
        Testing();

        /**
         * @brief loads ground truth 
         *
         * @param ground_truth object with ground truth records
         */
        void loadGroundTruth(
                GroundTruthInterface *ground_truth );

        /**
         * @brief set up the pointer to class, that decides whether detected bounding 
         * box of detected word is similar to the one in ground truth
         * if it is, detection is considered to be true positive, negative positive 
         * otherwise
         *
         * @param decider_ptr shared_ptr to class implementing TruePositiveInterface
         *
         * @return 
         */
        void setTruePositiveDecider
            ( TruePositiveInterface * decider_ptr);

        void resetCounters();

        /**
         * @brief updates statistics about recognition
         * with recognition of text in image
         *
         * @param image name of image, words have been recognized from
         * @param detected_words detected words from recognition
         */
        void updateScores( const std::string &image, 
                const std::vector<TranslatedWord> &detected_words );

        /**
         * @brief return current precision based on 
         * provided results
         *
         * @return precision of recognition
         */
        double getPrecision() const;

        /**
         * @brief return current recall based on 
         * provided results
         *
         * @return recall of recognition
         *
         * @throws TestingException
         */
        double getRecall() const;

        void makeRecord( std::ostream &oss ) const;

        void notifyResize(const std::string & name, double scale);

    private:
        std::map<std::string, ImageGroundTruth> ground_truth_;

        TruePositiveInterface * decider_ptr_;

        unsigned int true_positives_;
        unsigned int number_results_;
        unsigned int number_ground_truth_;

        void updatePositiveScore( const cv::Rect &test_bound_box, 
                const cv::Rect & detected_bound_box );
};

struct LetterGT
{
    typedef std::pair<std::string, cv::Rect> LetterGTRec;
    std::vector<LetterGTRec> records_;
    int letter_count_;
};


class LetterDetectionTesting
{
    public:
        LetterDetectionTesting() 
            : true_positives_(0), number_results_(0), number_ground_truth_(0) { }

        void loadGroundTruth( GroundTruthInterface * ground_truth_loader );

        std::vector<bool> updateScores( const std::string &image_name, 
                const std::vector<Letter> &letters );

        /**
         * @brief return mask with components belonging to letters
         *
         * @param image_name
         * @param components
         *
         * @return 
         */
        std::vector<bool> checkLetterComponent(
                const std::string &image_name,
                const std::vector<Component> & components);

        /**
         * @brief return current precision based on 
         * provided results
         *
         * @return precision of recognition
         */
        double getPrecision() const;

        /**
         * @brief return current recall based on 
         * provided results
         *
         * @return recall of recognition
         *
         * @throws TestingException
         */
        double getRecall() const;

        void makeRecord( std::ostream &oss ) const;

        void notifyResize( const std::string & name, double d );
    private:
        std::map<std::string, ImageGroundTruth> ground_truth_;

        unsigned int true_positives_;
        unsigned int number_results_;
        unsigned int number_ground_truth_;
};

class Icdar2013Train
{
    public:
        void setUpGroundTruth( const std::string & file_name );

        typedef std::pair<char, Component> ComponentRecord;
        std::vector<ComponentRecord> 
            getGtComponent( const cv::Mat &image );


    private:
        struct LetterGT
        {
            cv::Rect bbox_;
            char letter_;
            cv::Vec3b bgr_color_;
            cv::Point center_;
        };

        std::vector<LetterGT> letter_ground_truths_;

};


#endif

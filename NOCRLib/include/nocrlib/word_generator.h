/**
 * @file word_generator.h
 * @brief contains declaration of class for generating optimal words from dictionary and given letters 
 * and its output class
 * @author Tran Tuan Hiep
 * @version 1.0
 * @date 2014-06-21
 */

#ifndef NOCRLIB_WORD_GENERATOR_H
#define NOCRLIB_WORD_GENERATOR_H

#include "structures.h"
#include "letter_equiv.h"
#include "ocr.h"
#include "dictionary.h"
#include "trie_node.h"
#include "word_deformation.h"


#include <vector>
#include <string>
#include <tuple>
#include <unordered_map>
#include <ostream>



#define WORD_GENERATOR_DEBUG 0



/**
 * @brief WordGenerator encapsulates algorithm proposed by and co. This approach generates
 * from given candidates letters words that are most likely in the picture.
 * The instance is initialize with letters candidates. After that will generates 
 * best word with given dictionary using method process
 *      
 */
class WordGenerator
{
    public:
        typedef std::unique_ptr< AbstractOCR > OcrPtr;

        typedef std::vector<Letter> VecLetter;

        /**
         * @brief default constructor
         */
        WordGenerator() = default;

        /* --------------------------------------------------------------------------*/
        /**
         * @brief default constructor 
         *
         * constuctor for letters find optimal words from dictionary
         *
         * @param letters letters candidates, from which we generate words
         *
         * Computes all neccesery information about the letters in image
         */
        /* ----------------------------------------------------------------------------*/
        WordGenerator( const VecLetter &letters, const cv::Mat & image );

        // void initialize( const VecLetter &letters, const LetterWordEquiv &equivalence );

        /**
         * @brief computes neccesery information for detection of horizontal words
         *
         * @param letters letters candidates, from which we generate words
         *
         * Computes all neccesery information about letters in image for the algorithm
         */
        void initHorizontalDetection( const VecLetter &letters,  
                const cv::Mat & image );

        /**
         * @brief computes neccesery information for detection of vertical words
         *
         * @param letters letters candidates, from which we generate words
         *
         * Computes all neccesery information about letters in image for the algorithm
         */
        void initVerticalDetection( const VecLetter &letters, 
                const cv::Mat & image );
 
        /**
         * @brief return optimal configuration from letters for word
         *
         * @param word word we find configuration for  
         *
         * @return instance of Word, best configuration for \p word
         *
         * Generate best configuration for \p word from letters candidates 
         * and return geometric and visual information about it uses the 
         * algorithm from .
         */

        std::vector<TranslatedWord> detectWords(const std::vector<std::string> & words);

        /**
         * @brief return words, that are in the image 
         * and dictionary
         *
         * @param dictionary user given dictionary with words
         *
         * @return vector of TranslatedWord
         *
         * Method computes, which words from dictionary are in input image.
         * Using the algorithm described in bachelor thesis from .
         */
        std::vector<TranslatedWord> process( const Dictionary &dictionary );

        /**
         * @brief returns letters not belonging to output 
         * words from method WordGenerator::process
         *
         * @return unused letters in output words 
         *
         * If WordGenerator::process wasn't called, method would return empty vector.
         */
        std::vector<Letter> getRemainingLetters() const ;

        void setDeformationCostFactor(double deformation_cost_factor)
        {
            deformation_cost_factor_ = deformation_cost_factor;
        }

        void setSpaceStdDevFactor(double space_stddev_factor)
        {
            space_stddev_factor_ = space_stddev_factor;
        }
        
    private:

        // private classes  ===============================================
        //
        struct EdgeWeights
        {
            double deformation_cost;
            double space_dist;

            EdgeWeights() : deformation_cost(0), space_dist(0) { }

            EdgeWeights(double _deformation_cost, double _space_dist)
                : deformation_cost(_deformation_cost), space_dist( _space_dist) { }
        };

        struct WordRecord
        {
            WordRecord( double _score, const std::string & _text, 
                    const std::vector<int> & _indices, int _edit_dist)
                : score(_score), text(_text), indices(_indices), edit_dist(_edit_dist)
            {

            }

            double score;
            std::string text;
            std::vector<int> indices;
            int edit_dist;
        };


        struct ScoreRecord
        {
            ScoreRecord()
                : i(-1), j(-1), 
                score(std::numeric_limits<double>::lowest())
            {
            }

            ScoreRecord( int _i, int _j, double _score, const std::vector<int> & _indices )
                : i(_i), j(_j), score(_score), indices(_indices)
            {

            }

            void tie( int & _i, int & _j , double & _score )
            {
                _i = i;
                _j = j;
                _score = score;
            }

            int i, j;
            double score;
            std::vector<int> indices;

            friend std::ostream& operator<<( std::ostream &oss, const ScoreRecord &rec )
            {
                oss << rec.i << ':' << rec.j << ':' << rec.score;
                return oss;
            }

            friend bool operator<(const ScoreRecord & a, const ScoreRecord & b)
            {
                return a.score < b.score;
            }
        };

        class WordDescriptors
        {
        public:
            float dist_sum, dist_sqr;

            int succesor;
            std::size_t letters_count;

            // centroid of first letter of word and the second one
            void merge(const WordDescriptors & other, double space_dist);

            float getDistStDeviation() const;
        };


        // ============================= private members and methods
        double deformation_cost_factor_ = 0.25;
        double space_stddev_factor_ = 0.1;
        const double k_epsilon = 0.9;
        // const double k_theta = .0;
        // const double k_epsilon = .0;
        const static int k_max_missing_letters = 3;

        int current_depth_;

        int max_length_;
        int rows_, cols_;
        cv::Mat image_;

        VecLetter letters_;

        std::string text_;
        // empty_score_[i] = sum from 0 to i score(letters_[i], epsilon)
        std::vector<double> empty_score_;
        // Matrix<double> optimal_score_;
        std::vector<double> optimal_score_;
        // Matrix<int> succesors_;
        // std::vector<int> succesors_;
        std::vector<WordDescriptors> descriptors_informations_;

        // storing current maxima for the column
        std::vector<ScoreRecord> maxima_;
        // mask of used letters 
        std::vector<bool> used_letters_;

        std::vector<WordDescriptors> descriptor_prototypes_;

        std::unordered_multimap<int, std::pair<int, EdgeWeights> > edges_;
        // std::map<double, WordRecord> detected_words_;
        std::vector<WordRecord> detected_words_;

        double computeDeformationCost( int i, int j );

        template <typename EdgeEval> 
        void fillRelationTables(EdgeEval && eval)
        {
            edges_.clear();
            detected_words_.clear();

            for ( size_t i = 0; i < letters_.size(); ++i ) 
            {
                double max_distance = getMaxDistance(i);
                for ( size_t j = i+1; j < letters_.size(); ++j ) 
                {
                    if (getDistance(i,j) >= max_distance)
                    {
                        continue;
                    }

                    // if ( equivalence.areEquivalent( letters_[i], letters_[j] ) )
                    cv::Rect a_rect = letters_[i].getRectangle();
                    cv::Rect b_rect = letters_[j].getRectangle();
                    cv::Point a_centroid = letters_[i].getCentroid();
                    cv::Point b_centroid = letters_[j].getCentroid();
                    
                    double height_ratio = 
                        a_rect.height < b_rect.height ? (double)b_rect.height/a_rect.height : (double)a_rect.height/b_rect.height;

                    double intersection_area = (a_rect & b_rect).area();

                    if (intersection_area/a_rect.area() < 0.8
                            && intersection_area/b_rect.area() < 0.8
                            && (a_centroid.x < b_centroid.x) 
                            && height_ratio < 3.5
                            && (a_rect.br().x < b_rect.br().x))
                        // && equivalence.areEquivalent(letters_[i], letters_[j]))
                    {
                        EdgeWeights edge_weights = eval(letters_[i], letters_[j]);
                        edges_.insert( std::make_pair( i, std::make_pair( j, edge_weights) ) ); 
                    }
                }
            }

        }

        void fillEmptyScore();

        // return sum from start to end score(letters_[i],epsilon)
        double getEmptyScore( int start, int end );
// ==============================table initialization and updating =========================
        void findConfiguration( const std::string &word );
        void initializeTable();
        void updateTable();
// =============================finding element in scores table ==========================

        std::tuple<int, int, double> findMaxSuccesor( int base_index, int start_optimal_row, int start_word_col, WordDescriptors & descriptors );

        std::vector<ScoreRecord> findMaxConfiguration( int start_row, int start_col, std::size_t min_length);

        std::vector<int> reconstruct( int i, int j ); // reconstruct word from position;

//===================================traversing dictionary trie ===========================
        void traverse( TrieNode *root, std::string &word );
        
        void updateTables(char current_letter);

        void updateMaximal();
        double getMaxDistance( int i );
        double getDistance( int i, int j );
        double getCharacterScoreOnly( const std::vector<int> &indices, double score );

        static std::size_t getMaxEditDist(std::size_t size);

        // ==========================word descriptor =======================
        //
        void initWordDescPrototypes(const cv::Mat & image);
        WordDescriptors mergeDescriptors(const WordDescriptors & a, 
                const WordDescriptors & b, double space_dist);
        // void mergeDescInformation(std::size_t i, std::size_t j, float dist);
        //
        double spaceDist(const Letter & a, const Letter & b);
};


#endif /* WordGenerator.h */

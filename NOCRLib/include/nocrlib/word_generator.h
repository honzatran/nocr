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


#include <vector>
#include <string>
#include <tuple>
#include <unordered_map>
#include <ostream>

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
         * @param equivalence pointer to determine if two letter can be in one word
         *
         * Computes all neccesery information about the letters in image
         */
        /* ----------------------------------------------------------------------------*/
        WordGenerator( const VecLetter &letters, 
                const LetterWordEquiv &equivalence );

        // void initialize( const VecLetter &letters, const LetterWordEquiv &equivalence );

        /**
         * @brief computes neccesery information for detection of horizontal words
         *
         * @param letters letters candidates, from which we generate words
         * @param equivalence pointer to determine if two letter can be in one word
         *
         * Computes all neccesery information about letters in image for the algorithm
         */
        void initHorizontalDetection( const VecLetter &letters, const LetterWordEquiv &equivalence );

        /**
         * @brief computes neccesery information for detection of vertical words
         *
         * @param letters letters candidates, from which we generate words
         * @param equivalence pointer to determine if two letter can be in one word
         *
         * Computes all neccesery information about letters in image for the algorithm
         */
        void initVerticalDetection( const VecLetter &letters, const LetterWordEquiv &equivalence );
 
        /* --------------------------------------------------------------------------*/
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
        /* ----------------------------------------------------------------------------*/
        Word findConfiguration( const std::string &word );

        /* --------------------------------------------------------------------------*/
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
        /* ----------------------------------------------------------------------------*/
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
        
    private:

        // private classes  ===============================================
        struct WordRecord
        {
            WordRecord( const std::string &text, const std::vector<int> &indices )
                : text_(text), indices_(indices)
            {

            }

            std::string text_;
            std::vector<int> indices_;
        };


        struct ScoreRecord
        {
            ScoreRecord( int i, int j, double score )
                : i_(i), j_(j), score_(score)
            {

            }

            void tie( int &i, int &j , double &score )
            {
                i = i_;
                j = j_;
                score = score_;
            }

            int i_, j_;
            double score_;

            friend std::ostream& operator<<( std::ostream &oss, const ScoreRecord &rec )
            {
                oss << rec.i_ << ':' << rec.j_ << ':' << rec.score_;
                return oss;
            }
        };

        // ============================= private members and methods
        const double k_theta = 1.75;
        const double k_epsilon = 0.9;
        const static int k_max_missing_letters = 3;

        int current_depth_;

        int max_length_;

        template <typename T> using Matrix = std::vector< std::vector<T> >;
        VecLetter letters_;

        std::string text_;
        // empty_score_[i] = sum from 0 to i score(letters_[i], epsilon)
        std::vector<double> empty_score_;
        // table to update
        Matrix<double> optimal_score_;
        // succesors 
        Matrix<int> succesors_;
        // storing current maxima for the column
        std::vector<ScoreRecord> maxima_;
        // mask of used letters 
        std::vector<bool> used_letters_;

        std::unordered_multimap<int, std::pair<int,double> > edges_;
        std::map<double,WordRecord> detected_words_;

        double computeDeformationCost( int i, int j );

        template <typename EdgeEval> 
        void fillRelationTables(const LetterWordEquiv &equivalence, const EdgeEval &eval )
        {
            edges_.clear();
            detected_words_.clear();

            for ( size_t i = 0; i < letters_.size(); ++i ) 
            {
                double max_distance = getMaxDistance(i);
                for ( size_t j = i+1; j < letters_.size(); ++j ) 
                {
                    if ( getDistance(i,j) >= max_distance )
                    {
                        continue;
                    }

                    if ( equivalence.areEquivalent( letters_[i], letters_[j] ) )
                    // if ( true )
                    {
                        double deformation_cost = eval( letters_[i], letters_[j] );
                        edges_.insert( std::make_pair( i, std::make_pair( j, deformation_cost ) ) ); 
                    }
                }
            }

        }

        void fillEmptyScore();

        // return sum from start to end score(letters_[i],epsilon)
        double getEmptyScore( int start, int end );
// ==============================table initialization and updating =========================
        void initializeTable();
        void updateTable();
// =============================finding element in scores table ==========================

        std::tuple<int, int, double> findMaxSuccesor( int base_index, int start_optimal_row, int start_word_col );

        std::tuple<int, int, double> findMaxConfiguration( int start_row, int start_col );

        std::vector<int> reconstruct( int i, int j ); // reconstruct word from position;

//===================================traversing dictionary trie ===========================
        void traverse( TrieNode *root, std::string &word );
        
        void updateTables(char current_letter);

        void updateMaximal();
        double getMaxDistance( int i );
        double getDistance( int i, int j );
        double getCharacterScoreOnly( const std::vector<int> &indices, double score );
};


/// @cond
class WangWordGenerator
{
    public:
        typedef std::unique_ptr< AbstractOCR > OcrPtr;
        
        WangWordGenerator( const std::vector<Letter> &letters, const LetterWordEquiv &equivalence, 
                const cv::Size &image_size);
        Word findConfiguration( const std::string &word );
    private:
        template <typename T> using Matrix = std::vector< std::vector<T> >;

        const double gamma_ = 1.0; 
        const double epsilon_ = 1;
        const double theta_ = 0.05;
        std::string text_;

        std::vector< Letter> letters_;
        Matrix<double> word_equivalance_;
        Matrix<int> configurations_;
        std::vector<double> configurations_scores_;
        Matrix<double> distances_;

        std::vector<int> getCharacterIndices( const std::string &word );
        
        void fillRelationTables(const LetterWordEquiv &equivalence);
        void setUpDistances( int i, int j );
        double computeDistance( int i, int j );

        void generatePossibleConfiguration();
        bool notContaining( const std::vector<int> &vector, int k ); 
        void updateConfigurationTables( int depth, char current_letter );

};

/// @endcond
#endif /* WordGenerator.h */

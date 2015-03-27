/* 
 * Tran Tuan Hiep
 * Implementation of methods and classes declared in word_generator.h
 *
 * Compiler: g++ 4.8.3
 */

#include "../include/nocrlib/word_generator.h"
#include "../include/nocrlib/letter_equiv.h"
#include "../include/nocrlib/dictionary.h"
#include "../include/nocrlib/trie_node.h"
#include "../include/nocrlib/assert.h"
#include "../include/nocrlib/utilities.h"

#include <iostream>
#include <locale>
#include <limits>
#include <algorithm>

#include <opencv2/core/core.hpp>

using namespace std;

WordGenerator::WordGenerator( const VecLetter &letters, 
    const LetterWordEquiv &equivalence )
{
    initHorizontalDetection( letters, equivalence );
}

// =================== initializations for detection =============
void WordGenerator::initHorizontalDetection
    ( const VecLetter &letters, const LetterWordEquiv &equivalence )
{
    letters_ = letters;

    std::sort( letters_.begin(), letters_.end(),[] 
        ( const Letter &a, const Letter &b )
    {
        return a.getLeftBorder() < b.getLeftBorder();
    });
    
    fillRelationTables( equivalence, [] (const Letter &a, const Letter &b) 
            {
                cv::Point a_top_right( a.getRightBorder(), a.getUpperBorder() );
                cv::Point b_top_left( b.getLeftBorder(), b.getUpperBorder() );
                cv::Point2d diff = a_top_right - b_top_left;

                double tmp = diff.x * diff.x / a.getWidth() + 
                    diff.y * diff.y / a.getHeight();
                return std::sqrt( tmp );
            });

    fillEmptyScore();
}

void WordGenerator::initVerticalDetection
    ( const VecLetter &letters, const LetterWordEquiv &equivalence )
{
    letters_ = letters;

    std::sort( letters_.begin(), letters_.end(),[] 
        ( const Letter &a, const Letter &b )
    {
        return a.getUpperBorder() < b.getUpperBorder();
    });
    
    fillRelationTables( equivalence, [] (const Letter &a, const Letter &b) 
            {
                cv::Point a_bottom_left( a.getLeftBorder(), a.getLowerBorder() );
                cv::Point b_top_left( b.getLeftBorder(), a.getUpperBorder() );
                cv::Point2d diff = a_bottom_left - b_top_left;

                double tmp = diff.x * diff.x / a.getWidth() + 
                    diff.y * diff.y / a.getHeight();
                return std::sqrt( tmp );
            });

    fillEmptyScore();
}


double WordGenerator::computeDeformationCost( int i, int j )
{
    cv::Point a( letters_[i].getRightBorder(), letters_[i].getUpperBorder() );
    cv::Point b( letters_[j].getLeftBorder(), letters_[j].getUpperBorder() );
    cv::Point2d diff = a - b;

    double tmp = diff.x * diff.x / letters_[i].getWidth() + 
        diff.y * diff.y / letters_[i].getHeight();
    return std::sqrt( tmp );
}

void WordGenerator::fillEmptyScore()
{
    // we fill empty scores
    // empty_score_[i] = sum from i = 0 to i characterScore(letters_[i], empty label )
    empty_score_.resize( letters_.size() ); 

    double prev_sum = 0;
    for ( size_t i = 0; i < empty_score_.size(); ++i ) 
    {
        double confidence = letters_[i].getConfidence();
        double score = 1 - confidence; 
        empty_score_[i] = score + prev_sum;
        prev_sum += score;
    }
}

// ====================== finding best configuration for one word =================
Word WordGenerator::findConfiguration( const std::string &word )
{
    // deprecated
    text_ = word;
    max_length_ = text_.size();
    initializeTable();
    updateTable();
    int i,j; 
    double max;
    std::tie( i, j, max ) = findMaxConfiguration( 0, 0 );
    vector<int> configuration = reconstruct(i,j);
    if ( configuration.size() > 1 )
    {
        detected_words_.insert( std::make_pair( max, WordRecord( word, configuration ) ) );
    }
    Word output( letters_[ configuration[0] ].getRectangle() );
    for ( int i : configuration )
    {
        output.addLetter( letters_[i] );
    }
    
    return output; 

}

void WordGenerator::initializeTable()
{
    // 1 phase of algorithm
    int size = letters_.size();
    double empty_score_sum = empty_score_.back();
    succesors_ = Matrix<int>( size, vector<int>( max_length_,-1 ));
    optimal_score_ = Matrix<double>( size, vector<double>( max_length_, 0 ) );

    for ( int p = max_length_ - 1; p >= 0; --p ) 
    {
        char c = text_[p];
        for ( int i = size-1; i >= 0; --i ) 
        {
            double empty_score = empty_score_sum - empty_score_[i];
            optimal_score_[i][p] = letters_[i].getProbability(c)+ empty_score; 
        }
    }
}


void WordGenerator::updateTable()
{
    // 2 phase of algorithm
    int size = letters_.size();
    for ( int p = max_length_ - 1; p >= 0; --p )
    {
        char c = text_[p];
        for( int i = size-1; i >= 0; --i )
        {
            int max_row, max_col;
            double max_score; 
            std::tie( max_row, max_col, max_score ) = findMaxSuccesor( i, i+1, p+1 );
            double max_value = letters_[i].getProbability(c) + max_score ;
            if ( optimal_score_[i][p] < max_value )
            {
                optimal_score_[i][p] = max_value;
                succesors_[i][p] = max_row * max_length_ + max_col;
            }
        }
    }
}

// ============== methods for table information extraction ==============

std::tuple<int, int, double> WordGenerator::findMaxSuccesor( int base_index, int start_optimal_row, int start_optimal_col )
{
    double max_score = std::numeric_limits<double>::lowest();
    double max_probability = max_score;
    int num_letters = letters_.size();
    if ( start_optimal_row >= num_letters || start_optimal_col >= max_length_ )
    {
        return std::make_tuple( -1, -1, max_score ); 
    }
    int max_row = 0;
    int max_col = 0;

    auto range = edges_.equal_range(base_index);

    // check all possible neighbours
    for ( auto it = range.first; it != range.second; ++it )
    {
        int j;
        double deformation_cost;
        std::tie(j, deformation_cost) = it->second;
        // int j = it->second;
        double empty_score = getEmptyScore( base_index + 1, j - 1 ); 
        for ( int q = start_optimal_col; q < max_length_; ++q )
        {
            double score = optimal_score_[j][q] + empty_score - k_theta * deformation_cost;
            if ( max_score < score )
            {
                max_score = score; 
                max_row = j;
                max_col = q;
                max_probability = optimal_score_[j][q] + empty_score;
            }
        }
    }
    return std::make_tuple( max_row, max_col, max_probability ); 
}


std::tuple< int, int, double > WordGenerator::findMaxConfiguration( int start_row, int start_col )
{
    int size = letters_.size();
    int max_row = 0;
    int max_col = 0;
    double max_score = std::numeric_limits<double>::lowest();
    for ( int i = start_row; i < size; ++i )
    {
        double empty_score = getEmptyScore( start_row, i - 1 ); 
        for ( int j = start_col; j < max_length_; ++j )
        {
            double val = optimal_score_[i][j] + empty_score;
            if ( val > max_score )
            {
                max_row = i;
                max_col = j;
                max_score = val; 
            }
            
        }
    }
    return std::make_tuple( max_row, max_col, max_score );
}


double WordGenerator::getEmptyScore( int start, int end )
{
    if ( start > end )
    {
        return 0;
    }

    double end_sum = empty_score_[end];

    if ( start == 0 )
    {
        return end_sum;                 
    }

    return ( end_sum - empty_score_[start-1] );
}

vector<int> WordGenerator::reconstruct( int start_row, int start_col )
{
    int i = start_row; 
    int j = start_col;

    int val = succesors_[i][j];
    vector<int> word( 1, i );
    while ( val != -1 ) 
    {
        i = val / max_length_; 
        j = val % max_length_; 
        val = succesors_[i][j];
        word.push_back(i);
    }

    return word;
}




// ================== processing dictionary tree =================
std::vector<TranslatedWord> WordGenerator::process( const Dictionary &dictionary )
{
    if ( letters_.empty() )
    {
        return std::vector<TranslatedWord>();
    }

    // initialization table size
    current_depth_ = dictionary.getMaxLength();
    max_length_ = dictionary.getMaxLength();

    optimal_score_ = Matrix<double>( letters_.size(), vector<double>( current_depth_, 0 ) );
    succesors_ = Matrix<int>( letters_.size(), vector<int>( current_depth_, -1 ) );
    maxima_ = std::vector<ScoreRecord>( max_length_ + 1, 
           ScoreRecord(-1, -1, std::numeric_limits<double>::lowest()) ); 


    std::string word;
    auto root_children = dictionary.getRoot()->getChildren();
    for ( const auto &p : root_children ) 
    {
        --current_depth_;
        word.push_back( p.first );
        traverse( p.second, word );
        word.pop_back();
        ++current_depth_;
    }


    // choose best words from lexicon
    used_letters_ = vector<bool>( letters_.size(), false );
    vector<TranslatedWord> output;
    for ( auto it = detected_words_.rbegin(); it != detected_words_.rend(); ++it )
    {
        vector<int> indices = it->second.indices_; 
        bool new_word = true;
        for ( int i : indices )
        {
            if ( used_letters_[i] ) 
            {
                new_word = false;
                break;
            }
        }

        if ( new_word )
        {
            Word w( letters_[ indices[0] ].getRectangle() );
            for ( int i : indices )
            {
                used_letters_[i] = true;
                w.addLetter( letters_[i] );
            }
            output.push_back( TranslatedWord( w, it->second.text_ ) );
        }
    }

    return output;
}

void WordGenerator::traverse( TrieNode *node, std::string &word )
{
    // fill current_depth column of optimal score matrix and updates succesors;
    updateTables(word.back());
    // find current maximal value in optinal score table
    updateMaximal();

    
    if ( node->isEndWordNode() ) 
    {
        int max_row, max_col;
        double max_value;

        maxima_[current_depth_].tie( max_row, max_col, max_value );
        vector<int> indices = reconstruct( max_row, max_col );
        if ( indices.size() > 1 )
        {
            if ( indices.size() < k_max_missing_letters )
            {
                double character_score = getCharacterScoreOnly( indices, max_value );
                if ( character_score >  word.size() * k_epsilon )
                {
                    std::string text = word;
                    std::reverse( text.begin(), text.end() );
                    detected_words_.insert( 
                            std::make_pair( max_value, WordRecord( text, indices ) ) );
                }
            }
            else if ( indices.size() > word.size() - k_max_missing_letters )
            {
                std::string text = word;
                std::reverse( text.begin(), text.end() );
                detected_words_.insert( 
                        std::make_pair( max_value, WordRecord( text, indices ) ) );
            }
        }
    }

    
    auto children  = node->getChildren();
    for ( const auto &pair : children ) 
    {
        current_depth_--;
       
        word.push_back( pair.first ); 
        traverse( pair.second, word );
        word.pop_back();
        current_depth_++;
    }
}

void WordGenerator::updateTables( char current_letter )
{
    // inicialization that letter_[i] is the last letter of current word
    double empty_score_sum = empty_score_.back();
    for ( int i = letters_.size() - 1; i >= 0; --i )
    {
        optimal_score_[i][current_depth_] = letters_[i].getProbability(current_letter)
            + empty_score_sum - empty_score_[i];
    }

    for ( int i = letters_.size() - 1; i >= 0; --i )
    {
        int max_row, max_col;
        double max_score; 
        std::tie( max_row, max_col, max_score ) = findMaxSuccesor( i, i+1, current_depth_ +1 ); 
        double max_value = letters_[i].getProbability(current_letter) + max_score ;
        if ( optimal_score_[i][current_depth_] < max_value )
        {
            optimal_score_[i][current_depth_] = max_value;
            succesors_[i][current_depth_] = max_row * max_length_ + max_col;
        }
    }

}

void WordGenerator::updateMaximal()
{
    maxima_[current_depth_] = maxima_[current_depth_+1];
    for ( size_t i = 0; i < letters_.size(); ++i )
    {
        double score = getEmptyScore(0, i-1) + optimal_score_[i][current_depth_];
        if ( maxima_[current_depth_].score_ < score )
        {
            maxima_[current_depth_] = ScoreRecord( i, current_depth_, score );
        }
    }
}

double WordGenerator::getMaxDistance( int i )
{
    double diagonal = letters_[i].getDiagonal();
    const double k_epsilon = 1.75;
    return diagonal * k_epsilon;
}

double WordGenerator::getDistance( int i, int j )
{
    cv::Point2d a_centroid = letters_[i].getCentroid();
    cv::Point2d b_centroid = letters_[j].getCentroid();

    return norm( a_centroid - b_centroid );
}

double WordGenerator::getCharacterScoreOnly( const std::vector<int> &indices, double configuration_score )
{
    double empty_score_sum = empty_score_.back();
    for ( int i : indices ) 
    {
        empty_score_sum -= ( 1 - letters_[i].getConfidence() );
    }
    return configuration_score - empty_score_sum;
}

std::vector<Letter> WordGenerator::getRemainingLetters() const
{
    vector<Letter> remaining_letters;
    for ( int i = 0; i < (int) used_letters_.size(); ++i )
    {
        if ( !used_letters_[i] )
        {
            remaining_letters.push_back( letters_[i] );
        }
    }

    return remaining_letters;
}


//======================Wang word generator====================
//
//
WangWordGenerator::WangWordGenerator( const std::vector<Letter>  &letters, 
        const LetterWordEquiv &equivalence, const cv::Size &image_size )
    : letters_( letters )
{
    UNUSED(image_size);
    size_t size = letters_.size();
    word_equivalance_ = Matrix<double>( size , vector<double>(size, 0) ); 
    distances_ = Matrix<double>( size, vector<double>(size,0) );
    
    // fillWordEquivalence(equivalence);
    fillRelationTables( equivalence );
}

void WangWordGenerator::fillRelationTables( const LetterWordEquiv &equivalence )
{
    for ( size_t i = 0; i < letters_.size(); ++i ) 
    {
        for ( size_t j = i+1; j < letters_.size(); ++j ) 
        {
            double val = equivalence.computeProbability( letters_[i], letters_[j] );
            word_equivalance_[i][j] = word_equivalance_[j][i] = val;
            setUpDistances( i, j );
        }
    }
}

void WangWordGenerator::setUpDistances( int i, int j )
{
    distances_[i][j] = computeDistance( i, j );
    distances_[j][i] = computeDistance( j, i );
}
    
double WangWordGenerator::computeDistance( int i, int j )
{
    cv::Point a( letters_[i].getRightBorder(), letters_[i].getUpperBorder() );
    cv::Point b( letters_[j].getLeftBorder(), letters_[j].getUpperBorder() );
    cv::Point2d diff = a - b;

    double tmp = diff.x * diff.x / letters_[i].getWidth() + 
        diff.y * diff.y / letters_[i].getHeight();
    /*
     * return norm( diff ); 
     */
    //
    return std::sqrt( tmp );
}

Word WangWordGenerator::findConfiguration( const std::string &text )
{
    text_ = text;
    generatePossibleConfiguration(); 
    double max_score = configurations_scores_[0]; 
    int output = 0;
    for ( size_t i = 1; i < configurations_scores_.size(); ++i )
    {
        if ( max_score < configurations_scores_[i] ) 
        {
            output = i;
            max_score = configurations_scores_[i];
        }
    }
    std::vector<int> conf = configurations_[output];
    cv::Rect rect = letters_[ conf[0] ].getRectangle();
    Word w( rect );
    for ( int i : conf ) 
    {
        w.addLetter( letters_[i] );
    }
    return w;
}

void WangWordGenerator::generatePossibleConfiguration()
{
    int size = letters_.size();
    configurations_scores_ = std::vector<double>( size ); 
    configurations_ = vector< vector<int> >( size, vector<int>(text_.size(), -1) );  
    // initiaze score 
    // in step 0 score of letter[i] = probability( letter[i] = word_indices_[0] )
    for ( int i = 0; i < size; ++i )
    {
        configurations_scores_[i] = gamma_ * letters_[i].getProbability( text_[0] ); 
        configurations_[i][0] = i;
        // cout << configurations_scores_[i] << endl;
    }
    // domyslet nejaky parametr pro deformaci 


    for ( size_t i = 1; i < text_.size(); ++i )
    {
        updateConfigurationTables( i, text_[i] );// i depth
    }
}

void WangWordGenerator::updateConfigurationTables( int depth, char current_letter )
{
    /* * if ( depth < 2 )
     *
     *
     * {
     *     // fillPairTables(int depth);
     *     return;
     * }
     */
    // cout << alpha_[current_index] << endl;
    int size = letters_.size();
    for ( int j = 0; j < size; ++j )
    {
        double val = -std::numeric_limits<double>::max();
        /*
         * cv::Point2d a = letters_[depth-1].getCentroid();
         * cv::Point2d b = letters_[depth-2].getCentroid();
         */
        int last_letter = configurations_[j][depth-1];

        for ( int k = 0; k < size; ++k )
        {
            if( j != k && notContaining( configurations_[j], k ) )
            {
                double tmp =  gamma_ * letters_[k].getProbability(current_letter) + 
                    epsilon_ * word_equivalance_[last_letter][k] - theta_ * distances_[last_letter][k];
                    //+ angleDiff( a, b, letters_[k].getCentroid() ); 
                    //
                if ( tmp > val )
                {
                    val = tmp;
                    configurations_[j][depth] = k;
                }
            }
        }
        configurations_scores_[j] += val;
    }
}


bool WangWordGenerator::notContaining( const std::vector<int> &vector, int val )
{
    return std::find( vector.begin(), vector.end(), val ) == vector.end();
}





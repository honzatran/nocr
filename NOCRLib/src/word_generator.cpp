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
#include "../include/nocrlib/drawer.h"
#include "../include/nocrlib/levenstein_distance.h"

#include <iostream>
#include <locale>
#include <limits>
#include <algorithm>

#include <opencv2/core/core.hpp>

#define WORD_DESCRIPTOR 0
#define MAX_SUCCESSOR_CAPACITY 5
#define MAX_CONFIGURATION_CAPACITY 5

typedef std::tuple<int, int, double, double> SuccessorType;

template <typename T, std::size_t CAPACITY, typename COMPARATOR = std::less<T> >
class SortedVector
{
public:
    SortedVector() = default;

    SortedVector(const COMPARATOR & comp)
        : comp_(comp)
    {
        data_.reserve(CAPACITY + 1);
    }

    bool insert(const T & val)
    {
        auto it = std::lower_bound(data_.begin(), data_.end(), val, comp_);
        if (it == data_.end())
        {
            if (data_.size() >= CAPACITY)
            {
                return false;
            }
            else
            {
                data_.push_back(val);
                return true;
            }
        }
        else
        {
            data_.insert(it, val);
            if (data_.size() >= CAPACITY)
            {
                data_.pop_back();
            }

            return true;
        }
    }

    const std::vector<T> & getInnerVector() const
    {
        return data_;
    }

    std::vector<T> & getInnerVector() 
    {
        return data_;
    }

private:
    std::vector<T> data_;
    COMPARATOR comp_;
};

template < typename T, std::size_t I>
struct TupleComparator
{
    bool operator() (const T & lhs, const T & rhs)
    {
        return std::get<I>(lhs) > std::get<I>(rhs);
    }
};



using namespace std;

WordGenerator::WordGenerator( const VecLetter &letters, 
    const cv::Mat & image )
{
    initHorizontalDetection( letters, image );
}

// =================== initializations for detection =============
void WordGenerator::initHorizontalDetection
    ( const VecLetter &letters, const cv::Mat & image )
{
    letters_ = letters;

    std::sort( letters_.begin(), letters_.end(),[] 
        ( const Letter &a, const Letter &b )
    {
        return a.getLeftBorder() < b.getLeftBorder();
    });

    initWordDescPrototypes(image);

    WordDeformation deformation;
    deformation.setImage(image);


    fillRelationTables(  
            [this] (const Letter &a, const Letter &b) -> EdgeWeights
            {
                cv::Point a_top_right( a.getRightBorder(), a.getUpperBorder());
                cv::Point b_top_left( b.getLeftBorder(), b.getUpperBorder());
                cv::Point2d diff = a_top_right - b_top_left;

                double tmp = diff.x * diff.x / a.getWidth() + 
                    diff.y * diff.y / a.getHeight();

                double space_dist = spaceDist(a, b);

                return { std::sqrt( tmp ), space_dist};
            });

#if WORD_GENERATOR_DEBUG
    std::unique_ptr<DrawerInterface> drawer( new BinaryDrawer() );
    drawer->init(image);
    for ( const auto &l : letters )
    {
        drawer->draw(l);
    }

    // std::unique_ptr<DrawerInterface> rect_drawer( new RectangleDrawer() );
    // rect_drawer->init(drawer->getImage());
    //
    // for ( const auto &l : letters )
    // {
    //     rect_drawer->draw(l);
    // }
    //
    cv::Mat edge_img = drawer->getImage();
    for (const auto &pair : edges_)
    {
        cv::line(edge_img, letters_[pair.first].getCentroid(),
                letters_[pair.second.first].getCentroid(), 255);
    }

    image_ = edge_img;
    // gui::showImage(edge_img, "hrany");
#endif

    fillEmptyScore();

   // evaluator_.loadConfiguration("boost_dccost.xml");
}

void WordGenerator::initVerticalDetection
    ( const VecLetter &letters, const cv::Mat & image )
{
    letters_ = letters;

    std::sort( letters_.begin(), letters_.end(),[] 
        ( const Letter &a, const Letter &b )
    {
        return a.getUpperBorder() < b.getUpperBorder();
    });
    
    initWordDescPrototypes(image);

    WordDeformation deformation;
    deformation.setImage(image);

    fillRelationTables( 
            [] (const Letter &a, const Letter &b) -> EdgeWeights
            {
                cv::Point a_bottom_left( a.getLeftBorder(), a.getLowerBorder() );
                cv::Point b_top_left( b.getLeftBorder(), a.getUpperBorder() );
                cv::Point2d diff = a_bottom_left - b_top_left;

                double tmp = diff.x * diff.x / a.getWidth() + 
                    diff.y * diff.y / a.getHeight();
                return { std::sqrt(tmp), 0};
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
//
std::vector<TranslatedWord> WordGenerator::detectWords(
        const std::vector<std::string> & words)
{
    for (const string & w : words)
    {
        findConfiguration(w);
    }

    std::sort(detected_words_.begin(), detected_words_.end(),
            [](const WordRecord & a, const WordRecord & b)
            {
                if (a.score == b.score) 
                {
                    return a.edit_dist > b.edit_dist;
                }
                else 
                {
                    return a.score < b.score;
                }
            });


    used_letters_ = vector<bool>( letters_.size(), false );
    vector<TranslatedWord> output;
    for ( auto it = detected_words_.rbegin(); it != detected_words_.rend(); ++it )
    {
        vector<int> & indices = it->indices;

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
            output.push_back( TranslatedWord( w, it->text));
        }
    }

    return output;



}

void WordGenerator::findConfiguration( const std::string &word )
{
    // deprecated
    text_ = word;
    max_length_ = text_.size();
    initializeTable();
    updateTable();
    std::size_t word_length = word.size();

    std::size_t min_length = word_length < k_max_missing_letters ?
        word_length : 3 * word_length / 4;

    auto max_configurations = findMaxConfiguration( 0, 0, min_length);

    int max_row, max_col;
    double max_value;


    for (ScoreRecord & rec : max_configurations)
    {
        rec.tie(max_row, max_col, max_value);
        vector<int> & indices= rec.indices;
        cv::Rect word_rec = letters_[indices[0]].getRectangle();

        std::size_t area = word_rec.area();
        for (std::size_t i = 1; i < indices.size(); ++i)
        {
            cv::Rect characted_rect  = letters_[indices[i]].getRectangle();
            word_rec |= characted_rect;
            area += characted_rect.area();
        }

        double area_ratio = (double)area/word_rec.area();

        if ( indices.size() > 1)
        {
            if ( indices.size() < k_max_missing_letters )
            {
                double character_score = getCharacterScoreOnly( indices, max_value );
                if ( character_score >  word.size() * k_epsilon &&
                        area_ratio > 0.45)
                {
                    std::string text = word;
                    detected_words_.emplace_back(max_value, text, indices, 0);

#if WORD_DESCRIPTOR
                    cout << text << " " << max_value 
                        << " " << area_ratio << endl;
#endif
                }
            }
            else if (area_ratio > 0.3)
            {
                std::string text = word;

                std::string tmp;
                tmp += letters_[indices[0]].getTranslation();

                for (std::size_t i = 1; i < indices.size(); ++i) 
                {
                    tmp += letters_[indices[i]].getTranslation();
                }

                std::vector<int> tmp_labels = TranslationInfo::getLabels(tmp);
                std::vector<int> text_labels = TranslationInfo::getLabels(text);

                LevensteinDistance<int> levenstein;
                std::size_t edit_dist = levenstein(text_labels, tmp_labels);

                std::size_t max_edit_dist = getMaxEditDist(text.size());
                if (edit_dist <= max_edit_dist)
                {
                    // detected_words_.insert( 
                    //         std::make_pair( max_value, WordRecord( text, indices ) ) );
                    detected_words_.emplace_back(max_value, text, indices, edit_dist);

#if WORD_DESCRIPTOR
                    cout << text << " " << tmp << " " << max_value 
                        << " " << area_ratio << endl;
#endif
                }
            }

        }
    }

}

void WordGenerator::initializeTable()
{
    // 1 phase of algorithm
    int size = letters_.size();
    rows_ = size;
    cols_ = max_length_;
    double empty_score_sum = empty_score_.back();

    optimal_score_ = std::vector<double>( rows_ * cols_ , 0);
    descriptors_informations_ = std::vector<WordDescriptors>(rows_ * cols_);

    for ( int p = max_length_ - 1; p >= 0; --p ) 
    {
        char c = text_[p];
        for ( int i = size-1; i >= 0; --i ) 
        {
            double empty_score = empty_score_sum - empty_score_[i];
            optimal_score_[i + p * rows_] = letters_[i].getProbability(c)+ empty_score; 
        }
    }

}


void WordGenerator::updateTable()
{
    // 2 phase of algorithm
    for ( int p = cols_ - 1; p >= 0; --p )
    {
        char c = text_[p];
        for ( int i = rows_ - 1; i >= 0; --i )
        {
            int max_row, max_col;
            double max_score; 
            WordDescriptors tmp = descriptor_prototypes_[i];
           
            std::tie( max_row, max_col, max_score ) = findMaxSuccesor( i, i+1, p + 1,
                    tmp);

            double max_value = letters_[i].getProbability(c) + max_score ;
            if ( optimal_score_[i + p * rows_] < max_value )
            {
                optimal_score_[i + p * rows_] = max_value;
                tmp.succesor = max_row * max_length_ + max_col;
                
                descriptors_informations_[i + p * rows_] = tmp;

                // descriptors_informations_[i + current_depth_ * rows_].merge(
                //         descriptors_informations_[max_col * rows_ + max_row]);
                // mergujeme informace o slove
            }
            else
            {
                descriptors_informations_[i + p * rows_] = descriptor_prototypes_[i];
            }
        }
    }
}

// ============== methods for table information extraction ==============
     

std::tuple<int, int, double> WordGenerator::findMaxSuccesor( int base_index, int start_optimal_row, int start_optimal_col, 
        WordDescriptors & word_descriptors )
{
    double max_score = std::numeric_limits<double>::lowest();
    double max_probability = max_score;
    if ( start_optimal_row >= rows_ || start_optimal_col >= cols_ )
    {
        return std::make_tuple( -1, -1, max_score ); 
    }

    int max_row = 0;
    int max_col = 0;

    WordDescriptors max_descriptor;

    auto range = edges_.equal_range(base_index);

    // check all possible neighbours
    //
#if WORD_GENERATOR_DEBUG
    // std::size_t dist = std::distance(range.first, range.second);

#endif
    for ( auto it = range.first; it != range.second; ++it )
    {
        int j;
        EdgeWeights edge_weights;
        std::tie(j, edge_weights) = it->second;
        // int j = it->second;
        //
        double empty_score = getEmptyScore( base_index + 1, j - 1 ); 
        for ( int q = start_optimal_col; q < max_length_; ++q )
        {
            double score = optimal_score_[j + q * rows_] + empty_score;

            auto tmp = mergeDescriptors(word_descriptors, 
                    descriptors_informations_[j + q * rows_], 
                    edge_weights.space_dist);


            score -= (deformation_cost_factor_) * edge_weights.deformation_cost;

            if ( max_score < score )
            {
                max_score = score; 
                max_row = j;
                max_col = q;
                max_probability = optimal_score_[j + q * rows_] + empty_score;
                max_descriptor = tmp;
            }
        }
    }

    word_descriptors = max_descriptor;
    
    return std::make_tuple( max_row, max_col, max_probability ); 
}


auto WordGenerator::findMaxConfiguration( int start_row, int start_col, std::size_t min_length )
    -> std::vector<ScoreRecord>
{
    vector<ScoreRecord> output_records;
    output_records.reserve(MAX_CONFIGURATION_CAPACITY + 1);

    double min_enabled_score = std::numeric_limits<double>::lowest();

    for ( int i = start_row; i < rows_; ++i )
    {
        double empty_score = getEmptyScore( start_row, i - 1 ); 
        for ( int j = start_col; j < max_length_; ++j )
        {
            auto & curr_desc = descriptors_informations_[i + j * rows_];

            if (curr_desc.letters_count < min_length)
            {
                continue;
            }

            double score = optimal_score_[i + j * rows_] + empty_score;
            if (curr_desc.letters_count > 2)
            {
                score -= space_stddev_factor_ * curr_desc.getDistStDeviation();
            }

            if (score < min_enabled_score)
            {
                continue;
            }

            auto r_it = std::find_if(output_records.begin(), output_records.end(), 
                    [score] (const ScoreRecord & sr)
                    {
                        return sr.score <= score;
                    });

            vector<int> indices = reconstruct(i, j);
            bool common = false;
            decltype(r_it) it = output_records.begin();

            for (; it != r_it; ++it)
            { 
                vector<int> & it_indices = it->indices;

                if (nonEmptyIntersection(indices.begin(), indices.end(), 
                        it_indices.begin(), it_indices.end()))
                {
                    common = true;
                    break;
                }
            }

            if (!common)
            {
                auto new_it = output_records.emplace(r_it, i, j, score, indices);

                auto it = new_it + 1;
                bool exist_lesser = false;
                for (; it != output_records.end(); ++it)
                {
                    std::vector<int> & it_indices = it->indices;
                    if (nonEmptyIntersection(indices.begin(), indices.end(), 
                            it_indices.begin(), it_indices.end()))
                    {
                        exist_lesser = true;
                        break;
                    }
                }

                if (exist_lesser)
                {
                    output_records.erase(it);
                }
                else
                {
                    if (output_records.size() > MAX_CONFIGURATION_CAPACITY)
                    {
                        output_records.pop_back();
                    }
                }

                min_enabled_score = output_records.back().score;
            }
        }
    }


    return output_records;
}


double WordGenerator::getEmptyScore( int start, int end )
{
    if ( start > end )
    {
        NOCR_ASSERT(start - 1 == end,  "empty score of neighbour letters");
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

    int val = descriptors_informations_[i + j * rows_].succesor;
    vector<int> word( 1, i );
    while ( val != -1 ) 
    {
        i = val / max_length_; 
        j = val % max_length_; 
        val = descriptors_informations_[i + j * rows_].succesor;
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

    current_depth_ = dictionary.getMaxLength();
    max_length_ = dictionary.getMaxLength();

    rows_ = letters_.size();
    cols_ = current_depth_;

    optimal_score_ = std::vector<double>(rows_ * cols_ , 0);
    descriptors_informations_.resize(rows_ * cols_);
    
    maxima_ = std::vector<ScoreRecord>( max_length_ + 1);
           


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
    //
    std::sort(detected_words_.begin(), detected_words_.end(),
            [](const WordRecord & a, const WordRecord & b)
            {
                if (a.score == b.score) 
                {
                    return a.edit_dist > b.edit_dist;
                }
                else 
                {
                    return a.score < b.score;
                }
            });


    used_letters_ = vector<bool>( letters_.size(), false );
    vector<TranslatedWord> output;
    for ( auto it = detected_words_.rbegin(); it != detected_words_.rend(); ++it )
    {
        vector<int> & indices = it->indices;

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
            output.push_back( TranslatedWord( w, it->text));
        }
    }

    return output;
}

void WordGenerator::traverse( TrieNode *node, std::string &word )
{
    // fill current_depth column of optimal score matrix and updates succesors;
    updateTables(word.back());
    // find current maximal value in optinal score table
    // updateMaximal();

    
    if ( node->isEndWordNode() ) 
    {
        int max_row, max_col;
        double max_value;


        std::size_t min_length = word.size() <= k_max_missing_letters ? word.size() : 3 * word.size() /4;

        auto max_configurations = findMaxConfiguration(0, current_depth_, min_length);

        for (ScoreRecord & rec : max_configurations)
        {

            rec.tie(max_row, max_col, max_value);
            vector<int> & indices= rec.indices;
            cv::Rect word_rec = letters_[indices[0]].getRectangle();

            std::size_t area = word_rec.area();
            for (std::size_t i = 1; i < indices.size(); ++i)
            {
                cv::Rect characted_rect  = letters_[indices[i]].getRectangle();
                word_rec |= characted_rect;
                area += characted_rect.area();
            }

            double area_ratio = (double)area/word_rec.area();

            if ( indices.size() > 1)
            {
                if ( indices.size() < k_max_missing_letters )
                {
                    double character_score = getCharacterScoreOnly( indices, max_value );
                    if ( character_score >  word.size() * k_epsilon &&
                            area_ratio > 0.45)
                    {
                        std::string text = word;
                        std::reverse( text.begin(), text.end() );
                        // detected_words_.insert( 
                        //         std::make_pair( max_value, WordRecord( text, indices ) ) );
                        //
                        detected_words_.emplace_back(max_value, text, indices, 0);

#if WORD_DESCRIPTOR
                        cout << text << " " << max_value 
                            << " " << area_ratio << endl;
#endif
                    }
                }
                else if (area_ratio > 0.3)
                {
                    std::string text = word;
                    std::reverse( text.begin(), text.end() );

                    std::string tmp;
                    tmp += letters_[indices[0]].getTranslation();

                    for (std::size_t i = 1; i < indices.size(); ++i) 
                    {
                        tmp += letters_[indices[i]].getTranslation();
                    }

                    std::vector<int> tmp_labels = TranslationInfo::getLabels(tmp);
                    std::vector<int> text_labels = TranslationInfo::getLabels(text);

                    LevensteinDistance<int> levenstein;
                    std::size_t edit_dist = levenstein(text_labels, tmp_labels);

                    std::size_t max_edit_dist = getMaxEditDist(text.size());
                    if (edit_dist <= max_edit_dist)
                    {
                        // detected_words_.insert( 
                        //         std::make_pair( max_value, WordRecord( text, indices ) ) );
                        detected_words_.emplace_back(max_value, text, indices, edit_dist);

#if WORD_DESCRIPTOR
                        cout << text << " " << tmp << " " << max_value 
                            << " " << area_ratio << endl;
#endif
                    }
                }

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
        optimal_score_[i + current_depth_ * rows_] = letters_[i].getProbability(current_letter)
            + empty_score_sum - empty_score_[i];
        // tady se inicializujou hodnoty
        // descriptors_informations_[i + current_depth_ * rows_] = descriptor_prototypes_[i];
    }

    for ( int i = letters_.size() - 1; i >= 0; --i )
    {
        int max_row, max_col;
        double max_score; 
        WordDescriptors tmp = descriptor_prototypes_[i];
       
        std::tie( max_row, max_col, max_score ) = findMaxSuccesor( i, i+1, current_depth_ +1,
                tmp);

        double max_value = letters_[i].getProbability(current_letter) + max_score ;
        if ( optimal_score_[i + current_depth_ * rows_] < max_value )
        {
            optimal_score_[i + current_depth_ * rows_] = max_value;
            tmp.succesor = max_row * max_length_ + max_col;
            
            descriptors_informations_[i + current_depth_ * rows_] = tmp;

            // descriptors_informations_[i + current_depth_ * rows_].merge(
            //         descriptors_informations_[max_col * rows_ + max_row]);
            // mergujeme informace o slove
        }
        else
        {
            descriptors_informations_[i + current_depth_ * rows_] = descriptor_prototypes_[i];
        }
    }

}

void WordGenerator::updateMaximal()
{
    maxima_[current_depth_] = maxima_[current_depth_+1];
    for ( size_t i = 0; i < letters_.size(); ++i )
    {
        WordDescriptors & curr_desc = descriptors_informations_[i + current_depth_ * rows_];

        double score = getEmptyScore(0, i-1) + optimal_score_[i + current_depth_ * rows_];
        if (curr_desc.letters_count > 2)
        {
            score -= curr_desc.getDistStDeviation();
        }

        vector<int> indices = reconstruct(i, current_depth_);

        if ( maxima_[current_depth_].score < score )
        {
            maxima_[current_depth_] = ScoreRecord( i, current_depth_, score, indices );
        }
    }
}

double WordGenerator::getMaxDistance( int i )
{
    double diagonal = letters_[i].getDiagonal();
    const double k_epsilon = 3;
    return std::min<double>(diagonal * k_epsilon, image_.rows/4);
}

double WordGenerator::getDistance( int i, int j )
{
    return spaceDist(letters_[i], letters_[j]);
    // return cv::norm(a_centroid - b_centroid);
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

void WordGenerator::initWordDescPrototypes(const cv::Mat & image)
{
    image_ = image;
    descriptor_prototypes_.reserve(letters_.size());
    WordDeformation word_deformation;
    word_deformation.setImage(image);

    for (std::size_t i = 0; i < letters_.size(); ++i)
    {
        WordDescriptors prototype;

        prototype.succesor = -1;
        prototype.letters_count = 1;
        prototype.dist_sum = prototype.dist_sqr = 0;

        descriptor_prototypes_.push_back(prototype);
    }
}

double WordGenerator::spaceDist(const Letter & a, const Letter & b)
{
    cv::Point a_centroid = a.getCentroid();
    cv::Point b_centroid = b.getCentroid();


    double a_rx = a.getRightBorder();
    double b_lx = b.getLeftBorder();

    cv::Point tmp = b_centroid - a_centroid;
    cv::Point diff(-tmp.y, tmp.x);
    
    float c = diff.x * a_centroid.x + diff.y * a_centroid.y;

    double a_ry = (c - diff.x * a_rx)/diff.y;
    if (a_ry < a.getUpperBorder())
    {
        a_ry = a.getUpperBorder();
        a_rx = (c - diff.y * a_ry)/diff.x;
    } 
    else if (a_ry > a.getLowerBorder())
    {
        a_ry = a.getLowerBorder();
        a_rx = (c - diff.y * a_ry)/diff.x;
    }


    double b_ly = (c - diff.x * b_lx)/diff.y;
    if (b_ly < b.getUpperBorder())
    {
        b_ly = b.getUpperBorder();
        b_lx = (c - diff.y * b_ly)/diff.x;
    }
    else if (b_ly > b.getLowerBorder())
    {
        b_ly = b.getLowerBorder();
        b_lx = (c - diff.y * b_ly)/diff.x;
    }


#if WORD_GENERATOR_DEBUG
    // if (centroid_dist < shorter_dist)
    // {
    //     cv::Rect a_rect = a.getRectangle();
    //     a_rect |= b.getRectangle();
    //     cv::Mat tmp;
    //     image_(a_rect).copyTo(tmp);
    //     
    //     cv::rectangle(tmp, a.getRectangle() - a_rect.tl(), 170);
    //     cv::rectangle(tmp, b.getRectangle() - a_rect.tl(), 100);
    //
    //     gui::showImage(tmp, "wrong");
    //
    //     std::cout << centroid_dist << " " << shorter_dist  << std::endl;
    //     std::cout << a.getTranslation() << " " << b.getTranslation() << std::endl;
    // }
#endif

    return cv::norm(cv::Point(a_rx, a_ry) - cv::Point(b_lx, b_ly));
    
    // return cv::norm(a_centroid - b_centroid);
}

void WordGenerator::WordDescriptors::merge(const WordDescriptors & other, double space_dist)
{
    letters_count += other.letters_count;
    
    dist_sum += (space_dist+ other.dist_sum);
    dist_sqr += (space_dist * space_dist + other.dist_sqr);
}

float WordGenerator::WordDescriptors::getDistStDeviation() const
{
    int k = letters_count - 1;
    float dist_mean = dist_sum/k;
    float dist_variation = (dist_sqr + k * dist_mean * dist_mean - 2 * dist_mean * dist_sum)/(k - 1);

    return std::sqrt(dist_variation);
}

auto WordGenerator::mergeDescriptors(const WordGenerator::WordDescriptors & a,
        const WordGenerator::WordDescriptors & b, double space_dist) -> WordDescriptors
{
    WordDescriptors tmp = a;
    tmp.merge(b, space_dist);

    return tmp;
}


std::size_t WordGenerator::getMaxEditDist(std::size_t size)
{
    if (size <= 3)
    {
        return 0;
    }
    else if (size <= 5)
    {
        return 1;
    }
    else if (size <= 10)
    {
        return size/4 + 1;
    }
    else
    {
        return size/4;
    }
}





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

#define WORD_DESCRIPTOR 1

using namespace std;

WordGenerator::WordGenerator( const VecLetter &letters, 
    const LetterWordEquiv &equivalence, const cv::Mat & image )
{
    initHorizontalDetection( letters, equivalence, image );
}

// =================== initializations for detection =============
void WordGenerator::initHorizontalDetection
    ( const VecLetter &letters, const LetterWordEquiv &equivalence,
      const cv::Mat & image )
{
    letters_ = letters;

    std::sort( letters_.begin(), letters_.end(),[] 
        ( const Letter &a, const Letter &b )
    {
        return a.getLeftBorder() < b.getLeftBorder();
    });

    initWordDescPrototypes(image);

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

    cv::Mat edge_img = drawer->getImage();
    image_ = edge_img;
#endif
    
    fillRelationTables( equivalence, [this] (const Letter &a, const Letter &b) -> EdgeWeights
            {
                cv::Point a_top_right( a.getRightBorder(), a.getUpperBorder());
                cv::Point b_top_left( b.getLeftBorder(), b.getUpperBorder());
                cv::Point2d diff = a_top_right - b_top_left;

                double tmp = diff.x * diff.x / a.getWidth() + 
                    diff.y * diff.y / a.getHeight();



                double space_dist = spaceDist(a, b);

                return { std::sqrt( tmp ), space_dist};
            });

    fillEmptyScore();

   // evaluator_.loadConfiguration("boost_dccost.xml");
}

void WordGenerator::initVerticalDetection
    ( const VecLetter &letters, const LetterWordEquiv &equivalence, const cv::Mat & image )
{
    letters_ = letters;

    std::sort( letters_.begin(), letters_.end(),[] 
        ( const Letter &a, const Letter &b )
    {
        return a.getUpperBorder() < b.getUpperBorder();
    });
    
    initWordDescPrototypes(image);

    fillRelationTables( equivalence, [] (const Letter &a, const Letter &b) -> EdgeWeights
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
    rows_ = size;
    cols_ = max_length_;
    double empty_score_sum = empty_score_.back();
    optimal_score_ = std::vector<double>( rows_ * cols_ , 0);

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
    int size = letters_.size();
    for ( int p = max_length_ - 1; p >= 0; --p )
    {
        char c = text_[p];
        for( int i = size-1; i >= 0; --i )
        {
            int max_row, max_col;
            double max_score; 
            std::tie( max_row, max_col, max_score ) = findMaxSuccesor( i, i+1, p+1, descriptors_informations_[0] );
            double max_value = letters_[i].getProbability(c) + max_score ;
            if ( optimal_score_[i + p * rows_] < max_value )
            {
                optimal_score_[i + p * rows_] = max_value;

                descriptors_informations_[i + current_depth_ * rows_].succesor = max_row * max_length_ + max_col;
                // descriptors_informations_[i + current_depth_ * rows_].merge(
                //         descriptors_informations_[max_col * rows_ + max_row]);
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
    int num_letters = letters_.size();
    if ( start_optimal_row >= num_letters || start_optimal_col >= max_length_ )
    {
        return std::make_tuple( -1, -1, max_score ); 
    }
    int max_row = 0;
    int max_col = 0;

    WordDescriptors max_descriptor;

    auto range = edges_.equal_range(base_index);
    cv::Rect base_bbox =  letters_[base_index].getRectangle();
    cv::Point base_tl = base_bbox.tl();
    cv::Point base_bl = base_tl + cv::Point(0, base_bbox.height);

    // check all possible neighbours
    for ( auto it = range.first; it != range.second; ++it )
    {
        int j;
        EdgeWeights edge_weights;
        std::tie(j, edge_weights) = it->second;
        // int j = it->second;
        //
        cv::Rect n_bbox = letters_[j].getRectangle();
        cv::Point n_tl = n_bbox.tl();
        cv::Point n_bl = n_tl + cv::Point(0, n_bbox.height);

        double empty_score = getEmptyScore( base_index + 1, j - 1 ); 
        for ( int q = start_optimal_col; q < max_length_; ++q )
        {
            double score = optimal_score_[j + q * rows_] + empty_score;
            std::size_t succesor = descriptors_informations_[j + q * rows_].succesor/cols_;
            cv::Rect n2_bbox = letters_[succesor].getRectangle();
            cv::Point n2_tl = n2_bbox.tl();
#if WORD_GENERATOR_DEBUG
            if (descriptors_informations_[j + q * rows_].letters_count > 1)
            {
                double angle_tl = angle(base_tl, n_tl, n2_tl);
                double angle_bl = angle(base_bl, n_bl, n2_tl + cv::Point(0, n2_bbox.height));
                
                if (std::max(angle_tl, angle_bl) < 2.5)
                {
                    continue;
                }
            }
#endif


            auto tmp = mergeDescriptors(word_descriptors, descriptors_informations_[j + q * rows_], edge_weights.space_dist);
            // uhel
            // if (tmp.letters_count > 5)
            // {
            //     double word_score = evaluator_.getCost(tmp.getDescriptor());
            //     score -= word_score;
            // }
            // else 
            // {
                // score -= (1 - k_theta) * edge_weights.deformation_cost;
                // score -= 0.2 * edge_weights.deformation_cost;
            // }
            //
//             if (tmp.letters_count > 2)
//             {
//                 score -= 0.5 * tmp.getDistStDeviation();
//             }
//
//             if (tmp.letters_count > 3)
//             {
//                 score -= tmp.getAngleStdDeviation();
// #if WORD_GENERATOR_DEBUG
//                 cout << tmp.getAngleStdDeviation() << endl;
// #endif
//             }
//             else if (tmp.letters_count == 3)
//             {
//                 score -= (CV_PI - tmp.angle)/(CV_PI);
//             }

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
            double val = optimal_score_[i + j * rows_] + empty_score;
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

    // initialization table size
    current_depth_ = dictionary.getMaxLength();
    max_length_ = dictionary.getMaxLength();

    rows_ = letters_.size();
    cols_ = current_depth_;

    optimal_score_ = std::vector<double>(rows_ * cols_ , 0);
    descriptors_informations_.resize(rows_ * cols_);
    
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
        cv::Rect word_rec = letters_[indices[0]].getRectangle();

        std::size_t area = word_rec.area();
        for (std::size_t i = 1; i < indices.size(); ++i)
        {
            cv::Rect characted_rect  = letters_[indices[i]].getRectangle();
            word_rec |= characted_rect;
            area += characted_rect.area();
        }

        if ( indices.size() > 1)
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
            else if ( indices.size() >= 3 * word.size()/4)
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
                
                if (edit_dist <= std::max<std::size_t>(text.size()/4, 2) + 1)
                {
                    detected_words_.insert( 
                            std::make_pair( max_value, WordRecord( text, indices ) ) );

#if WORD_DESCRIPTOR
                cout << text << " " << tmp << " " << endl;
#endif
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
            score -= 0.1 * curr_desc.getDistStDeviation();
        }

        if ( maxima_[current_depth_].score_ < score )
        {
            maxima_[current_depth_] = ScoreRecord( i, current_depth_, score );
        }
    }
}

double WordGenerator::getMaxDistance( int i )
{
    double diagonal = letters_[i].getDiagonal();
    const double k_epsilon = 3;
    return diagonal * k_epsilon;
}

double WordGenerator::getDistance( int i, int j )
{
    cv::Point2d a_centroid = letters_[i].getCentroid();
    cv::Point2d b_centroid = letters_[j].getCentroid();

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

        cv::Vec3f color_medians = word_deformation.getColorMedians(letters_[i]);
        prototype.color_sums = color_medians;
        prototype.color_sums_sqr = color_medians.mul(color_medians);
    
        prototype.succesor = -1;
        prototype.letters_count = 1;
        prototype.dist_sum = prototype.dist_sqr = 0;
        prototype.height_sum = letters_[i].getHeight();
        prototype.height_sqr = prototype.height_sum * prototype.height_sum;
        prototype.center1 = letters_[i].getCentroid();
        prototype.angle_sqr = 0;
        prototype.angle_sum = 0;

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

    double centroid_dist = cv::norm(b_centroid - a_centroid);
    double shorter_dist = cv::norm(cv::Point2d(a_rx, a_ry) - cv::Point2d(b_lx, b_ly));

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
    color_sums += other.color_sums;
    color_sums_sqr += other.color_sums_sqr;
    height_sum += other.height_sum;
    height_sqr += other.height_sqr;
    
    cv::Point v1 = center1 - other.center1;
    
    dist_sum += (space_dist+ other.dist_sum);
    dist_sqr += (space_dist * space_dist + other.dist_sqr);

    center2 = other.center1;
    if (letters_count > 2)
    {
        cv::Point v2 = other.center2 - other.center1;
        angle = std::acos(v1.ddot(v2)/(cv::norm(v1) * cv::norm(v2)));

        angle_sum += angle;
        angle_sqr += angle * angle;
    }
    else
    {
        angle = CV_PI;
    }

}

float WordGenerator::WordDescriptors::getDistStDeviation() const
{
    int k = letters_count - 1;
    float dist_mean = dist_sum/k;
    float dist_variation = (dist_sqr + k * dist_mean * dist_mean - 2 * dist_mean * dist_sum)/(k - 1);

    return std::sqrt(dist_variation);
}

float WordGenerator::WordDescriptors::getAngleStdDeviation() const
{
    int k = letters_count - 2;

    float angle_mean = angle_sum/k;
    float angle_variation = (angle_sqr+ k * angle_mean* angle_mean - 2 * angle_mean * angle_sum)/(k - 1);

    return std::sqrt(angle_variation);
}

auto WordGenerator::mergeDescriptors(const WordGenerator::WordDescriptors & a,
        const WordGenerator::WordDescriptors & b, double space_dist) -> WordDescriptors
{
    WordDescriptors tmp = a;
    tmp.merge(b, space_dist);

    return tmp;
}

std::vector<float> WordGenerator::WordDescriptors::getDescriptor() const
{
    cv::Vec3f means, variance;
    means = color_sums * (1./letters_count);
    variance = color_sums_sqr + (int)letters_count * means.mul(means) - 2 * means.mul(color_sums);
    variance *= (1./(letters_count - 1));
    
    cv::Vec3f coef_variation;

    coef_variation[0] = std::sqrt(variance[0])/means[0];
    coef_variation[1] = std::sqrt(variance[1])/means[1];
    coef_variation[2] = std::sqrt(variance[2])/means[2];

    coef_variation *= (1. + 1./(4 * letters_count));

    int k = letters_count - 1;
    float dist_mean = dist_sum/k;
    float dist_variation = (dist_sqr + k * dist_mean * dist_mean - 2 * dist_mean * dist_sum)/(k - 1);

    // cout << dist_mean << " " << dist_variation << endl;
    float dist_coef_variation = std::sqrt(dist_variation)/dist_mean * (1. + 1./(4 * k));

    k++;
    float height_mean = (float)height_sum/k;
    float height_variation= (height_sqr + k * height_mean * height_mean - 2 * height_mean * height_sum)/(k - 1);

    float height_coef_variation = std::sqrt(height_variation)/height_mean * (1. + 1./(4 * k));

    return { coef_variation[0], coef_variation[1], coef_variation[2], 
        angle, dist_coef_variation, height_coef_variation };
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





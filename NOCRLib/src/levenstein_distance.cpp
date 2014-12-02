/* 
 * Tran Tuan Hiep
 * Implementation of methods and classes declared in levenstein_distance.h
 *
 * Compiler: g++ 4.8.3
 */

#include "../include/nocrlib/levenstein_distance.h"
#include "../include/nocrlib/dictionary.h"

#include <vector>
#include <string>
#include <algorithm>

using namespace std;

int LevensteinDistance::operator() ( const string &a, const string &b )
{
    int distances[a.size()+1][b.size()+1];
    int rows = a.size() + 1;
    int cols = b.size() + 1; 
    
    for ( int i = 0; i < rows; ++i )
    {
        distances[i][0] = i;
    }

    for ( int i = 0; i < cols ; ++i )
    {
        distances[0][i] = i;
    }

    for ( int j = 1; j < cols; ++j )
    {
        for ( int i = 1; i < rows; ++i ) 
        {
            if ( a[i-1] == b[j-1] ) 
            {
                distances[i][j] = distances[i-1][j-1];
            }
            else
            {
                distances[i][j] = minimum( distances[i-1][j], distances[i][j-1], distances[i-1][j-1] ) + 1;
            }
        }
    }
    return distances[a.size()][b.size()];
}

int LevensteinDistance::minimum( int a, int b, int c )
{
    return std::min( a, std::min( b,c ) );
}



LevensteinDistanceTrie::LevensteinDistanceTrie( const std::string &word, TrieNode *root, int max_length )
    : word_(word), root_(root)
{
    rows_ = max_length + 1; 
    cols_ = word.size() + 1; 
    distances_ = vector< vector<int> >( rows_ , vector<int>( cols_ , 0) );
    std::transform( word_.begin(), word_.end(), word_.begin(), ::tolower );
    output_.clear();
}

void LevensteinDistanceTrie::findTranslation() 
{
    //fill 0 row of distance map
    for ( int j = 0; j < cols_; ++j )
    {
        distances_[0][j] = j;
    }

    min_ = std::numeric_limits<int>::max(); 

    findClosestMatch( root_, 1 );
}

void LevensteinDistanceTrie::findClosestMatch
    ( TrieNode *node, int fill_row ) 
{
    // check if node is word end
    if ( node->word_node_ )
    {
        int distance = distances_[fill_row-1][cols_-1];
        if ( distance < min_ )
        {
            output_.clear();
            output_.push_back( tmp_ );
            min_ = distance; 
        }
        else if ( distance == min_ ) 
        {
            output_.push_back(tmp_);
        }
    }
    
    for ( auto &child: node->childs_ ) 
    {
        fillRow( fill_row, child.first );
        tmp_.push_back(child.first);
        findClosestMatch( child.second, fill_row + 1 );
        tmp_.pop_back();
    }
    
}

void LevensteinDistanceTrie::fillRow( int row, char c )
{
    distances_[row][0] = row;
    for( int j = 1; j < cols_; ++j )
    {
        if ( c == word_[j-1] ) 
        {
            distances_[row][j] = distances_[row-1][j-1];
        }
        else
        {
            distances_[row][j] = minimum( distances_[row-1][j], distances_[row][j-1], distances_[row-1][j-1] ) + 1;
        }
    }
}

/**
 * @file levenstein_distance.h
 * @brief computing levenstein distance
 * @author Tran Tuan Hiep
 * @version 1.0
 * @date 2014-09-16
 */


#ifndef NOCRLIB_LEVENSTEIN_DISTANCE_H
#define NOCRLIB_LEVENSTEIN_DISTANCE_H

#include <vector>
#include <string>
#include "dictionary.h"
#include <algorithm>

/**
 * @brief algorithm computing levenstein distance between 
 * two words 
 * minimum levenstein distance between word and
 * a dictionary.
 */
class StringLevensteinDistance
{
    public:
        StringLevensteinDistance() { }
        ~StringLevensteinDistance() { }
        int operator() ( const std::string &a, const std::string &b );
    private:
        int minimum( int a, int b, int c);

        // root of dictionary tree;
        // 

};


/**
 * @brief finds closest match do word in 
 * dictionary trie using levenstein distance.
 */
class LevensteinDistanceTrie
{
    public:
        /**
         * @brief constructor
         *
         * @param word word we search for
         * @param root root of dictionary trie
         * @param max_length length of longes word in dictionary trie
         */
        LevensteinDistanceTrie( const std::string &word, TrieNode *root, int max_length ); 

        /**
         * @brief run levenstein distance algorithm 
         * on dictionary trie
         */
        void findTranslation();

        /**
         * @brief returns results after running the algorithm 
         *
         * @return closest matches
         *
         * If we haven't run the algorithm on dictionary trie, empty output will be returned.
         */
        std::vector<std::string> getResult() const { return output_; }
        /**
         * @brief return minimum 
         *
         * @return 
         */
        int getMinLDistance() const { return min_; }
    private:
        std::string word_;
        TrieNode *root_;
        std::vector< std::vector<int> > distances_;
        int rows_,cols_;
        int min_;
        std::string tmp_;

        std::vector<std::string> output_;

        void findClosestMatch( TrieNode *node, int fill_row );
        void fillRow( int row, char c );

        int minimum( int a, int b, int c ) {  return std::min( a, std::min(b,c) ); }

};

template <typename T, typename DIST = std::size_t>
class LevensteinDistance
{
    public:
    DIST operator() (const std::vector<T> & a, const std::vector<T> & b)
    {
        const T * a_ptr = &a[0];
        const T * b_ptr = &b[0];

        std::size_t cols = a.size();
        std::size_t rows = b.size();
        if (a.size() > b.size()) 
        {
            std::swap(a_ptr, b_ptr);
            std::swap(cols, rows);
        }

        std::vector<DIST> table_row(cols);
        for (std::size_t i = 0; i < table_row.size(); ++i)
        {
            table_row[i] = i + 1;
        }

        for (std::size_t i = 0; i < rows; ++i)
        {
            DIST last_upper = i;
            DIST last_left = i + 1;
            for (std::size_t j = 0; j < table_row.size(); ++j)
            {
                DIST dist1 = std::min(last_left, table_row[j]) + 1;
                DIST diag = last_upper + (a_ptr[j] == b_ptr[i] ? 0 : 1);

                last_upper = table_row[j];
                table_row[j] = last_left = std::min(dist1, diag);
            }
        }

        return table_row.back();
    }
};

#endif /* levenstein_distance.h */


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

/**
 * @brief algorithm computing levenstein distance between 
 * two words 
 * minimum levenstein distance between word and
 * a dictionary.
 */
class LevensteinDistance
{
    public:
        LevensteinDistance() { }
        ~LevensteinDistance() { }
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

#endif /* levenstein_distance.h */


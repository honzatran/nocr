/**
 * @file dictionary.h
 * @brief Contains declaration of dictionary class.
 * @author Tran Tuan Hiep
 * @version 1.0
 * @date 2014-09-14
 */

#ifndef NOCRLIB_DICT_H
#define NOCRLIB_DICT_H

#include <string>
#include <vector>
#include <unordered_map>

#include "trie_node.h"



/**
 * @brief class for dictionary
 *
 * Class representing dictionary. 
 * Dictionary isn't copyable and copy-assignable !!
 */
class Dictionary 
{
    public:
        /**
         * @brief default constructor
         */
        Dictionary();

        /**
         * @brief constructor, that loads words from file \p dictionary_file
         *
         * @param dictionary_file path to the dictionary
         */
        Dictionary( const std::string &dictionary_file );
        /**
         * @brief forbidden copy constructor
         *
         * @param dictionary
         */
        Dictionary( const Dictionary &dictionary ) = delete;

        /**
         * @brief forbidden copy assigment constructor
         *
         * @param dictionary
         */
        Dictionary& operator=( const Dictionary &dictionary ) = delete;


        ~Dictionary() { delete root_; }

        /**
         * @brief add word to dictionary
         *
         * @param word new word 
         */
        void addWord( const std::string &word );

        /**
         * @brief loads all words in file \p dictionary_file
         *
         * @param dictionary_file path to the dictionary file
         *
         * See programming documentation for further details on 
         * dictionary file.
         */
        void loadWords( const std::string &dictionary_file );


        /**
         * @brief return root of dictionary trie 
         *
         * @return 
         */
        TrieNode* getRoot() const { return root_; }

        /**
         * @brief print all words in dictionary to stdout
         */
        void print();

        /**
         * @brief get length of longest word in dictionary
         *
         * @return maximal lenght of word in dictionary
         */
        size_t getMaxLength() const { return max_length_; }

        /**
         * @brief removes all words from dictionary
         */
        void clearDictionary();

        /**
         * @brief finds the closest translation to the \p word
         *
         * @param word 
         *
         * @return closest translation to the \p word
         *
         * Method use Levenstein distance to find the closest match in 
         * dictionary.
         */
        std::vector<std::string> findClosestTranslation(const std::string &word);

        /**
         * @brief gets all words from dictionary
         *
         * @return all words in dictionary 
         */
        std::vector<std::string> getAllWords() const;
    private:
        TrieNode *root_;
        size_t max_length_;

        typedef std::vector< std::vector<int> > vecVecInt;
        std::vector<std::string> data_;

};


#endif

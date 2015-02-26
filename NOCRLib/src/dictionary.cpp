/* 
 * Tran Tuan Hiep
 * Implementation of methods and classes declared in specified header
 *
 * Compiler: g++ 4.8.3
 */
#include "../include/nocrlib/levenstein_distance.h"
#include "../include/nocrlib/dictionary.h"
#include "../include/nocrlib/exception.h"

#include <string>
#include <algorithm>
#include <limits>
#include <fstream>
#include <iostream>
#include <cmath>


using namespace std;

Dictionary::Dictionary()
    : max_length_(0)
{
    root_ = new TrieNode();
}


Dictionary::Dictionary( const std::string &dictionary_file )
    : max_length_(0)
{
    root_ = new TrieNode();
    loadWords( dictionary_file );
}

void Dictionary::loadWords( const std::string &dictionary_file )
{
    ifstream input( dictionary_file );
    if ( !input.is_open() )
    {
        throw FileNotFoundException(dictionary_file+ " dictionary not found");
    }
    std::string s;
    while( input >> s ) 
    {
        addWord(s);
        data_.push_back(s);
        // cout << s << endl;
    }
    input.close();
}

void Dictionary::addWord( const std::string &word )
{
    TrieNode *node = root_;

    int tmp = 0;
    for ( auto it = word.rbegin(); it != word.rend(); ++it )
    {
        char c = *it;
        if ( !isdigit(c) && !isalpha(c) )
        {
            ++tmp;
            continue;
        }
        TrieNode *next_node = node->contain(c); 
        if ( next_node ) 
        {
            node = next_node;    
            continue;
        }
        node = node->addNode(c);
    }
    
    max_length_ = std::max( max_length_, word.size() - tmp );
    if ( node != root_ )
    {
        node->setEndWord(true);
    }
}

void Dictionary::clearDictionary()
{
    delete root_;
    root_ = new TrieNode();
}


void Dictionary::print() 
{
    std::string s;
    s.reserve( max_length_ );
    root_->print(s);
}

std::vector<std::string> Dictionary::findClosestTranslation( const std::string &word )
{
    // inicialization of distances matrix
    LevensteinDistanceTrie finder( word, root_, max_length_ );
    
    finder.findTranslation(); 
    return finder.getResult();
}

std::vector<std::string> Dictionary::getAllWords() const 
{
    std::string tmp = "";
    vector<string> output;
    root_->visit( tmp, output );
    return output;
}






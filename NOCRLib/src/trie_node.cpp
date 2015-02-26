/* 
 * Tran Tuan Hiep
 * Implementation of methods and classes declared in trie_node.h
 *
 * Compiler: g++ 4.8.3
 */

#include "../include/nocrlib/trie_node.h"

#include <string>
#include <map>
#include <iostream>
#include <algorithm>


using namespace std;

TrieNode::~TrieNode() 
{
    for ( auto &p: childs_ )
    {
        delete p.second;
    }
}

TrieNode* TrieNode::contain( char letter ) 
{
    auto it = childs_.find( letter );
    if ( it == childs_.end() ) 
    {
       return nullptr; 
    }
    return it->second;
}

TrieNode* TrieNode::addNode( char letter )
{
    TrieNode* newChild = new TrieNode();
    auto it = childs_.insert( std::pair<char,TrieNode*>( letter, newChild ) );
    return it.first->second; 
}

void TrieNode::print( std::string &tmp )
{
    if ( word_node_ )
    {
        std::string s = tmp;
        std::reverse(s.begin(),s.end());
        cout << s << endl;
    }

    for ( const auto &child: childs_ )
    {
        tmp.push_back( child.first );
        child.second->print( tmp );
        tmp.pop_back();
    }
}

void TrieNode::visit( std::string &tmp, std::vector<std::string> &output )
{
    if ( word_node_ )
    {
        std::string s = tmp;
        std::reverse(s.begin(),s.end());
        output.push_back(s);
    }

    for ( const auto &child: childs_ )
    {
        tmp.push_back( child.first );
        child.second->visit( tmp, output );
        tmp.pop_back();
    }
}



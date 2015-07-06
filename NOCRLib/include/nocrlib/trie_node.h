/**
 * @file trie_node.h 
 * @brief contains declaration of nodes in dictionary trie
 * @author Tran Tuan Hiep
 * @version 1.0
 * @date 2014-09-14
 */


#ifndef NOCRLIB_TRIE_NODE_H
#define NOCRLIB_TRIE_NODE_H


#include <string>
#include <vector>
#include <unordered_map>

/**
 * @brief represents node in dictionary trie
 *
 * TrieNode is node in dictionary trie, it has children and its childrens
 * representing letters.
 */
class TrieNode 
{
    public:
        /**
         * @brief constructor
         */
        TrieNode() : word_node_(false) { }
        ~TrieNode();

        /**
         * @brief finds out if nodes has any children representing with 
         * letter.
         *
         * @param letter
         *
         * @return true if there is node representing \p letter 
         */
        TrieNode* contain(char letter);

        /**
         * @brief add new node to children and its representing \p letter
         *
         * @param letter letter that represents new child
         *
         * @return pointer to the new child.
         */
        TrieNode* addNode(char letter);

        /**
         * @brief unmark or mark this node as word ending
         *
         * @param val if true then we are marking, if false then we are unmarking
         */
        void setEndWord(bool val) { word_node_ = val; }

        /// @cond
        void print(std::string &tmp);
        void visit(std::string &tmp, std::vector<std::string> &output);
        /// @endcond

        /**
         * @brief finds out if in this node ends word
         *
         * @return true if is word node else false
         */
        bool isEndWordNode()
        {
            return word_node_;
        }

        /**
         * @brief returns children
         *
         * @return unordered map of children
         * The key in unordered map is representing letter
         * and value is pointer to the TrieNode*
         */
        std::unordered_map< char, TrieNode* > getChildren() 
        //
        // std::vector< std::pair<char, TrieNode> > getChildren()
        {
            return childs_;
        }

    private:
        std::unordered_map< char, TrieNode* > childs_;
        // std::vector< std::pair<char, TrieNode> > childs_;
        bool word_node_;

        friend class LevensteinDistanceTrie;
};



#endif /* TrieNode.h */

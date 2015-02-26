/* 
 * Tran Tuan Hiep
 * Implementation of methods and classes declared in specified header
 *
 * Compiler: g++ 4.8.3, 
 */

#include "../include/nocrlib/structures.h"

#include <vector>
#include <string>
#include <unordered_map>


using namespace std;

const std::unordered_map<char, int> TranslationInfo::alpha_label_ =
{
    { '1', 1 },
    { '0', 0 },
    { '3', 3 },
    { '2', 2 },
    { '5', 5 },
    { '4', 4 },
    { '7', 7 },
    { '6', 6 },
    { '9', 9 },
    { '8', 8 },
    { 'A', 33 },
    { 'C', 12 },
    { 'B', 34 },
    { 'E', 36 },
    { 'D', 35 },
    { 'G', 38 },
    { 'F', 37 },
    { 'I', 1 },
    { 'H', 39 },
    { 'K', 40 },
    { 'J', 18 },
    { 'M', 42 },
    { 'L', 41 },
    { 'N', 43 },
    { 'Q', 44 },
    { 'P', 22 },
    { 'O', 0 },
    { 'S', 25 },
    { 'R', 45 },
    { 'U', 27 },
    { 'T', 46 },
    { 'W', 29 },
    { 'V', 28 },
    { 'Y', 47 },
    { 'X', 30 },
    { 'Z', 32 },
    { 'a', 10 },
    { 'c', 12 },
    { 'b', 11 },
    { 'e', 14 },
    { 'd', 13 },
    { 'g', 16 },
    { 'f', 15 },
    { 'i', 1 },
    { 'h', 17 },
    { 'k', 19 },
    { 'j', 18 },
    { 'm', 20 },
    { 'l', 1 },
    { 'o', 0 },
    { 'n', 21 },
    { 'q', 23 },
    { 'p', 22 },
    { 's', 25 },
    { 'r', 24 },
    { 'u', 27 },
    { 't', 26 },
    { 'w', 29 },
    { 'v', 28 },
    { 'y', 31 },
    { 'x', 30 },
    { 'z', 32 },
};


double TranslationInfo::getProbability( char c ) const
{
    auto it = alpha_label_.find(c);
    if ( it == alpha_label_.end() )
    {
        //TODO throw exception
        return 0;
    }

    return probabilities_[it->second];
}

bool TranslationInfo::haveSameLabels( char a, char b )
{
   int a_label = alpha_label_.find(a)->second; 
   int b_label = alpha_label_.find(b)->second; 

   return a_label == b_label;
}

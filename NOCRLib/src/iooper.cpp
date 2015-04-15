
/* 
 * Tran Tuan Hiep
 * Implementation of methods and classes declared in specified header
 *
 * Compiler: g++ 4.8.3 
 */

#include <vector>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/ml/ml.hpp> 
#include <opencv2/core/core.hpp>

#include "../include/nocrlib/iooper.h"

using namespace std;
using namespace cv;



auto loader::loadDataToFloatMatrix( const string &textFile )
    -> floatVecVec
{
    floatVecVec output; 
    std::ifstream in;
    in.open( textFile );
    std::string s;
    while( std::getline( in,s ) ) 
    {
        auto values = parse( s );       
        output.push_back( values );
    }
    return output;
}

vector<float> loader::parse( const string &lineWithNumbers )
{
    vector<float> output;
    stringstream buffer( lineWithNumbers );
    string tmpVal;
    while( getline( buffer, tmpVal, delim_ ) ) 
    {
        // std::cout << tmpVal << " ";
        output.push_back( std::stof(tmpVal) );
    }
    // std::cout << std::endl;

    return output;
}

// vector< fileInfo> loader::getFileList( const string &textFile ) 
// {
//     ifstream in(textFile);
//     string tmpVal;
//     vector< fileInfo > output;
//     while( getline( in,tmpVal ) ) 
//     {
//         int tmpDelimPos = tmpVal.find( delim_ );
//         string filePath = tmpVal.substr( 0, tmpDelimPos );
//         string fileLabel = tmpVal.substr( tmpDelimPos + 1, string::npos );
//         
//         output.push_back( fileInfo( filePath, fileLabel) );
//     }
//     return output;
// }

vector<string> loader::getFileContent( const std::string &file )
{
    vector<string> output; 
    std::ifstream in;
    in.open( file );
    std::string s;
    while( std::getline( in,s ) ) 
    {
        output.push_back( s );
    }
    in.close();
    return output;
}

//========================================================


void OutputWriter::write( const vector<string> &data )
{
    for ( const string &s : data )
    {
        *out_ << s << endl;
    }
}





/**
 * @file iooper.h
 * @brief utilities class for IO operations
 * @author Tran Tuan Hiep
 * @version 1.0 
 * @date 2014-09-16
 */

#ifndef NOCRLIB_IOOPER_H
#define NOCRLIB_IOOPER_H

#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <ostream>
#include <iostream>
#include <algorithm>
#include <functional>

#include <opencv2/core/core.hpp>

/// @cond
class fileInfo
{
    public:
        fileInfo( const std::string &pathToFile, const std::string &strLabelOfFile )
            : pathToFile_(pathToFile)
        {
            label_ = std::stof( strLabelOfFile );
        }

        std::string getPathToFile() const { return pathToFile_; }
        float getLabel() const { return label_; }
    private:
        std::string pathToFile_;
        float label_;
};

class loader
{
    public:
        loader(const char &delim = ':'):delim_(delim) { }
        ~loader() { }

        void setDelim( const char &delim )
        {
            delim_ = delim;
        }

        typedef std::vector< std::vector<float> > floatVecVec;
        floatVecVec loadDataToFloatMatrix(const std::string &textFile);
   
        typedef std::vector< fileInfo > fileInfoVec;
        fileInfoVec getFileList(const std::string &textFile);
        std::vector<std::string> getFileContent( const std::string &file );
    private:
        char delim_;
        //typedef std::map< std::string, std::string > mapStr;

        std::vector<float> parse( const std::string &lineWithNumbers );
};

/**
 * @brief loads data from file to vector< vector<T> >
 *
 * @tparam T specifies type, we are loading
 * @tparam C specifies convertor class, that convert std::string to T
 */
template <typename T, typename C > 
class Loader
{
    public:
        typedef std::vector<std::vector<T> > ReturnType;

        /**
         * @brief constructor
         *
         * @param delim delimeter
         * @param convertor convertor class
         */
        Loader( char delim,  const C &convertor ) 
            : delim_(delim), convertor_(convertor) { }

        /**
         * @brief set delimeter 
         *
         * @param delim new seperator
         */
        void setDelimeter( char delim ) { delim_ = delim; }

        /**
         * @brief loads data from \p file
         *
         * @param file_name
         *
         * @return vector< vector<T> > with loaded datas
         */
        ReturnType loadData( const std::string &file )
        {
            std::ifstream in;
            in.open( file );
            std::string s;
            ReturnType output;
            while( std::getline( in,s ) ) 
            {
                output.push_back( parse(s) ); 
            }
            return output;
        }


    private:
        char delim_;
        C convertor_;

        std::vector<T> parse( const std::string &line )
        {
            std::stringstream ss( line ); 
            std::string buffer; 
            std::vector<T> output;
            while( getline( ss, buffer, delim_ ) ) 
            {
                output.push_back( convertor_(buffer) ); 
            }
            return output;
        }
};


/**
 * @brief class for writing vector to outputs
 */
class OutputWriter
{
    public:
        OutputWriter() = default;

        OutputWriter( std::ostream *out, char delim = ':' ) 
            : out_(out), delim_(delim)
        { 
        }
        
        template<typename T> void write( const std::vector<T> &values, float label )
        {
            write( values );
            *out_ << delim_ << label << std::endl;
        }

        template <typename T> void write( const std::vector<T> &values )
        {
            std::for_each( values.begin(), values.end() - 1,  
                    [this] (const T &val) { *out_ << val << delim_; });
            *out_ << values.back();
        }
        
        template <typename T> void writeln( const std::vector<T> &values )
        {
            write( values );
            *out_ << std::endl;
        }

        void write( const std::vector<std::string> &data );

    private:
        std::ostream *out_;
        char delim_;
};
/// @endcond

#endif /*iooper.hpp*/


/**
 * @file utilities.h
 * @brief utility classes 
 * @author Tran Tuan Hiep
 * @version 
 * @date 2014-09-16
 */


#ifndef NOCRLIB_UTILITIES_H
#define NOCRLIB_UTILITIES_H

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <vector>
#include <iostream>
#include <cmath>
#include <sstream>
#include <string>
#include <limits>
#include <set>
#include <cassert>
#include <chrono>
#include <algorithm>
#include <iterator>
#include <fstream>
#include <type_traits>

/// @cond

using namespace std;

enum class connectivity { fourpass , eightpass };


template < connectivity C > struct Neighbourhood
{
   static std::vector< cv::Point > Neighbourhoodeigbours( const cv::Point &p ); 
};


template < > struct Neighbourhood<connectivity::fourpass>
{

   static std::vector<cv::Point> getNeighbours( const cv::Point &p ) 
   {
       std::vector<cv::Point> output = 
       {
           cv::Point( p.x - 1, p.y ),
           cv::Point( p.x , p.y + 1 ),
           cv::Point( p.x , p.y - 1 ),
           cv::Point( p.x + 1, p.y ),
       };
       return output;
   }

   static cv::Point getNeighbour( const int &index, const cv::Point &p )
   {
       switch ( index )
       {
           case 0: return cv::Point( p.x , p.y - 1 );
           case 1: return cv::Point( p.x + 1 , p.y );
           case 2: return cv::Point( p.x , p.y + 1 );
           default: return cv::Point( p.x - 1, p.y );
       };
   }

   
};

template < > struct Neighbourhood<connectivity::eightpass>
{
   static std::vector<cv::Point> getNeighbours( const cv::Point &p ) 
   {
       std::vector<cv::Point> output = 
       {
           cv::Point( p.x - 1, p.y - 1 ),
           cv::Point( p.x - 1, p.y ),
           cv::Point( p.x - 1, p.y + 1 ),
           cv::Point( p.x , p.y + 1 ),
           cv::Point( p.x , p.y - 1 ),
           cv::Point( p.x + 1, p.y - 1 ),
           cv::Point( p.x + 1, p.y ),
           cv::Point( p.x + 1, p.y + 1 ),
       };

       return output;
   }


   static cv::Point getNeighbour( int index, const cv::Point &p )
   {
       switch ( index )
       {
           case 0: return cv::Point( p.x - 1, p.y - 1 );
           case 1: return cv::Point( p.x - 1, p.y + 1 );
           case 2: return cv::Point( p.x + 1, p.y - 1 );
           case 3: return cv::Point( p.x + 1, p.y + 1 );
           case 4: return cv::Point( p.x - 1, p.y );
           case 5: return cv::Point( p.x , p.y + 1 );
           case 6: return cv::Point( p.x , p.y - 1 );
           case 7: return cv::Point( p.x + 1, p.y );
           default : return cv::Point(0,0);
       };
   }
};


struct gui 
{
    static void showImage( const cv::Mat &image, const std::string &name )
    {
        cv::namedWindow( name, CV_WINDOW_NORMAL );
        cv::imshow( name, image );
        cv::waitKey(0);
    }
};



template < typename T > class statistic
{
    private:
    public:
        static float computeMedian( std::vector<T> &data ) 
        {
            T med;
            std::sort( data.begin(), data.end() );

            if ( data.size() % 2 == 1 )
                med = data[ data.size()/2 ];
            else 
                med = ( data[ data.size()/2 - 1 ] + data[ data.size() /2 ] ) / 2;
            return med;
        }

        static float computeMean(const std::vector<T> &data)
        {
            float mean = 0;
            for ( const T &i : data )
                mean += i;

            mean = (float) mean / (float) data.size();
            return mean;
        }

        static float computeVariance ( const std::vector<T> &data )
        {
            float mean = mean( data );
            return variance( data, mean );
        }

        static float computeVariance ( const std::vector<T> &data, const float &mean )
        {
            float var = 0;
            for (const T &x : data ) 
                var += ( x - mean ) * ( x - mean );

            var /= (float) ( data.size() );
            return var;
        }
        
        static float computeStandardDeviation ( const std::vector<T> &data, const float &mean ) 
        {
            float deviantion = 0;
            for ( const T& x: data ) 
            {
                deviantion += ( x - mean ) * ( x - mean );
            }
            deviantion = std::sqrt( deviantion/(data.size() -1) );
            return deviantion;
        }

        static std::tuple<float,float,float,float> computeAll(std::vector<T> &data) 
        {
            float median = computeMedian( data );
            float mean = computeMean( data );
            float variance = computeVariance( data, mean );
            float standardDeviation = computeStandardDeviation( data,mean );
            return std::tuple<float,float,float,float>(median,mean,variance,standardDeviation);
        }

        static T median( const T &a, const T &b, const T &c )
        {
            if ( (a-b) * (c-a) >= 0 )
                return a;
            else if ( (b-a) * (c-b) >= 0 )
                return b;
            else return c;

        }

};

class helper
{
    private:
    public:
        helper() { }
        ~helper() { }

        template < typename T> static std::string convToString( const T& val ) 
        {

            std::ostringstream oss;
            oss << val;
            return oss.str();
        }

        static std::vector<bool> getAccesibilityMaskWithNegativeBorder( const cv::Mat &bitmap )
        {
            const int size = bitmap.rows * bitmap.cols;
            std::vector<bool> out( size, false ); 
            //set false first row
            for( int i = 0; i < bitmap.cols; ++i ) 
            {
                out[i] = true;
            }

            // set true first and last element in row 1 - bitmap.rows - 1
            int first = 0; int last = bitmap.cols - 1;
            for ( int i = 1; i < bitmap.rows - 1 ; ++i )
            {
                first += bitmap.cols; 
                last += bitmap.cols;
                out[first] = out[last] = true; 
            }

            // set true last row 
            for ( int i = last + 1; i < size; ++i )
            {
                out[i] = true;
            }

            return out; 
        }
};


/*
 * template < typename T, connectivity C = connectivity::eightpass > struct pomTraits
 * {
 *     typedef Point2D<T> myPoint;
 *     typedef std::vector< myPoint > vecPoint;
 *     typedef std::set< myPoint > setPoint;
 *     
 *     static vecPoint Neighbourhoodeighbours( const myPoint &p ) { return Neighbourhood<C>::Neighbourhoodeighbours(p); }
 * 
 * };
 */


/*
 * template < typename T = int > class PointTransposition
 * {
 *     public:
 *         PointTransposition(const cv::Point_<T> &transpositionPoint ):
 *                 transpositionPoint_( transpositionPoint ) { }
 *         ~PointTransposition() { }
 * 
 *         cv::Point_<T> getTransposedPoint(const cv::Point_<T> &x)  
 *         {
 *             return x + transpositionPoint_;
 *         }
 *     private:
 *         cv::Point_<T> transpositionPoint_;
 * };
 * 
 */


template <typename Clock, typename Type > class timeCounter 
{
    private:
        typedef std::chrono::time_point< Clock > timepoint;
    public:
        timeCounter( ) { };
        ~timeCounter( ) { };
        Type operator() ( const timepoint &t1, const timepoint &t2 ) 
        {
            return std::chrono::duration_cast< Type >( t2 - t1 );
        }
};


class ImageSaver
{
    public:
        ImageSaver() = default;
        ImageSaver( const std::vector<int> &compression ) 
            : compression_(compression) { }

        void saveImage( const std::string &name, const cv::Mat &image )
        {
            cv::imwrite( name, image, compression_ );
        }

    private:
        std::vector<int> compression_;

};


template <typename T>
T sum( const std::vector<T> &sum_vector )
{
    static_assert( std::is_arithmetic<T>::value, "type T must be arithmetic for summing" );
    T sum = 0;
    for ( T a: sum_vector )
    {
        sum += a;
    }
    return sum;
}


template <typename T> 
std::vector<T> linspace( T a, T b, size_t n )
{
    static_assert( std::is_floating_point<T>::value, "type T must be floating point" );
    T step = (b-a)/(n-1);
    std::vector<T> output(n);
    T val = a;
    for ( int i = 0; i < n; ++i )
    {
        output[i] = val;
        val += step;
    }
    return output;
}

template <typename T>
std::vector<T> getColumn( int j, 
        const std::vector< std::vector<T> > &matrix )
{
    std::vector<T> col( matrix.size() );
    for ( int i = 0; i < col.size(); ++i )
    {
        col[i] = matrix[i][j];
    }
    return col;
}

inline bool fileExists( const std::string &file_path )
{
    ifstream input(file_path);
    bool exists = input.is_open();
    input.close();
    return exists;
}

inline std::string getFileName(const std::string &path)
{
    unsigned pos = path.find_last_of("/\\");
    if ( pos == string::npos )
    {
        return std::string();
    }

    return path.substr( pos + 1 );
}

class Resizer
{
    public:
        Resizer() : size_(0), last_scale_(0) { }

        Resizer( int size )
            :size_(size), last_scale_(0)
        {
        }

        void setSize(int size)
        {
            size_ = size;
        }

        cv::Mat resizeKeepAspectRatio( const cv::Mat &image )
        {
            bool aspect = image.rows > image.cols;
            cv::Mat resized_image;
            if (image.rows != size_ && image.cols != size_)
            {
                if (aspect)
                {
                    last_scale_ = (double)size_/image.rows;
                    cv::resize(image, resized_image,
                            cv::Size(image.cols * last_scale_, size_));
                }
                else
                {
                    last_scale_ = (double)size_/image.cols;
                    cv::resize(image, resized_image,
                            cv::Size(size_, image.rows * last_scale_));
                }
            }

            return resized_image;
        }

        double getLastScale() const
        {
            return last_scale_;
        }

    private:
        int size_;
        double  last_scale_;
};







/// @endcond



#endif /* utilities.h */

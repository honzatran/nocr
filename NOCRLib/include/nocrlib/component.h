

/**
 * @file component.h
 * @brief contains declaration of classes representing connected component in 
 * image 
 * @author Tran Tuan Hiep
 * @version 1.0 
 * @date 2014-09-14
 */

#ifndef NOCRLIB_COMP_H
#define NOCRLIB_COMP_H

#include "utilities.h"

#include <map>
#include <set>
#include <vector>
#include <queue>
#include <iostream>
#include <tuple>
#include <string>
#include <sstream>
#include <memory>
#include <chrono>
#include <ostream>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>


/// do

class Stat; 
class Component;

// struct compTraits 
// {
//     typedef std::pair< Component, Stat > compPair;
//     typedef std::vector< Component > vecComp;
//     typedef std::vector< compPair > vecCompStats;
//     typedef std::shared_ptr< Component > ptrComp;
//     typedef std::shared_ptr< Stat > ptrStats;
//     typedef std::vector< ptrComp > vecPtrComp;
//     typedef std::vector< ptrStats > vecPtrStats;
//     typedef std::pair< vecPtrComp, vecPtrStats > pairVecPtr;
// };


/**
 * @brief represens connected component 
 */
class Component 
{
    public:
        typedef std::vector< cv::Point > VecPoint;

        /**
         * @brief constructor
         */
        Component();

        /**
         * @brief constructor
         *
         * @param point point belonging to the component
         */
        Component(const cv::Point &point)
            : upper_( point.y ), lower_( point.y ), right_( point.x ), 
            left_( point.x ), sumX_(point.x), sumY_(point.y)
        { 
            points_.push_back(point); 
        }


        /**
         * @brief constructor, initializing the instance with the list of 
         * points
         *
         * @param points list of components points
         */
        Component( const VecPoint &points ): points_(points)
        {
            init();
            for ( const auto &p :points_ ) 
            {
                updateSize(p);
            }
        }

        /**
         * @brief add point and update bounding box of components
         *
         * @param point new point
         */
        void addPoint( const cv::Point &point ); 

        /**
         * @brief add point, but doesn't update bounding box of component
         *
         * @param p
         */
        void addPointWithoutUpdatingSize( const cv::Point &p )
        {
            points_.push_back( p );
            sumX_ += p.x;
            sumY_ += p.y;
        }

        /**
         * @brief return left border 
         *
         * @return x coordinate of left border
         */
        int getLeft() const { return left_; }

        /**
         * @brief return right border 
         *
         * @return x coordinate of right border
         */
        int getRight() const { return right_; }

        /**
         * @brief return upper border 
         *
         * @return y coordinate of upper border
         */
        int getUpper() const { return upper_; }

        /**
         * @brief return lower border 
         *
         * @return y coordinate of lower border
         */
        int getLower() const { return lower_; }

        /**
         * @brief set left border
         *
         * @param left x coordinate of left border
         */
        void setLeft( int left ) 
        { 
            left_ = left; 
        } 

        /**
         * @brief set right border
         *
         * @param right x coordinate of right border
         */
        void setRight( int right ) 
        { 
            right_ = right; 
        } 

        /**
         * @brief set upper border
         *
         * @param upper y coordinate of upper border
         */
        void setUpper( int upper ) 
        { 
            upper_ = upper; 
        } 

        /**
         * @brief set lower border
         *
         * @param lower y coordinate of lower border
         */
        void setLower( int lower ) 
        { 
            lower_ = lower;
        } 

        /**
         * @brief return height of component
         *
         * @return height of component
         */
        int getHeight() const { return lower_ - upper_ + 1; }

        /**
         * @brief return width of component
         *
         * @return width of component
         */
        int getWidth() const { return right_ - left_ + 1; }

        /**
         * @brief return length of diagonal of bounding box of 
         * component
         *
         * @return length of diagonal
         */
        double getDiagonal() const;

        /**
         * @brief return centroid of component
         *
         * @return centroid of component
         */
        cv::Point2d centroid() const 
        {
            int size = points_.size();
            return cv::Point2d( (double)sumX_/size, (double)sumY_/size );
        }

        /**
         * @brief return all components points
         *
         * @return vector of components points
         */
        VecPoint getPoints() const { return points_; }

        /**
         * @brief returns number of components points
         *
         * @return number of components points
         */
        int size() const { return points_.size(); }


        /**
         * @brief returns bounding box of component
         *
         * @return bounding box of component
         */
        cv::Rect rectangle() const 
        { 
            return cv::Rect(left_, upper_, getWidth(), getHeight());
        }

        /**
         * @brief cut component from image
         *
         * @param image input image
         * @param zeroPadding if set to true cutted image will have additional border 
         * of zeros
         *
         * @return component cropped from image
         */
        cv::Mat cutComponentFromImage( const cv::Mat &image, 
                bool zeroPadding = false ) const;


        /**
         * @brief finds out if \p c is in bounding box of this component
         *
         * @param c component
         *
         * @return true if this component contain c else false
         */
        bool contain( const Component &c ) const ;

        /**
         * @brief reserve the size of component points 
         *
         * @param size size we reserve the vector of components points to
         */
        void reserve( size_t size ) 
        {
            points_.reserve(size);
        }

        /**
         * @brief return binary image of component
         *
         * @return binary image of component
         *
         * Component pixels have value 255 and background has value 0.
         */
        cv::Mat getBinaryMat()  
        {
            if ( binary_mat_.empty() ) 
            {
                cv::copyMakeBorder( createBinaryMat(255,0), binary_mat_, 1, 1, 1, 1, cv::BORDER_CONSTANT, 0);
            }
            return binary_mat_;
        }

        /**
         * @brief return minimal rotated bounding box of component
         *
         * @return minimal rotated bounding box of component
         */
        cv::RotatedRect getMinAreaRect() 
        {
            if ( min_area_rect_.size.area() == 0  ) 
            {
                min_area_rect_ = cv::minAreaRect(points_);
            }
            return min_area_rect_; 
        }


        friend std::ostream& operator<<( std::ostream &oss, const Component &c )
        {
            oss << c.left_ << ':' << c.upper_ << ':' << c.getWidth() 
                << ':' << c.getHeight();
            return oss;
        }

    private:
        VecPoint points_;
        int upper_;
        int lower_;
        int right_;
        int left_;

        int sumX_;
        int sumY_;
        cv::Mat binary_mat_;

        cv::RotatedRect min_area_rect_;

        void updateSize(const cv::Point &point);
        // cv::Point indexTransposition( const cv::Point &point ) const ;

        void init() 
        {
            upper_ = std::numeric_limits<int>::max();
            lower_ = std::numeric_limits<int>::min();
            left_ = upper_;
            right_ = lower_; 
            sumX_ = 0;
            sumY_ = 0;
        }

        cv::Mat createBinaryMat(uchar component_value, uchar background_value) const;        
        cv::Point indexTransposition( const cv::Point &point ) const;
};

std::vector<cv::Point> getPerimeterPoints
                       ( Component &c, const cv::Size &bounds );



//=====================Component finder ======================

/**
 * @brief encapsulates BFS approach to find connected components
 *
 * @tparam MergeRule rule by which we determine if two neighbouring pixel
 * can be in one component, if there is an edge between them
 * @tparam C connectivity eightpass or fourpass
 *
 * ComponentFinder use BFS to find connected components. To determine
 * if there is an edge between two neighbouring pixels in bitmap, we use class 
 * MergeRule and its method canBeMerged.
 */
template < typename MergeRule, connectivity C = connectivity::eightpass > 
class ComponentFinder 
{
    public: 
        /**
         * @brief constructor
         */
        ComponentFinder() = default;

        /**
         * @brief constructor 
         *
         * @param bitmap input with connected components
         */
        ComponentFinder( const cv::Mat &bitmap ) :bitmap_(bitmap) 
        { 
            visitedMap_ = cv::Mat( bitmap_.rows, bitmap_.cols, CV_8UC1, 1);
            rule_ = MergeRule( bitmap_ );
            bounds_ = cv::Rect( cv::Point(0,0), bitmap_.size() );
        }                                   

        /**
         * @brief constructor
         *
         * @param bitmap input with connected components
         * @param rule instance of MergeRule
         */
        ComponentFinder( const cv::Mat &bitmap, const MergeRule &rule ) 
            : rule_( rule ), bitmap_(bitmap)
        {
            visitedMap_ = cv::Mat( bitmap_.rows, bitmap_.cols, CV_8UC1, 1);
            bounds_ = cv::Rect( cv::Point(0,0), bitmap_.size() );  
        }

        /**
         * @brief finds component from point p
         *
         * @param p starting point 
         *
         * @return connected component containing pixel p 
         */
        Component findComp(const cv::Point &p);

        /**
         * @brief finds all component in bitmap loaded in constructor
         *
         * @return all components in bitmap
         */
        std::vector<Component> findAllComponents();

        /**
         * @brief return bitmap with connected components
         *
         * @return bitmap with connected components
         */
        cv::Mat bitmap() const { return bitmap_; } 

        /**
         * @brief return MergeRule instance
         *
         * @return MergeRule instance
         */
        MergeRule getMergeRule() const { return rule_; }

    private:
        typedef cv::Point point;
        typedef std::tuple<float,float,float> tupleFloat3;

        MergeRule rule_;
        cv::Mat bitmap_;
        cv::Mat visitedMap_; 
        std::queue<point> queue;
        cv::Rect bounds_;

        void control( const point &pointOfComponent, const point &p, Component &c ); 

        bool isVisited( const int &i, const int &j ) 
        {
            return visitedMap_.at<uchar>(i,j) == 0;
        }

};

template <typename MergeRule, connectivity C>
std::vector<Component> ComponentFinder<MergeRule,C>::findAllComponents()
{
    std::vector<Component> output;

    for( int i = 0; i < bitmap_.rows; ++i ) 
    {
        for( int j = 0; j < bitmap_.cols; ++j ) 
        {
            point p(j,i);
            if ( !isVisited(i,j) && rule_.isStartPointOfComponent(p) ) 
            {
                Component comp = findComp(p);
                output.emplace_back( comp );
            }
        }
    }
    return output;
}


template <typename MergeRule, connectivity C>
Component ComponentFinder<MergeRule,C>::findComp(const point &p) 
{
    visitedMap_.at<uchar>(p.y,p.x) = 0;
    Component c(p);
    queue.push(p);
    while ( !queue.empty() ) 
    {
        auto top = queue.front();
        auto neighbours = Neighbourhood<C>::getNeighbours(top); 
        for ( const auto &np : neighbours )
        {
            control( top, np, c ); 
        }

        queue.pop();    
    }
    return c;      
}

template <typename MergeRule, connectivity C>
void ComponentFinder<MergeRule,C>::control( const point &pointOfComponent, 
        const point &p, Component &c )
    // p is neighbour of pointOfComponent
{
    if ( bounds_.contains(p) ) 
    {
        if ( rule_.canBeMerged( pointOfComponent, p ) && !isVisited(p.y,p.x) ) 
        {
            c.addPoint(p);
            queue.push(p);
        }
        visitedMap_.at<uchar>(p.y,p.x) = 0;
    }
}

/// @cond
struct ComponentMergeRule
{
    cv::Mat image_;
    ComponentMergeRule() = default;
    
    ComponentMergeRule( const cv::Mat &image ) 
        : image_(image)
    {
        /*
         * TODO 
         * throw exception( bad matrix type, CV_8UC1 required )
         */
    }

    bool isStartPointOfComponent( const cv::Point &p )
    {
        return image_.at<uchar>(p.y,p.x) > 200;
    }

    bool canBeMerged( const cv::Point &comp_point, const cv::Point &p ) 
    {
        (void)(comp_point);
        return image_.at<uchar>(p.y,p.x) > 200;
    }
};
/// @endcond

//
//
//











#endif

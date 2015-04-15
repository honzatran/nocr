/**
 * @file er_region.h
 * @brief Contains declaration of class Region, that represents image information
 * about node in component tree node used with class ExtremalRegion.
 * @author Tran Tuan Hiep
 * @version 1.0
 * @date 2014-09-13
 */

#ifndef NOCRLIB_ER_REGION_H
#define NOCRLIB_ERREGION_H

#include "component.h"
#include "structures.h"

#include <algorithm>
#include <queue>
#include <cstdint>
#include <bitset>


struct LinkedPoint 
{
    LinkedPoint()
        : next_(nullptr), prev_(nullptr), val_(cv::Point(-1,-1)) { } 

    LinkedPoint(cv::Point val)
        : next_(nullptr), prev_(nullptr), val_(val) { } 

    LinkedPoint *next_;
    LinkedPoint *prev_;
    cv::Point val_;
};


/**
 * @brief Class that contain information about probability of certain 
 * node, and depth from one of his ancestor. This classed is used for
 * keeping track for local extrem of child delta-neighbourhood of some 
 * node.
 */
struct ProbabilityRecord
{
    ProbabilityRecord() : depth_(0), probability_(0) { }
    ProbabilityRecord( int depth, float probability )
        : depth_( depth ), probability_( probability ) { } 

    int depth_;
    float probability_;

    friend bool operator< ( ProbabilityRecord a, ProbabilityRecord b )
    {
        return a.probability_ < b.probability_;
    }

    friend bool operator> ( ProbabilityRecord a, ProbabilityRecord b )
    {
        return a.probability_ > b.probability_;
    }
};

// predelat bez enumu



class ERStat;

/**
 * @brief Region represents component in image. Class contain 
 * list of all components pixel, its gray level and 
 * probability. Region can also convert itself to class Component.
 * For futher details see programming documentation.
 */
class ERRegion
{
    public:
        typedef std::shared_ptr<Component> CompPtr;

        /**
         * @brief constructor
         *
         * @param grayLevel level the node has been added to the stack
         */
        ERRegion( int grayLevel = 256 );
        /**
         * @brief constructor
         *
         * @param grayLevel level the node has been added to the stack
         * @param p, initializing position of pixel 
         *
         * p isn't added to the list of points, p just initialize basic geometric 
         * properties such as position, bounding box
         */
        ERRegion( int grayLevel, cv::Point p );

        ~ERRegion();

        /**
         * @brief add point to the region
         *
         * @param point pointer to point 
         * @param horizontalCrossingChange horizontal crossing change
         * @param quads 3*3 neighbourhood of the pixel, where false if pixel hasn't been added
         * else true
         *
         * This method will update all neccessery information and adds the point 
         * to the region.
         */
        // void addPoint( LinkedPoint *point, int horizontalCrossingChange, bool *quads );
        void addPoint( LinkedPoint *point, int horizontalCrossingChange);

        /**
         * @brief merge region to this region
         *
         * @param other region to merge with
         *
         * This method will update all information after merging.
         */
        void merge( ERRegion &other );

        /**
         * @brief return level of region
         *
         * @return level of region
         */
        int getLevel() const { return grayLevel_; }
        /**
         * @brief return area of region
         *
         * @return area of region
         */
        int getSize() const { return size_; } 
        /**
         * @brief return width of region
         *
         * @return width of region
         */
        int getWidth() const { return x_max_ - x_min_ +1; }

        /**
         * @brief return height of region
         *
         * @return height of region
         */
        int getHeight() const { return y_max_ - y_min_ +1; }

        /**
         * @brief return bounding box of region
         *
         * @return bounding box of region
         */
        cv::Rect getRectangle() const 
        { 
            return cv::Rect( x_min_, y_min_, getWidth(), getHeight() );
        }

        /**
         * @brief set probability of region being letter candidate
         *
         * @param probability new probability
         */
        void setProbability( float probability );

        /**
         * @brief return probability of region being letter candaidate
         *
         * @return probability of region being letter candaidate
         */
        float getProbability() const 
        {
            return probability_;
        }

        /**
         * @brief convert region to component
         *
         * @return converted component
         */
        Component toComponent() const;

        /**
         * @brief convert region to component
         *
         * @return shared pointer to converted component
         *
         * This function will change the content of region instance.
         */
        CompPtr toCompPtr(); 

        /**
         * @brief set median crossing of region, futher details
         * in programming documentation
         */
        void setMedianCrossing(); 

        /**
         * @brief return features for first stage of er filtering 
         *
         * @return vector of features
         */
        std::vector<float> getFeatures() const;

        /**
         * @brief returns minimum probability from childs delta-neighbourhood 
         *
         * @return ProbabilityRecord, the probability and depth from this node 
         */
        ProbabilityRecord getChildMinProbability() { return minProb_; }

        /**
         * @brief returns maximum probability from childs delta-neighbourhood 
         *
         * @return ProbabilityRecord, the probability and depth from this node 
         */
        ProbabilityRecord getChildMaxProbability() { return maxProb_; }

        /**
         * @brief set child max probability
         *
         * @param maxProb, max probability
         */
        void setChildMaxProbability( ProbabilityRecord maxProb )
        {
            maxProb_ = maxProb;
        }
        
        /**
         * @brief set child min probability
         *
         * @param minProb, max probability
         */
        void setChildMinProbability( ProbabilityRecord minProb )
        {
            minProb_ = minProb;
        }

        /**
         * @brief return true if this node is local maximum in 
         * childs delta-neighbourhood, 
         *
         * @return 
         */
        bool isLocalChildMaximum() const 
        {
            return maxProb_.depth_ == 0;
        }

        /**
         * @brief return true if region is similar to reg
         *
         * @param reg, other region
         *
         * @return 
         */
        bool isSimilarParent( const ERRegion &reg );
        
        /**
         * @brief updates means of color
         *
         * @param blue 
         * @param red
         * @param green
         * @param gray
         */
        void updateMeans( uchar blue, uchar red, uchar green, uchar gray );

        /**
         * @brief updates means of color
         *
         * @param new_values
         */
        void updateMeans( const cv::Vec4b &new_values ); 

        void updateEulerBit(std::uint16_t quads);

        /**
         * @brief create ERStat based on means
         *
         * @return ERStat
         */
        ERStat createERStat() const;

        int hole_area_;
    private:
        // private classes
        struct EulerQuadRecord
        {
            EulerQuadRecord() 
                : q1Count_(0), q2Count_(0), q2DCount_(0), q3Count_(0) 
            { 
            }
            
            const static float c;

            static const int start_zero[];
            static const int start_one[];
            static const int start_three[];
            static const int start_four[];

            void update( bool *q );
            // void changeAtQuad( int start, int elemToChange, bool *quad );
            void changeAtQuad(const int *indices, int elemToChange, bool *quad );

            int getEulerNumber() const 
            {
                return ( q1Count_ - q3Count_ + 2 * q2DCount_ )/4;
            }

            float getPerimeterLength() const 
            {
                // return ( q1Count_ + q2Count_ + q3Count_ + 2*q2DCount_ );
                return q2Count_ + c * ( q1Count_ + 2 * q2DCount_ + q3Count_ );
            }

            void merge( const EulerQuadRecord &a )
            {
                q1Count_ += a.q1Count_;
                q2Count_ += a.q2Count_;
                q2DCount_ += a.q2DCount_;
                q3Count_ += a.q3Count_;
            }

            int q1Count_, q2Count_, q2DCount_, q3Count_;
        };

        struct EulerQuadRecordBit
        {
            EulerQuadRecordBit() 
                : q1_count(0), q2_count(0), q2d_count(0), q3_count(0) 
            { 
            }

            int q1_count, q2_count, q2d_count, q3_count;
            const static double k_c;

            void update(std::uint16_t quads);

            int getEulerNumber() const 
            {
                return ( q1_count - q3_count + 2 * q2d_count )/4;
            }

            float getPerimeterLength() const 
            {
                return q2_count + k_c * ( q1_count + 2 * q2d_count + q3_count );
            }

            int getBitCount(std::uint16_t k)
            {
                int c = 0;
                for (; k; ++c)
                {
                    k &= k - 1;
                }
                
                return c;
            }

            void merge( const EulerQuadRecordBit &a )
            {
                q1_count += a.q1_count;
                q2_count += a.q2_count;
                q2d_count += a.q2d_count;
                q3_count += a.q3_count;
            }
        };

        class HorizontalCrossingTracker
        {
            public:
                HorizontalCrossingTracker() : HorizontalCrossingTracker(-1) { }

                HorizontalCrossingTracker( int row ) : y_max_(row), y_min_(row) 
                { 
                    crossings_.push_back(0); 
                }
                
                ~HorizontalCrossingTracker() { }

                void updateHorizontalCrossing( int y, int change );
                void merge( const HorizontalCrossingTracker &other );
                int getMedian() const;

                int getSize() const { return crossings_.size(); }
                void swap( HorizontalCrossingTracker & other );
            private:
                int y_max_;
                int y_min_;

                std::deque<int> crossings_;
        };


        struct PerimeterLengthTracker
        {
            static const int quad_indices[];

            PerimeterLengthTracker()
            {
                val_ = 0;
            }
                
            int val_;

            void updateChange(bool *quad)
            {
                int change = 0;
                for ( int i = 0; i < 4; ++i )
                {
                    if (quad[quad_indices[i]])
                    {
                        change++;
                    }
                }
                val_ += 4 - 2 * change;
            }

            void merge(const PerimeterLengthTracker &other)
            {
                val_ += other.val_;
            }
        };
        /// @endcond

        int grayLevel_;

        int x_max_;
        int x_min_;
        int y_max_;
        int y_min_;

        CompPtr c_ptr_;

        // EulerQuadRecord rec_;
        EulerQuadRecordBit bit_rec_;
        HorizontalCrossingTracker horizontal_crossings_;

        int med_crossing;
        float probability_;

        size_t size_;
        LinkedPoint *head_;
        LinkedPoint *tail_;

        cv::Vec4i sums_;

        ProbabilityRecord minProb_;
        ProbabilityRecord maxProb_;

        void init();
        void updateSize( cv::Point p );

        friend class ERFilter1Stage;
        friend class ERFilter2Stage;
};

/**
 * @brief class containing information about color means of 
 * certain components
 *
 * This class is used as a storage for information computed in ER extraction, 
 * but used in generating words for letter equivalence.
 */
class ERStat
{
    public:
        ERStat() = default;

        ERStat( float blue_mean, float green_mean, 
                float red_mean, float intensity_mean )
            : blue_mean_(blue_mean), green_mean_(green_mean),
            red_mean_(red_mean), intensity_mean_(intensity_mean )
        {

        }
        
        ~ERStat() { }

        float blue_mean_;
        float green_mean_;
        float red_mean_;
        float intensity_mean_;

    private:
};




#endif /* ERRegion.h */


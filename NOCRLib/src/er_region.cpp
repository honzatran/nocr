/* 
 * Tran Tuan Hiep
 * Implementation of methods and classes declared in er_region.h
 *
 * Compiler: g++ 4.8.3
 */
#include "../include/nocrlib/er_region.h"
#include "../include/nocrlib/features.h"

#include <vector>
#include <bitset>
#include <cassert>

#include <opencv2/core/core.hpp>

#define _DEBUG 0


using namespace std;
using namespace cv;

const double ERRegion::EulerQuadRecordBit::k_c = 1/sqrt(2);
const int ERRegion::PerimeterLengthTracker::quad_indices[] = { 1,3,4,6 };


void ERRegion::EulerQuadRecordBit::update(std::uint16_t quads)
{

#if _DEBUG
    std::uint16_t quad0 = quads & 0xF;
    std::uint16_t quad1 = (quads & 0xF0) >> 4;
    std::uint16_t quad2 = (quads & 0xF00) >> 8;
    std::uint16_t quad3 = (quads & 0xF000) >> 12;

    switch(getBitCount(quad0))
    {
        case 0:
            ++q1_count;
            break;
        case 1:
            --q1_count;
            if (quad0 != 1)
                ++q2_count;
            else
                ++q2d_count;
            break;
        case 2:
            if(quad0 != 6)
                --q2_count;
            else
                --q2d_count;

            ++q3_count;
            break;
        default:
            --q3_count;
            break;
    }

    switch (getBitCount(quad1))
    {
        case 0:
            ++q1_count;
            break;
        case 1:
            --q1_count;
            if (quad1 != 2)
                ++q2_count;
            else
                ++q2d_count;
            break;
        case 2:
            if(quad1 != 9)
                --q2_count;
            else
                --q2d_count;

            ++q3_count;
            break;
        default:
            --q3_count;
            break;
    }
    
    switch(getBitCount(quad2))
    {
        case 0:
            ++q1_count;
            break;
        case 1:
            --q1_count;
            if (quad2 != 4)
                ++q2_count;
            else
                ++q2d_count;
            break;
        case 2:
            if(quad2 != 9)
                --q2_count;
            else
                --q2d_count;

            ++q3_count;
            break;
        default:
            --q3_count;
            break;
    }

    switch(getBitCount(quad3))
    {
        case 0:
            ++q1_count;
            break;
        case 1:
            --q1_count;
            if (quad3 != 8)
                ++q2_count;
            else
                ++q2d_count;
            break;
        case 2:
            if(quad3 != 6)
                --q2_count;
            else
                --q2d_count;

            ++q3_count;
            break;
        default:
            --q3_count;
            break;
    }

#else
    std::bitset<4> quad0 = quads & 0xF;
    switch(quad0.count())
    {
        case 0:
            ++q1_count;
            break;
        case 1:
            --q1_count;
            if (quad0 != 1)
                ++q2_count;
            else
                ++q2d_count;
            break;
        case 2:
            if(quad0 != 6)
                --q2_count;
            else
                --q2d_count;

            ++q3_count;
            break;
        default:
            --q3_count;
            break;
    }

    std::bitset<4> quad1 = (quads & 0xF0) >> 4;

    switch(quad1.count())
    {
        case 0:
            ++q1_count;
            break;
        case 1:
            --q1_count;
            if (quad1 != 2)
                ++q2_count;
            else
                ++q2d_count;
            break;
        case 2:
            if(quad1 != 9)
                --q2_count;
            else
                --q2d_count;

            ++q3_count;
            break;
        default:
            --q3_count;
            break;
    }

    std::bitset<4> quad2 = (quads & 0xF00) >> 8;

    switch(quad2.count())
    {
        case 0:
            ++q1_count;
            break;
        case 1:
            --q1_count;
            if (quad2 != 4)
                ++q2_count;
            else
                ++q2d_count;
            break;
        case 2:
            if(quad2 != 9)
                --q2_count;
            else
                --q2d_count;

            ++q3_count;
            break;
        default:
            --q3_count;
            break;
    }

    std::bitset<4> quad3 = (quads  & 0xF000) >> 12;
    switch(quad3.count())
    {
        case 0:
            ++q1_count;
            break;
        case 1:
            --q1_count;
            if (quad3 != 8)
                ++q2_count;
            else
                ++q2d_count;
            break;
        case 2:
            if(quad3 != 6)
                --q2_count;
            else
                --q2d_count;

            ++q3_count;
            break;
        default:
            --q3_count;
            break;
    }
#endif
}




void ERRegion::HorizontalCrossingTracker::
        updateHorizontalCrossing( int y, int change )
{
    int actualChange = 2 - 2* change;
    if ( y < y_min_ ) 
    {
        crossings_.push_front(2); 
#if PRINT_INFO
        if (y != y_min_ - 1)
        {
            cout << y << " " << y_min_ << endl;
        }
#endif
        --y_min_;
        return;
    }

    if ( y > y_max_ )
    {
        crossings_.push_back(2);
#if PRINT_INFO
        if (y != y_max_ + 1)
        {
            cout << y << " " << y_max_ << endl;
        }
#endif
        ++y_max_;
        return;
    }

    crossings_[y - y_min_ ] += actualChange;
}


/**
 * @brief 
 *
 * @param other
 */
void ERRegion::HorizontalCrossingTracker::
            merge( const HorizontalCrossingTracker &other )
{
    for ( int i = std::max( y_min_, other.y_min_ ); i <= std::min( y_max_, other.y_max_); ++i )
    {
        crossings_[  i-y_min_ ] += other.crossings_[ i-other.y_min_ ];
    }
    // add crossings not included in parent 
    int pom = y_min_ - 1;
    if ( y_min_ > other.y_max_ ) 
    {
        while( (unsigned)( pom- other.y_min_ ) >= other.crossings_.size() )
        {
            crossings_.push_front(0);
            --pom;
        }
    }

    for ( int i = pom; i >= other.y_min_; --i )
    {
        int tmp = other.crossings_[  i - other.y_min_ ];
        crossings_.push_front( tmp ); 
    }

    // add crossing after y_max_
    pom = y_max_ + 1;
    if ( y_max_ < other.y_min_ )
    {
        while( ( pom - other.y_min_ ) < 0 )
        {
            crossings_.push_back(0);
            ++pom;
        }
    }

    for ( int i = pom; i <= other.y_max_; ++i )
    {
        int tmp = other.crossings_[ i - other.y_min_ ];
        crossings_.push_back( tmp );
    }

    y_max_ = std::max( y_max_, other.y_max_ );
    y_min_ = std::min( y_min_, other.y_min_ );
}


/**
 * @brief 
 *
 * @return 
 */
int ERRegion::HorizontalCrossingTracker::getMedian() const
{
    int height = y_max_ - y_min_ + 1;
    int a = crossings_[height/6];
    int b = crossings_[height/2];
    int c = crossings_[height*5/6];
    if ( (a-b)*(c-a) >= 0 )
        return a;
    else if ( (b-a)*(c-b) >= 0 )
        return b;
    else 
        return c;
}

void ERRegion::HorizontalCrossingTracker::swap( HorizontalCrossingTracker &other )
{
    std::swap( y_min_, other.y_min_ );
    std::swap( y_max_, other.y_max_ );
    crossings_.swap( other.crossings_ );
}

ERRegion::ERRegion( int grayLevel ) :
    // parent_(nullptr), child_(nullptr), next_(nullptr), 
    // prev_(nullptr), depth_from_parent_(0), last_child_(nullptr),
    grayLevel_( grayLevel ), size_(0), 
    head_(nullptr), tail_(nullptr)
{
    init();
} 


ERRegion::ERRegion( int grayLevel, cv::Point p ) :
    // parent_(nullptr), child_(nullptr), next_(nullptr), 
    // prev_(nullptr), depth_from_parent_(0), last_child_( nullptr ),
    grayLevel_( grayLevel ), size_(0), 
    head_(nullptr), tail_(nullptr)
{
    x_max_ = p.x;
    x_min_ = p.x;
    y_max_ = p.y;
    y_min_ = p.y;
    // horizontalCrossing_ = std::deque<int>(1,0);
    probability_ = 0;
    // parent_ = 0;
    horizontal_crossings_ = HorizontalCrossingTracker(p.y);
    // ...
    c_ptr_ = nullptr;
} 


ERRegion::~ERRegion() 
{
    /*
     * parent_ = nullptr;
     * child_ = nullptr;
     * next_ = nullptr;
     * prev_ = nullptr;
     * last_child_ = nullptr;
     */
    head_ = nullptr;
    tail_ = nullptr;
}



// void ERRegion::addPoint( LinkedPoint *point, int horizontalCrossingChange, bool *quad )
void ERRegion::addPoint( LinkedPoint *point, int horizontalCrossingChange)
{
    if ( size_ > 0 ) 
    {
        point->prev_ = tail_;
        tail_->next_ = point;
        point->next_ = nullptr; 
    }
    else 
    {
        point->prev_ = nullptr;
        point->next_ = nullptr;
        head_ = point;
    }
    // std::cout << point->val_ << std::endl;

    
    tail_ = point;
    ++size_;
    // rec_.update( quad ); 
    horizontal_crossings_.updateHorizontalCrossing( point->val_.y, horizontalCrossingChange );
    // perim_.updateChange( quad );
    updateSize( point->val_ );
}

void ERRegion::setProbability( float probability )
{
    probability_ = probability;
}

void ERRegion::updateSize(cv::Point p)
{
    if ( x_max_ < p.x ) x_max_ = p.x;
    else if ( x_min_ > p.x ) x_min_ = p.x;

    if ( y_max_ < p.y ) y_max_ = p.y;
    else if ( y_min_ > p.y ) y_min_ = p.y;
}

bool ERRegion::isSimilarParent( const ERRegion &reg )
{
    int parent_size = reg.size_;
    const double similarity_ratio = 0.95;
    const double min_diff = 10;

    if (parent_size - size_ < min_diff)
    {
        return true;
    }

    return (float) size_/parent_size > similarity_ratio;
}


void ERRegion::updateEulerBit(std::uint16_t quads)
{
    bit_rec_.update(quads);
}

void ERRegion::merge( ERRegion &child )
{
    if ( child.size_ > 0 && size_ > 0 )
    {
        tail_->next_ = child.head_;
        child.head_->prev_ = tail_;

        tail_ = child.tail_;
    }

    if ( size_ == 0 )
    {
        head_ = child.head_;
        tail_ = child.tail_;
    }

    // horizontal_crossings_.merge( child.horizontal_crossings_ );
    if ( horizontal_crossings_.getSize() >= child.horizontal_crossings_.getSize() )
    {
        horizontal_crossings_.merge( child.horizontal_crossings_ );
    }
    else
    {
        child.horizontal_crossings_.merge( horizontal_crossings_ );
        horizontal_crossings_.swap( child.horizontal_crossings_ );
    }

    // update size
    x_max_ = std::max( x_max_, child.x_max_ );
    x_min_ = std::min( x_min_, child.x_min_ );

    y_max_ = std::max( y_max_, child.y_max_ );
    y_min_ = std::min( y_min_, child.y_min_ );

    size_ += child.size_;
    // rec_.merge( child.rec_ );
    bit_rec_.merge( child.bit_rec_);
    // perim_.merge( child.perim_ );
}

void ERRegion::setMedianCrossing() 
{
    med_crossing = horizontal_crossings_.getMedian();
}

vector<float> ERRegion::getFeatures() const
{
    float aspect_ratio = (float) getWidth()/getHeight();
    float compactness = (float) (std::sqrt( size_ ))/bit_rec_.getPerimeterLength();

    return { aspect_ratio, compactness, 
        (float)1 - bit_rec_.getEulerNumber(), (float)med_crossing };
}

Component ERRegion::toComponent() const
{
    Component out;
    out.reserve( size_ );
    cv::Point offset( 1, 1);
    
    for ( LinkedPoint* p = head_; p != tail_->next_ ; p = p->next_ ) 
    {
        out.addPointWithoutUpdatingSize( p->val_  - offset ); 
    }

    out.setLeft( x_min_ - 1 );
    out.setRight( x_max_ - 1 );
    out.setUpper( y_min_ - 1);
    out.setLower( y_max_ -1 );

    return out;
}

auto ERRegion::toCompPtr() 
    -> CompPtr
{
    if ( c_ptr_ == nullptr )
    {
        c_ptr_ = make_shared<Component>( toComponent() );
    }

    return c_ptr_; 
}

void ERRegion::init()
{
    x_max_ = std::numeric_limits<int>::min();
    x_min_ = std::numeric_limits<int>::max();
    y_max_ = x_max_;
    y_min_ = x_min_;
    probability_ = 0;
}


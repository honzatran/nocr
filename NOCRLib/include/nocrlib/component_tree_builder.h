/**
 * @file component_tree_builder.h
 * @brief Contains algorithm proposed by Nisterius and co 
 * for computing component tree builder from domain bitmap.
 * See programming documentation for details.
 *
 * @author Tran Tuan Hiep
 * @version 1.0
 * @date 2014-09-11
 */


#ifndef NOCRLIB_COMPONENT_TREE_BUILDER_H
#define NOCRLIB_COMPONENT_TREE_BUILDER_H

#include <stack>
#include <opencv2/core/core.hpp>

/**
 * @brief policy for algorithm to build the component tree
 * with class ComponentTreeBuilder<E>
 *
 * @tparam E type of class, that implements structural steps of
 * building the tree
 */
template < typename E > 
struct ComponentTreePolicy;

/**
 * @brief ComponentTreeBuilder encapsulates algorithm proposed by
 * Nisterius and co. for building component tree
 *
 * @tparam E type of class, that implements structural steps of
 * building the tree and from which we get the domain bitmap.
 *
 * Class E will take care of structural steps of building tree such
 * as connecting children node to the parent node. It also 
 * provide domain Image for building the component tree. Instead of 
 * passing the parameter. For further details see the programming 
 * documentation. ComponentTreeBuilder only specifies the algorithm, or plan 
 * of building the tree.
 */
template < typename E > 
class ComponentTreeBuilder
{
    public:
        /**
         * @brief loads class E, that will take care of structural part
         * of building the component tree 
         *
         * @param extraction raw pointer to the instance of class Extraction
         */
        ComponentTreeBuilder( E *extraction )
            : extraction_( extraction )
        {

        }

        ~ComponentTreeBuilder()
        {
            while( !stack_.empty() )
            {
                NodeType *top = stack_.top();
                stack_.pop();
                delete top;
            }
        }

        /**
         * @brief method builds component tree 
         *
         * @return root node builded component tree
         *
         * Method builds component tree. The input image is passed through class E.
         * For further details see the programming documentation.
         * Output is root of component tree. ComponentTreeBuilder is not resposible 
         * for deallocating the ComponentTree, that is an user responsibility.
         */
        typename ComponentTreePolicy<E>::NodeType* buildTree()
        // void buildTree()
        {
            //assert( image je v GrayScale)
            //

            // step 1 - 2 of algorithm
            // initialization
            cv::Mat domain_bitmap = extraction_->getDomain();
            int init_pixel;
            ComponentTreePolicy<E>::init( domain_bitmap, accessiblePixels_, stack_, &init_pixel );
            const uchar* image_data = domain_bitmap.data;
            cols_ = domain_bitmap.cols; 
            rows_ = domain_bitmap.rows;
            currPixel_ = PixelRecord(init_pixel,0); 
            currLevel_ = image_data[init_pixel];

            for(;;)
            {
                pushRegion( currLevel_, currPixel_.getCodePosition() ); // step 3 

                // step 4
                for (;; )
                {
                    bool goToStep3Required = false;

                    for ( int i = currPixel_.getEdgeToDiscover(); i < 4 ; ++i )
                    {
                        int ncode = getNeighbour( currPixel_ );
                        if ( accessiblePixels_[ncode] )
                        {
                            currPixel_.setEdgeToDiscover( i + 1 );
                            continue; 
                            // was reached continue
                        }

                        accessiblePixels_[ncode] = true;
                        int nlevel = image_data[ncode];

                        PixelRecord pixInfo(ncode, 0); 
                        currPixel_.setEdgeToDiscover( i + 1); 
                        if ( nlevel >= currLevel_ )
                        {
                            boundaryPixels_.push( pixInfo,nlevel );
                        }
                        else 
                        {
                            // go down in stream
                            goToStep3Required = true;
                            boundaryPixels_.push( currPixel_, currLevel_ );
                            currPixel_ = pixInfo;
                            currLevel_ = nlevel; 
                            break;
                        }
                    }

                    // use this instead of go_to
                    if ( goToStep3Required )
                    {
                        break;
                    }

                    // add current pixel to component on stack
                    extraction_->accumulate( stack_.top(), currPixel_.getCodePosition() );
                    if ( boundaryPixels_.empty() )
                    {
                        // we are done
                        NodeType* root = stack_.top();
                        stack_.pop();
                        ComponentTreePolicy<E>::setRoot(root, extraction_);

                        return root;
                    }

                    // pop pixel from heap and then get new top of the heap
                    setNewCurrent();   
                    int newCurrLevel = image_data[currPixel_.getCodePosition()];

                    if ( newCurrLevel > currLevel_ )
                    {
                        // go up filling the basin
                        currLevel_ = newCurrLevel;
                        processStackRoutine();
                    }
                }
            }
        }

    private:
        // private class declaration ===============================
        class PixelRecord 
        {
            public:
                PixelRecord()
                    :codePosition_(0), edgeToDiscover_(0)
                {
                    
                }

                PixelRecord( const int &codePosition, const int &edgeToDiscover = 0 )
                    : codePosition_( codePosition ), edgeToDiscover_(edgeToDiscover)
                {
                }
                

                int getCodePosition() const 
                {
                    return codePosition_;
                }

                int getEdgeToDiscover() const 
                {
                    return edgeToDiscover_;
                }

                void setEdgeToDiscover(const int newEdgeToDiscover)
                {
                    edgeToDiscover_ = newEdgeToDiscover;
                }

                bool isExplored() const  
                {
                    return edgeToDiscover_ > 7;
                }


            private:
                int codePosition_;
                int edgeToDiscover_;
        };


        template <typename T> class BitmapHeap
        {
            public:
                BitmapHeap() 
                    // :priority_(256)
                {
                    heap_.clear();
                    heap_.resize(256);
                }

                void setUp(const cv::Mat &image)
                {
                    std::vector<int> values_occurence = getOccurences(image);
                    heap_.resize(256);
                    for ( int i = 0; i < 256; ++i )
                    {
                        heap_[i].reserve( values_occurence[i] );
                    }
                }

                ~BitmapHeap() { }
                const T& top() const 
                {
                    // return heap_[priority_].back();
                    int priority = -priority_heap_.top();
                    return heap_[priority].back();
                }

                bool empty() const
                {
                    // return priority_ == 256; 
                    return priority_heap_.empty();
                }

                void pop() 
                {
                    int priority = -priority_heap_.top();
                    // heap_[hep].pop_back();
                    //
                    // while (  priority_ < 256 && heap_[priority_].empty()  ) 
                    // {
                    //     ++priority_;
                    // }
                    heap_[priority].pop_back();
                    if ( heap_[priority].empty() )
                    {
                        priority_heap_.pop();
                    }
                }

                void push( const T &value, int priority )
                {
                    // heap_[priority].push_back(value);
                    // priority_ = std::min( priority_, priority );
                    bool empty = heap_[priority].empty();
                    heap_[priority].push_back(value);
                    if ( empty )
                    {
                        priority_heap_.push( -priority );
                    }
                }

            private:
                // int priority_;
                std::vector< std::vector< T > > heap_;
                std::priority_queue<int> priority_heap_;
                // std::vector< std::vector<T> > heap_[256];
                
                std::vector<int> getOccurences( const cv::Mat &image )
                {
                    std::vector<int> occurences(256,0);
                    std::for_each( image.begin<uchar>(), image.end<uchar>(), 
                            [&occurences] ( int i ) 
                            {
                                occurences[i] += 1;
                            });
                    return occurences;
                }
        };

        //=================== private class members =====================================

        E *extraction_;

        typedef typename ComponentTreePolicy<E>::NodeType NodeType;

        std::vector<bool> accessiblePixels_;
        BitmapHeap<PixelRecord> boundaryPixels_; 
        int currLevel_;
        PixelRecord currPixel_;
        std::stack< NodeType* > stack_;

        int rows_, cols_;

        int getNeighbour( const PixelRecord &rec )
        {
            int pos = rec.getCodePosition();
            switch( rec.getEdgeToDiscover() )
            {
                case 0: return pos - cols_; 
                case 1: return pos - 1; 
                case 2: return pos + 1; 
                default: return pos + cols_; 
            }
        }

        void setNewCurrent()
        {
            currPixel_ = boundaryPixels_.top();
            boundaryPixels_.pop();
        }

        void pushRegion( int level, int pixel_code )
        {
            cv::Point pixel = getPoint(pixel_code);
            NodeType* node = ComponentTreePolicy<E>::createNode( level, pixel );
            stack_.push( node );
        }

        cv::Point getPoint( int pixel_code )
        {
            return cv::Point( pixel_code % cols_, pixel_code / cols_); 
        }

        void processStackRoutine()
        {
            int top_level;
            do 
            {
                NodeType *child = stack_.top(); 
                stack_.pop();
                top_level = ComponentTreePolicy<E>::getLevel( stack_.top() ); 
                if ( currLevel_ < top_level ) 
                {
                    pushRegion( currLevel_, currPixel_.getCodePosition() );
                    extraction_->merge( child, stack_.top() ); 
                    return;
                }

                extraction_->merge( child, stack_.top() ); 
            }
            while( currLevel_ > top_level ); 
        }

};




#endif /* component_tree_builder.h */


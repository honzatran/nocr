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
#include <queue>

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
            // step 1 - 2 of algorithm
            // initialization
            cv::Mat domain_bitmap = extraction_->getDomain();
            int init_pixel;
            ComponentTreePolicy<E>::init( domain_bitmap, accessible_pixels_, stack_, &init_pixel );
            const uchar* image_data = domain_bitmap.data;
            cols_ = domain_bitmap.cols; 
            rows_ = domain_bitmap.rows;
            curr_pixel_ = { init_pixel,0 };
            curr_level_ = image_data[init_pixel];

            for(;;)
            {
                pushRegion( curr_level_, curr_pixel_.code_position ); // step 3 
                // step 4
                for (;; )
                {
                    bool goToStep3Required = false;

                    for ( int i = curr_pixel_.edge_to_discover; i < 4 ; ++i )
                    {
                        int ncode = getNeighbour( curr_pixel_ );
                        if ( accessible_pixels_[ncode] )
                        {
                            ++curr_pixel_.edge_to_discover;
                            continue; 
                            // was reached continue
                        }

                        accessible_pixels_[ncode] = true;
                        int nlevel = image_data[ncode];

                        PixelRecord pixInfo = { ncode, 0 }; 
                        ++curr_pixel_.edge_to_discover;
                        if ( nlevel >= curr_level_ )
                        {
                            boundary_pixels_.push( pixInfo, nlevel );
                        }
                        else 
                        {
                            // go down in stream
                            goToStep3Required = true;
                            boundary_pixels_.push( curr_pixel_, curr_level_ );
                            curr_pixel_ = pixInfo;
                            curr_level_ = nlevel; 
                            break;
                        }
                    }

                    // use this instead of go_to
                    if ( goToStep3Required )
                    {
                        break;
                    }

                    // add current pixel to component on stack
                    extraction_->accumulate( stack_.top(), curr_pixel_.code_position);
                    if ( boundary_pixels_.empty() )
                    {
                        // we are done
                        NodeType* root = stack_.top();
                        stack_.pop();
                        ComponentTreePolicy<E>::setRoot(root, extraction_);
                        return root;
                    }

                    // pop pixel from heap and then get new top of the heap
                    setNewCurrent();   
                    int newCurrLevel = image_data[curr_pixel_.code_position];

                    if ( newCurrLevel > curr_level_ )
                    {
                        // go up filling the basin
                        curr_level_ = newCurrLevel;
                        processStackRoutine();
                    }
                }
            }
        }

    private:
        // private class declaration ===============================
        struct PixelRecord 
        {
            int code_position;
            int edge_to_discover;
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

        std::vector<bool> accessible_pixels_;
        BitmapHeap<PixelRecord> boundary_pixels_; 
        int curr_level_;
        PixelRecord curr_pixel_;
        std::stack< NodeType* > stack_;

        int rows_, cols_;

        int getNeighbour( const PixelRecord &rec )
        {
            int pos = rec.code_position;
            switch( rec.edge_to_discover )
            {
                case 0: return pos - cols_; 
                case 1: return pos - 1; 
                case 2: return pos + 1; 
                default: return pos + cols_; 
            }
        }

        void setNewCurrent()
        {
            curr_pixel_ = boundary_pixels_.top();
            boundary_pixels_.pop();
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
                if ( curr_level_ < top_level ) 
                {
                    pushRegion( curr_level_, curr_pixel_.code_position );
                    extraction_->merge( child, stack_.top() ); 
                    return;
                }

                extraction_->merge( child, stack_.top() ); 
            }
            while( curr_level_ > top_level ); 
        }

};

#endif /* component_tree_builder.h */


/**
 * @file extremal_region.h
 * @brief File contains declaration of classes, that takes care of
 * structural operation of building the component tree, extraction
 * ER from component tree and specialized ComponentTreePolicy for
 * integration of classes ExtramalRegion and ComponentTreeBuilder.
 * @author Tran Tuan Hiep
 * @version 1.0
 * @date 2014-09-13
 */

#ifndef NOCRLIB_EXTREMAL_REGION_H
#define NOCRLIB_EXTREMAL_REGION_H

#include "er_region.h"
#include "component_tree_builder.h"
#include "component_tree_node.h"
#include "component.h"
#include "classifier_wrap.h"
#include "feature_traits.h"
#include "utilities.h"

#include <opencv2/core/core.hpp>

#include <boost/pool/pool.hpp>
#include <boost/pool/object_pool.hpp>
#include <boost/pool/pool_alloc.hpp>

#include <vector>
#include <memory>

/**
 * @brief Interface for function used in first phase of ER extraction
 * in component tree to evalueste P( r | letter )
 */
class ERFunctionInterface
{
    public:
        virtual ~ERFunctionInterface() { }

        virtual void setImage( const cv::Mat &image ) { UNUSED(image); }
        virtual float getProbability( const ERRegion &r ) = 0;
};


///@cond
template <>
struct FeatureTraits<feature::ERGeom> 
{
    static const int features_length = 4;
    typedef ERFilter1StageFactory FactoryType;
};
///@endcond

/**
 * @brief implementation of interface ERFunctionInterface, 
 * following the paper from Neumann and Matas
 *
 * This class isn't copyable and copy-assignable.
 */
class ERFilter1Stage : public ERFunctionInterface
{
    public:
        /**
         * @brief constructor
         */
        ERFilter1Stage()
        { 
            boost_.setReturningSum(true);
        }

        /**
         * @brief forbidden copy constructor
         *
         * @param other
         */
        ERFilter1Stage( const ERFilter1Stage &other ) = delete; 
        /**
         * @brief forbidden assigment constructor
         *
         * @param other
         *
         * @return 
         */
        ERFilter1Stage& operator=( const ERFilter1Stage &other ) = delete;

        /**
         * @brief loads configuration for boosting algorithm, to 
         * classify the regions
         *
         * @param conf path to configuration file
         */
        void loadConfiguration( const std::string &conf )
        {
            boost_.loadConfiguration( conf ); 
        }

        /**
         * @brief compute the probability P( r | character ) for the
         * \p r
         *
         * @param r region, we compute the probability for
         *
         * @return probability P( r | character )
         *
         * Function returns probability $P(r|character)$. If no 
         * configuration file will be loaded, assertion will fail.
         * For further see the programming documentation.
         * */
        float getProbability( const ERRegion &r ) override;
    private:
        Boost<feature::ERGeom> boost_; 

};


///@cond
template <>
struct FeatureTraits<feature::ERGeom1> 
{
    static const int features_length = 7;
    typedef ERFilter2StageFactory FactoryType;
};
///@endcond

/**
 * @brief Second stage filtering in ER algorithm, see the 
 * programming documentation for further details.
 *
 * This class isn't copyable and copy-assignable.
 *
 */
class ERFilter2Stage
{
    public:
        /**
         * @brief constructor
         */
        ERFilter2Stage()
        {
            ERFilter2StageFactory factory;
            features_extractor_ = decltype(features_extractor_) 
                                        (factory.getOnly2StageFeatureExtractor());
        }

        /**
         * @brief forbidden copy constructor
         *
         * @param other
         */
        ERFilter2Stage( const ERFilter2Stage &other ) = delete; 

        /**
         * @brief forbidden assigment constructor
         *
         * @param other
         *
         * @return 
         */
        ERFilter2Stage& operator=( const ERFilter2Stage &other ) = delete;

        /**
         * @brief loads configuration for SVM to perform 
         * second stage filtering
         *
         * @param conf path to the configuration file
         */
        void loadConfiguration( const std::string &conf )
        {
            // svm_.loadConfiguration( conf, "scaling_er2stage.conf");
            svm_.loadConfiguration( conf );
        }


        /**
         * @brief decides if region r is a letter candidate or not
         *
         * @param r region we make decision for
         *
         * @return true if r is letter candidate else false
         */
        bool isLetter( ERRegion &r );
    private:
        std::unique_ptr<AbstractFeatureExtractor> features_extractor_;

        // LibSVM<feature::ERGeom1> svm_;
        ScalingLibSVM<feature::ERGeom1> svm_;
};

// ==================================Extremal region=========================

 /**
  * @brief Extremal Region takes care of structural steps of building the tree
  * and updating tree nodes. Class also extracts ER from component tree 
  * based on approach described in work from Neumann and Matas.
  * This class isn't copyable or copy-assignable.
  */
class ERTree
{
    public:
        typedef ComponentTreeNode<ERRegion> NodeType;

        /**
         * @brief default constructor sets min area to 0 and max
         * area to 1, which will result in all nodes being marked 
         * as letter candidates.
         */
        ERTree() 
            : root_(nullptr), min_area_ratio_(0), max_area_ratio_(1) 
        { 
        }

        /**
         * @brief constructor 
         *
         * @param min_area_ratio sets the lower bound of region size, 
         * min area is set to \p min_area_ratio * domain image area 
         * @param max_area_ratio sets the upper bound of region size,
         * max area is set to \p max_area_ratio * domain image area 
         *
         * Constructor initialize min area and max area, using the ratio of 
         * the bounds and the domain image area. 
         * Following inequality must hold * 0 < min_area_ratio < max_area_ratio < 1.
         */
        ERTree( double min_area_ratio, double max_area_ratio );

        /**
         * @brief forbidden copy constructor
         *
         * @param other
         */
        ERTree( const ERTree& other ) = delete;

        /**
         * @brief forbidden assigment constructor
         *
         * @param other
         *
         * @return 
         */
        ERTree& operator=( const ERTree& other) = delete;

        /**
         * @brief destructor
         */
        ~ERTree() 
        {
            if (root_ != nullptr)
            {
                deallocateTree();
            }
        }

        /**
         * @brief load configuration file for the second stage
         *
         * @param second_stage_conf path to the configuration file
         */
        void loadSecondStageConf( const std::string &second_stage_conf );

        /**
         * @brief set domain image, from which we will build the component tree
         *
         * @param image domain image
         * 
         * The domain image must have BGR format, CV8UC3 type for cv::Mat
         */
        void setImage( const cv::Mat &image );

        /**
         * @brief set min area ratio
         *
         * @param min_area_ratio sets the lower bound of region size, 
         * min area is set to \p min_area_ratio * domain image area 
         *
         * Following inequality must hold 0 < min_area_ratio < max_area_ratio < 1.
         */
        void setMinAreaRatio( double min_area_ratio )
        {
            min_area_ratio_ = min_area_ratio;
        }

        /**
         * @brief set max area ratio
         *
         * @param max_area_ratio sets the upper bound of region size,
         * max area is set to \p max_area_ratio * domain image area 
         *
         * Following inequality must hold 0 < min_area_ratio < max_area_ratio < 1.
         */
        void setMaxAreaRatio( double max_area_ratio )
        {
            max_area_ratio_ = max_area_ratio;
        }

        /**
         * @brief minimum probability for er detection
         *
         * @param max_area_ratio sets the upper bound of region size,
         * max area is set to \p max_area_ratio * domain image area 
         *
         * Following inequality must hold for every region r in tree  \f$P( r\mid character) > \p min_global_prob)
         */
        void setMinGlobalProbability( double min_global_prob )
        {
            min_global_prob_ = min_global_prob;
        }

        /**
         * @brief minimum difference between local minumum
         * and maximum
         *
         * @param min_difference difference between local mininum
         * and maximum probabily in Tree
         *
         */
        void setMinDifference(double min_difference)
        {
            min_delta_ = min_difference;
        }

        /**
         * @brief sets delta, which gaves the neighbourhood size for extreme search
         *
         * @param delta must be greater then 0
         *
         */
        void setDelta(int delta)
        {
            delta_ = delta;
        }

        /**
         * @brief set er function for evaluating \f$ P(r \mid character ) \f$ in
         * the first stage filtering 
         * @param er_function unique pointer to the function
         */
        void setERFunction( std::unique_ptr<ERFunctionInterface> er_function )
        {
            er_function_ = std::move( er_function );
        }

        /**
         * @brief release unique pointer of er function for the first phase
         * sets the current er function to nullptr
         *
         * @return unique pointer to er function
         */
        std::unique_ptr<ERFunctionInterface> releaseERFunction() 
        {
            std::unique_ptr<ERFunctionInterface> tmp;
            tmp.swap( er_function_ );
            return tmp;
        }

        /**
         * @brief return domain image
         *
         * @return domain image
         */
        cv::Mat getDomain() const { return bitmap_; }
        /**
         * @brief invert domain image I = 255 - I;
         */
        void invertDomain();



        /**
         * @brief deallocate the all tree nodes from root to lists
         *
         * @param root root of component tree
         */
        // void deallocateTree( NodeType *root );
        void deallocateTree();

        /**
         * @brief extract letter candidates from from component tree by performing 
         * the second stage of filtering of ER approach
         *
         * @param root root of component tree
         * @param deallocate if true the tree will be after extraction of ER deallocated
         *
         * @return vector of letter candidates 
         *
         * This function perform only second stage of filtering of the algorithm. Following 
         * the steps described in my bachalor thesis following the work of Neumann and Matas.
         */
        std::vector< Component > getLetters(bool deallocate = true);
        // std::vector< LetterStorage<ERStat> > getLetters( NodeType * root, bool deallocate = true );

        /**
         * @brief convert all nodes in tree to class component
         *
         * @param root root of component tree
         *
         * @return vector of converted components
         */
        // std::vector<Component> toComponent( NodeType * root );
        std::vector<Component> toComponent();

        /**
         * @brief deletes all nodes, that doesn't meet the condition of being
         * ER in the first stage of filtering
         * 
         * @param root root of component tree
         *
         * The regions, that doesn't meet the required conditions fo being 
         * accepted in first stage of filtering are rejected. See the condition
         * and further details in programming documentation.
         */
        // void transformExtreme( NodeType *root );
        void transformExtreme();

        /**
         * @brief transform tree using the er 2 stage classifier,
         *
         * Removes all nodes from tree ,that are classified as non letter by 
         * er 2 stage classifier
         */
        void transform2StageFiltering();

        /**
         * @brief deletes all nodes, that are similar to their parent or children
         *
         * @param root root of the component tree
         */
        void rejectSimilar(); 

        std::vector< std::vector<float> > getAllFirstStageDesc() const;


        template <typename Functor>
        void processTree(Functor & functor)
        {
            return root_->visit(functor);
        }

    private:
        friend class ComponentTreeBuilder<ERTree>;
        friend class ComponentTreePolicy<ERTree>;

        // class methods
        ComponentTreeNode<ERRegion> * root_;
        boost::fast_pool_allocator<
            NodeType, 
            boost::default_user_allocator_new_delete, 
            boost::details::pool::null_mutex,
            32> memory_pool_allocator_;
        
        int cols_, rows_;
        cv::Mat bitmap_;
        size_t processed_points_;

        // cv::Mat4b value_mat_;
        std::vector<LinkedPoint> points_;
        std::vector<bool> accumulated_pixels_;

        std::unique_ptr<ERFunctionInterface> er_function_;
        // ERFilter1Stage filter1_;
        ERFilter2Stage filter2_;

        float min_global_prob_ = 0.2f;
        float min_delta_ = 0.1f;

        double min_area_ratio_, max_area_ratio_;
        const int min_area_limit = 20;
        int min_area_, max_area_;

        int delta_ = 5;

        //  method for component builder 
        /**
         * @brief add pixel coded with \p code to the 
         * to reg
         *
         * @param reg node we update in component tree
         * @param code code of pixel we add to the \p reg
         *
         * Pixels are coded by following formula \f$code = y * domain.width + x \f$ 
         */
        void accumulate( NodeType *reg, int code ); 

        /**
         * @brief connect node child to parent node as his new child.
         *
         * @param child new child of parent
         * @param parent node we connect the child
         *
         * This function takes care of structural connecting child to the parent, 
         * and also updates parent node after adding new node \p child.
         */
        void merge( NodeType *child, NodeType *parent );


        NodeType * createNode(cv::Point pixel, int level);

        NodeType * createRootNode();

        void destroyNode(NodeType * node);


        // private methods of class
        cv::Point getPoint( int code )
        {
            return cv::Point( code % cols_, code / cols_); 
        }


        void saveTree( NodeType *root, std::vector<NodeType*> &nodes ) const;
        

        // template <typename Functor> 
        // bool transformTree( NodeType *root, Functor &&fn )
        // {
        //     bool result = fn( root );
        //     NodeType *child = root->child_;
        //     while( child != nullptr )
        //     {
        //         NodeType *tmp = child->next_;
        //         if ( !transformTree( child, fn ) )
        //         {
        //             child->remove();
        //             delete child;
        //         }
        //         child = tmp;
        //     }
        //
        //     return result;
        // }
        
        template <typename Functor>
        bool transform(Functor && functor, NodeType * root)
        {
            bool result = functor(root);
            root->size_ = 1;
             
            NodeType * child = root->child_;
            while( child != nullptr )
            {
                NodeType * tmp = child->next_;
                if ( !transform(functor, child) )
                {
                    root->size_ += child->size_;
                    child->remove();
                    // memory_pool_.destroy(child);
                    memory_pool_allocator_.destroy(child);
                    memory_pool_allocator_.deallocate(child);
                }
                else
                {
                    root->size_ += child->size_;
                }

                child = tmp;
            }

            return result;
        }

        bool isExtremeRegion( NodeType *reg );
        std::pair<float,float> findExtremeParentProb(NodeType *child_region);

        bool testSimilarChildren( NodeType *r );
        bool checkChildren( NodeType *reg, int min_area, float probability );
        bool testSimilarParent( NodeType *r );

        static std::size_t getMinSizeDiff(std::size_t size);
};

/**
 * @brief Specified policy class ComponentTreePolicy, used for integration
 * of class ERTree and ComponentTreeBuilder.
 */
template <> class ComponentTreePolicy<ERTree>
{
    public:
        typedef ComponentTreeNode<ERRegion> NodeType;
    
        static void init( const cv::Mat &bitmap, std::vector<bool> &accesible_mask, 
                int * init_pixel_code )
        {
            accesible_mask = helper::getAccesibilityMaskWithNegativeBorder( bitmap );
            // stack.push( new NodeType(256) );
            *init_pixel_code = bitmap.cols + 1;
            accesible_mask[*init_pixel_code] = true;
        }

        static NodeType* createNode( int level, cv::Point p )
        {
            ERRegion r( level, p );
            return new NodeType(r);
        }

        static int getLevel( NodeType* node )
        {
            return node->getVal().getLevel();
        }

        static void setRoot( NodeType * root, ERTree * extremal_region )
        {
            extremal_region->root_ = root;
        }
};

// ===========policy class for segment integration of extremal regions ==========

/**
 * @brief method class for er text extraction 
 * used for segmentantion returns vector of letter 
 */
class ERTextDetection 
{
    public:
        typedef Component Storage; 

        /**
         * @brief default constructor
         */
        ERTextDetection() = default;
        /**
         * @brief initialize object with configuration file
         * object is fully capable of letter segmentation
         *
         * @param first_stage_conf configuration file for first stage of ER
         * @param second_stage_conf configuration file for second stage of ER
         *
         * Loads configuration file for both phases of extremal region approach.
         * For further details see the programming documentation.
         */
        ERTextDetection(const std::string &first_stage_conf, const std::string &second_stage_conf);
      

        /**
         * @brief finds letter candidates in image 
         *
         * @param image input image, CV8UC3 required format
         *
         * @return vector of letters candidates 
         *
         * Finds letter candidates in image using extremal approach algorithm.
         * If configs are not loaded, then assertion will fail.
         */
        std::vector< Storage > getLetters(const cv::Mat &image);

        ERTree & getTree() 
        {
            return extremal_region_;
        }

    private:
        ERTree extremal_region_;
};


/**
 * @brief Creates composite used in computing if two letters 
 * can be in one word
 */
struct LetterMergingFactory : public AbstractFeatureFactory
{
    LetterMergingFactory( const cv::Mat &image )
        : image_(image)
    {

    }
    
    FeaturePtr createFeatureExtractor() const 
    {
        CompositeFeatureExtractor* composite = new CompositeFeatureExtractor();
        composite->addFeatureExtractor( new SwtMean() );
        composite->addFeatureExtractor( new PerimeterValuesMean( image_ ) );
        return FeaturePtr( composite ); 
    }

    cv::Mat image_;
};

template <> 
class SegmentationPolicy<ERTextDetection>
{
    public:
        typedef Component MethodOutput;

        static const bool k_perform_nm_suppresion = true;

        static std::vector<MethodOutput> extract
            ( ERTextDetection * er_text_detection,
              const cv::Mat &image )
        {
            return er_text_detection->getLetters(image);
        }

        static bool haveSignificantOverlap( const Component &a, 
                const Component &b )
        {
            cv::Rect a_rect = a.rectangle();
            cv::Rect b_rect = b.rectangle();
            cv::Rect intersection = a_rect & b_rect;

            if ( intersection.area() == 0 )
            {
                return false;
            }

            double overlap_ratio_a = (double) intersection.area()/a_rect.area(); 
            double overlap_ratio_b = (double) intersection.area()/b_rect.area(); 
            return (overlap_ratio_a > k_epsilon) && (overlap_ratio_b > k_epsilon);
        }

        static Letter convert( const Component &a, 
                const TranslationInfo &translation )
        {
            return Letter( std::make_shared<Component>(a), translation );
        }

    private:
        const static double k_epsilon;
};



class ComponentExtractor
{
public:
    ComponentExtractor() = default;

    void operator() (const ERRegion & er_region);

    std::vector<Component> getExtractedComponents() const;
private:
    std::vector<Component> extracted_components_;
};

class ErLimitSize
{
public:
    static std::pair<double, double> getErSizeLimits(const cv::Size & size);
};

/**
 * @brief builds er_tree, and process the tree with \p functor
 *
 * @tparam SECOND_STAGE_FILTER if true nodes that doesn't pass er 2stage are deleted from tree
 * @tparam REJECT_SIMILAR if true similar nodes in tree are deleted
 * @tparam F
 *
 * @param er_tree er_tree, class used for building er tree, must have initialized classifiers
 * @param image input image, required format CV_8UC3
 * @param functor functor for processing tree
 *
 * Build an er tree, rejects all non extreme nodes, optionally perform second stage filtering and rejecting similar,
 * afterward process \p functor on \p er_tree.
 */
template <bool SECOND_STAGE_FILTER = false, bool REJECT_SIMILAR = false, typename F> 
void process(ERTree & er_tree, const cv::Mat & image, F && functor)
{
    double min_area_ratio, max_area_ratio;

    std::tie(min_area_ratio, max_area_ratio) =
        ErLimitSize::getErSizeLimits(image.size());

    er_tree.setMinAreaRatio(min_area_ratio);
    er_tree.setMaxAreaRatio(max_area_ratio);
    er_tree.setImage(image);

    ComponentTreeBuilder<ERTree> builder( &er_tree );

    builder.buildTree();
    er_tree.transformExtreme();
    if (SECOND_STAGE_FILTER)
    {
        er_tree.transform2StageFiltering();
    }

    if (REJECT_SIMILAR)
    {
        er_tree.rejectSimilar();
    }

    er_tree.processTree(functor);
    er_tree.deallocateTree();

    er_tree.invertDomain();

    builder.buildTree();
    er_tree.transformExtreme();
    if (SECOND_STAGE_FILTER)
    {
        er_tree.transform2StageFiltering();
    }

    if (REJECT_SIMILAR)
    {
        er_tree.rejectSimilar();
    }
    er_tree.processTree(functor);
    er_tree.deallocateTree();
}



#endif /* extremal_region.h */

/* 
 * Tran Tuan Hiep
 * Implementation of methods and classes declared in extremal_region.h
 *
 * Compiler: g++ 4.8.3
 */
#include "../include/nocrlib/extremal_region.h"
#include "../include/nocrlib/assert.h"
#include <opencv2/core/core.hpp>

using namespace std;

float ERFilter1Stage::getProbability( const ERRegion &r )
{
    vector<float> data = r.getFeatures();
    cv::Mat features_mat( data );
    float sum = boost_.predict( features_mat ); 
    return 1/(1 + std::exp( -2 * sum ) );
}


bool ERFilter2Stage::isLetter( ERRegion &r )
{
    auto c = r.toComponent();
    vector<float> features = r.getFeatures();
    vector<float> data = features_extractor_->compute(c);
    features.insert( features.end(), data.begin(), data.end() );
    /*
     * for( float f : features )
     * {
     *     cout << f << ':';
     * }
     * cout << endl;
     */
    
    return svm_.predict( features ) == 1;
}

void ERFilter2Stage::operator() (ERRegion & err)
{
    auto c = err.toComponent();
    vector<float> features = err.getFeatures();
    vector<float> data = features_extractor_->compute( c);
    features.insert( features.end(), data.begin(), data.end() );

    if (svm_.predict(features) == 1)
    {
        storages_.push_back( LetterStorage<ERStat>( std::make_shared<Component>(
                        std::move(c)), err.createERStat() ));
    }
}

std::vector<LetterStorage<ERStat> > ERFilter2Stage::getLetters() const
{
    return storages_;
}

void ERFilter2Stage::clearLetters()
{
    storages_.clear();
}

// ==================================extremal region============================

ERTree::ERTree( double min_area_ratio, double max_area_ratio ) 
    : root_(nullptr), min_area_ratio_(min_area_ratio),max_area_ratio_(max_area_ratio)
    
{
}

void ERTree::loadSecondStageConf( const string &second_stage_conf )
{
    filter2_.loadConfiguration( second_stage_conf );
}

void ERTree::setImage( const cv::Mat &image )
{
    vector<cv::Mat> bgr_mat;
    if ( image.type() == CV_8UC3 )
    {
        // image has BGR format
        cv::split( image, bgr_mat );
        cv::Mat gray_image;
        cv::cvtColor( image, gray_image, CV_BGR2GRAY ); 
        bgr_mat.push_back( gray_image );
        cv::copyMakeBorder( gray_image, bitmap_, 1, 1, 1, 1, cv::BORDER_CONSTANT, 255 );
    }
    else if ( image.type() == CV_8UC1 )
    {
        // image has grayscale format
        cv::copyMakeBorder( image, bitmap_, 1, 1, 1, 1, cv::BORDER_CONSTANT, 255 );
        cv::Mat bgr_image;
        cv::cvtColor( image, bgr_image, CV_GRAY2BGR );
        cv::split( bgr_image, bgr_mat );
        bgr_mat.push_back( image );
    }
    else
    {
        NOCR_ASSERT(false, "wrong input image format"); 
    }

    cv::merge( bgr_mat, value_mat_ );
    rows_ = image.rows + 2; 
    cols_ = image.cols + 2;

    // zero padding
    processed_points_ = 0;
    int image_size = image.rows * image.cols;
    min_area_ = std::max( min_area_limit, (int)(min_area_ratio_ * image_size) );

    max_area_ = max_area_ratio_ * image_size;

    accumulated_pixels_ = std::vector<bool>( rows_ * cols_, false ); 
    points_.resize( image_size );
    er_function_->setImage( image );
}

void ERTree::invertDomain()
{
    processed_points_ = 0;
    cv::Rect domain_rect(1, 1, cols_ - 2, rows_ - 2 );
    cv::Mat domain = bitmap_( domain_rect );
    cv::bitwise_not( domain, domain );
    std::replace( accumulated_pixels_.begin(), accumulated_pixels_.end(),
            true, false );
}

void ERTree::accumulate( NodeType *reg, int code )
{
    const uchar* image_data = bitmap_.data;
    cv::Point accumulated_point = getPoint( code );
    points_[processed_points_].val_ = accumulated_point; 

    int horiz_cross_change = 0;
    uchar pVal = image_data[code];
    std::uint16_t mask = 0;
    // case 0: 
        int ncode0 = code - 1 - cols_; 
        bool acc_neighbour0 = ((accumulated_pixels_[ncode0])
                && (image_data[ncode0] <= pVal));
    // case 1: 
        int ncode1 = code - cols_; 
        bool acc_neighbour1 = ((accumulated_pixels_[ncode1])
                && (image_data[ncode1] <= pVal));

    // case 2: 
        int ncode2 = code + 1 - cols_; 
        bool acc_neighbour2 = ((accumulated_pixels_[ncode2])
                && (image_data[ncode2] <= pVal));
         
    // case 3: 
        int ncode3 = code - 1; 
        bool acc_neighbour3 = ((accumulated_pixels_[ncode3])
                && (image_data[ncode3] <= pVal));

        if (acc_neighbour3)
            horiz_cross_change++;

    // case 4: 
        int ncode4 = code + 1; 
        bool acc_neighbour4 = ((accumulated_pixels_[ncode4])
                && (image_data[ncode4] <= pVal));

        if (acc_neighbour4)
            horiz_cross_change++;

    // case 5: 
        int ncode5 = code - 1 + cols_; 
        bool acc_neighbour5 = ((accumulated_pixels_[ncode5])
                && (image_data[ncode5] <= pVal));
    // case 6: 
        int ncode6 = code + cols_; 
        bool acc_neighbour6 = ((accumulated_pixels_[ncode6])
                && (image_data[ncode6] <= pVal));

    // case 7: 
        int ncode7 = code + 1 + cols_; 
        bool acc_neighbour7 = ((accumulated_pixels_[ncode7])
                && (image_data[ncode7] <= pVal));

        std::uint16_t q1 = acc_neighbour0 | acc_neighbour1 << 1 | acc_neighbour3 << 2;
        std::uint16_t q2 = acc_neighbour1 << 4| acc_neighbour2 << 5 | acc_neighbour4 << 7;
        std::uint16_t q3 = acc_neighbour3 << 8 | acc_neighbour5 << 10 | acc_neighbour6 << 11;
        std::uint16_t q4 = acc_neighbour4 << 13| acc_neighbour6 << 14 | acc_neighbour7 << 15;

        mask = q1 | q2 | q3 | q4;


    ERRegion * ptr = &reg->getVal();

    // ptr->addPoint( &points_[processed_points_++], 
    //                         horiz_cross_change , quad );
    ptr->addPoint( &points_[processed_points_++], horiz_cross_change);
    // minus offset (1,1) because of add zero border
    ptr->updateMeans( 
            value_mat_.at<cv::Vec4b>( accumulated_point.y - 1, accumulated_point.x -1 ) );
    ptr->updateEulerBit(mask);
    accumulated_pixels_[code] = true;
}

void ERTree::merge( NodeType *child, NodeType *parent )
{
    child->getVal().setMedianCrossing();
    // connect children to its parent
    parent->getVal().merge( child->getVal() );
    // check geometric requirement for region for probability evaluation
    int size = child->getVal().getSize();
    if ( size > min_area_ && size < max_area_
            && child->getVal().getHeight() > 2 && child->getVal().getWidth() > 2 )
    {

        // check probability requirement
        float prob = er_function_->getProbability( child->getVal() ); 
        child->getVal().setProbability( prob );
        if ( prob > min_global_prob_ )
        {
            // child node meets conditions to be an ER
            parent->addChild( child );
            // update parent child probabilities
            setChildrensProbabilities( parent,child );
            return;
        }
    }

    // else we deallocate children and connect childs children to parent
    // and update parent child probabilities
    parent->reconnectChildren( child );
    ProbabilityRecord child_max = child->getVal().getChildMaxProbability();
    ProbabilityRecord parent_max = parent->getVal().getChildMaxProbability();
    if ( child_max > parent_max )
    {
        parent->getVal().setChildMaxProbability( child_max );
    }

    ProbabilityRecord child_min = child->getVal().getChildMinProbability();
    ProbabilityRecord parent_min = parent->getVal().getChildMinProbability();
    if ( child_min < parent_min )
    {
        parent->getVal().setChildMinProbability( child_min );
    }

    // child node isn't needed anymore 
    // delete child;
    memory_pool_allocator_.destroy(child);
    memory_pool_allocator_.deallocate(child);
    // memory_pool_.destroy(child);
}

auto ERTree::createNode(cv::Point p, int level)
    -> NodeType *
{
    ERRegion r( level, p );
    NodeType * chunk = memory_pool_allocator_.allocate();
    NodeType tmp(level, p);
    memory_pool_allocator_.construct(chunk, tmp);
    return chunk;
}

auto ERTree::createRootNode()
    -> NodeType *
{
    NodeType * chunk = memory_pool_allocator_.allocate();
    NodeType tmp(256);
    memory_pool_allocator_.construct(chunk, tmp);
    return chunk;
}

void ERTree::destroyNode(NodeType * node)
{
    // memory_pool_.destroy(node);
    memory_pool_allocator_.destroy(node);
    memory_pool_allocator_.deallocate(node);
}


void ERTree::setChildrensProbabilities( NodeType *parent, NodeType *child )
{
    ProbabilityRecord parentMax = parent->getVal().getChildMaxProbability();
    ProbabilityRecord childMax = child->getVal().getChildMaxProbability();
    bool minSearch = false;
    bool maxSearch = false;

    // if children could give us a new local maximum
    if ( parentMax < childMax ) 
    {
        // if children maximum in our reach 
        // set parent maximum to the childrens 
        // with incremented depth
        if ( childMax.depth_ + 1 <= delta_ )
        {
            ++childMax.depth_;
            parent->getVal().setChildMaxProbability( childMax );
        }
        // children maximum is out of our reach 
        else 
        {
            maxSearch = true;
            // bread first search is required
        }
    }

    ProbabilityRecord parentMin = parent->getVal().getChildMinProbability();
    ProbabilityRecord childMin = child->getVal().getChildMinProbability();
    // if children could give us a new local minimum
    if ( parentMin > childMin ) 
    {
        // if children minimum in our reach 
        // set parent mininum to the childrens 
        // with incremented depth
        if ( childMin.depth_ + 1 <= delta_ )
        {
            ++childMin.depth_;
            parent->getVal().setChildMinProbability( childMin );
        }
        // the children minimum is out of our reach 
        else 
        {
            minSearch = true;
            // bread first search is required
        }
    }

    if ( minSearch || maxSearch )
    {
        ProbabilityRecord min,max;
        // finds minimum and maximum using bread first search from child node
        // and then update parent max and min child probabilities
        breadthFirstSearchMinMax( child, delta_ -1, parentMax, parentMin ); 
        parent->getVal().setChildMaxProbability( parentMax );
        parent->getVal().setChildMinProbability( parentMin );
    }
}


void ERTree::breadthFirstSearchMinMax( NodeType *root, int maxDepth, 
        ProbabilityRecord &max, ProbabilityRecord &min )
// classical BFS from node to its childrens
// returns min and max probability from all children closer to root then delta
{
    typedef std::pair<int, NodeType* > depthRegionPair;
    std::queue< depthRegionPair > region_queue; 
    region_queue.push( depthRegionPair( 0,root ));

    while( !region_queue.empty() )
    {
        depthRegionPair back = region_queue.back();
        int depth = back.first;
        NodeType *reg = back.second;
        region_queue.pop();

        float probability = reg->getVal().getProbability();
        if ( max.probability_ < probability ) 
        {
            max = ProbabilityRecord( depth, probability );
        }

        if ( min.probability_ > probability )
        {
            min = ProbabilityRecord( depth, probability );
        }

        for( NodeType* child = reg->child_; child != nullptr; child = child->next_ )
        {
            int depth_from_root = depth + child->depth_from_parent_;
            if ( depth_from_root <= delta_ )
            {
                region_queue.push( depthRegionPair( depth_from_root, child ));
            }
        }
    }
}


vector< LetterStorage<ERStat> > ERTree::getLetters( bool deallocate )
{
    transformExtreme(); 
    
#ifdef PRINT_INFO
    std::cout << "first stage " << root_->getNodeCount() << endl;
    std::cout << sizeof(cv::Vec4i) << std::endl;
#endif
     
    transform([this] (NodeType * node) -> bool
            {
                return filter2_.isLetter(node->getVal());
            }, root_);


#ifdef PRINT_INFO
    std::cout << "second stage " << root_->getNodeCount() << endl;
#endif

    /*
     * transform([] (NodeType * node) -> bool
     *         {
     *             std::size_t parent_size = node->parent_->getVal().getSize();
     *             std::size_t size = node->getVal().getSize();
     *             std::size_t min_diff = std::max((std::size_t) (size * 0.002), (std::size_t) 10);

     *             return (parent_size - size > min_diff);
     *         }, root_);
     */

    rejectSimilar();

#ifdef PRINT_INFO
    std::cout << "second stage after rejecting similar " << root_->getNodeCount() << endl;
#endif


    vector<LetterStorage<ERStat> > storages;

    vector<NodeType*> nodes;
    saveTree( root_, nodes );

    for (NodeType  * node : nodes)
    {
        storages.push_back( LetterStorage<ERStat>(node->getVal().toCompPtr(), 
                    node->getVal().createERStat() ));
    }

    if ( deallocate )
    {
        for ( NodeType * node : nodes )
        {
            // delete node;
            memory_pool_allocator_.destroy(node);
            memory_pool_allocator_.deallocate(node);
            // memory_pool_.destroy(node);
        }
        // delete root_;
        // memory_pool_.destroy(root_);
        memory_pool_allocator_.destroy(root_);
        memory_pool_allocator_.deallocate(root_);
        root_ = nullptr;
    }

    return storages;
}


void ERTree::deallocateTree()
{
    vector<NodeType*> nodes;
    saveTree(root_, nodes);
    for ( NodeType *node: nodes )
    {
        // delete node;
        memory_pool_allocator_.destroy(node);
        memory_pool_allocator_.deallocate(node);
        // memory_pool_.destroy(node);
    }
    // delete root_;
    memory_pool_allocator_.destroy(root_);
    memory_pool_allocator_.deallocate(root_);
    // memory_pool_.destroy(root_);
    root_ = nullptr;
}

void ERTree::saveTree( NodeType *root, std::vector<NodeType*> &nodes ) const
{
    stack<NodeType *> stack_node;

    for ( NodeType *node = root->child_; node != nullptr; node = node->next_ )
    {
        stack_node.push(node);
    }

    while(!stack_node.empty())
    {
        NodeType *top_stack = stack_node.top();
        nodes.push_back(stack_node.top());
        stack_node.pop();

        for ( NodeType *node = top_stack->child_; node != nullptr; node = node->next_ )
        {
            stack_node.push(node);
        }
    }
}

void ERTree::transformExtreme()
{
    transform([this] (NodeType * node) -> bool
            {
                return isExtremeRegion(node);
            }, root_);
}


bool ERTree::isExtremeRegion( NodeType *node )
{
    /*
     * if ( node->getVal().isLocalChildMaximum() ) 
     * {
     */
        auto extremeProb = findExtremeParentProb(node);
        // if nodeion is local maximum
        float childMaxProb = node->getVal().getProbability();
        if (extremeProb.second <= childMaxProb) 
        {
            auto child_min = node->getVal().getChildMinProbability();
            // float local_min = std::min( extremeProb.first, child_min.probability_ );
            float local_min = extremeProb.first;
            float val = childMaxProb - local_min;
            return ( val >= min_delta_ || val == 0 );
            // return ( val >= min_delta_ );
        }
    // }
    return false;
}

pair<float,float> ERTree::findExtremeParentProb(NodeType *child_node)
{
    NodeType* node = child_node;
    float prob = node->getVal().getProbability();
    std::pair<float,float> output( prob, prob ); 

    int counter = 0;
    while( counter < delta_ && node->parent_ )
    {
        node = node->parent_; 
        // node = &records_[reg->parent_];
        prob = node->getVal().getProbability();
        if ( prob < output.first ) 
        {
            output.first = prob;
        }
        else if ( prob > output.second ) 
        {
            output.second = prob;
        }
         
        // counter += node->depth_from_parent_;
        counter += 1;
    }

    return output;
}

vector<Component> ERTree::toComponent()
{
    vector<NodeType *> nodes;
    saveTree( root_, nodes );
    vector<Component> components;
    components.reserve( nodes.size() );
    for ( NodeType * node: nodes )
    {
        components.push_back( node->getVal().toComponent() );
    }
    return components;
}

void ERTree::rejectSimilar()
{
    transform([] (NodeType * node) -> bool
            {
                if (!node->parent_)
                {
                    return true;
                }

                std::size_t parent_size = node->parent_->getVal().getSize();
                std::size_t size = node->getVal().getSize();
                std::size_t min_diff = std::max((std::size_t) (size * 0.002), (std::size_t) 10);

                return (parent_size - size > min_diff);
            }, root_);

}

bool ERTree::testSimilarChildren( NodeType *node )
{
    const double k = 0.1;
    int maxChildArea = node->getVal().getSize()*( 1 - k );
    for( NodeType *child = node->child_; child != nullptr; child = child->child_ )
    {
        if ( !checkChildren( child, maxChildArea, node->getVal().getProbability() ) )
        {
            return false;
        }
    }

    return true;
}

bool ERTree::checkChildren( NodeType * node, int minArea, float probability )
{
    if ( node->getVal().getSize() < minArea )
    {
        return true;
    }

    if ( node->getVal().getProbability() > probability )
    {
        return false;
    }

    for( NodeType *child = node->child_; child != nullptr; child = child->child_ )
    {
        if ( !checkChildren( child, minArea, probability ) )
        {
            return false;
        }
    }

    return true;
}

bool ERTree::testSimilarParent(NodeType *node) 
{
    if ( !node->parent_ )
    {
        return true;
    }

    float prob = node->getVal().getProbability();
    NodeType *parent = node->parent_; 

    bool parent_similarity = node->getVal().isSimilarParent( parent->getVal() );
    while( parent->parent_ && parent_similarity ) 
    {
        float parent_prob = parent->getVal().getProbability();
        if ( parent_prob > prob )
        {
            return false;
        }
        parent = parent->parent_; 
        parent_similarity = node->getVal().isSimilarParent( parent->getVal() );
    }

    return true;
}

std::vector< std::vector<float> > ERTree::getAllFirstStageDesc() const
{
    std::vector<NodeType *> nodes;
    saveTree( root_, nodes );

    vector< vector<float> > first_stage_desc;
    first_stage_desc.reserve( nodes.size() );
    for ( NodeType *node : nodes )
    {
        first_stage_desc.push_back( node->getVal().getFeatures() );
    }

    return first_stage_desc;
}

// ======== extremal region integration text detection=====
ERTextDetection::ERTextDetection
    ( const std::string &first_stage_conf, const std::string &second_stage_conf )
    // ocr_( std::move(ocr) )
{
    // knn_ocr_.loadTrainData( "distHistogramTrain" );
    double min_area_ratio = 0.00003;
    double max_area_ratio = 0.1;

   
    extremal_region_.setMinAreaRatio(min_area_ratio);
    extremal_region_.setMaxAreaRatio(max_area_ratio);
    extremal_region_.loadSecondStageConf( second_stage_conf );
    extremal_region_.setDelta(2);

    std::unique_ptr<ERFilter1Stage> er_function( new ERFilter1Stage() );
    er_function->loadConfiguration( first_stage_conf );
    
    extremal_region_.setERFunction( std::move( er_function ) );
}

auto ERTextDetection::getLetters( const cv::Mat &image ) 
    -> vector<Storage>
{
    double min_area_ratio;
    double max_area_ratio;

    std::tie(min_area_ratio, max_area_ratio) = ErLimitSize::getErSizeLimits(image.size());
    extremal_region_.setMinAreaRatio(min_area_ratio);
    extremal_region_.setMaxAreaRatio(max_area_ratio);
    extremal_region_.setImage( image );

    ComponentTreeBuilder<ERTree> builder( &extremal_region_ );
    // ComponentTreeNode<ERRegion> *root = builder.buildTree();
    builder.buildTree();
    auto letters_storages = extremal_region_.getLetters(); 
    extremal_region_.invertDomain();

    builder.buildTree();
    auto tmp = extremal_region_.getLetters();
    letters_storages.reserve( letters_storages.size() + tmp.size() );
    letters_storages.insert( letters_storages.end(), tmp.begin(), tmp.end() );
    
    return letters_storages;
}

void ComponentExtractor::operator() (const ERRegion & er_region)
{
    extracted_components_.push_back(er_region.toComponent());
}

std::vector<Component> ComponentExtractor::getExtractedComponents() const 
{
    return extracted_components_;
}

std::pair<double, double> ErLimitSize::getErSizeLimits(const cv::Size & size)
{
    double min_area_ratio = 0.000035;
    double max_area_ratio = 0.1;
    if (size.area() <= 640 * 480)
    {
        min_area_ratio = 0.00005;
        max_area_ratio = 0.4;
    }
    else if (size.area() <= 1024* 768)
    {
        min_area_ratio = 0.000035;
        max_area_ratio = 0.3;
    }
    else if (size.area() <= 1280 * 1024)
    {
        min_area_ratio = 0.00003;
        max_area_ratio = 0.1;
    }
    else if (size.area() <= 1600 * 1200)
    {
        min_area_ratio = 0.00003;
        max_area_ratio = 0.1;
    }
    else
    {
        min_area_ratio = 0.00002;
        max_area_ratio = 0.05;
    }


    return std::make_pair(min_area_ratio, max_area_ratio);
}


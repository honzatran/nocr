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
    auto c_ptr = r.toCompPtr();
    vector<float> features = r.getFeatures();
    vector<float> data = features_extractor_->compute( c_ptr );
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
    bool quad[9];
    for ( int i = 0; i < 8; ++i )
    {
        int ncode = 0;
        switch ( i )
        {
            case 0: ncode = code - 1 - cols_; break; 
            case 1: ncode = code - cols_; break; 
            case 2: ncode = code + 1 - cols_; break;
            case 3: ncode = code - 1; break;
            case 4: ncode = code + 1; break;
            case 5: ncode = code - 1 + cols_; break; 
            case 6: ncode = code + cols_; break; 
            case 7: ncode = code + 1 + cols_; break;
            default: break;
        }

        int index;
        index = i < 4 ?  i :  i+1; 

        quad[index] = ((accumulated_pixels_[ncode])
                        && (image_data[ncode] <= pVal));

        if ( ( i == 3 || i == 4 ) && ( quad[index] ) )
        {
            ++horiz_cross_change;
        }
    }

    quad[4] = false; 

    reg->getVal().addPoint( &points_[processed_points_++], 
                            horiz_cross_change ,quad );
    // minus offset (1,1) because of add zero border
    reg->getVal().updateMeans( 
            value_mat_.at<cv::Vec4b>( accumulated_point.y - 1, accumulated_point.x -1 ) );
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
    delete child;
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
    vector< LetterStorage<ERStat> > storages;
    transformExtreme(); 
    rejectSimilar();
    vector<NodeType*> nodes;
    saveTree( root_, nodes );

    for ( NodeType *node: nodes )
    {
        ERRegion reg = node->getVal();
        if ( filter2_.isLetter(reg) )
        {
            auto comp_ptr = reg.toCompPtr();
            // gui::showImage(comp_ptr->getBinaryMat(), "componenta");
            storages.push_back( LetterStorage<ERStat>( comp_ptr, reg.createERStat() ) );
        }
    }

    if ( deallocate )
    {
        for ( NodeType *node : nodes )
        {
            delete node;
        }
        delete root_;
        root_ = nullptr;
    }

#if PRINT_INFO
    cout << "first stage filtered:" << nodes.size() << endl;
    cout << "second stage filtered:" << storages.size() << endl;
#endif
    return storages;
}


void ERTree::deallocateTree()
{
    vector<NodeType*> nodes;
    saveTree(root_, nodes);
    for ( NodeType *node: nodes )
    {
        delete node;
    }
    delete root_;
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
    transformTree( root_, [this] ( NodeType * node ) -> bool 
            {
                return isExtremeRegion(node);
            });
}


bool ERTree::isExtremeRegion( NodeType *node )
{
    if ( node->getVal().isLocalChildMaximum() ) 
    {
        auto extremeProb = findExtremeParentProb(node);
        // if nodeion is local maximum
        float childMaxProb = node->getVal().getProbability();
        if ( extremeProb.second <= childMaxProb ) 
        {
            auto child_min = node->getVal().getChildMinProbability();
            float local_min = std::min( extremeProb.first, child_min.probability_ );
            float val = childMaxProb - local_min;
            // return ( val >= min_delta || val == 0 );
            return ( val >= min_delta_ );
        }
    }
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
        counter += node->depth_from_parent_;
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
    transformTree( root_, [this] (NodeType * node) 
            {
                return testSimilarParent(node);
            });
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
    const double min_area_ratio = 0.00007;
    const double max_area_ratio = 0.3;
    extremal_region_.setMinAreaRatio(min_area_ratio);
    extremal_region_.setMaxAreaRatio(max_area_ratio);
    extremal_region_.loadSecondStageConf( second_stage_conf );
    extremal_region_.setDelta(5);

    std::unique_ptr<ERFilter1Stage> er_function( new ERFilter1Stage() );
    er_function->loadConfiguration( first_stage_conf );
    
    extremal_region_.setERFunction( std::move( er_function ) );
}

auto ERTextDetection::getLetters( const cv::Mat &image ) 
    -> vector<Storage>
{
    extremal_region_.setImage( image );

    /*
     * int size_image = image.rows * image.cols;

     * if ( size_image <= )
     * {
     * } 
     * else if (size_image <= )
     * {
     * } 
     * else
     * {
     * }
     */

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



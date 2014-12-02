#include "../include/nocrlib/testing.h"
#include "../include/nocrlib/assert.h"
#include "../include/nocrlib/iooper.h"

#include <cctype>



using namespace std;

void ImageGroundTruth::addGroundTruth( const string &word,
        const cv::Rect &bound_box )
{
    records_.insert( std::make_pair( word, bound_box ) );
}

cv::Rect ImageGroundTruth::getRectangle( const string &word )
{ 
    auto it = records_.find( word );

    if ( it != records_.end() )
    {
        return it->second;
    }

    return cv::Rect(0,0,0,0);
}

size_t ImageGroundTruth::getLetterCount() const
{
    size_t letter_count = 0;
    for ( const auto &rec : records_ )
    {
        letter_count += rec.first.size();
    }
    return letter_count;
}

void ImageGroundTruth::resize( double scale )
{
    for ( auto & rec : records_ )
    {
        cv::Point tl_rect = rec.second.tl();
        cv::Point br_rect = rec.second.br();

        cv::Rect scaled_rect = cv::Rect(scale * tl_rect,scale * br_rect);

        rec.second = scaled_rect;
    }
}

bool ImageGroundTruth::containLetter( const Letter & l) const
{
    cv::Rect letter_bbox = l.getRectangle();
    auto it = findContainingRecord(letter_bbox);
    
    if (it == records_.end())
    {
        return false;
    }

    char l_char = l.getTranslation();
    for (char c : it->first)
    {
        if ( TranslationInfo::
                haveSameLabels(c, l_char) )
        {
            return true;
        }
    }

    return false;
}

bool ImageGroundTruth::isLetterComponent(const Component & c) const
{
    cv::Rect comp_bbox = c.rectangle();
    return (findContainingRecord(comp_bbox) != records_.end());
}

auto ImageGroundTruth::findContainingRecord(const cv::Rect & rect) const
    -> decltype(records_)::const_iterator
{
    const double ratio = 0.8;
    const double height_ratio =  0.25;
    for (auto it = records_.begin(); it != records_.end(); ++it)
    {
        cv::Rect word_rec = it->second;
        cv::Rect intersection = word_rec & rect;

        if (intersection.area() == 0)
        {
            continue;
        }

        if (rect.height < height_ratio * word_rec.height)
        {
            continue;
        }

        if (intersection.area() >= ratio * rect.area())
        {
            return it;
        }
    }

    return records_.end();
}

bool ImageGroundTruth::containWord(
        const TranslatedWord & word,
        TruePositiveInterface * tp_decider)
{
    auto pair_it = records_.equal_range(word.translation_);

    cv::Rect word_bbox = word.visual_information_.getRectangle();

    if ( pair_it.second == records_.end() 
            && pair_it.first == records_.end())
    {
        return false;
    }

    for ( auto it = pair_it.first; it != pair_it.second; ++it )
    {
        if (tp_decider->isTruePositive(word_bbox, it->second))
        {
            return true;
        }
    }

    return false;
}

//==============================================
//
bool TruePositiveTest::isTruePositive( 
        const cv::Rect &detected_bbox,
        const cv::Rect &gt_bbox ) 
{
    cv::Rect intersection = detected_bbox & gt_bbox;

    if (intersection.area() == 0)
    {
        return false;
    }

    bool gt_condition = (double)intersection.area()/gt_bbox.area() > 0.8;
    bool detected_condition = (double)intersection.area()/detected_bbox.area() > 0.4;

    return gt_condition && detected_condition;
}

///==========================================================


Testing::Testing()
{
    decider_ptr_ = nullptr;
    resetCounters();
}

void Testing::loadGroundTruth( GroundTruthInterface *gt_ptr )
{
    gt_ptr->storeGroundTruth(ground_truth_);
}

void Testing::setTruePositiveDecider( const DeciderPtr &decider_ptr )
{
    decider_ptr_ = decider_ptr;
}

void Testing::resetCounters()
{
    true_positives_ = 0;
    number_results_ = 0;
    number_ground_truth_ = 0;
}

void Testing::updateScores( 
        const string &image, 
        const vector<TranslatedWord> &words )
{
    auto it = ground_truth_.find( image );
    NOCR_ASSERT( decider_ptr_ != nullptr, 
            "pointer for true positive decision not loaded");

    if ( it != ground_truth_.end() )
    {
        // ground truth founded 
        ImageGroundTruth ground_truth = it->second;

        // update counts
        number_results_ += words.size();
        number_ground_truth_ += ground_truth.getCount();

        // update positive truth
        for ( const auto &word: words )
        {
            if ( ground_truth.containWord(word, decider_ptr_.get()) )
            {
                true_positives_++;
            }
        }

        return;
    }
    // no ground truth provided throw exception
    throw TestingException( image );
}

double Testing::getPrecision() const  
{
    return (double) true_positives_/number_results_;
}

double Testing::getRecall() const 
{
    return (double) true_positives_/number_ground_truth_;
}

void Testing::makeRecord( std::ostream &oss ) const 
{
    oss << "Text recognition session:" << endl;

    oss << "Precision:" << getPrecision() 
        << " (#true positive/#detected words)" << endl;

    oss << "Recall:" << getRecall() 
        << " (#true positive/#ground truth)" << endl;
}

void Testing::notifyResize( 
        const std::string &file_name,
        double scale) 
{
    auto it = ground_truth_.find(file_name);
    it->second.resize(scale);
}

//========================================================

void LetterDetectionTesting::loadGroundTruth( GroundTruthInterface *gt_ptr )
{
    gt_ptr->storeGroundTruth(ground_truth_);
}

std::vector<bool> LetterDetectionTesting::updateScores( 
        const std::string & image_name, 
        const std::vector<Letter> &letters )
{
    auto it = ground_truth_.find(image_name);

    if (it == ground_truth_.end())
    {
        throw TestingException( image_name);
    }

    number_results_ += letters.size();

    number_ground_truth_ += it->second.getLetterCount();
    
    vector<bool> true_positive_mask( letters.size(), false );

    int i = 0;
    for (const auto & l : letters)
    {
        if ( it->second.containLetter(l) )
        {
            true_positives_++;
            true_positive_mask[i] = true;
        }

        ++i;
    }

    return true_positive_mask;
}

std::vector<bool> LetterDetectionTesting::checkLetterComponent(
        const std::string & image_name, 
        const std::vector<Component> &components)
{
    auto it = ground_truth_.find(image_name);

    if (it == ground_truth_.end())
    {
        throw TestingException( image_name);
    }

    vector<bool> true_positive_mask( components.size(), false );
    int i = 0;
    for (const Component & c : components)
    {
        if ( it->second.isLetterComponent(c) )
        {
            true_positive_mask[i] = true;
        }
        ++i;
    }

    return true_positive_mask;
}

double LetterDetectionTesting::getPrecision() const  
{
    return (double) true_positives_/number_results_;
}

double LetterDetectionTesting::getRecall() const 
{
    return (double) true_positives_/number_ground_truth_;
}

void LetterDetectionTesting::makeRecord( std::ostream &oss ) const 
{
    oss << "Text recognition session:" << endl;

    oss << "Precision:" << getPrecision() 
        << " (#true positive/#detected words)" << endl;

    oss << "Recall:" << getRecall() 
        << " (#true positive/#ground truth)" << endl;
}

void LetterDetectionTesting::notifyResize( 
        const std::string &file_name,
        double scale )
{
    auto it = ground_truth_.find(file_name);
    it->second.resize(scale);
}

bool isWSLine(const string &line)
{
    if ( line.empty())
    {
        return true;
    }

    for (char c: line)
    {
        if (!isspace(c))
        {
            return false;
        }
    }

    return true;
}


void Icdar2013Train::setUpGroundTruth( const std::string &file_name )
{
    loader ld;
    auto lines = ld.getFileContent(file_name);

    letter_ground_truths_.clear();
    letter_ground_truths_.reserve(lines.size());
    for (const string & line: lines)
    {
        if (isWSLine(line) || line.front() == '#')
        {
            continue;
        }

        cout << line << endl;

        stringstream ss(line);

        LetterGT letter_gt;
        int b, g, r; 
        ss >> r;
        ss >> g;
        ss >> b;
        letter_gt.bgr_color_= cv::Vec3b(b, g, r);

        ss >> letter_gt.center_.x;
        ss >> letter_gt.center_.y;

        cv::Point tl, br;
        ss >> tl.x;
        ss >> tl.y;
        ss >> br.x;
        ss >> br.y;

        letter_gt.bbox_ = cv::Rect(tl, br);

        string tmp;
        ss >> tmp;

        letter_gt.letter_ = tmp[1];

        letter_ground_truths_.push_back(letter_gt);
    }
}

auto Icdar2013Train::getGtComponent(const cv::Mat & image)
    -> std::vector<ComponentRecord>
{
    vector<ComponentRecord> output;
    for (const auto &letter_gt : letter_ground_truths_ )
    {
        Component c;
        cv::Mat crop_comp_image = image(letter_gt.bbox_);

        auto begin = crop_comp_image.begin<cv::Vec3b>();
        auto end = crop_comp_image.end<cv::Vec3b>();

        for (auto it = begin; it != end; ++it)
        {
            if ( *it == letter_gt.bgr_color_ )
            {
                c.addPoint( it.pos());
            }
        }

        output.push_back(
                std::make_pair(letter_gt.letter_, c));
    }

    return output;
}


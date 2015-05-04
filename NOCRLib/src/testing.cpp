#include "../include/nocrlib/testing.h"
#include "../include/nocrlib/assert.h"
#include "../include/nocrlib/iooper.h"

#include <cctype>

#define TEXT_DETECTION_TAG "text-detection-eval"
#define IMAGE_TAG "image"
#define FILE_NAME_TAG "file-name"
#define WORD_TAG "word"
#define TEXT_TAG "text"
#define TRUE_POSITIVE "true-positive"
#define LETTER_NODE "letter-node"
#define LETTER_CHAR_TAG "letter-node"
#define LETTER_CONFIDENCE_TAG "letter-node"

using namespace std;

void ImageGroundTruth::addGroundTruth( const string &word,
        const cv::Rect &bound_box )
{
    auto labels = TranslationInfo::getLabels(word);
    records_.push_back( { word, labels, bound_box} );
}

cv::Rect ImageGroundTruth::getRectangle( const string &word )
{ 
    auto it = std::find_if(records_.begin(), records_.end(), 
            [&word] (const WordRecord & record)
            {
                return record.word == word;
            });

    if ( it != records_.end() )
    {
        return it->bbox;
    }


    return cv::Rect(0,0,0,0);
}

size_t ImageGroundTruth::getLetterCount() const
{
    size_t letter_count = 0;
    for ( const auto &rec : records_ )
    {
        letter_count += rec.word.size();
    }
    return letter_count;
}

void ImageGroundTruth::resize( double scale )
{
    for ( auto & rec : records_ )
    {
        cv::Point tl_rect = rec.bbox.tl();
        cv::Point br_rect = rec.bbox.br();

        cv::Rect scaled_rect = cv::Rect(scale * tl_rect,scale * br_rect);

        rec.bbox = scaled_rect;
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
    for (char c : it->word)
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
        cv::Rect word_rec = it->bbox;
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
    std::vector<int> labels = TranslationInfo::getLabels(word.translation_);
    cv::Rect word_bbox = word.visual_information_.getRectangle();
    std::vector<std::size_t> word_match;

    for (std::size_t i = 0; i < records_.size(); ++i) 
    {
        std::vector<int> & rec_labels = records_[i].labels;

        if (rec_labels.size() != labels.size())
        {
            continue;
        }

        if (std::equal(rec_labels.begin(), rec_labels.end(), labels.begin()))
        {
            word_match.push_back(i);
        }
    }



    if (word_match.empty())
    {
        return false;
    }

    for (std::size_t i : word_match)
    {
        if (tp_decider->isTruePositive(word_bbox, records_[i].bbox))
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

    root_ = doc_.append_child(TEXT_DETECTION_TAG);
}

void Testing::loadGroundTruth( GroundTruthInterface *gt_ptr )
{
    gt_ptr->storeGroundTruth(ground_truth_);
}

void Testing::setTruePositiveDecider( TruePositiveInterface * decider_ptr)
{
    decider_ptr_ = decider_ptr;
}

void Testing::resetCounters()
{
    true_positives_ = 0;
    number_results_ = 0;
    number_ground_truth_ = 0;
}

std::vector<bool> Testing::updateScores( 
        const string &image, 
        const vector<TranslatedWord> &words )
{
    auto it = ground_truth_.find( image );
    NOCR_ASSERT( decider_ptr_ != nullptr, 
            "pointer for true positive decision not loaded");

    std::vector<bool> results(words.size(), false);

    if ( it != ground_truth_.end() )
    {
        auto image_node = root_.append_child(IMAGE_TAG);
        image_node.append_attribute(FILE_NAME_TAG).set_value(image.c_str());
        // ground truth founded 
        ImageGroundTruth ground_truth = it->second;
        // pugi::xml_node = 

        // update counts
        number_results_ += words.size();
        number_ground_truth_ += ground_truth.getCount();

        // update positive truth
        for (std::size_t i = 0; i < words.size(); ++i) 
        {
            pugi::xml_node word_node = image_node.append_child(WORD_TAG);
            createWordNode(word_node, words[i]);

            if ( ground_truth.containWord(words[i], decider_ptr_) )
            {
                true_positives_++;
                results[i] = true;

                word_node.append_attribute(TRUE_POSITIVE) = true;
            }
            else
            {
                word_node.append_attribute(TRUE_POSITIVE) = false;
            }
        }

        return results;
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

void Testing::printXmlOutput(std::ostream & oss) const
{
    doc_.save(oss);
}

void Testing::notifyResize( 
        const std::string &file_name,
        double scale) 
{
    auto it = ground_truth_.find(file_name);
    it->second.resize(scale);
}

void Testing::createWordNode(pugi::xml_node & word_node, const TranslatedWord & word)
{
    word_node.append_attribute(TEXT_TAG).set_value(word.translation_.c_str());
    cv::Rect rect = word.visual_information_.getRectangle();
    word_node.append_attribute( "x" ) = rect.x;
    word_node.append_attribute( "y" ) = rect.y;
    word_node.append_attribute( "width" ) = rect.width;
    word_node.append_attribute( "height" ) = rect.height;

    auto letters = word.visual_information_.getLetters();

    for (const auto & l : letters)
    {
        auto letter_node = word_node.append_child(LETTER_NODE);
        cv::Rect l_rect = l.getRectangle();

        letter_node.append_attribute( "x" ) = l_rect.x;
        letter_node.append_attribute( "y" ) = l_rect.y;
        letter_node.append_attribute( "width" ) = l_rect.width;
        letter_node.append_attribute( "height" ) = l_rect.height;
        std::string letter_str;
        letter_str += l.getTranslation();
        letter_node.append_attribute( LETTER_CHAR_TAG).set_value(letter_str.c_str());
        letter_node.append_attribute( LETTER_CONFIDENCE_TAG) = l.getConfidence();
    }
}
//========================================================
//
void EvaluationDrawer::loadXml(const pugi::xml_document & doc)
{
    pugi::xml_node root = doc.first_child();

    for (pugi::xml_node image_node : root.children(IMAGE_TAG))
    {
        string file_name = image_node.attribute(FILE_NAME_TAG).as_string();
        std::vector<WordRecord> word_records;
        for (pugi::xml_node word_node : image_node.children(WORD_TAG))
        {
            int x = word_node.attribute("x").as_int();
            int y = word_node.attribute("y").as_int();
            int width = word_node.attribute("width").as_int();
            int height = word_node.attribute("height").as_int();
            bool true_positive = word_node.attribute(TRUE_POSITIVE).as_bool();

            cv::Rect word_bbox = cv::Rect(x, y, width, height);

            vector<LetterRecord> letter_records = getLetterRecords(word_node);

            word_records.emplace_back(word_bbox, letter_records, true_positive);
        }

        detection_results_.insert(std::make_pair(file_name, word_records));
    }
}

void EvaluationDrawer::loadXml(const std::string & xml_file)
{
    std::ifstream ifs(xml_file);
    if (!ifs.is_open())
    {
        // throw FileNot
    }

    pugi::xml_document doc;
    auto parse_result = doc.load(ifs);

    if (!parse_result)
    {
    }

    loadXml(doc);

}

auto EvaluationDrawer::getLetterRecords(const pugi::xml_node & word_node)
    -> std::vector<LetterRecord> 
{
    std::vector<LetterRecord> letters;
    for (pugi::xml_node letter_node : word_node.children(LETTER_NODE))
    {
        int x = letter_node.attribute("x").as_int();
        int y = letter_node.attribute("y").as_int();
        int width = letter_node.attribute("width").as_int();
        int height = letter_node.attribute("height").as_int();
        char character = letter_node.attribute(LETTER_CHAR_TAG).as_string()[0];
        double confidence = letter_node.attribute(LETTER_CONFIDENCE_TAG).as_double();

        cv::Rect letter_bbox = cv::Rect(x, y, width, height);

        letters.emplace_back(letter_bbox, character, confidence);
    }

    return letters;
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


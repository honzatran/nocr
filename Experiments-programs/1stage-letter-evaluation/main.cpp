

#include <iostream>
#include <string>
#include <memory>
#include <vector>
#include <fstream>
#include <sstream>

#include <nocrlib/extremal_region.h>
#include <nocrlib/component_tree_builder.h>
#include <nocrlib/testing.h>
#include <nocrlib/iooper.h>

#include <pugi/pugixml.hpp>

using namespace std;

string er1_conf_file = "../boost_er1stage.conf";
string icdar_gt_file = "";
double min_area_ratio = 0.00007;
double max_area_ratio = 0.3;

class LetterSegmentTesting
{
public:
    LetterSegmentTesting() 
        : true_positives_(0), number_results_(0), number_ground_truth_(0) { }

    void loadGroundTruthXML(const std::string & xml_gt_file);

    void updateScores( const std::string &image_name, 
            const std::vector<Component> &letters );

    double getPrecision() const;

    double getRecall() const;

    void makeRecord( std::ostream &oss ) const;
private:
    std::map<std::string, std::vector<cv::Rect> > ground_truth_;

    bool areMatching(const cv::Rect & c_rect, const cv::Rect & gt_rect);

    unsigned int true_positives_;
    unsigned int number_results_;
    unsigned int number_ground_truth_;
};

void LetterSegmentTesting::loadGroundTruthXML(const std::string & xml_gt_file)
{
    std::ifstream ifs; 
    ifs.open(xml_gt_file);

    pugi::xml_document doc;
    pugi::xml_parse_result result = doc.load( ifs );

    pugi::xml_node root = doc.first_child();

    for (auto img_node : root.children())
    {
        string name = img_node.child("image-name").child_value();
        vector<cv::Rect> rectangles;

        for (auto rect_node : img_node.children("rectangle"))
        {
            int x = rect_node.attribute("x").as_int();
            int y = rect_node.attribute("y").as_int();
            int width = rect_node.attribute("width").as_int();
            int height = rect_node.attribute("height").as_int();
            rectangles.push_back(cv::Rect( x, y, width, height ));
        }

        ground_truth_.insert(make_pair(name, rectangles));
    }
}

void 
LetterSegmentTesting::updateScores(const std::string & image_name,
        const std::vector<Component> & components)
{
    auto gt_it = ground_truth_.find(image_name);

    if (gt_it == ground_truth_.end())
    {
        return;
    }

    vector<cv::Rect> gt_rectangles = gt_it->second;
    vector<bool> detected(gt_rectangles.size(), false);

    for (const Component & c : components)
    {
        cv::Rect c_rect = c.rectangle();

        for (size_t i = 0; i < gt_rectangles.size(); ++i)
        {
            if (areMatching(c_rect, gt_rectangles[i]))
            {
                if (!detected[i])
                {
                    ++true_positives_;
                    detected[i] = true;
                }
                break;
            }
        }
    }

    number_ground_truth_ += gt_rectangles.size();
    number_results_ += components.size();
}

bool 
LetterSegmentTesting::areMatching(
        const cv::Rect & c_rect, 
        const cv::Rect & gt_rect)
{
    cv::Rect intersection = c_rect & gt_rect;

    if (intersection.area() == 0)
    {
        return false;
    }

    bool gt_condition = (double)intersection.area()/gt_rect.area() > 0.8;
    bool detected_condition = (double)intersection.area()/c_rect.area() > 0.4;

    return gt_condition && detected_condition;
}


double LetterSegmentTesting::getPrecision() const  
{
    return (double) true_positives_/number_results_;
}

double LetterSegmentTesting::getRecall() const 
{
    return (double) true_positives_/number_ground_truth_;
}

void LetterSegmentTesting::makeRecord( std::ostream &oss ) const 
{
    oss << "Text recognition session:" << endl;

    oss << "Precision:" << getPrecision() 
        << " (#true positive/#detected words)" << endl;

    oss << "Recall:" << getRecall() 
        << " (#true positive/#ground truth)" << endl;
}

pair<string, string> parse(const string & line)
{
    stringstream ss(line);
    string path, gt_file_name;
    std::getline(ss, path, ':');
    std::getline(ss, gt_file_name, ':');

    return make_pair(path, gt_file_name);
}

int main(int argc, char ** argv)
{
    if (argc != 3)
    {
        cerr << "argument error " << argv[0] << " [xml ground truth] [list of  test files]" << endl;
        return 1;
    }

    ERTree er_tree(min_area_ratio, max_area_ratio);
    er_tree.setMinGlobalProbability(0.2);
    er_tree.setMinDifference(0.1);

    std::unique_ptr<ERFilter1Stage> er_function( new ERFilter1Stage() );
    er_function->loadConfiguration(er1_conf_file);

    er_tree.setERFunction(std::move(er_function));
    ComponentTreeBuilder<ERTree> builder( &er_tree );

    LetterSegmentTesting letter_test;

    letter_test.loadGroundTruthXML(argv[1]);

    loader input;
    vector<string> lines = input.getFileContent(argv[2]);

    for (const string & line : lines)
    {
        string file_path, gt_file_name;
        std::tie(file_path, gt_file_name) = parse(line);



        cv::Mat image = cv::imread(file_path, CV_LOAD_IMAGE_GRAYSCALE);
        er_tree.setImage(image);

        builder.buildTree();
        er_tree.transformExtreme();
        er_tree.rejectSimilar();
        auto components = er_tree.toComponent();
        er_tree.deallocateTree();

        er_tree.invertDomain();

        builder.buildTree();
        er_tree.transformExtreme();
        er_tree.rejectSimilar();
        auto tmp = er_tree.toComponent();
        er_tree.deallocateTree();

        components.insert(components.end(), tmp.begin(), tmp.end());
        letter_test.updateScores(gt_file_name,  components);
        // update statistics components;
    }

    letter_test.makeRecord(std::cout);

    // print results
}

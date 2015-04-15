
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <map>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <nocrlib/train_data.h>
#include <nocrlib/component.h>
#include <nocrlib/features.h>
#include <nocrlib/feature_factory.h>
#include <nocrlib/classifier_wrap.h>
#include <nocrlib/utilities.h>

#include <pugi/pugixml.hpp>

#define MIN_PROB 0.2
#define MIN_AREA 45

using namespace std;


struct LetterRecord
{
    cv::Rect rect;
    cv::Vec3b color;
};

class TestLetter
{
public:
    void loadXml(const std::string & xml_file);
    std::vector<Component> scanForComponents(const std::string & image_name, const cv::Mat & image);
private:
    std::map<string, std::vector<LetterRecord> > records_;
};

void TestLetter::loadXml(const std::string & xml_file)
{
    std::ifstream ifs(xml_file);
    pugi::xml_document doc;
    doc.load(ifs);

    pugi::xml_node root = doc.first_child();

    for (auto img_node : root.children())
    {
        string name = img_node.child("image-name").child_value();
        vector<LetterRecord> letters;

        for (auto rect_node : img_node.children("rectangle"))
        {
            int x = rect_node.attribute("x").as_int();
            int y = rect_node.attribute("y").as_int();
            int width = rect_node.attribute("width").as_int();
            int height = rect_node.attribute("height").as_int();

            auto color_node = rect_node.child("rect-color");
            cv::Vec3b color;
            color[0] = color_node.attribute("b").as_int();
            color[1] = color_node.attribute("g").as_int();
            color[2] = color_node.attribute("r").as_int();

            letters.push_back({ cv::Rect(x, y, width, height), color });
        }

        records_.insert(make_pair(name, letters));
    }
}

std::vector<Component> 
TestLetter::scanForComponents(const std::string & img_name, const cv::Mat & image)
{
    std::vector<Component> output;
    auto rec_it = records_.find(img_name);
    if (rec_it == records_.end())
    {
        return output;
    }

    for (const auto & rec : rec_it->second)
    {
        Component c;
        cv::Mat cropped_img = image(rec.rect);

        for (auto it = cropped_img.begin<cv::Vec3b>(); 
                it != cropped_img.end<cv::Vec3b>(); ++it)
        {
            if (*it == rec.color)
            {
                c.addPoint(it.pos());
            }
        }

        output.push_back(c);
    }


    return output;
}



int main(int argc, char ** argv)
{
    if (argc != 3)
    {
        std::cerr << "paramaters error" << std::endl;
        std::cout << argv[0] 
            << " [xml gt file] [list of gt images]" << std::endl;

        return 1;
    }
    const string k_boost_conf = "../boost_er1stage.conf";

    string line;
    ERFilter1StageFactory factory;
    auto ptr = factory.createFeatureExtractor();

    TestLetter test;
    test.loadXml(argv[1]);

    int detected = 0;
    int count = 0;

    ifstream input(argv[2]);

    auto boosting = create<Boost, feature::ERGeom>();
    boosting->loadConfiguration(k_boost_conf);
    boosting->setReturningSum(true);

    while (getline(input,line))
    {
        int comp_detected = 0;

        cv::Mat image = cv::imread(line, cv::IMREAD_COLOR);

        vector<float> scores;
        TrainExtractionPolicy<extraction::BWMaxComponent> policy;
        std::string file_name = getFileName(line);
        auto comps = test.scanForComponents(file_name, image);

        for (auto & c: comps)
        {
            if (c.size() <= MIN_AREA)
            {
                continue;
            }

            vector<float> desc = ptr->compute(c);
            float sum = boosting->predict(cv::Mat(desc));

            float prob = 1/(1 + std::exp( -2 * sum ));

            if (prob > MIN_PROB)
            {
                comp_detected++;
            }

            scores.push_back(prob);
        }

        count += comps.size();
        detected += comp_detected;

        auto it_med = scores.begin() + scores.size()/2;
        std::nth_element(scores.begin(), it_med, scores.end());
        cout << file_name << " median: " << *it_med << 
            " detection rate: " <<  (double) comp_detected/comps.size() << endl;
    }



    double detected_ratio = (double)detected/count;
    cout << "detection ratio: " << detected_ratio << endl;
}




#include <iostream>
#include <string>
#include <memory>
#include <vector>
#include <fstream>
#include <sstream>

#include <pugi/pugixml.hpp>

#include <nocrlib/extremal_region.h>
#include <nocrlib/component_tree_builder.h>
#include <nocrlib/testing.h>
#include <nocrlib/iooper.h>
#include <nocrlib/assert.h>

#include "segmentation_test.hpp"


using namespace std;

string er1_conf_file = "../boost_er1stage.conf";
string er2_conf_file = "../svm_er2stage.conf";
string icdar_gt_file = "";
double min_area_ratio = 0.00007;
double max_area_ratio = 0.3;


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
    bool second_stage_filter = false;
    if (argc == 4 || argc == 3)
    {
        if (argc == 4)
        {
            string s = argv[3];
            if (s != "--2stage-filter")
            {
                cerr << "argument error " << argv[0] << " [xml ground truth] [list of  test files] --2stage-filter" << endl;
            }
            else
            {
                second_stage_filter = true;
            }
        }
    }
    else
    {
        cerr << "argument error " << argv[0] << " [xml ground truth] [list of  test files] --2stage-filter" << endl;
        return 1;
    }


    ERTree er_tree(min_area_ratio, max_area_ratio);
    er_tree.setMinGlobalProbability(0.2);
    er_tree.setMinDifference(0.1);
    er_tree.setDelta(1);

    std::unique_ptr<ERFilter1Stage> er_function( new ERFilter1Stage() );
    er_function->loadConfiguration(er1_conf_file);

    er_tree.setERFunction(std::move(er_function));
    ComponentTreeBuilder<ERTree> builder( &er_tree );

    std::unique_ptr<LetterSegmentTesting> letter_test;
    if (second_stage_filter)
    {
        SecondStageTesting * second_filter = new SecondStageTesting();
        second_filter->setSvmConfiguration(er2_conf_file);
        letter_test = std::unique_ptr<LetterSegmentTesting>(second_filter);
    }
    else
    {
        letter_test = std::unique_ptr<LetterSegmentTesting>(new LetterSegmentTesting());
    }

    letter_test->loadGroundTruthXML(argv[1]);

    loader input;
    vector<string> lines = input.getFileContent(argv[2]);

    for (const string & line : lines)
    {
        string file_path, gt_file_name;
        std::tie(file_path, gt_file_name) = parse(line);

        cv::Mat image = cv::imread(file_path, CV_LOAD_IMAGE_GRAYSCALE);
        er_tree.setImage(image);
        letter_test->setImageName(gt_file_name);

        builder.buildTree();
        er_tree.transformExtreme();
        er_tree.rejectSimilar();
        er_tree.processTree(*letter_test);
        er_tree.deallocateTree();

        er_tree.invertDomain();

        builder.buildTree();
        er_tree.transformExtreme();
        er_tree.rejectSimilar();
        er_tree.processTree(*letter_test);
        er_tree.deallocateTree();
    }

    letter_test->makeRecord(std::cout);

    // print results
}

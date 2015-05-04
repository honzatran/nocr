

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
#include <nocrlib/utilities.h>

#include "segmentation_test.hpp"

#include <boost/program_options.hpp>

#define SIZE 1024

using namespace std;

string er1_conf_file = "../boost_er1stage_handpicked.xml";
string er2_conf_file = "../scaled_svmEr2_1.xml";
string icdar_gt_file = "";
double min_area_ratio = 0.000035;
double max_area_ratio = 0.1;

// pair<string, string> parse(const string & line)
// {
//     stringstream ss(line);
//     string path, gt_file_name;
//     std::getline(ss, path, ':');
//     std::getline(ss, gt_file_name, ':');
//
//     return make_pair(path, gt_file_name);
// }

int main(int argc, char ** argv)
{
    bool second_stage_filter = false;

    string xml_file, test_file, dir;
    
    
    namespace po = boost::program_options;
    po::variables_map vm;
    po::options_description desc("Usage");
    desc.add_options()
        ("help,h","display help message")
        ("dir,d", po::value<string>(&dir),"output directory of extracted letters")
        ("xml,x", po::value<string>(&xml_file), "icdar 2013 test set ground truth")
        ("test,t", po::value<string>(&test_file), "file with test images")
        ("2stage-filter", "run second stage filter")
        ("boost-conf", po::value<string>(&er1_conf_file), "path to boost first stage classifier")
        ("svm-conf", po::value<string>(&er2_conf_file), "path to svm second stage classifier");

    Resizer resizer(SIZE);
         
    try 
    {
        po::parsed_options parsed = po::parse_command_line(argc, argv, desc);
        po::store( parsed , vm ); 
        po::notify(vm);
    } 
    catch ( po::error &e )
    {
        std::cerr << "Parsing cmd line error:" << std::endl;
        std::cerr << e.what() << std::endl;

        return 1;
    }

    if ( vm.count("help") || vm.count("xml") == 0 || vm.count("test") == 0)
    {
        std::cerr << "experiment filtering letters" << std::endl;
        std::cerr << desc << std::endl;
        return 1;
    }

    
    if (vm.count("2stage-filter") > 0)
    {
        second_stage_filter = true;
    }



    ERTree er_tree(min_area_ratio, max_area_ratio);
    er_tree.setMinGlobalProbability(0.2);
    er_tree.setMinDifference(0.1);
    er_tree.setDelta(8);

    std::unique_ptr<ERFilter1Stage> er_function( new ERFilter1Stage() );
    er_function->loadConfiguration(er1_conf_file);

    er_tree.setERFunction(std::move(er_function));

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

    letter_test->loadGroundTruthXML(xml_file);

    loader input;
    vector<string> lines = input.getFileContent(test_file);
    ImageSaver saver;

    for (const string & line : lines)
    {
        string file_path, gt_file_name;
        std::tie(file_path, gt_file_name) = parse(line);

        cv::Mat image = cv::imread(file_path, CV_LOAD_IMAGE_GRAYSCALE);

        if ( image.rows < SIZE && image.cols < SIZE )
        {
            image = resizer.resizeKeepAspectRatio(image);
            letter_test->notifyResize(gt_file_name, resizer.getLastScale());
        }

        std::tie(min_area_ratio, max_area_ratio) =
            ErLimitSize::getErSizeLimits(image.size());


        letter_test->setImageName(gt_file_name, image);
        process(er_tree, image, *letter_test);

        cv::Mat img = letter_test->getCurrentImage();
        if (!dir.empty())
        {
            saver.saveImage(dir + "/" + getFileName(file_path), img);
        }
        else
        {
            gui::showImage(img, "detected letters");
        }
    }

    letter_test->makeRecord(std::cout);

    // print results
}

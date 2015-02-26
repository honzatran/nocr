
#include <iostream>
#include <memory>
#include <utility>
#include <string>
#include <sstream>
#include <algorithm>

#include <nocrlib/extremal_region.h>
#include <nocrlib/component_tree_builder.h>
#include <nocrlib/component.h>
#include <nocrlib/utilities.h> 
#include <nocrlib/feature_factory.h>
#include <nocrlib/iooper.h>
#include <nocrlib/street_view_scene.h>
#include <nocrlib/testing.h>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp> 
#include <boost/program_options.hpp>

#define SIZE 1024

using namespace std;

double min_area_ratio = 0.00007;
double max_area_ratio = 0.3;

string directory = "./";

int parseCmd(int argc, char ** argv)
{
    namespace po = boost::program_options;
    po::variables_map vm;
    po::options_description desc("Usage");
    desc.add_options()
        ("help,h","display help message")
        ("directory,d", po::value<string>(&directory), "output directory");
         

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

    if ( vm.count("help") || argc == 1 ) 
    {
        std::cout << desc << std::endl;
        return 1;
    }


    if (directory.back() != '/')
    {
        directory += '/';
    }
    return 0;
}

/*
 * void saveComponents( std::vector<Component> &components, const std::string & file_name)
 * {
 *     int i = 0;
 *     for ( auto & c : components )
 *     {
 *         cv::Mat comp_img = c.getBinaryMat();
 *         std::stringstream ss;
 *         ss << directory << '/' << file_name << '_' << c.size() << '_' << i << ".jpg";
 *         ++i;
 *         cv::imwrite(ss.str(), comp_img);
 *     }
 * }
 */

std::string getFileNameWithoutEnd(const std::string & file_path)
{
    std::string file_name = getFileName(file_path);
    unsigned int pos = file_name.find('.');
    return file_name.substr(0, pos);
}

void buildTree(ERTree & er_tree, 
        ComponentTreeBuilder<ERTree> & builder)
{
    builder.buildTree();
    er_tree.transformExtreme();
    er_tree.rejectSimilar();
}

void computeERDescriptor( 
        const ERTree & er_tree,  
        OutputWriter & output)
{
    auto descriptors = er_tree.getAllFirstStageDesc();
    for (const auto &d : descriptors)
    {
        output.write(d, 0);
    }
}

void computeER2StageDesc(
        vector<Component> & components,
        OutputWriter & output)
{
    ERFilter2StageFactory factory;
    auto desc_extractor = factory.createFeatureExtractor();

    for (auto &c: components)
    {
        vector<float> desc = desc_extractor->compute(c);
        output.write(desc, 0);
    }
}

std::vector<Component> extractLetterComponent(
        ERTree & er_tree,
        LetterDetectionTesting &testing,
        const std::string &file_name)
{
    vector<Component> letter_comps;
    auto comps = er_tree.toComponent();
    auto mask = testing.checkLetterComponent(file_name, comps);

    for (unsigned int i = 0; i < comps.size(); ++i)
    {
        if (mask[i])
        {
            letter_comps.push_back(comps[i]);
        }
    }

    return letter_comps;
}


std::string formatFileName(char letter, int id, int size, 
        const std::string & file_name)
{
    std::stringstream ss;
    ss << directory << letter << '_' << file_name << '_' << size << '_' << id << ".jpg";

    return ss.str();
}

int main( int argc, char **argv )
{
    if (parseCmd(argc, argv) == 1)
    {
        return 1;
    }

    string line;

    Icdar2013Train icdar_train;
    while(std::getline(std::cin, line))
    {
        uint pos = line.find(':');
        string file_name = line.substr(0, pos);
        string gt_file = line.substr(pos + 1);

        cout << file_name << " " << gt_file << endl;
        icdar_train.setUpGroundTruth(gt_file);
        cv::Mat image  = cv::imread(file_name, CV_LOAD_IMAGE_COLOR);
        auto component_records = icdar_train.getGtComponent(image);

        // save image
        int i = 0;
        for ( auto &cr : component_records )
        {
            string full_output_path = formatFileName(cr.first,
                    i, cr.second.size(), getFileNameWithoutEnd(file_name));

            cv::imwrite(full_output_path, cr.second.getBinaryMat());
            ++i;
        }
    }

    return 0;
}

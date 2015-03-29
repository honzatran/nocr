
#include <iostream>
#include <memory>
#include <utility>
#include <string>
#include <sstream>
#include <algorithm>
#include <map>
#include <vector>
#include <fstream>

#include <nocrlib/extremal_region.h>
#include <nocrlib/component_tree_builder.h>
#include <nocrlib/component.h>
#include <nocrlib/utilities.h>
#include <nocrlib/feature_factory.h>
#include <nocrlib/iooper.h>
#include <nocrlib/street_view_scene.h>
#include <nocrlib/assert.h>
#include <nocrlib/features.h>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp> 
#include <boost/program_options.hpp>


#include "extractor.hpp"

#define SIZE 1024

class DescriptorOutput : public AbstractOutput
{
public:
    DescriptorOutput(std::ostream * oss, int label) 
        : oss_(oss), label_(label)
    {

    }

    void save(ERRegion & err, 
            const std::string & file_name, char c) override
    {
        UNUSED(file_name)
        UNUSED(c)

        std::vector<float> desc = err.getFeatures();
        for (float f : desc) 
        {
            *oss_ << f << ":";
        }

        *oss_ << label_ << std::endl;
    }
private:
    std::ostream * oss_;
    int label_;
};


class ComponentOutput : public AbstractOutput
{
public:
    ComponentOutput(const std::string & dir)
        : dir_(dir)
    {
    }

    void save(ERRegion & err, 
            const std::string & file_name, char c) override
    {
        auto it = lists_.find(file_name);
        if ( it == lists_.end())
        {
            it = lists_.insert(std::make_pair(file_name, 1)).first;
        }

        auto err_comp = err.toComponent();

        // gui::showImage(err_comp.getBinaryMat(), "bla");
        std::ostringstream ss;
        ss << dir_ << '/' << file_name << '_' << 
           c << '_' << err.getSize() << '_' << it->second << ".jpg";
        // ss << dir_ << '/' << file_name << "_" << err.getSize() << "_" << it->second;
        
        cv::Mat binary = err_comp.getBinaryMat();
        cv::imwrite(ss.str(), binary, { cv::IMWRITE_JPEG_QUALITY, 100 });
    }

private:
    std::string dir_;
    ImageSaver saver_;
    std::map<string, int> lists_;
};

class ER2StageDesc : public AbstractOutput
{
public:
    ER2StageDesc(const std::string & output) 
    {
        ERFilter2StageFactory factory;
        extractor_ = std::unique_ptr<AbstractFeatureExtractor>(
                factory.getOnly2StageFeatureExtractor());

        ofs_.open(output);
    }

    void save(ERRegion & err, 
            const std::string & file_name, char c) override
    {
        UNUSED(file_name)
        UNUSED(c)

        std::vector<float> basic_desc = err.getFeatures();
        Component comp_err = err.toComponent();
        std::vector<float> additional_2stage_desc = extractor_->compute(comp_err);
        basic_desc.insert(basic_desc.end(), 
                additional_2stage_desc.begin(), additional_2stage_desc.end());

        for (float f : basic_desc)
        {
            ofs_ << f << ':';
        }

        for (float f : additional_2stage_desc)
        {
            ofs_ << f << ':';
        }

        ofs_ << 0 << std::endl;
    }
private:
    std::unique_ptr<AbstractFeatureExtractor> extractor_;
    std::ofstream ofs_;
};


using namespace std;

// string er2_conf_file = "conf/boostGeom.conf";
// const string er2_conf_file = "conf/svmERGeom.conf";


int main( int argc, char **argv )
{

    string er1_conf_file = "../boost_er1stage.conf";
    double min_area_ratio = 0.00007;
    double max_area_ratio = 0.3;

    string er1_neg = "";
    string er2_neg = "";

    bool component_type_saving = false;
    string xml_input;
    bool detect_positive = true;
    std::string extract_2stage_desc = "";
    string output;

    namespace po = boost::program_options;
    po::variables_map vm;
    po::options_description desc("Usage");

    std::string output_file;
    desc.add_options()
        ("help,h","display help message")
        ("er1-conf-file", po::value<string>(&er1_conf_file),"path to er 1 stage Boosting conf")
        ("min-area", po::value<double>(&min_area_ratio), "minimal area of nodes")
        ("max-area", po::value<double>(&max_area_ratio), "maximal area of node")
        ("first-stage", po::value<string>(&er1_neg), "output file with all negative descriptors")
        ("detect-negative", "detecting false component")
        ("component-output", po::value<string>(&output), 
             "save detected er tree node to directory")
        ("desc-output", po::value<string>(&output))
        ("xml-input,x", po::value<string>(&xml_input))
        ("2stage-desc", po::value<string>(&extract_2stage_desc),"extract second stage descriptor");
    
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

    if ( vm.count("help") || argc == 1)
    {
        std::cout << desc << std::endl;
        return 1;
    }

    if (vm.count("xml-input") == 0)
    {
        std::cerr << "xml input neccesery" << std::endl;
        return 1;
    }

    if (vm.count("detect-negative"))
    {
        detect_positive = false;
    }

    bool save_component = vm.count("component-output") > 0;
    bool save_desc = vm.count("desc-output") > 0;

    if (save_desc && save_component)
    {
        std::cerr << "only one possibility enabled" << std::endl;
        return 1;
    }

    if (save_desc)
    {
        component_type_saving = false;
    } 
    else if (save_component)
    {
        component_type_saving = true;
    }

    string line;

    ERTree er_tree(min_area_ratio, max_area_ratio);

    // er_tree.setMinGlobalProbability(0.2);
    // er_tree.setMinDifference(0.1);
    er_tree.setMinGlobalProbability(0.0);
    er_tree.setMinDifference(0.0);

    std::unique_ptr<ERFilter1Stage> er_function( new ERFilter1Stage() );
    er_function->loadConfiguration(er1_conf_file);

    er_tree.setERFunction(std::move(er_function));
    ComponentTreeBuilder<ERTree> builder( &er_tree );

    TruePositiveExtractor extractor;
    extractor.setDetectPositive(detect_positive);
    extractor.loadXml(xml_input);

    std::unique_ptr<AbstractOutput> ptr;
    std::ofstream ofs;
    if (component_type_saving)
    {
        ptr = std::unique_ptr<AbstractOutput>(new ComponentOutput(output));
    }
    else 
    {
        if (!extract_2stage_desc.empty())
        {
            ptr = std::unique_ptr<AbstractOutput>(new ER2StageDesc(extract_2stage_desc));
            er_tree.setMinGlobalProbability(0.2);
            er_tree.setMinDifference(0.1);
        }
        else
        {
            int label = detect_positive ? 1 : 0;
            DescriptorOutput * desc_output;
            std::ostream * tmp;
            if (output.empty())
            {
                tmp = &cout;
            }
            else
            {
                ofs.open( output);
                tmp = &ofs;
            }

            ptr = std::unique_ptr<AbstractOutput>(new DescriptorOutput(tmp, label));
        }
    }

    extractor.setOutput(ptr.get());

    bool transform_extreme = !extract_2stage_desc.empty();

    while( std::getline(cin, line))
    {
        cv::Mat image = cv::imread(line, cv::IMREAD_GRAYSCALE);
        string file_name = getFileName(line);
        extractor.setFileImage(file_name);
        cout << line << endl;
        if (image.empty())
        {
            continue;
        }

        er_tree.setImage(image);
        builder.buildTree();
        if (transform_extreme)
        {
            er_tree.transformExtreme();
        }

        er_tree.processTree(extractor);


        er_tree.invertDomain();
        er_tree.deallocateTree();

        builder.buildTree();
        if (transform_extreme)
        {
            er_tree.transformExtreme();
        }

        er_tree.processTree(extractor);
        er_tree.deallocateTree();
    }

    return 0;
}

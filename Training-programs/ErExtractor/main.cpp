
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

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp> 
#include <boost/program_options.hpp>

#define SIZE 1024

class FirstStageDesc
{
public:
    FirstStageDesc(std::ostream * oss, char delim = ':')
        : output_writer_(oss, delim) { }

    
    void operator() (const ERRegion & err)
    {
        output_writer_.write(err.getFeatures());
    }

private:
    OutputWriter output_writer_;
};

using namespace std;

// string er1_conf_file = "conf/boostGeom.conf";
string er1_conf_file = "../boost_er1stage.conf";
// const string er2_conf_file = "conf/svmERGeom.conf";
string directory = "";


double min_area_ratio = 0.00007;
double max_area_ratio = 0.3;

string er1_neg = "";
string er2_neg = "";
string icdar_xml = "";
string icdar_dir = "";

int parseCmd(int argc, char ** argv)
{
    namespace po = boost::program_options;
    po::variables_map vm;
    po::options_description desc("Usage");
    desc.add_options()
        ("help,h","display help message")
        ("er1-conf-file", po::value<string>(&er1_conf_file),"path to er 1 stage Boosting conf")
        ("directory,d", po::value<string>(&directory),"output directory of extracted letters")
        ("min-area", po::value<double>(&min_area_ratio), "minimal area of nodes")
        ("max-area", po::value<double>(&max_area_ratio), "maximal area of node")
        ("first-stage", po::value<string>(&er1_neg), "output file with all negative descriptors")
        ("second-stage", po::value<string>(&er2_neg), "output file with all negative descriptors")
        ("icdar-xml", po::value<string>(&icdar_xml), "path to xml icdar file")
        ("icdar-dir", po::value<string>(&icdar_dir), "path to xml icdar file");
         
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

    if ( vm.count("help"))
    {
        std::cout << desc << std::endl;
        return 1;
    }

    return 0;
}

void saveComponents( std::vector<Component> &components, const std::string & file_name)
{
    int i = 0;
    for ( auto & c : components )
    {
        cv::Mat comp_img = c.getBinaryMat();
        std::stringstream ss;
        ss << directory << '/' << file_name << '_' << c.size() << '_' << i << ".jpg";
        ++i;
        cv::imwrite(ss.str(), comp_img);
    }
}

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

template <typename T>
void draw( const std::vector<T> &objects, 
        DrawerInterface * drawer)
{
    for ( const T & o : objects )
    {
        drawer->draw(o);
    }
}

int main( int argc, char **argv )
{
    if (parseCmd(argc, argv) != 0)
    {
        return 1;
    }

    string line;

    ERTree er_tree(min_area_ratio, max_area_ratio);
    er_tree.setMinGlobalProbability(0.2);
    er_tree.setMinDifference(0.1);

    std::unique_ptr<ERFilter1Stage> er_function( new ERFilter1Stage() );
    er_function->loadConfiguration(er1_conf_file);

    er_tree.setERFunction(std::move(er_function));
    ComponentTreeBuilder<ERTree> builder( &er_tree );

    std::unique_ptr<DrawerInterface> bin_drawer( new BinaryDrawer() );
    std::unique_ptr<DrawerInterface> rect_drawer( new RectangleDrawer() );

    std::ofstream er1_desc_out, er2_desc_out;

    if ( !er1_neg.empty() )
    {
        er1_desc_out.open(er1_neg);
    }

    if ( !er2_neg.empty() )
    {
        er2_desc_out.open(er2_neg);
    }

    while( std::getline(cin, line))
    {
        cv::Mat image = cv::imread(line, CV_LOAD_IMAGE_GRAYSCALE);
        string file_name = getFileName(line);

        cout << line << endl;

        er_tree.setImage(image);
        buildTree( er_tree, builder );
        // auto letter_comp = extractLetterComponent(er_tree, letter_testing, file_name);
        auto components = er_tree.toComponent();

        if ( !er1_neg.empty() )
        {
            OutputWriter output(&er1_desc_out);
            computeERDescriptor(er_tree, output);
        }

        er_tree.invertDomain();
        er_tree.deallocateTree();

        buildTree( er_tree, builder );
        auto tmp = er_tree.toComponent();

        // auto tmp = extractLetterComponent(er_tree, letter_testing, file_name);
        // letter_comp.insert(letter_comp.end(), tmp.begin(), tmp.end());
        components.insert(components.end(), tmp.begin(), tmp.end());
        
        cout << components.size() << endl;

        if (!er1_neg.empty() )
        {
            OutputWriter output(&er1_desc_out);
            computeERDescriptor(er_tree, output);
        }

        er_tree.deallocateTree();

        if (!er2_neg.empty())
        {
            OutputWriter output(&er2_desc_out);
            computeER2StageDesc(components, output);
        }

        BinaryDrawer binary_drawer;
        binary_drawer.init(image);
        draw(components, &binary_drawer);
        RectangleDrawer rectangle_drawer;
        rectangle_drawer.init(binary_drawer.getImage());
        draw(components, &rectangle_drawer);

        if (!directory.empty())
        {
            stringstream ss;
            ss << directory << "/" << file_name;
            cout << ss.str() << endl;
            cv::imwrite( ss.str(), rectangle_drawer.getImage());
        }
        else
        {
            gui::showImage(rectangle_drawer.getImage(), "image");
        }
    }

    return 0;
}

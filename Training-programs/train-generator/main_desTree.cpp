#include <nocrlib/swt_segmentation.h>
#include <nocrlib/train_data.h>
#include <nocrlib/ocr.h>
#include <nocrlib/extremal_region.h>

#include <libsvm/svm.h>

#include <vector>
#include <iostream>
#include <limits>
#include <string>

#include <opencv2/ml/ml.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>

#include <boost/program_options.hpp>
//using namespace cv;
using namespace std;


template <feature F>
void generate(const std::string & tests_file, const std::string & output)
{
    TrainDataCreator<F, extraction::BWMaxComponent> generator;
    generator.loadAndProcessSamples( tests_file, output );
}

int main( int argc, char** argv )
{

    std::string input, output;

    const std::string er_geom = "ergeom";
    const std::string er_geom1 = "ergeom-1";
    const std::string swt = "swt";
    const std::string hog = "hogOcr";

    namespace po = boost::program_options;
    po::options_description desc("Usage");
    desc.add_options()
        ("help,h","display help message")
        ("input,i", po::value<string>(&input),"list of paths to images")
        ("output,o", po::value<string>(&output), "specifies output file")
        (er_geom.c_str(), "generate first er geom phase descriptors")
        (er_geom1.c_str(), "generate second er geom phase descriptors")
        (swt.c_str(), "generate swt filtering phase descriptors")
        (hog.c_str(), "generate hog ocr descriptors");

    po::variables_map vm;

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
        return 0;
    }

    if (vm.count(er_geom.c_str()) )
    {
        generate<feature::ERGeom>(input, output);
        return 0;
    }

    if (vm.count(er_geom1.c_str()))
    {
        generate<feature::ERGeom1>(input, output);
        return 0;
    }

    if (vm.count(swt.c_str()))
    { 
        generate<feature::SwtGeom1>( input, output);
        return 0;
    }

    if (vm.count(hog.c_str()))
    {
        generate<feature::hogOcr>(input, output);
        return 0;
    }

}


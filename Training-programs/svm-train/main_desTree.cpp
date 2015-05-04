#include <nocrlib/swt_segmentation.h>
#include <nocrlib/component.h>
#include <nocrlib/classifier_wrap.h>
#include <nocrlib/drawer.h>
#include <nocrlib/feature_traits.h>
#include <nocrlib/key_point_extractor.h>
#include <nocrlib/bag_of_words.h>
#include <nocrlib/feature_factory.h>
#include <nocrlib/direction_histogram.h>
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

#define PROB_ESTIMATES "prob-estimates"


template <feature F>
void trainSVM(svm_parameter * param, const std::string & input, const std::string & output,
        const std::string &scaling )
{
    if (scaling.empty())
    {
        LibSVM<F> svm; 
        svm.train(input, param);
        svm.saveConfiguration(output);
    }
    else 
    {
        ScalingLibSVM<F> scaling_svm;
        scaling_svm.train(input, param);
        scaling_svm.saveConfiguration(output);
    }
}


int main( int argc, char** argv )
{
    std::string input, output;

    const std::string er_geom = "ergeom";
    const std::string er_geom1 = "ergeom-1";
    const std::string swt = "swt";
    const std::string hog = "hogOcr";

    svm_parameter param;
    param.svm_type = C_SVC;
    param.kernel_type = RBF;
    param.degree = 3;
    // param.gamma = 2;
    param.gamma = 1./144.;
    param.coef0 = 0;
    param.nu = 0.5;
    param.cache_size = 100;
    param.C = 4;
    param.eps = 1e-3;
    param.p = 0.1;
    param.shrinking = 0;
    param.probability = 0;
    param.nr_weight = 0;
    param.weight_label = NULL;
    param.weight = NULL;

    std::string scaling = "";
    bool prob_estimates = false;


    namespace po = boost::program_options;
    po::variables_map vm;
    po::options_description desc("Usage");
    desc.add_options()
        ("help,h","display help message")
        ("input,i", po::value<string>(&input),"list of paths to images")
        ("output,o", po::value<string>(&output), "specifies output file")
        (er_geom.c_str(), "first er geom phase descriptors")
        (er_geom1.c_str(), "second er geom phase descriptors")
        (swt.c_str(), "swt filtering phase descriptors")
        (hog.c_str(), "hog ocr descriptors")
        ("gamma", po::value<double>(&param.gamma), "gamma parameter for training")
        ("c-value", po::value<double>(&param.C), "C parameter for training")
        (PROB_ESTIMATES, "enable probability estimates")
        ("scaling", po::value<string>(&scaling), "turn on scaling of train data");

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

    if (vm.count(PROB_ESTIMATES))
    {
        param.probability = 1;
    }

    if (vm.count(er_geom.c_str()) )
    {
        trainSVM<feature::ERGeom>(&param, input, output, scaling);
        svm_destroy_param(&param);
        return 0;
    }

    if (vm.count(er_geom1.c_str()))
    {
        trainSVM<feature::ERGeom1>(&param, input, output, scaling);
        svm_destroy_param(&param);
        return 0;
    }



    if (vm.count(swt.c_str()))
    { 
        trainSVM<feature::SwtGeom1>(&param,  input, output, scaling);
        svm_destroy_param(&param);
        return 0;
    }

    if (vm.count(hog.c_str()))
    {
        trainSVM<feature::hogOcr>(&param, input, output, scaling);
        svm_destroy_param(&param);
        return 0;
    }
}


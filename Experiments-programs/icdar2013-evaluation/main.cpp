
#include <iostream>
#include <memory>
#include <utility>
#include <string>

#include <nocrlib/segment.h>
#include <nocrlib/opencv_mser.h>
#include <nocrlib/drawer.h>
#include <nocrlib/dictionary.h>
#include <nocrlib/word_generator.h>
#include <nocrlib/text_recognition.h>
#include <nocrlib/iooper.h>
#include <nocrlib/swt_segmentation.h>
#include <nocrlib/extremal_region.h>
#include <nocrlib/ground_truth_impl.h>
#include <nocrlib/utilities.h>
#include <nocrlib/structures.h>
#include <nocrlib/street_view_scene.h>
#include <nocrlib/word_deformation.h>
#include <nocrlib/knn_ocr.h>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>

#include <boost/program_options.hpp>

#define SIZE 1024
#define WORD_GENERATOR 0
#define IKSVM_OCR 0

using namespace std;

string er1_conf_file = "../boost_er1stage_handpicked.xml";
// const string er2_conf_file = "conf/svmERGeom.conf";
string er2_conf_file = "../scaled_svmEr2_handpicked.xml";

string icdar_test = "";
string svt = "";
string image_dir = "";
string xml_file = "";


int parseCmd(int argc, char ** argv)
{
    namespace po = boost::program_options;
    po::variables_map vm;
    po::options_description desc("Usage");
    desc.add_options()
        ("help,h","display help message")
        ("er1-conf-file", po::value<string>(&er2_conf_file),"path to er 1 stage Boosting conf")
        ("er2-conf-file", po::value<string>(&er2_conf_file),"path to er 2 stage SVM conf")
        ("test,t", po::value<string>(&icdar_test),"list of ICDAR test samples")
        ("directory,d", po::value<string>(&image_dir), "path to output directory")
        ("xml,x", po::value<string>(&xml_file), "enable xml evaluation output");
    
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

    return 0;
}


template <typename T, typename DRAWER> 
void drawAndSave( const std::vector<T> &objects, 
        const cv::Mat &image,
        DRAWER &drawer, const std::string & file_name,
        const std::vector<bool> & results)
{
    drawer.init(image);

    for (std::size_t i = 0; i < objects.size(); ++i) 
    {
        cv::Scalar color = results[i] ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255);

        drawer.setColor(color);
        drawer.draw(objects[i].visual_information_);
    }

    ImageSaver saver;
    saver.saveImage(image_dir + '/' + file_name, drawer.getImage());
}

template <typename M, typename OCR>
std::vector<TranslatedWord> runTest(
        Segment<M, OCR> & segmentation,
        Testing & testing,
        const LetterWordEquiv & word_equiv,
        const Dictionary & dict, 
        const cv::Mat & image,
        const std::string & file_name)
{
    auto words = recognizeWords(segmentation, word_equiv, 
            dict, image);

    auto results = testing.updateScores( file_name, words );
    

#if WORD_GENERATOR
    WordDeformation word_deformation;
    word_deformation. setImage(image);
    for (std::size_t i = 0; i < words.size(); ++i) 
    {
        auto desc = word_deformation.getDescriptor(words[i].visual_information_);        
        for (float f  : desc)
        {
            std::cout << f << " ";
        }
        std::cout << std::endl;
    }

#endif

    if (!image_dir.empty())
    {
        RectangleDrawer drawer;
        drawAndSave(words, image, drawer, file_name, results);
    }

    return words;
}


        
int main( int argc, char **argv )
{
    parseCmd(argc, argv);
    
    // const string ocr_conf = "../conf/iksvm.conf";
    const string dict = "../conf/dict";
    const string merge_conf = "../conf/svmMerge.conf";

#if IKSVM_OCR
    const std::string ocr_conf = "../conf/iksvm.conf";
    unique_ptr<MyOCR> ocr( new MyOCR(ocr_conf) );
    Segment<ERTextDetection, MyOCR> er_text_segment;

#else
    const std::string ocr_conf = "../training/svm_hog.xml";
    unique_ptr<AbstractOCR> ocr( new HogRBFOcr(ocr_conf) );
    Segment<ERTextDetection, AbstractOCR> er_text_segment;
#endif

    // const string er1_conf_file = "conf/boostGeom.conf";

    // Segment<CvMSERDetection> segmentation; 
    //
    // CvMSERDetection detection_method;
    // detection_method.loadConfiguration(filter_conf);
    //
    // segmentation.loadMethod( &detection_method );

    //
    // segmentation.loadOcr( ocr.get() );

    LetterWordEquiv word_equivalence( merge_conf );

    Dictionary dictionary(dict);

    // Testing mser_testing;

    auto decider = std::make_shared<TruePositiveTest>();
    // mser_testing.setTruePositiveDecider(decider.get());

    ERTextDetection er_text_detection( er1_conf_file, er2_conf_file );
    er_text_segment.loadMethod(&er_text_detection);
    er_text_segment.loadOcr( ocr.get() );

    Testing er_testing;
    er_testing.setTruePositiveDecider(decider.get());

    Resizer resizer;
    resizer.setSize(SIZE);

    if ( !icdar_test.empty() )
    {
        XmlGroundTruth xml;
        xml.loadData("icdar2013.xml");

        // mser_testing.loadGroundTruth(&xml);
        er_testing.loadGroundTruth(&xml);

        loader ld;
        auto input = ld.getFileContent(icdar_test);
        for ( const string &file_path : input )
        {
            cout << file_path << endl;
            cv::Mat image = cv::imread( file_path, CV_LOAD_IMAGE_COLOR);
            string file_name = getFileName( file_path );

            if ( image.rows < SIZE && image.cols < SIZE )
            {
                image = resizer.resizeKeepAspectRatio(image);
                er_testing.notifyResize(file_name, resizer.getLastScale());
            }


            runTest( er_text_segment, er_testing, word_equivalence, dictionary, image, file_name );
            // runTest( segmentation, mser_testing, word_equivalence, dictionary, image, file_name );
        }

    }

    // cout << "MSER results:" << endl;
    // mser_testing.makeRecord( cout );
    cout << "ER results:" << endl;
    er_testing.makeRecord( cout );
    if (xml_file.empty())
    {
        er_testing.printXmlOutput(cout);
    }
    else
    {
        std::ofstream ofs(xml_file);
        er_testing.printXmlOutput(ofs);
        ofs.close();

        EvaluationDrawer eval_drawer;
        eval_drawer.loadXml(xml_file);
    }

    
    return 0;
}

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>
#include <vector>
#include <utility>
#include <chrono>
#include <algorithm>
#include <exception>
#include <ostream>

#include <nocrlib/structures.h>
#include <nocrlib/drawer.h>
#include <nocrlib/dictionary.h>
#include <nocrlib/ocr.h>
#include <nocrlib/word_generator.h>
#include <nocrlib/text_recognition.h>

#include "xml_creator.h"
#include "recorder_interface.h"
#include "translation_recorder.h"

#include <boost/program_options.hpp>


using namespace std;
using namespace cv;

int main ( int argc, char** argv ) 
{
    // ====================== setting up command line parameters ====================
    std::string output;
    std::string dict = "conf/dict";
    vector<string> input_lists;


    std::string svm_ER2Phase = "scaled_svmEr2_handpicked.xml";
    namespace po = boost::program_options;
    po::options_description desc("Usage");
    desc.add_options()
        ("help,h","display help message")
        ("input,i", po::value< vector<string> >(&input_lists),"list of paths to images")
        ("output,o", po::value<std::string>(&output), "specifies output file")
        ("dictionary,d", po::value<std::string>(&dict), "specifies dictionary")
        ("xml","enable xml output")
        ("display-words", "enable displaying of detected words")
        ("display-letters", "enable displaying of detected letters")
        ("svm-er-2stage", po::value<string>(&svm_ER2Phase), "specifies svm config path");
    

    po::variables_map vm;
    std::vector<std::string> input;

    try 
    {
        po::parsed_options parsed = po::parse_command_line(argc, argv, desc);
        po::store( parsed , vm ); 

        po::notify(vm);
        input = po::collect_unrecognized( parsed.options, po::include_positional );
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

    std::unique_ptr<RecorderInterface> recorder( new TranslationRecorder() );
    if ( vm.count("xml") )
    {
        recorder = std::unique_ptr<XmlCreator>( new XmlCreator() );
    }

    bool display_letters = vm.count("display-letters") != 0; 
    bool display_words = vm.count("display-words") != 0;

    std::ostream *oss = &std::cout;
    if ( !output.empty() )
    {
        oss = new std::ofstream(output);
    }

    loader ld;
    for ( const string &input_list: input_lists )
    {
        auto tmp = ld.getFileContent(input_list);
        input.insert( input.end(), tmp.begin(), tmp.end() );
    }


    //==================== recognize text from input images =========================
    // const std::string boost_ER1Phase = "conf/boostGeom.conf";
    // std::string boost_ER1Phase = "boost_er1stage.conf";
    std::string boost_ER1Phase = "boost_er1stage_handpicked.xml";
    // const std::string svm_ER2Phase = "conf/svmERGeom.conf";
    // std::string svm_ER2Phase = "svm_er2stage.conf";
    const std::string svm_merge = "conf/svmMerge.conf";
    // const std::string ocr_conf = "conf/iksvm.conf";
    // const std::string ocr_conf = "training/train_hog_ocr.xml";

    const std::string ocr_conf = "training/svm_hog.xml";
    cout << svm_ER2Phase << endl;
    TextRecognition<ERTextDetection, AbstractOCR> image_reader;
    image_reader.setShowingLetters( display_letters );
    image_reader.setShowingWords( display_words );
    try 
    {
        Dictionary dictionary(dict);
        // image_reader.loadConfiguration( boost_ER1Phase, svm_ER2Phase, svm_merge );
        image_reader.constructExtractionMethod( boost_ER1Phase, svm_ER2Phase);
        image_reader.loadEquivConfiguration(svm_merge);
        // unique_ptr<MyOCR> ocr( new MyOCR(ocr_conf) );
        unique_ptr<AbstractOCR> ocr( new HogRBFOcr(ocr_conf) );
        image_reader.loadOcr( ocr.get() );
        
        for ( const std::string &file_path : input )
        {
            vector<TranslatedWord> words = image_reader.recognize( file_path, 
                    dictionary );
            recorder->makeRecord( file_path, words ); 
        }

        recorder->save( *oss ); 
    }
    catch ( std::exception &exp )
    {
        cerr << "exception thrown during text recognition:" << endl;
        cerr << exp.what();
    }

    if ( !output.empty() )
    {
        delete oss;
    }

    return 0;
}

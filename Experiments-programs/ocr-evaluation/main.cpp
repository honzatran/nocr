
#include <iostream>
#include <memory>
#include <utility>
#include <string>
#include <sstream>
#include <algorithm>
#include <map>
#include <vector>
#include <fstream>
#include <limits>

#include <nocrlib/extremal_region.h>
#include <nocrlib/component_tree_builder.h>
#include <nocrlib/component.h>
#include <nocrlib/utilities.h>
#include <nocrlib/feature_factory.h>
#include <nocrlib/iooper.h>
#include <nocrlib/street_view_scene.h>
#include <nocrlib/assert.h>
#include <nocrlib/features.h>
#include <nocrlib/extremal_region.h>
#include <nocrlib/ocr.h>
#include <nocrlib/knn_ocr.h>
#include <nocrlib/text_recognition.h>
#include <nocrlib/exception.h>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp> 
#include <boost/program_options.hpp>

#include "../common/segmentation_base.hpp"

#define IKSVM_OCR 0
#define DEBUG 1



#define SIZE 1024
#define ER1_CONF_FILE "../boost_er1stage_handpicked.xml"
#define ER2_CONF_FILE "../scaled_svmEr2_handpicked.xml"


using namespace std;

string lalpha = "1032547698ACBEDGFIHKJMLNQPOSRUTWVYXZacbedgfihkjmlonqpsrutwvyxz";

class OcrSegmentation : public LetterSegmentBase
{
public:
    std::vector<bool> filter(const std::vector<Letter> & letters, const std::string & file_name);
};

std::vector<bool>
OcrSegmentation::filter(const std::vector<Letter> & letters, const std::string & file_name)
{
    vector<bool> mask(letters.size(), false);
    auto curr_it = ground_truth_.find(file_name);
    if (curr_it == ground_truth_.end())
    {
        throw FileNotFoundException("no record for " + file_name);
    }

    vector<cv::Rect> & gt_rects = curr_it->second;

    for (std::size_t i = 0; i < letters.size(); ++i)
    {
        cv::Rect l_rect = letters[i].getRectangle();
        auto it = std::find_if(gt_rects.begin(), gt_rects.end(), 
                [this, &l_rect](const cv::Rect & rect) -> bool
                {
                    return areMatching(l_rect, rect);
                });

        if (it != gt_rects.end())
        {
            mask[i] = true;
            cout << "detected " << endl;
        }
    }

    return mask;
}

int main( int argc, char **argv )
{
    string line;

#if IKSVM_OCR
    const std::string ocr_conf = "../conf/iksvm.conf";
    unique_ptr<MyOCR> ocr( new MyOCR(ocr_conf) );

    Segment<ERTextDetection, MyOCR> segmentation;
#else
    const std::string ocr_conf = "../training/svm_hog.xml";
    unique_ptr<AbstractOCR> ocr( new HogRBFOcr(ocr_conf) );

    Segment<ERTextDetection, AbstractOCR> segmentation;
#endif
    // const std::string ocr_conf = "../training/ocr_dist_data";
    //
    ERTextDetection er_text_detection(ER1_CONF_FILE, ER2_CONF_FILE);
    segmentation.loadMethod(&er_text_detection);

    // unique_ptr<AbstractOCR> ocr( new KNNOcr(ocr_conf) );
    segmentation.loadOcr(ocr.get());
    Resizer resizer;
    resizer.setSize(SIZE);

    OcrSegmentation ocr_gt;
    ocr_gt.loadGroundTruthXML(argv[1]);

    while (std::getline(cin, line))
    {
        string file_path, gt_file_name;
        std::tie(file_path, gt_file_name) = parse(line);

        cv::Mat image = cv::imread(file_path, CV_LOAD_IMAGE_COLOR);
        if (image.rows < SIZE && image.cols < SIZE)
        {
            image = resizer.resizeKeepAspectRatio(image);
            ocr_gt.notifyResize(gt_file_name, resizer.getLastScale());
        }

        auto letters = segmentation.segment(image);

        auto mask = ocr_gt.filter(letters, gt_file_name);

        for (std::size_t i = 0; i < mask.size(); ++i) 
        {
            if (mask[i])
            {
                char c = letters[i].getTranslation();

                std::cout << c << " " <<
                    letters[i].getConfidence() << std::endl;
                gui::showImage(letters[i].getBinaryMat(), "letter");
            }
        }

    }


    return 0;
}

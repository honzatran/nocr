
#include "LetterExtractor.h"
#include <nocrlib/iooper.h>

#include <opencv2/core/core.hpp>

#define SIZE 1024

LetterExtractor::LetterExtractor()
{
    directory_ = "";
}

LetterExtractor::storeGroundTruth( GroundTruthInterface * gt_ptr )
{
    letter_testing_.storeGroundTruth(gt_ptr);
}

LetterExtractor::setDirectory(const std::string & directory)
{
    if (directory.back() != '/')
    {
        directory += '/';
    }

    directory_ = directory;
}

LetterExtractor::storeLetters(
        const std::string &file_list, 
        AbstractFeatureExtractor * extractor)
{
    loader ld;
    auto content = ld.getFileContent(file_list);

    Resizer resizer(SIZE);
    for (const string & s: content)
    {
        cv::Mat image = cv::imread(s, CV_LOAD_IMAGE_GRAYSCALE);
        if (image.rows < SIZE && image.cols < SIZE)
        {
            image = resizer.resizeKeepAspectRatio(image);
        }



    }
}






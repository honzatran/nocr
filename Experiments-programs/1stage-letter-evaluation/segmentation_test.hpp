
#ifndef _SEGMENTATION_TEST_HPP
#define _SEGMENTATION_TEST_HPP

#include <iostream>
#include <string>
#include <memory>
#include <vector>
#include <fstream>
#include <sstream>

#include <pugi/pugixml.hpp>

#include <nocrlib/component.h>
#include <nocrlib/extremal_region.h>
#include <nocrlib/component_tree_builder.h>
#include <nocrlib/testing.h>
#include <nocrlib/iooper.h>
#include <nocrlib/assert.h>
#include <nocrlib/classifier_wrap.h>

#include <opencv2/core/core.hpp>



class LetterSegmentTesting
{
public:
    LetterSegmentTesting() 
        : true_positives_(0), number_results_(0), number_ground_truth_(0) { }

    ~LetterSegmentTesting() { };

    void loadGroundTruthXML(const std::string & xml_gt_file);

    void updateScores( const std::string &image_name, 
            const std::vector<Component> &letters );

    double getPrecision() const;

    double getRecall() const;

    void makeRecord( std::ostream &oss ) const;

    void setImageName(const std::string & image_name, const cv::Mat & image);

    void virtual operator() (const ERRegion & err);

    bool operator== (const LetterSegmentTesting & other)
    {
        return number_results_ == other.number_results_ 
            && number_ground_truth_ == other.number_ground_truth_
            && true_positives_ == other.true_positives_;
    }

    cv::Mat getCurrentImage()
    {
        return curr_img_;
    }

    void notifyResize(const std::string & name, double scale);

protected:
    std::map<std::string, std::vector<cv::Rect> > ground_truth_;

    bool areMatching(const cv::Rect & c_rect, const cv::Rect & gt_rect);
    void draw(const Component & c);

    unsigned int true_positives_;
    unsigned int number_results_;
    unsigned int number_ground_truth_;

    decltype(ground_truth_.begin()) curr_image_it_;
    cv::Mat curr_img_;

    std::vector<bool> detected_;
};

class SecondStageTesting : public LetterSegmentTesting
{
public:
    SecondStageTesting();

    void setSvmConfiguration(const std::string & svm_conf);
    void operator() (const ERRegion & err) override;
private:
    std::unique_ptr<AbstractFeatureExtractor> extractor_;
    std::ofstream ofs_;

    void drawError(const Component & c);

    ScalingLibSVM<feature::ERGeom1> svm_;
    Boost<feature::ERGeom1> boost_;
};



#endif

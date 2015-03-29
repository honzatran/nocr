
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <map>
#include <cmath>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <nocrlib/train_data.h>
#include <nocrlib/component.h>
#include <nocrlib/features.h>
#include <nocrlib/feature_factory.h>
#include <nocrlib/classifier_wrap.h>
#include <nocrlib/utilities.h>

#define MIN_PROB 0.2

using namespace std;


struct LetterRecord
{
    LetterRecord() : count(0), detected(0) { }
    std::vector<double> scores;
    int count;
    int detected;
};

std::pair<char, string> getLetter(const string & line)
{
    size_t delim = line.find(':');
    string path = line.substr(0, delim);
    char c = line.back();
    
    return make_pair(c, path);
}



int main(int argc, char ** argv)
{
    const string k_boost_conf = "../boost_er1stage.conf";

    string line;
    ERFilter1StageFactory factory;
    auto ptr = factory.createFeatureExtractor();


    auto boosting = cv::ml::StatModel::create<cv::ml::Boost>(k_boost_conf);

    std::map<char, LetterRecord> letter_records;
    cout << std::fixed;

    while (getline(cin,line))
    {
        string path;
        char letter_char;

        std::tie(letter_char, path) = getLetter(line);

        auto letter_it = letter_records.find(letter_char);

        if (letter_it == letter_records.end())
        {
            auto it = letter_records.insert(std::make_pair(letter_char, LetterRecord()));
            letter_it = it.first;
        }

        cv::Mat image = cv::imread(path, cv::IMREAD_GRAYSCALE);

        if (image.empty())
        {
            continue;
        }

        TrainExtractionPolicy<extraction::BWMaxComponent> policy;
        Component c = policy.extract(image)[0];
        vector<float> desc = ptr->compute(c);
        float sum = boosting->predict(cv::Mat(desc), cv::noArray(), cv::ml::Boost::PREDICT_SUM| cv::ml::Boost::RAW_OUTPUT);

        float prob = 1/(1 + std::exp( -2 * sum ));
        cout << getFileName(line) << " " << prob << endl;

        if (prob > MIN_PROB)
        {
            letter_it->second.detected++;
        }

        letter_it->second.count++;

        letter_it->second.scores.push_back(prob);
    }


    for (const auto & p : letter_records)
    {
        auto scores = p.second.scores;

        auto it_med = scores.begin() + scores.size()/2;
        std::nth_element(scores.begin(), 
                it_med, scores.end());

        double detected_ratio = (double)p.second.detected/p.second.count;
        cout << "letter: " << p.first << " " << detected_ratio 
            << " median: " << *it_med << endl;
        
    }
}


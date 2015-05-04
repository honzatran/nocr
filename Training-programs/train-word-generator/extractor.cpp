
#include <pugi/pugixml.hpp>
#include "extractor.hpp"
#include <algorithm>
#include <cmath>

#include <nocrlib/assert.h>
#include <nocrlib/exception.h>
#include <nocrlib/utilities.h>
#include <nocrlib/word_deformation.h>

#include <opencv2/core/core.hpp>

#define NEGATIVE_RANDOM 2
#define PRINT_INFO 1

void 
Extractor ::loadXml(const std::string & xml_file)
{
    std::ifstream ifs; 
    ifs.open(xml_file);

    pugi::xml_parse_result result = doc_.load( ifs );

    if (!result)
    {
        throw FileNotFoundException(xml_file + " not found");
    }

    pugi::xml_node root = doc_.first_child();
    pugi::xml_node gt_train = root.first_child();

    for(pugi::xml_node img_node : gt_train.children("image"))
    {
        string img_name = getFileName(img_node.child("image-path").child_value());


        gt_records_.insert(std::make_pair(img_name, img_node));
    }
}

void 
Extractor::setUpErTree(const std::string & er1_conf_file, const std::string & er2_conf_file)
{
    er_tree_.setMinGlobalProbability(0.2);
    er_tree_.setMinDifference(0.1);
    er_tree_.setDelta(8);

    std::unique_ptr<ERFilter1Stage> er_function( new ERFilter1Stage() );
    er_function->loadConfiguration(er1_conf_file);

    er_tree_.setERFunction(std::move(er_function));
    er_tree_.loadSecondStageConf(er2_conf_file);
}

void 
Extractor::findImage(const std::string & file_name)
{
    auto it = gt_records_.find(file_name);
    if (it == gt_records_.end())
    {
        std::cout << "no record for image " << file_name << std::endl;
        return;
    }

    pugi::xml_node img_node = it->second;
    pugi::xml_node gt_node = img_node.child("image-gt");

    auto words = getWords(gt_node);
    string gt_path = gt_node.child("image-gt-path").text().as_string();
    string img_path = img_node.child("image-path").text().as_string();

    cv::Mat gt_img = cv::imread(gt_path, CV_LOAD_IMAGE_COLOR);
    cv::Mat img = cv::imread(img_path, CV_LOAD_IMAGE_COLOR);

    WordDeformation word_deformation;
    word_deformation.setImage(img);

    std::vector< Component > word_components;
    std::vector<cv::Rect> rectangles;

    std::size_t min_length = words[0].letters.size();
    std::size_t max_length = min_length;

    for (std::size_t i = 0; i < words.size(); ++i)
    {
        auto components = extractWord(words[i], gt_img);
        min_length = std::min(min_length, components.size());
        max_length = std::max(max_length, components.size());
        cv::Rect word_rect = getRectangle(words[i].letters);
        
        word_components.insert(word_components.end(), components.begin(), components.end());
        
        if (words[i].letters.size() > 2)
        {
            auto desc = word_deformation.getDescriptor(components);
            // for (float f : desc) 
            // {
            //     *os_ << f << " ";
            // }
            // *os_ << 1 <<  endl;

        }

        rectangles.push_back(word_rect);
    }



    vector<bool> mask;
    if (words.size() > 1)
    {
        std::size_t avg_length = word_components.size()/words.size() + 1;
        vector<Component> random_components = randomNegative(word_components, rectangles, avg_length, mask);
        cv::Rect random_rect = getRectangle(random_components);

        for (const auto & c : random_components)
        {
            cv::rectangle(img, c.rectangle(), cv::Scalar(0, 0, 255));
        }

        if (random_components.size() > 2)
        {
            auto desc = word_deformation.getDescriptor(random_components);
            // for (float f : desc) 
            // {
            //     *os_ << f << " ";
            // }
            // *os_ << 0 << endl;
        }
    }

    ComponentExtractor component_extractor;
    process<true, true>(er_tree_, img, component_extractor);
    auto components = component_extractor.getExtractedComponents();
    mask = vector<bool>(components.size(), false);
    std::size_t tmp = max_length - min_length;

    for (int i = 0; i < NEGATIVE_RANDOM; ++i)
    {
        std::size_t length = tmp == 0 ? min_length : min_length + dist_(generator_) % tmp;
        auto negative_components = randomNegative(components, rectangles, length, mask);
        if (negative_components.size() > 2)
        {
            auto desc = word_deformation.getDescriptor(negative_components);
            // for (float f : desc) 
            // {
            //     *os_ << f << " ";
            // }
            // *os_ << 0 << endl;
        }
    }

    min_length = std::max(min_length, (std::size_t)3);

    for (int i = 0; i < NEGATIVE_RANDOM; ++i)
    {
        std::size_t length = tmp == 0 ? min_length : min_length + dist_(generator_) % tmp;
        auto negative_components = randomNegative(components, rectangles, length, mask);
        if (negative_components.size() > 2)
        {
            auto desc = word_deformation.getDescriptor(negative_components);
            // for (float f : desc) 
            // {
            //     *os_ << f << " ";
            // }
            // *os_ << 0 << endl;
        }
    }
}

auto 
Extractor::getWords(const pugi::xml_node & gt_node) 
    -> std::vector<WordRecord>
{
    std::vector<WordRecord> words;

    for (auto & node : gt_node.children("word"))
    {
        string name = node.attribute("text").as_string();
        std::vector<LetterRecord> letters;

        for (const auto & letter : node.children("letter"))
        {
            char c = letter.child_value()[0];
            int b = letter.attribute("b").as_int();
            int g = letter.attribute("g").as_int();
            int r = letter.attribute("r").as_int();

            cv::Point center;
            center.x = letter.attribute("center-x").as_int();
            center.y = letter.attribute("center-y").as_int();

            cv::Point tl, br;
            tl.x = letter.attribute("tl-x").as_int();
            tl.y = letter.attribute("tl-y").as_int();

            br.x = letter.attribute("br-x").as_int();
            br.y = letter.attribute("br-y").as_int();

            cv::Vec3b color;
            color[0] = b;
            color[1] = g;
            color[2] = r;

            cv::Rect rect(tl, br);
            letters.push_back({ color, rect, center, c});
        }

        words.push_back({ name, letters });
    }

    return words;
}

std::vector<Component> 
Extractor::extractWord(const WordRecord & word_record, 
        const cv::Mat & gt_img)
{
    auto letters = word_record.letters;
    std::vector<Component> components;

    for (const auto & lr : letters)
    {
        cv::Mat cropped_image = gt_img(lr.rect);
        Component l_component;

        for (auto it = cropped_image.begin<cv::Vec3b>(); it != cropped_image.end<cv::Vec3b>(); ++it) 
        {
            if (*it  == lr.color)
            {
                cv::Point p = it.pos() + lr.rect.tl();
                l_component.addPoint(p);
            }
        }

        components.push_back(l_component);
    }

    std::sort(components.begin(), components.end(), 
            [](const Component & a, const Component & b)
            {
                int a_left = a.getLeft();
                int b_left = b.getLeft();

                if (a_left == b_left) return  a.getUpper() < b.getUpper();
                else return a_left < b_left;
            });

#if PRINT_INFO
    if (components.size() > 2)
    {
        for (std::size_t i = 0; i < components.size() - 2; ++i)
        {
            cv::Rect a = components[i].rectangle();
            cv::Rect b = components[i + 1].rectangle();
            cv::Rect c = components[i + 2].rectangle();

            cv::Rect r = a | b | c;

            double angle_tl = angle(a.tl(), b.tl(), c.tl());
            double angle_bl = angle(a.tl() + cv::Point(0, a.height), 
                    b.tl() + cv::Point(0, b.height), c.tl() + cv::Point(0,c.height));

            double max_angle = std::max(angle_bl, angle_tl);
            ImageSaver saver;
            if (max_angle < 2.5)
            {
                cout << max_angle << endl;
                stringstream ss;
                ss << "angle_" << id++ << "_" << max_angle << ".jpg";
                saver.saveImage(ss.str(), gt_img(r));
            }
        }
    }
#endif

    return components;
}

cv::Vec3f 
Extractor::getColorInformation(
        const WordRecord & word_record, 
        const std::vector<Component> & components, 
        const cv::Mat & img)
{
    cv::Vec3f sums(0, 0, 0);
    cv::Vec3f sums_square(0, 0, 0);
    std::size_t k = 0;
    for (const auto & c : components)
    {
        auto points = c.getPoints();
        for (const cv::Point & p : points)
        {
            cv::Vec3b pixel_val = img.at<cv::Vec3b>(p.y, p.x);
            sums += pixel_val;
            sums_square[0] += (int)pixel_val[0] * (int)pixel_val[0];
            sums_square[1] += (int)pixel_val[1] * (int)pixel_val[1];
            sums_square[2] += (int)pixel_val[2] * (int)pixel_val[2];
        }

        k += points.size();
    }

    cv::Vec3f means = sums*(1./k);
    cv::Vec3f variance(0, 0, 0);

    variance[0] = k * means[0] * means[0] - 2 * means[0] * sums[0];
    variance[1] = k * means[1] * means[1] - 2 * means[1] * sums[1];
    variance[2] = k * means[2] * means[2] - 2 * means[2] * sums[2];

    variance += sums_square;
    variance *= (double)1/(k - 1);

    cv::Vec3f coef_variation;
    coef_variation[0] = std::sqrt(variance[0])/means[0];
    coef_variation[1] = std::sqrt(variance[1])/means[1];
    coef_variation[2] = std::sqrt(variance[2])/means[2];

    double fact = 1. + 1./(4 * k);
    
    return coef_variation*fact;
}

float
Extractor::getAngle(cv::Point p1, cv::Point p2, cv::Point p3)
{
    cv::Point v1 = p1 - p2;
    cv::Point v2 = p3 - p2;

    float dot_product = v1.x * v2.x + v1.y * v2.y;


    return std::acos(dot_product/(cv::norm(v1)*cv::norm(v2)));
}

float
Extractor::coefVarianceDist(const WordRecord & word_record)
{
    auto letters = word_record.letters;
    float sum_dist = 0;
    float sum_sqr_dist = 0;

    for (std::size_t i = 1; i < letters.size(); ++i) 
    {
        float dist = cv::norm(letters[i-1].center - letters[i].center);
        sum_dist += dist;
        sum_sqr_dist += dist * dist;
    }

    std::size_t k = letters.size();
    float mean = sum_dist/k;
    float variance = (sum_sqr_dist + k * mean * mean - 2 * mean * sum_dist)/(k-1);
    
    return (1.f + 1.f/(4*k))*std::sqrt(variance)/mean;
}

cv::Rect
Extractor::getRectangle(const std::vector<LetterRecord> & letters)
{
    cv::Rect r = letters.front().rect;

    for (std::size_t i = 1; i < letters.size(); ++i) 
    {
        r |= letters[i].rect;
    }

    return r;
}

cv::Rect
Extractor::getRectangle(const std::vector<Component> & components)
{
    cv::Rect r = components.front().rectangle();

    for (std::size_t i = 1; i < components.size(); ++i) 
    {
        r |= components[i].rectangle();
    }

    return r;
}

std::vector<Component> 
Extractor::randomNegative(const std::vector< Component > & components,
        const std::vector<cv::Rect> & rectangles, std::size_t length, 
        std::vector<bool> & mask)
{
    std::size_t letter_count = components.size();
    vector<Component> negative;
    negative.reserve(length);
    if (mask.size() != components.size())
    {
        mask = std::vector<bool>(components.size(), false);
    }


    for (std::size_t i = 0; i < length; ++i)
    {
        std::size_t random = dist_(generator_) % letter_count;
        while (mask[random])
        {
            random = dist_(generator_) % letter_count;
        }

        mask[random] = true;

        negative.push_back(components[random]);
    }

    std::sort(negative.begin(), negative.end(), 
            [](const Component & a, const Component & b)
            {
                int a_left = a.getLeft();
                int b_left = b.getLeft();

                if (a_left == b_left) return  a.getUpper() < b.getUpper();
                else return a_left < b_left;
            });

    return negative;
}




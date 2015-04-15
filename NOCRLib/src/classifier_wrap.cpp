/* 
 * Tran Tuan Hiep
 * Implementation of methods and classes declared in classifier_wrap.h
 *
 * Compiler: g++ 4.8.3, 
 */

#include "../include/nocrlib/classifier_wrap.h"

#include <libsvm/svm.h>
#include <liblinear/linear.h>

#include <opencv2/core/core.hpp>

#include <vector>
#include <memory>
#include <string>
#include <functional>
#include <algorithm>
#include <sstream>
#include <fstream>


#define ROOT_TAG "lib-svm"
#define SVM_MODEL_TAG "svm-model"
#define SVM_TYPE_TAG "svm-type"
#define NR_CLASS_TAG "nr-class"
#define KERNEL_TYPE_TAG "kernel_type"
#define GAMMA_TAG "gamma"
#define DEGREE_TAG "degree"
#define COEF0_TAG "coef0"
#define RHO_TAG "rho"
#define LABEL_TAG "label"
#define PROB_A_TAG "probA"
#define PROB_B_TAG "probB"
#define NR_SV_TAG "nr-sv"
#define TOTAL_SV_TAG "total-sv"
#define SVS_TAG "support-vectors"
#define SV_TAG "sv"
#define SV_COEF_TAG "sv-coef"
#define SV_POINT_TAG "sv-point"
#define INDX_TAG "indx"
#define COUNT_TAG "count"

#define SCALERS_TAG "scalers"
#define SCALER_TAG "scaler"
#define MIN_TAG "min"
#define INTERVAL_LENGTH_TAG "interval-length"
#define SCALED_MIN_TAG "scaled-min"
#define SCALED_INTERVAL_LENGTH_TAG "scaled-interval-length"


using namespace std;

svm_model* LibSVMTrainBridge::train( const cv::Mat &train_data, const cv::Mat &labels, 
        svm_parameter *params )
{
    constructProblem( train_data, labels ); 
    std::cout << "start training" << std::endl;
    
    cout << problem_->l << endl;
    svm_model *model = svm_train( problem_, params );

    /*
     * std::cout << "delete done" << std::endl;
     * delete[] problem.y;
     * for ( int i = 0; i < train_data.rows; ++i )
     * {
     *     delete[] problem.x[i];
     * }
     * delete[] problem.x;
     */

    return model; 
}

void LibSVMTrainBridge::constructProblem
    ( const cv::Mat &train_data, const cv::Mat &labels )
{
    problem_ = new svm_problem;
    problem_->l = train_data.rows;
    problem_->y = new double[train_data.rows]; 
    problem_->x = new svm_node*[train_data.rows];
    for ( int i = 0; i < train_data.rows; ++i )
    {
        problem_->y[i] = labels.at<float>(i,0);
        problem_->x[i] = new svm_node[train_data.cols+1];
        for ( int j = 0; j < train_data.cols; ++j )
        {
            problem_->x[i][j] = { j, train_data.at<float>(i,j) }; 
        }

        problem_->x[i][train_data.cols] = { -1 , 25 };
    }
}

svm_node* LibSVMTrainBridge::constructSample( const std::vector<float> &data ) const
{
    svm_node *sample = new svm_node[data.size()+1];
    for ( size_t i = 0; i < data.size(); ++i )
    {
        sample[i].index = i;
        sample[i].value = data[i];
    }
    sample[data.size()].index = -1;
    return sample;
}

void LibSVMTrainBridge::constructSample( const std::vector<float> &data, svm_node * nodes) const
{
    for ( size_t i = 0; i < data.size(); ++i )
    {
        nodes[i].index = i;
        nodes[i].value = data[i];
    }
    nodes[data.size()].index = -1;
}

void LibSVMTrainBridge::save(const std::string & file_name, const svm_model * model)
{
    std::ofstream ofs(file_name);
    if (!ofs)
    {
        //throw exception
    }

    pugi::xml_document doc;
    auto root = doc.append_child(ROOT_TAG);
    auto svm_node = root.append_child(SVM_MODEL_TAG);
    saveSvmModel(svm_node, model);
    
    doc.save(ofs);
}

void LibSVMTrainBridge::save(const std::string & file_name, const svm_model * model,
        const std::vector<FeatureScaler> & scalers)
{
    std::ofstream ofs(file_name);
    if (!ofs)
    {
        //throw exception
    }

    pugi::xml_document doc;
    auto root = doc.append_child(ROOT_TAG);
    auto svm_node = root.append_child(SVM_MODEL_TAG);
    saveSvmModel(svm_node, model);

    auto scalers_node = root.append_child(SCALERS_TAG);

    for (std::size_t i = 0;i < scalers.size(); ++i)
    {
        auto s_node = scalers_node.append_child(SCALER_TAG);
        s_node.append_attribute(INDX_TAG).set_value(i);
        
        s_node.append_child(MIN_TAG).text().set(scalers[i].min_);
        s_node.append_child(INTERVAL_LENGTH_TAG).text().set(scalers[i].interval_length_);

        s_node.append_child(SCALED_MIN_TAG).text().set(scalers[i].scaled_min_);
        s_node.append_child(SCALED_INTERVAL_LENGTH_TAG).text()
            .set(scalers[i].scaled_interval_length_);
    }

    
    doc.save(ofs);
}

void LibSVMTrainBridge::saveSvmModel(pugi::xml_node & xml_svm_node, 
        const svm_model * model)
{
    const svm_parameter& param = model->param;

    xml_svm_node.append_child(SVM_TYPE_TAG)
        .text().set(svm_get_type(param.svm_type));

    xml_svm_node.append_child(KERNEL_TYPE_TAG)
        .text().set(svm_get_kernel_type(param.kernel_type));

    if(param.kernel_type == POLY)
    {
        xml_svm_node.append_child(DEGREE_TAG)
            .text().set(param.degree);
    }

    if(param.kernel_type == POLY || param.kernel_type == RBF || param.kernel_type == SIGMOID)
    {
        xml_svm_node.append_child(GAMMA_TAG)
            .text().set(param.gamma);
    }

    if(param.kernel_type == POLY || param.kernel_type == SIGMOID)
    {
        xml_svm_node.append_child(COEF0_TAG)
            .text().set(param.coef0);
    }

    int nr_class = model->nr_class;
    int l = model->l;
    xml_svm_node.append_child(NR_CLASS_TAG)
        .text().set(nr_class);

    xml_svm_node.append_child(TOTAL_SV_TAG)
        .text().set(l);

    std::stringstream ss;
    int num_classifiers = nr_class * (nr_class - 1)/2;

    auto rho = xml_svm_node.append_child(RHO_TAG);

    rho.text().set(format(model->rho, num_classifiers).data());

    if(model->label)
    {
        auto label = xml_svm_node.append_child(LABEL_TAG);
        label.text().set(format(model->label, nr_class).data());
    }

    if(model->probA) // regression has probA only
    {
        auto probA = xml_svm_node.append_child(PROB_A_TAG);
        probA.text().set(format(model->probA, num_classifiers).data());
    }

    if(model->probB)
    {
        auto probB = xml_svm_node.append_child(PROB_B_TAG);
        probB.text().set(format(model->probB, num_classifiers).data());
    }

    if(model->nSV)
    {
        auto nSV = xml_svm_node.append_child(NR_SV_TAG);
        nSV.text().set(format(model->nSV, nr_class).data());
    }

    auto support_vectors = xml_svm_node.append_child(SVS_TAG);
    const double * const *sv_coef = model->sv_coef;
    const svm_node * const * SV = model->SV;

    int count = 0;
    for(int i=0;i<l; ++i)
    {
        std::stringstream ss;
        auto sv = support_vectors.append_child(SV_TAG);

        sv.append_attribute(INDX_TAG).set_value(i);
        auto coef = sv.append_child(SV_COEF_TAG);

        ss.precision(16);

        for(int j=0;j<nr_class-1;j++)
        {
            ss << sv_coef[j][i] << " ";
        }

        coef.text().set(ss.str().data());
        ss.str(std::string());
        
        const svm_node *p = SV[i];

        auto sv_point = sv.append_child(SV_POINT_TAG);
        if(param.kernel_type == PRECOMPUTED)
        {
            ss << "0:" << (int)(p->value) << " "; 
            ++count;
        }
        else
        {
            ss.precision(8);
            while(p->index != -1)
            {
                ss << p->index << ":" << p->value << " ";
                p++;
                ++count;
            }
            ++count;
        }

        sv_point.text().set(ss.str().data());
    }

    support_vectors.append_attribute(COUNT_TAG).set_value(count);
}

svm_model * LibSVMTrainBridge::load(const std::string & file_name)
{
    std::ifstream ifs(file_name);

    pugi::xml_document doc;
    pugi::xml_parse_result result = doc.load(ifs);

    if (!result)
    {
        throw FileNotFoundException(file_name);
    }

    pugi::xml_node root = doc.child(ROOT_TAG);

    return loadSvmModel(root.child(SVM_MODEL_TAG));
}

svm_model * LibSVMTrainBridge::load(const std::string & file_name, std::vector<FeatureScaler> & scalers)
{
    std::ifstream ifs(file_name);

    pugi::xml_document doc;
    pugi::xml_parse_result result = doc.load(ifs);

    if (!result)
    {
        throw FileNotFoundException(file_name);
    }

    pugi::xml_node root = doc.child(ROOT_TAG);

    svm_model * model = loadSvmModel(root.child(SVM_MODEL_TAG));

    auto scalings_node = root.child(SCALERS_TAG);

    auto scalers_enum = scalings_node.children(SCALER_TAG);
    std::size_t scalers_count = std::distance(scalers_enum.begin(),
            scalers_enum.end());

    scalers.resize(scalers_count);
    for (auto scaler_node: scalers_enum)
    {
        int indx = scaler_node.attribute(INDX_TAG).as_int();
        scalers[indx].min_ = scaler_node.child(MIN_TAG).text().as_float();
        scalers[indx].interval_length_ = scaler_node.child(INTERVAL_LENGTH_TAG).text().as_float();
        scalers[indx].scaled_min_ = scaler_node.child(SCALED_MIN_TAG).text().as_float();
        scalers[indx].scaled_interval_length_ = scaler_node.child(SCALED_INTERVAL_LENGTH_TAG).text().as_float();
    }

    return model;
}

svm_model * LibSVMTrainBridge::loadSvmModel(const pugi::xml_node & xml_svm_node)
{
    svm_model * model = new svm_model;
    
    svm_parameter& param = model->param;
    model->rho = NULL;
    model->probA = NULL;
    model->probB = NULL;
    model->label = NULL;
    model->nSV = NULL;

    auto svm_type_node = xml_svm_node.child(SVM_TYPE_TAG);
    auto kernel_type_node = xml_svm_node.child(KERNEL_TYPE_TAG);
    param.svm_type = get_svm_type_indx(svm_type_node.text().as_string());
    param.kernel_type = get_kernel_type_indx(kernel_type_node.text().as_string());

    auto degree_node = xml_svm_node.child(DEGREE_TAG);
    if (degree_node)
    {
        param.degree = degree_node.text().as_double();
    }

    auto gamma_node=  xml_svm_node.child(GAMMA_TAG);
    if (gamma_node)
    {
        param.gamma = gamma_node.text().as_double();
    }

    auto coef0_node = xml_svm_node.child(COEF0_TAG);
    if (coef0_node)
    {
        param.coef0 = coef0_node.text().as_double();
    }

    auto nr_class_node = xml_svm_node.child(NR_CLASS_TAG);
    model->nr_class = nr_class_node.text().as_int();
    auto total_sv_node = xml_svm_node.child(TOTAL_SV_TAG);
    model->l = total_sv_node.text().as_int();

    int nr_class = model->nr_class;
    int num_classifiers = nr_class * (nr_class - 1)/2;

    auto rho_node = xml_svm_node.child(RHO_TAG);
    parse(&model->rho, num_classifiers, rho_node.text().as_string());

    auto label_node = xml_svm_node.child(LABEL_TAG);
    if (label_node)
    {
        parse(&model->label, nr_class, label_node.text().as_string());
    }

    auto probA_node = xml_svm_node.child(PROB_A_TAG);
    if (probA_node)
    {
        parse(&model->probA, num_classifiers, probA_node.text().as_string());
    }

    auto probB_node = xml_svm_node.child(PROB_B_TAG);
    if (probB_node)
    {
        parse(&model->probB, num_classifiers, probB_node.text().as_string());
    }

    auto nr_sv_node = xml_svm_node.child(NR_SV_TAG);
    if (nr_sv_node)
    {
        parse(&model->nSV, nr_class, nr_sv_node.text().as_string());
    }

    auto support_vectors_node = xml_svm_node.child(SVS_TAG);

    model->sv_coef = new double * [nr_class - 1];
    int l = model->l;

    for (int i = 0; i < nr_class - 1; ++i)
    {
        model->sv_coef[i] = new double[l];
    }

    model->SV = new svm_node * [l];
    int elements = support_vectors_node.attribute(COUNT_TAG).as_int();
    svm_node * x_space = NULL;

    if (l > 0)
    {
        x_space = new svm_node[elements];
    }

    int j = 0;
    for (auto sv_node : support_vectors_node.children(SV_TAG))
    {
        int indx = sv_node.attribute(INDX_TAG).as_int();
        stringstream ss(sv_node.child(SV_COEF_TAG).text().as_string());

        for (int k = 0; k < nr_class - 1; k++)
        {
            ss >> model->sv_coef[k][indx];
        }

        model->SV[indx] = &x_space[j];
        
        
        ss.str(sv_node.child(SV_POINT_TAG).text().as_string());
        std::string token;

        while (ss >> token)
        {
            std::size_t pos = token.find(':');
            NOCR_ASSERT(pos != string::npos, "chyba token parsovani");

            x_space[j].index = std::stoi(token.substr(0, pos));
            x_space[j].value = std::stod(token.substr(pos+1));
            ++j;
        }

        x_space[j++].index = -1;
    }

    model->free_sv = 1;

    return model;
}

void LibSVMTrainBridge::destroy_svm_model(svm_model ** model)
{
    svm_model * model_ptr = *model;
    if (model_ptr != nullptr)
    {
        if (model_ptr->free_sv && model_ptr->l > 0)
        {
            delete[] model_ptr->SV[0];
        }

        for (int i = 0; i < model_ptr->nr_class -1; ++i)
        {
            delete[] model_ptr->sv_coef[i];
        }

        delete[] model_ptr->SV;
        delete[] model_ptr->sv_coef;

        delete[] model_ptr->rho;
        delete[] model_ptr->label;
        delete[] model_ptr->probA;
        delete[] model_ptr->probB;
        delete[] model_ptr->nSV;
    }

    delete model_ptr;
}

// =================================================================

model* LibLINEARTrainBridge::trainModel( const cv::Mat &train_data, const cv::Mat &labels, 
        parameter *params)
{
    problem linear_problem = constructProblem( train_data, labels ); 
    std::cout << "start training" << std::endl;
    
    cout << linear_problem.l << endl;
    model *linear_model = train( &linear_problem , params );

    std::cout << "delete done" << std::endl;
    delete[] linear_problem.y;
    for ( int i = 0; i < train_data.rows; ++i )
    {
        delete[] linear_problem.x[i];
    }
    delete[] linear_problem.x;

    return linear_model;
}

problem LibLINEARTrainBridge::constructProblem
    ( const cv::Mat &train_data, const cv::Mat &labels ) const
{
    problem linear_problem;
    linear_problem.l = train_data.rows;
    linear_problem.n = train_data.cols;
    linear_problem.bias = -1;
    linear_problem.y = new double[train_data.rows]; 
    linear_problem.x = new feature_node*[train_data.rows];
    for ( int i = 0; i < train_data.rows; ++i )
    {
        linear_problem.y[i] = labels.at<float>(i,0);
        linear_problem.x[i] = new feature_node[train_data.cols+1];
        for ( int j = 0; j < train_data.cols; ++j )
        {
            linear_problem.x[i][j] = { j + 1, train_data.at<float>(i,j) }; 
        }
        linear_problem.x[i][train_data.cols] = { -1 , 25 };
    }

    return linear_problem;
}

feature_node* LibLINEARTrainBridge::constructSample( const std::vector<float> &data ) const
{
    feature_node *sample = new feature_node[data.size()+1];
    for ( size_t i = 0; i < data.size(); ++i )
    {
        sample[i].index = i+1;
        sample[i].value = data[i];
    }
    sample[data.size()].index = -1;
    return sample;
}

// void SVM::train( const std::string &data_file, svm_parameter *param )
// {
//     cv::Mat train_data, labels;
//
//     TrainDataLoader train_data_loader( length_ ); 
//     train_data_loader.prepareDataForTraining( data_file, train_data, labels );
//
//     svm_ = bridge_.train( train_data, labels, param );
//     number_of_classes_ = svm_get_nr_class( svm_ );
// }
//
// void SVM::saveConfiguration( const std::string &conf_file )
// {
//     int result = svm_save_model( conf_file.c_str(), svm_ );
//     //TODO
// }
//
// void SVM::loadConfiguration( const std::string &conf_file )
// {
//     svm_ = svm_load_model( conf_file.c_str() );
//     //TODO
//     if ( svm_ == nullptr )
//     {
//         throw FileNotFoundException(conf_file + ", libsvm configuration not found");
//     }
//     number_of_classes_ = svm_get_nr_class( svm_ );
// }
//
//
// float SVM::predict(const std::vector<float> &data ) const
// {
//     NOCR_ASSERT( svm_ != nullptr , "no configuration loaded yet" );
//
//     svm_node *nodes = bridge_.constructSample( data );
//     float out = svm_predict( svm_ , nodes );
//     delete[] nodes;
//     return out;
// }
//
// double SVM::predictProbabilities(const std::vector<float> &data, 
//                             std::vector<double> &probabilities ) const  
// {
//     NOCR_ASSERT( svm_ != nullptr , "no configuration loaded yet" );
//
//
//     svm_node *nodes = bridge_.constructSample( data );
//     probabilities.resize( number_of_classes_ );
//     double out = svm_predict_probability( svm_ , nodes, 
//                                         probabilities.data() ); 
//     delete[] nodes;
//     return out;
// }

FeatureScaler::FeatureScaler(float min, float max, float scaled_min, float scaled_max)
    :min_(min), scaled_min_(scaled_min)
{
    interval_length_ = max - min;
    scaled_interval_length_ = scaled_max - scaled_min;
}

float FeatureScaler::scale(float val) const 
{
    return scaled_min_ + (val-min_)/interval_length_ * scaled_interval_length_;
}

std::vector<float> 
    DataScaling::scale(const std::vector<float> &descriptor) const
{
    vector<float> scaled_descriptor(descriptor.size());
    for ( size_t i = 0; i < descriptor.size(); ++i )
    {
        scaled_descriptor[i] = 
                    scalers_[i].scale(descriptor[i]);
    }
    
    return scaled_descriptor;
}

void DataScaling::setUp( const cv::Mat &train_data )
{
    for ( int i = 0; i < train_data.cols; ++i )
    {
        cv::Mat curr_row = train_data.col(i);
        double min, max;
        cv::minMaxLoc(curr_row, &min, &max);
        FeatureScaler scaler( min, max);

        auto begin = curr_row.begin<float>();
        auto end = curr_row.end<float>();

        for ( auto it = begin; it != end; ++it )
        {
            *it = scaler.scale(*it);
        }

        scalers_.push_back(scaler);
    }
}

void DataScaling::saveScaling( const std::string &scaling_file )
{
    std::ofstream ofs(scaling_file);
    if ( !ofs.is_open() )
    {
    }

    for ( const auto s : scalers_ )
    {
        ofs << s.min_ << " " << s.min_ + s.interval_length_ << " " 
            << s.scaled_min_ << " " << s.scaled_min_ + s.scaled_interval_length_ << endl;
    }
    ofs.close();
}

void DataScaling::loadScaling( const std::string &scaling_file )
{
    std::ifstream ifs(scaling_file);
    if ( !ifs.is_open() )
    {
    }

    string line;
    while( std::getline(ifs, line) )
    {
        std::stringstream ss( line );
        float min, max, scaled_min, scaled_max;

        ss >> min; 
        ss >> max;
        ss >> scaled_min;
        ss >> scaled_max;
        ss.flush();

        scalers_.push_back( FeatureScaler(min, max, scaled_min, scaled_max) ) ;
    }
}



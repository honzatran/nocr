/* 
 * Tran Tuan Hiep
 * Implementation of methods and classes declared in iksvm.h
 *
 * Compiler: g++ 4.8.3
 */
#include "../include/nocrlib/iksvm.h"
#include "../include/nocrlib/exception.h"
#include "../include/nocrlib/utilities.h"

#include <limits>
#include <algorithm>
#include <tuple>
#include <fstream>
#include <sstream>

#include <pugi/pugixml.hpp>

#define DEBUG 0
#define FAST_EVAL 1

#define IKSVM_ROOT_TAG "iksvm"
#define IKSVM_NR_CLASS_TAG "nr-class"
#define IKSVM_FEAT_DIM_TAG "feat-dim"
#define IKSVM_PROB_A_TAG "prob-A"
#define IKSVM_PROB_B_TAG "prob-B"
#define IKSVM_LABELS_TAG "labels"
#define IKSVM_DFS_TAG "decision-functions-list"
#define IKSVM_DF_TAG "decision-function"
#define IKSVM_INDX_TAG "indx"
#define IKSVM_COUNT_TAG "count"
#define IKSVM_DIM_TAG "dimension"
#define IKSVM_STEP_TAG "step"
#define IKSVM_MIN_SAMPLE_TAG "min-sample"
#define IKSVM_DIM_FUNC_TAG "dim-function"


using namespace std;

///util function

template <typename T>
std::string toString(const std::vector<T> & values, char delim = ' ')
{
    std::stringstream ss;
    for (std::size_t i = 0; i < values.size() - 1; ++i)
    {
        ss <<  values[i] << delim;
    }
    ss << values.back();

    return ss.str();
}




const double ApproximatedFunction::epsilon_ = 0.000001;

ApproximatedFunction:: ApproximatedFunction
    ( double min_sample, double step_size,
      const std::vector<double> &function_values )
    : min_sample_(min_sample), step_size_(step_size),
    function_values_(function_values)
{
    // cout << min_sample << ':' << step_size << endl;
    num_steps_ = function_values_.size() - 1;
    function_values_.shrink_to_fit();
    /*
     * size_t size = function_values.size();
     * size_t i = 0;
     * while( true )
     * {
     *     double value = function_values[i];
     *     size_t j = i;
     *     while( std::abs( value - function_values[j]) < epsilon_
     *             && j < size )
     *     {
     *         ++j;
     *     }

     *     function_values_.push_back(value);
     *     indices_.push_back(i);
     *     i = j;

     *     if ( i >= size )
     *     {
     *         break;
     *     }
     * }
     */
}


double ApproximatedFunction::getValueAt( double d )
{
    double f_interpolation = ( d - min_sample_ )/step_size_;

    int left = std::floor(f_interpolation);
    int right = left + 1;
    left = std::max( std::min(left,num_steps_), 0);
    right = std::max( std::min(right,num_steps_), 0);
    double alpha = f_interpolation - left;
    double f_left = function_values_[left];
    double f_right = function_values_[right];
    // return interpolate_between( alpha, left, right );
    return ( 1 - alpha ) * f_left + alpha * f_right;
}

std::ostream& operator<<( std::ostream &oss, const ApproximatedFunction &fnc )
{
    size_t last = fnc.function_values_.size() - 1;
    oss << fnc.step_size_ << ";" << fnc.min_sample_ << ";";
    for ( size_t i = 0; i < last; ++i )
    {
        oss << fnc.function_values_[i] << ':';
    }
    oss << fnc.function_values_.back();

    return oss;
}

// ==========================Decision Function===========================

DecisionFunction::DecisionFunction( 
        const std::vector<ApproximatedFunction> &approximated_functions, 
        double b )
    : b_(b)
{
    for (const auto & fc: approximated_functions)
    {
        step_size_.push_back(std::make_pair(fc.getStepSize(), fc.getMinSample()));
        auto tmp = fc.getFunctionValues();
        function_values_.insert(function_values_.end(), tmp.begin(), tmp.end());
    }
}

double DecisionFunction::compute( const std::vector<double> &x )
{
    double value = b_;
    //tady pujde sse
    int num_steps = fuction_approx_count_ - 1;
    for ( size_t i = 0; i < x.size(); ++i )
    {
        // double tmp1 =  approximated_functions_[i].getValueAt(x[i]);
        // jsem se presune kod z ApproxFunction::compute
        std::size_t indx = i * fuction_approx_count_;
        double step_size, min_sample;
        std::tie(step_size, min_sample) = step_size_[i];

        double f_interpolation = (x[i] - min_sample)/step_size;

        int left = std::floor(f_interpolation);
        int right = left + 1;

        left = std::min(left, num_steps);
        left = left > 0 ? left : 0;
        right = std::min(right, num_steps);
        right = right > 0 ? right : 0;

        double alpha = f_interpolation - left;
        double f_left = function_values_[indx + left];
        double f_right = function_values_[indx + right];

        value += ( 1 - alpha ) * f_left + alpha * f_right;
    }

    return value;
}

std::ostream& operator<<( std::ostream &oss, const DecisionFunction &fnc )
{
    std::size_t fuction_approx_count = fnc.fuction_approx_count_;
    std::size_t features_dim = fnc.feature_dim_;

    for (std::size_t i = 0; i < features_dim; ++i)
    {
        int indx = i * fuction_approx_count;
        oss << fnc.step_size_[i].first << ";" << fnc.step_size_[i].second << ";";
         
        for (std::size_t j = 0; j < fuction_approx_count - 1; ++j)
        {
            oss << fnc.function_values_[indx + j] << ":";
        }
         
        oss << fnc.function_values_[indx + fuction_approx_count - 1] << " "; 
    }

    return oss;
}


void DecisionFunction::loadFromString( const std::string &s, int features_dim )
{
    std::stringstream ss(s);
    ss >> b_;

    feature_dim_ = features_dim;

    function_values_.reserve(features_dim * 15);
    string sample_function_str;
     
    for ( int i = 0; i < features_dim; ++i )
    {
        if (ss >> sample_function_str)
        {
            strToApproxFunction(sample_function_str);
        } 
        else
        {
            throw BadFileFormatting("");
        }
    }
}

void DecisionFunction::strToApproxFunction(const std::string &s)
{
    std::stringstream ss(s);
    std::string token;
    std::getline(ss,token, ';');
    double step_size = std::stod(token);
    std::getline(ss,token, ';');
    double min_sample = std::stod(token);
    std::getline(ss,token, ';');
    ss.clear();

    std::stringstream ss1( token );
    std::vector<double> function_values;
    string tmp;
    
    while( getline(ss1,tmp,':') )
    {
        function_values.push_back(std::stod(tmp));
    }

    function_values_.insert(function_values_.end(), function_values.begin(), function_values.end());
    
    step_size_.push_back(std::make_pair(step_size, min_sample));

    fuction_approx_count_ = function_values.size();
}
 



//========================FAST INTERSECTION KERNEL======================================
IKSVMConvertor::IKSVMConvertor( int features_dim )
    : features_dim_( features_dim )
{

}


IKSVM IKSVMConvertor::createFromSvmProblem( const std::string &problem_file, int num_segment )
{
    svm_model* model = svm_load_model( problem_file.c_str() );
    IKSVM iksvm = createFromSvmProblem( model, num_segment );
    svm_free_and_destroy_model( &model );
    return iksvm;
}

IKSVM IKSVMConvertor::createFromSvmProblem( svm_model *model, int num_segment )
{
    param_ = model->param;

    NOCR_ASSERT( param_.kernel_type == 5, "Wrong type of kernel in libsvm model" );

    num_segment_ = num_segment;
    nr_class_ = model->nr_class;
    int number_subproblems = nr_class_*( nr_class_ - 1 )/2;

    auto rho = std::vector<double>(model->rho, model->rho + number_subproblems );
    auto prob_a = std::vector<double>(model->probA, model->probA + number_subproblems );
    auto prob_b = std::vector<double>(model->probB, model->probB + number_subproblems );
    auto labels = std::vector<double>(model->label, model->label + nr_class_ );
    total_sv_ = model->l;
    number_sv_ = vector<int>(model->nSV, model->nSV + nr_class_);

    std::vector<int> start = getStartsOfSv();

    vector<DecisionFunction> desicion_functions;
    int p = 0;
    for ( int i = 0; i < nr_class_; ++i )
    {
        Matrix<double> i_support_vectors = getAllSv( i, model->SV, start );
        for ( int j = i + 1; j < nr_class_; ++j )
        {
            // aproximation for all classifiers(i,j)
           vector<double> i_sv_coef = getSvCoef( j-1, start[i], number_sv_[i], model->sv_coef );
           vector<double> j_sv_coef = getSvCoef( i, start[j], number_sv_[j], model->sv_coef );
           Matrix<double> j_support_vectors = getAllSv( j, model->SV, start );
           auto approximated_functions = approximateDecisionFunction
                                            ( i_support_vectors, i_sv_coef, 
                                              j_support_vectors, j_sv_coef );

           DecisionFunction decision_function( approximated_functions, -rho[p++] );
           desicion_functions.push_back( decision_function );
        }
    }

    return IKSVM( nr_class_, features_dim_, num_segment, prob_a, prob_b, desicion_functions, labels ); 
}

vector<int> IKSVMConvertor::getStartsOfSv()
{
    vector<int> start( nr_class_, 0 );
    for ( int i = 1; i < nr_class_; ++i )
    {
        start[i] = start[i-1] + number_sv_[i-1];
    }
    return start;
}

auto IKSVMConvertor::getAllSv( int i, svm_node **support_vectors, const std::vector<int> &start )
    -> Matrix<double>
{
    int start_index = start[i];
    int end_index = start[i] + number_sv_[i];
    Matrix<double> output;
    output.reserve( number_sv_[i] );
    for ( int j = start_index; j < end_index; ++j )
    {
        output.push_back( convertToVector( support_vectors[j] ) );
    }
    return output;
}

vector<double> IKSVMConvertor::convertToVector( svm_node *node )
{
    vector<double> output;
    output.resize( features_dim_, std::numeric_limits<double>::infinity() );
    int i = 0;
    int j = 0;
    while( node[j].index != -1 )
    {
        int index = node[j].index;
        if ( index == i )
        {
            output[i] = node[j].value; 
            ++j;
            ++i;
        }
        else ++j;
    }
    return output;
}

vector<double> IKSVMConvertor::getSvCoef( int label, int start, int count, double **sv_coef )
{
    vector<double> output;
    output.reserve( count );
    for ( int i = 0; i < count; ++i )
    {
        output.push_back( sv_coef[label][start+i] );
    }
    return output;
}


std::vector<ApproximatedFunction> IKSVMConvertor::approximateDecisionFunction
    ( const Matrix<double> &a_support_vectors, const std::vector<double> &a_sv_coef,
      const Matrix<double> &b_support_vectors, const std::vector<double> &b_sv_coef )
{
    std::vector<ApproximatedFunction> approximated_functions;
    approximated_functions.reserve( features_dim_ );
    for ( int i = 0; i < features_dim_; ++i )
    {
        vector<double> alpha = a_sv_coef;

        alpha.insert( alpha.end(), b_sv_coef.begin(), b_sv_coef.end() );

        VectorPair sorted_sequence = sortWithIndices( 
                getColumn(i, a_support_vectors), 
                getColumn(i, b_support_vectors) );

        vector<double> samples = linspace( sorted_sequence[0].first, 
                sorted_sequence.back().first, num_segment_ ); 

        ApproximatedFunction approximated_function = approximateFunction( alpha, sorted_sequence, samples );
        approximated_functions.push_back( approximated_function );
    }
    return approximated_functions;
}

auto IKSVMConvertor::sortWithIndices( const std::vector<double> &a, const std::vector<double> &b )
    -> VectorPair
{
    VectorPair output;
    output.reserve( a.size() + b.size() );
    int j = 0;
    auto lambda_insert = [&output,&j] ( double d ) 
    { 
        if ( d != std::numeric_limits<double>::infinity() )
        {
            output.push_back( std::make_pair(d,j) );
        }
        ++j;
    };

    std::for_each( a.begin(), a.end(), lambda_insert );
    std::for_each( b.begin(), b.end(), lambda_insert );
    std::sort( output.begin(), output.end(), 
            [] ( const std::pair<double,size_t> &a, const std::pair<double,size_t> &b )
            {
                return a.first < b.first;
            });

    return output;
}

ApproximatedFunction IKSVMConvertor::approximateFunction( 
        const std::vector<double> &alpha, const VectorPair &sorted_sequence, 
        const std::vector<double> &samples )
{
    vector<double> approx_function( samples.size() );
    size_t idx_sort = 0;
    double sa = 0; 
    double sax = 0;

    double sum_alpha = sum( alpha );

    for ( size_t i = 0; i < samples.size(); ++i )
    {
        while( sorted_sequence[idx_sort].first < samples[i] && idx_sort < sorted_sequence.size() ) 
        {
            int index;
            double value;
            std::tie(value, index) = sorted_sequence[idx_sort];
            sa += alpha[index];
            sax += alpha[index] * value;
            ++idx_sort;
        }
        approx_function[i] = sax + (sum_alpha - sa) * samples[i];
    }

    double step_size = (samples.back() - samples.front())/(samples.size()-1);
    return ApproximatedFunction(samples[0], step_size, approx_function);
}

//=========================saving and loading from file======================
//
const std::string IKSVM::number_class_text = "number of classes";
const std::string IKSVM::feature_dimension_text = "feature dimension";
const std::string IKSVM::approx_count = "approx count";
const std::string IKSVM::probA_text = "prob A";
const std::string IKSVM::probB_text = "prob B";
const std::string IKSVM::label_text = "labels";
const std::string IKSVM::decision_function_text = "decision functions";



IKSVM::IKSVM( int nr_class, int features_dim, int approx_count,
        const std::vector<double> prob_A, const std::vector<double> prob_B,
        const std::vector<DecisionFunction> decision_function, 
        const std::vector<double> labels )
    : nr_class_(nr_class), features_dim_(features_dim), approx_count_(approx_count),
    prob_A_(prob_A), prob_B_(prob_B), labels_(labels)
{
    int number_subproblems = decision_function.size();
    decision_values_b_.resize(number_subproblems);
    decision_function_.resize(number_subproblems * features_dim_ * approx_count_);
    decision_function_info_.resize(number_subproblems * features_dim_);

    for (std::size_t i = 0;i < decision_function.size(); ++i)
    {
        decision_values_b_[i] = decision_function[i].b_;
        auto info = decision_function[i].step_size_;
        decision_function_info_.insert(decision_function_info_.end(),
                info.begin(), info.end());

        auto values = decision_function[i].function_values_;
        decision_function_.insert(decision_function_.end(),
                values.begin(), values.end());
    }
}

void IKSVM::save( const std::string &file_name )
{
    std::ofstream ofs( file_name );

    /*
     * if ( !ofs.is_open() )
     * {
     *     throw Exception;

     * }
     */

    ofs << number_class_text << ':' << nr_class_ << endl;
    ofs << feature_dimension_text << ':' << features_dim_ << endl;
    ofs << approx_count  << ':' << approx_count_ << endl;


    ofs << probA_text << ':';
    for( double d : prob_A_ )
    {
        ofs << d << " ";
    }
    ofs << endl;

    ofs << probB_text << ':';
    for( double d : prob_B_ )
    {
        ofs << d << " ";
    }
    ofs << endl;

    ofs << label_text << ':';
    for ( double d: labels_ )
    { 
         ofs << d << " ";
    }
    ofs << endl;
    ofs << decision_function_text << endl;

    int num_class = nr_class_ * (nr_class_ - 1) / 2;
    double step_size, min_sample;

    for (int i = 0; i < num_class; ++i)
    {
        ofs << decision_values_b_[i] << " ";
        for (int j = 0; j < features_dim_; ++j)
        {
            std::tie(step_size, min_sample) = decision_function_info_[i * features_dim_ + j];
            ofs << step_size << ';' << min_sample << ';';
            std::size_t dec_fn_idx = (i * features_dim_  + j) * approx_count_;

            for (int k = 0; k < approx_count_ - 1; ++k)
            {
                ofs << decision_function_[dec_fn_idx + k] << ':';
            }

            ofs << decision_function_[dec_fn_idx + approx_count_ - 1] << " ";
        }

        ofs << endl;
    }

    ofs.close();
}

void IKSVM::saveXml( const std::string &file_name )
{
    pugi::xml_document doc;
    auto root = doc.append_child(IKSVM_ROOT_TAG);
    root.append_child(IKSVM_NR_CLASS_TAG).text().set(nr_class_);
    root.append_child(IKSVM_FEAT_DIM_TAG).text().set(features_dim_);
    root.append_child(IKSVM_PROB_A_TAG).text().set(toString(prob_A_).c_str());
    root.append_child(IKSVM_PROB_B_TAG).text().set(toString(prob_B_).c_str());
    root.append_child(IKSVM_LABELS_TAG).text().set(toString(labels_).c_str());
    auto decision_fnc_node = root.append_child(IKSVM_DFS_TAG);
    int num_class = nr_class_ * (nr_class_ - 1)/ 2;
    decision_fnc_node.append_attribute(IKSVM_COUNT_TAG).set_value(num_class);

    // for (std::size_t i = 0; i < desicion_functions_.size(); ++i)
    // {
    //     auto df_node = decision_fnc_node.append_child(IKSVM_DF_TAG);
    //     df_node.append_attribute(IKSVM_INDX_TAG).set_value(i);
    //     saveXmlDF(df_node, desicion_functions_[i]);
    // }
    



    std::ofstream ofs( file_name );

    /*
     * if ( !ofs.is_open() )
     * {
     *     throw Exception;

     * }
     */

    cout << num_class << endl;
    ofs << number_class_text << ':' << nr_class_ << endl;
    ofs << feature_dimension_text << ':' << features_dim_ << endl;

    ofs << probA_text << ':';
    for( double d : prob_A_ )
    {
        ofs << d << " ";
    }
    ofs << endl;

    ofs << probB_text << ':';
    for( double d : prob_B_ )
    {
        ofs << d << " ";
    }
    ofs << endl;

    ofs << label_text << ':';
    for ( double d: labels_ )
    { 
         ofs << d << " ";
    }
    ofs << endl;

    // ofs << decision_function_text << endl;
    // for ( const auto &des_fnc: desicion_functions_ )
    // {
    //     ofs << des_fnc << endl;
    // }

    doc.save(ofs);
    ofs.close();
}

void IKSVM::saveXmlDF(pugi::xml_node & df_node, const DecisionFunction & fn)
{
    // auto feature_dim_nodes = df_node.append_child(IKSVM_DIM_FUNC_TAG);
    /*
     * for (int i = 0; i < feature_dim_; ++i)
     * {
     *
     *     df_node.append_child(IKSVM_MIN_SAMPLE_TAG).text().set(fn.min_sample_);
     *     df_node.append_child(IKSVM_STEP_TAG).text().set(fn.step_);
     *     auto dim_fn = features_dim_nodes.append(child);
     *     dim_fn.append_attribute(IKSVM_DIM_TAG).set_value(i);
     *     //parseString dim_fn;
     * }
     */
    
}

void IKSVM::load( const std::string &file_name )
{
    std::ifstream ifs(file_name);
    if ( !ifs.good() )
    {
        throw FileNotFoundException( file_name + "iksvm configuration file not found" );
    }


    std::string line;

    std::getline(ifs, line);
    nr_class_ = std::stoi(parse( number_class_text, line ));
    std::getline(ifs, line);
    features_dim_ = std::stoi(parse( feature_dimension_text, line ));
    std::getline(ifs, line);
    approx_count_ = std::stoi(parse( approx_count, line));
    std::getline(ifs, line);
    prob_A_ = strToVec(parse( probA_text, line ));
    std::getline(ifs, line);
    prob_B_ = strToVec(parse( probB_text, line ));
    std::getline(ifs, line);
    labels_= strToVec(parse( label_text, line ));
    std::getline(ifs,line);
    
    if( line != decision_function_text )
    {
        throw BadFileFormatting("");
    }

    int number_subproblems = nr_class_ * (nr_class_ - 1) / 2;
    decision_values_b_.resize(number_subproblems);
    decision_function_.resize(number_subproblems * features_dim_ * approx_count_);
    decision_function_info_.resize(number_subproblems * features_dim_);

    for ( int i = 0; i < number_subproblems; ++i )
    {
        if ( std::getline( ifs, line ) )
        {
            parseDecisionFunction(line, i);
        }
        else 
        {
            throw BadFileFormatting("");
        }
    }

}

void IKSVM::parseDecisionFunction(const std::string & line, std::size_t indx)
{
    std::stringstream ss(line);
    ss >> decision_values_b_[indx];

    string sample_function_str;
     
    for ( int i = 0; i < features_dim_; ++i )
    {
        string dim_fn_line;
        ss >> dim_fn_line;

        std::stringstream ss1(dim_fn_line);
        string token;
        std::getline(ss1,token, ';');
        double step_size = std::stod(token);
        std::getline(ss1,token, ';');
        double min_sample = std::stod(token);
        std::getline(ss1,token, ';');


        decision_function_info_[indx * features_dim_ + i] = std::make_pair(step_size, min_sample);

#if DEBUG
        // std::cout << i << " "<< token  << std::endl;
        std::cout << token  << std::endl;
#endif
        
        std::stringstream parser_fn_dim(token);
        string tmp;
        int j = indx * features_dim_ * approx_count_ + i * approx_count_;
        for (int k = 0; k < approx_count_; ++k)
        {
            std::getline(parser_fn_dim, tmp, ':');
#if DEBUG
            std::cout << tmp << std::endl;
#endif
            decision_function_[j + k] = std::stod(tmp);
        }
    }
}
        

bool IKSVM::startsWith( const std::string &s, const std::string &start )
{
    if ( s.length() >= start.length() )
    {
        return start == s.substr(0, start.length());
    }
    
    return false;
}

std::string IKSVM::parse( const std::string &start, const std::string &s )
{
    int pos = s.find(':');
    if( pos == string::npos )
    {
        throw BadFileFormatting("error file");
    }
    string s_start = s.substr(0,pos);
    string s_end = s.substr(pos + 1, s.size() - pos - 1);
    if ( s_start != start )
    {
        throw BadFileFormatting( s + "doesn't start with" + start );
    }

    return s_end;
}

vector<double> IKSVM::strToVec( const std::string &s )
{
    std::stringstream ss(s);
    vector<double> output;
    string token;
    while( ss >> token )
    {
        output.push_back( std::stod(token) );
    }
    return output;
}
 
// ====================predicting =======================================
//

double IKSVM::predict( const std::vector<double> &x )
{
    std::vector<double> decision_values;
    return computeDecisionsValue(x, decision_values );
}
         
std::vector<double> IKSVM::predictMultiple(const std::vector<double> &x)
{
    std::vector<double> decision_values;
    return computeDecisionsValueMult(x, decision_values);
}

double IKSVM::computeDecisionsValue( const std::vector<double> &x, 
        std::vector<double> &decision_values )
{
    decision_values.resize(nr_class_ * (nr_class_ - 1 ) / 2);
    std::vector<int> votes( nr_class_, 0 );
    int p = 0;
    for ( int i = 0; i < nr_class_; ++i ) 
    {
        for ( int j = i + 1; j < nr_class_; ++j )
        {
            double decision_value = evalDecisionFunction(p, x);
            decision_values[p] = decision_value;

            if ( decision_value > 0 )
            {
                votes[i] += 1;
            }
            else 
            {
                votes[j] += 1;
            }
            ++p;
        }
    }

    size_t max_idx = 0;
    for ( size_t i = 1; i < votes.size(); ++i ) 
    {
        if ( votes[max_idx] < votes[i] )
        {
            max_idx = i;
        }
    }
    return labels_[max_idx];
}

std::vector<double> IKSVM::computeDecisionsValueMult(const std::vector<double> & descriptors,
        std::vector<double> & decision_values)
{
    std::size_t count = descriptors.size()/features_dim_;
    std::size_t num_classifiers = nr_class_ * (nr_class_ - 1 ) / 2;

    decision_values.resize(count * num_classifiers);
    vector<int> votes(count * nr_class_, 0);

    int p = 0;
    for ( int i = 0; i < nr_class_; ++i ) 
    {
        for ( int j = i + 1; j < nr_class_; ++j )
        {
            for (std::size_t k = 0; k < count; ++k)
            {
                double decision_value = evalDecisionFunction(p, descriptors, k * features_dim_);
                decision_values[k * num_classifiers + p] = decision_value;

                if ( decision_value > 0 )
                {
                    votes[k * nr_class_ + i] += 1;
                }
                else 
                {
                    votes[k * nr_class_ + j] += 1;
                }
            } 

            ++p;
        }
    }

    vector<double> results(count);

    std::size_t offset = 0;
    for (std::size_t k = 0; k < count; ++k)
    {
        std::size_t max_idx = 0;
        for (std::size_t i = 1; i < nr_class_; ++i)
        {
            if (votes[offset + max_idx] < votes[offset + i])
            {
                max_idx = i;
            }
        }

        results[k] = labels_[max_idx];
        offset += nr_class_;
    }

    return results;
}

double IKSVM::evalDecisionFunction(std::size_t indx, const std::vector<double> & x)
{
    double value = decision_values_b_[indx];
    //tady pujde sse
    int num_steps = approx_count_ - 1;
    for ( size_t i = 0; i < x.size(); ++i )
    {
        // double tmp1 =  approximated_functions_[i].getValueAt(x[i]);
        // jsem se presune kod z ApproxFunction::compute
        double step_size, min_sample;
        std::tie(step_size, min_sample) = decision_function_info_[indx * features_dim_ + i];

        double f_interpolation = (x[i] - min_sample)/step_size;

        int left = std::floor(f_interpolation);
        int right = left + 1;

        left = std::min(left, num_steps);
        left = left > 0 ? left : 0;
        right = std::min(right, num_steps);
        right = right > 0 ? right : 0;

        std::size_t dec_fn_indx = (indx * features_dim_ + i) * approx_count_;
        double alpha = f_interpolation - left;
        double f_left = decision_function_[dec_fn_indx + left];
        double f_right = decision_function_[dec_fn_indx + right];

        value += ( 1 - alpha ) * f_left + alpha * f_right;
    }

    return value;
}

double IKSVM::evalDecisionFunction(std::size_t indx, const std::vector<double> & x, std::size_t offset)
{
    double value = decision_values_b_[indx];
    //tady pujde sse
    int num_steps = approx_count_ - 1;
    for ( size_t i = 0; i < features_dim_; ++i )
    {
        // double tmp1 =  approximated_functions_[i].getValueAt(x[i]);
        // jsem se presune kod z ApproxFunction::compute
        double step_size, min_sample;
        std::tie(step_size, min_sample) = decision_function_info_[indx * features_dim_ + i];

        double f_interpolation = (x[i + offset] - min_sample)/step_size;

        int left = std::floor(f_interpolation);
        int right = left + 1;

        left = std::min(left, num_steps);
        left = left > 0 ? left : 0;
        right = std::min(right, num_steps);
        right = right > 0 ? right : 0;

        std::size_t dec_fn_indx = (indx * features_dim_ + i) * approx_count_;
        double alpha = f_interpolation - left;
        double f_left = decision_function_[dec_fn_indx + left];
        double f_right = decision_function_[dec_fn_indx + right];

        value += ( 1 - alpha ) * f_left + alpha * f_right;
    }

    return value;
}


std::pair<double, std::vector<double> > IKSVM::predictProbability
                                        ( const std::vector<double> &x )
{
    double **pairwise_prob = new double*[nr_class_];
    for ( int i = 0; i < nr_class_; ++i )
    {
        pairwise_prob[i] = new double[nr_class_];
    }

    vector<double> decision_values; 
    computeDecisionsValue( x, decision_values ); 
    static double min_prob = 1e-7;
    int p = 0;
    for ( int i = 0; i < nr_class_; ++i )
    {
        for ( int j = i + 1; j < nr_class_; ++j )
        {
            double sigmoid_predict_value = sigmoid_predict( decision_values[p], 
                    prob_A_[p], prob_B_[p] );
            double prob = std::min( std::max(sigmoid_predict_value, min_prob), 1 - min_prob );
            pairwise_prob[i][j] = prob;
            pairwise_prob[j][i] = 1 - prob;
            ++p;
        }
    }

    std::vector<double> prob_estimates( nr_class_, 0 );
    multiclass_probability( nr_class_, pairwise_prob, prob_estimates.data() ); 
    auto max_it = std::max_element( prob_estimates.begin(), prob_estimates.end() );
    double label = labels_[ max_it - prob_estimates.begin() ];
    for ( int i = 0; i < nr_class_; ++i )
    {
        delete [] pairwise_prob[i];
    }
    delete [] pairwise_prob;
    return std::make_pair( label, prob_estimates );
}


std::pair< std::vector<double>, std::vector<double> > IKSVM::predictProbabilityMultiple
                                        ( const std::vector<double> &x )
{
    double **pairwise_prob = new double*[nr_class_];
    for ( int i = 0; i < nr_class_; ++i )
    {
        pairwise_prob[i] = new double[nr_class_];
    }

    std::size_t count = x.size()/features_dim_;
    vector<double> decision_values; 
    computeDecisionsValueMult( x, decision_values ); 
     
    std::vector<double> prob_estimates( count * nr_class_, 0 );
    std::size_t num_classifiers = nr_class_ *(nr_class_ - 1)/2;

    std::vector<double> results(count, 0);

    static double min_prob = 1e-7;
    for (std::size_t k = 0; k < count; ++k)
    {
        int p = 0;
        for ( int i = 0; i < nr_class_; ++i )
        {
            for ( int j = i + 1; j < nr_class_; ++j )
            {
                double sigmoid_predict_value = sigmoid_predict(decision_values[k * num_classifiers + p], 
                        prob_A_[p], prob_B_[p] );
                double prob = std::min( std::max(sigmoid_predict_value, min_prob), 1 - min_prob );
                pairwise_prob[i][j] = prob;
                pairwise_prob[j][i] = 1 - prob;
                ++p;
            }
        }

        int indx = k * nr_class_;
        multiclass_probability( nr_class_, pairwise_prob, &prob_estimates[indx]);
        auto it = prob_estimates.begin() + indx;

        auto max_it = std::max_element( it, it + nr_class_);
        results[k] = labels_[ max_it - it];
    }


    for ( int i = 0; i < nr_class_; ++i )
    {
        delete [] pairwise_prob[i];
    }
    delete [] pairwise_prob;

    return std::make_pair( results, prob_estimates );
}



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


using namespace std;


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
    : approximated_functions_( approximated_functions ), b_(b)
{
    approximated_functions_.shrink_to_fit();
}

double DecisionFunction::compute( const std::vector<double> &x )
{
    double value = b_;
    for ( size_t i = 0; i < x.size(); ++i )
    {
        value += approximated_functions_[i].getValueAt(x[i]);
    }
    return value;
}

std::ostream& operator<<( std::ostream &oss, const DecisionFunction &fnc )
{
    oss << fnc.b_ << " ";
    for ( const auto &fn : fnc.approximated_functions_ )
    {
        oss << fn << " ";
    }
    
    return oss;
}


void DecisionFunction::loadFromString( const std::string &s, int features_dim )
{
    std::stringstream ss(s);
    ss >> b_;

    approximated_functions_.reserve( features_dim );
    for ( int i = 0; i < features_dim; ++i )
    {
        string sample_function_str;
        if (ss >> sample_function_str)
        {
            approximated_functions_.push_back ( 
                    strToApproxFunction(sample_function_str) ); 
        } 
        else
        {
            throw BadFileFormatting("");
        }
    }
}

ApproximatedFunction DecisionFunction::strToApproxFunction(const std::string &s)
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
    return ApproximatedFunction( min_sample, step_size ,function_values );
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

    return IKSVM( nr_class_, features_dim_, prob_a, prob_b, desicion_functions, labels ); 
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
const std::string IKSVM::probA_text = "prob A";
const std::string IKSVM::probB_text = "prob B";
const std::string IKSVM::label_text = "labels";
const std::string IKSVM::decision_function_text = "decision functions";



IKSVM::IKSVM( int nr_class, int features_dim,
        const std::vector<double> prob_A, const std::vector<double> prob_B,
        const std::vector<DecisionFunction> desicion_functions, 
        const std::vector<double> labels )
    : nr_class_(nr_class), features_dim_(features_dim),
    prob_A_(prob_A), prob_B_(prob_B),
    desicion_functions_(desicion_functions), labels_(labels)
{

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

    cout << desicion_functions_.size() << endl;
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

    ofs << decision_function_text << endl;
    for ( const auto &des_fnc: desicion_functions_ )
    {
        ofs << des_fnc << endl;
    }

    ofs.close();
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
    desicion_functions_.reserve( number_subproblems );
    for ( int i = 0; i < number_subproblems; ++i )
    {
        if ( std::getline( ifs, line ) )
        {
            DecisionFunction fnc;
            fnc.loadFromString(line, features_dim_);
            desicion_functions_.push_back( fnc ); 
        }
        else 
        {
            throw BadFileFormatting("");
        }
    }
    desicion_functions_.shrink_to_fit();
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
            double decision_value = desicion_functions_[p].compute(x);
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



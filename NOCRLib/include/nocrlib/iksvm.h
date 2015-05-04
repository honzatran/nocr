/**
 * @file iksvm.h
 * @brief svm with fast intersection kernel
 * @author Tran Tuan Hiep
 * @version 
 * @date 2014-09-14
 */

#ifndef NOCRLIB_IKSVM_H
#define NOCRLIB_IKSVM_H

#define DECISION_FUNC 1

#include <libsvm/svm.h>

#include <vector>
#include <string>
#include <ostream>
#include <thread>

#include <pugi/pugixml.hpp>

#include "assert.h"


/// @cond
class ApproximatedFunction 
{
    public:
        ApproximatedFunction( double min_sample, double step_size,
                const std::vector<double> & function_values );

        double getValueAt(double d);
        friend std::ostream& operator<< ( std::ostream &oss, const ApproximatedFunction &fnc );

        double getMinSample() const { return min_sample_; }
        double getStepSize() const { return step_size_; }
        std::vector<double> getFunctionValues() const { return function_values_; } 

    private:
        int num_steps_;
        double min_sample_;
        double step_size_;
        std::vector<double> function_values_;
        // std::vector<size_t> indices_;

        double getFunctionSampleAt( size_t index );

        const static double epsilon_;
};

class DecisionFunction
{
    public:
        DecisionFunction() : b_(0) { }

        DecisionFunction( 
                const std::vector<ApproximatedFunction> &approximated_functions, 
                double b );

        double compute( const std::vector<double> &x );
        
        void loadFromString( const std::string &s, int features_dim );

        void strToApproxFunction(const std::string &s);

        friend std::ostream& operator<<( std::ostream &oss, const DecisionFunction &fnc );
    private:
        std::size_t fuction_approx_count_;
        std::size_t feature_dim_;
        double b_;
         
        std::vector<double> function_values_;
        std::vector< std::pair<double, double> > step_size_;

        friend class IKSVM;
};


/// @endcond

class IKSVM;

/**
 * @brief class that converts class svm_problem from LibSVM to svm with fast 
 * intersection kernel proposed by Malik and co.
 */
class IKSVMConvertor 
{
    public:
        IKSVMConvertor() = default;
        /**
         * @brief constructor
         *
         * @param features_dim descriptor dimension
         */
        IKSVMConvertor( int features_dim );


        /**
         * @brief converts svm_problem to iksvm 
         *
         * @param model svm_problem we are converting 
         * @param num_segment number of line to one desion function
         *
         * @return class IKSVM prepared for classification
         *
         * Algorithm for converting follows algorithm proposed 
         * by Malik and co. \p model must be train using LibSVM 
         * with intersection kernel.
         */
        IKSVM createFromSvmProblem( svm_model *model, int num_segment );

        /**
         * @brief loads svm_model from file, converts svm_problem to iksvm
         *
         * @param problem_file file where svm_model is saved
         * @param num_segment number of line to one desion function
         *
         * @return instance of IKSVM
         *
         * Algorithm for converting follows algorithm proposed 
         * by Malik and co. \p model must be train using LibSVM 
         * with intersection kernel.
         */
        IKSVM createFromSvmProblem( const std::string &problem_file, int num_segment );

    private:
        template <typename T> using Matrix = std::vector< std::vector<T> >;


        int nr_class_;
        int total_sv_;
        int features_dim_;
        int num_segment_;
        svm_parameter param_;

        std::vector<int> number_sv_;

        std::vector<int> getStartsOfSv();
        Matrix<double> getAllSv( int i, svm_node **support_vectors, const std::vector<int> &start );
        std::vector<double> getSvCoef( int label, int start, int count, double **sv_coef_ );

        std::vector<double> convertToVector( svm_node *support_vector );

        std::vector<ApproximatedFunction> approximateDecisionFunction
            ( const Matrix<double> &a_support_vectors, const std::vector<double> &a_sv_coef,
              const Matrix<double> &b_support_vectors, const std::vector<double> &b_sv_coef );

        typedef std::vector< std::pair<double,size_t> > VectorPair;
        VectorPair sortWithIndices( const std::vector<double> &a, const std::vector<double> &d );

        ApproximatedFunction approximateFunction( const std::vector<double> &alpha,
                const VectorPair &sorted_sequence, const std::vector<double> &samples );


};

/**
 * @brief represents svm with fast intersection kernel
 */
class IKSVM
{
    public:
        IKSVM() : nr_class_(0), features_dim_(-1) { }

        /**
         * @brief save iksvm configuration to file 
         *
         * @param file path to file
         */
        void save( const std::string &file );

        void saveXml(const std::string & file);
        

        /**
         * @brief loads iksvm configuration from file
         *
         * @param file path to file
         */
        void load( const std::string &file );

        /**
         * @brief predict class for descriptor x
         *
         * @param x descriptor 
         *
         * @return label of predicted class
         */
        double predict( const std::vector<double> &x );

        
        /**
         * @brief predict class for all descriptors in x
         *
         * @param x vector matrix[count descriptors, descriptor dimension]
         *
         * @return vector of labels 
         */
        std::vector<double> predictMultiple(const std::vector<double> & x);

        /**
         * @brief predict class for descriptor x and its probability outputs
         *
         * @param x descriptor 
         *
         * @return label of predicted class and vector of probabilities for all classes
         */
        std::pair<double, std::vector<double> > predictProbability( const std::vector<double> &x );

        std::pair<std::vector<double>, std::vector<double> > predictProbabilityMultiple( const std::vector<double> &x );

        int getNumberOfClasses() const { return nr_class_; }
    private:
        IKSVM( int nr_class, int features_dim, int approx_count,
               const std::vector<double> prob_A, const std::vector<double> prob_B,
               const std::vector<DecisionFunction> desicion_functions, 
               const std::vector<double> labels ); 

        int nr_class_;
        int features_dim_; 
        int approx_count_;
        std::vector<double> prob_A_;
        std::vector<double> prob_B_;
        // std::vector<DecisionFunction> desicion_functions_;

        std::vector<double> decision_function_;
        std::vector<double> decision_values_b_;
        std::vector<std::pair<double, double> > decision_function_info_;

        std::vector<double> labels_;

        bool startsWith( const std::string &s, const std::string &start);
        std::string parse( const std::string &start, const std::string &line );
        void parseDecisionFunction(const std::string &line, std::size_t indx);
        std::vector<double> strToVec( const std::string &start );

        void saveXmlDF(pugi::xml_node & df_node, const DecisionFunction & fn);

        double computeDecisionsValue( const std::vector<double> &x, 
                std::vector<double> &decision_values );

        std::vector<double> computeDecisionsValueMult(const std::vector<double> & x,
                std::vector<double> & decision_values);

        double evalDecisionFunction(std::size_t indx, const std::vector<double> & x); 

        double evalDecisionFunction(std::size_t indx, const std::vector<double> & x, std::size_t offset);
        

        const static std::string number_class_text;
        const static std::string feature_dimension_text;
        const static std::string approx_count;
        const static std::string probA_text;
        const static std::string probB_text;
        const static std::string label_text;
        const static std::string decision_function_text;

        friend class IKSVMConvertor;
};


#endif /* iksvm.h */

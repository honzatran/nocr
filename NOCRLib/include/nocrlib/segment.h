/**
 * @file segment.h
 * @brief contains declaration of class Segment, that extracts letter candidates,
 * declaration of SegmentPolicy and classes that implements specific kind of extraction 
 * letter candidates from input image
 * @author Tran Tuan Hiep
 * @version 1.0
 * @date 2014-06-21
 */


#ifndef NOCRLIB_SEGMENT_H
#define NOCRLIB_SEGMENT_H

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>

#include <vector>
#include <map>
#include <iostream>
#include <ostream>
#include <chrono>
#include <type_traits>

#include "utilities.h"
#include "component.h"
#include "structures.h"
#include "abstract_ocr.h"
#include "assert.h"

/**
 * @brief policy class for Segment, see programming documentation for further details
 *
 * @tparam S type of class that implements specific kind of segmentation, 
 * \tp specifies this policy
 */
template <typename S> class SegmentationPolicy
{
    /*
     * required typedefs
     * MethodOutput type of method output after letter segmentation
     * VisualConvertor type of class that creates VisualInformation 
     * from MethodOutput
     */

    /* 
     * required static bool constant
     * static const bool k_perform_nm_suppresion
     */

    /*
     * following functions are neccesery to implement for integration 
     * with Segment<S>
     * 
     * static void initialize( VisualConvertor &visual_convertor, 
     *                         const cv::Mat &image );
     *
     * static std::vector<MethodOutput> extract
     *                       ( const std::unique_ptr<S> &method_ptr, 
     *                         const cv::Mat &image );
     *
     * static TranslationInfo translate
     *                      ( const std::unique_ptr<AbstractOCR> &ocr, 
     *                        const MethodOutput &output );
     *      
     * static bool haveSignificantOverlap
     *               ( const MethodOutput &a, 
     *                 const MethodOutput &b );
     *
     * static Letter convert
     *                  ( const VisualConvertor &convertor,
     *                    const MethodOutput &a, 
     *                    const TranslationInfo &translation );
     */
};

template <typename OCR, typename T>
struct SegmentOCRPolicy
{
    static std::vector<TranslationInfo> translate(OCR * ocr, 
            const std::vector<T> & objects);
};

/**
 * @brief creates mask for vector of letters using non max suppresion described in
 * programming documentation
 */
class MaskCreator
{
    public:
        /**
         * @brief initialize MaskCreater
         *
         * @param size size of vector of letters
         */
        MaskCreator( size_t size );

        /**
         * @brief updates mask with given TranslationInfos
         *
         * @param a first translation info
         * @param i index of letter in vector, to whom TranslationInfo \p a belongs to
         * @param b second translation info
         * @param j index of letter in vector, to whom TranslationInfo \p b belongs to
         */
        void update( const TranslationInfo &a, size_t i, 
                const TranslationInfo &b, size_t j );
        std::vector<bool> getMask() const { return mask_; }
    private:
        std::vector<bool> mask_;
};


/**
 * @brief segments character candidates from image, for further details see programming documentation
 *
 *
 * @tparam T type of class used for segmentation
 */
template <typename T, typename OCR> class Segment 
{
    public:
        typedef typename SegmentationPolicy<T>::MethodOutput MethodOutput; 
        typedef typename SegmentationPolicy<T>::VisualConvertor VisualConvertor;

        /**
         * @brief Default constructor
         */
        Segment() : method_ptr_(nullptr), ocr_(nullptr) { }

        /**
         * @brief loads unique pointer to class T implementing specific segmentation 
         *
         * @param ptr to instance of T 
         */
        void loadMethod( T *method_ptr )
        {
            method_ptr_ = method_ptr;
        }

        /**
         * @brief loads unique pointer to OCR
         *
         * @param ocr
         */
        void loadOcr(OCR *ocr)
        {
            ocr_ = ocr;
        }

        /**
         * @brief segment character canditates from image, cv::Mat image must have 
         * BGR format
         *
         * @param image, input image 
         *
         * @return vector of Letters 
         */
        std::vector<Letter> segment(const cv::Mat &image) 
        {
            NOCR_ASSERT( method_ptr_ != nullptr, "pointer to method isn't loaded yet" );
            NOCR_ASSERT( ocr_ != nullptr, "pointer to ocr isn't loaded yet" );

#if PRINT_TIME
            timeCounter<std::chrono::steady_clock, 
                std::chrono::milliseconds>  tC;
            auto begin = std::chrono::steady_clock::now();
#endif

            // extract candidates
            std::vector<MethodOutput> letter_candidates = 
                                SegmentationPolicy<T>::extract( method_ptr_, image );

#if PRINT_TIME
            auto end = std::chrono::steady_clock::now();
            std::cout << "segmentation takes: " << tC(begin,end).count()
                << " ms" << std::endl;
#endif 
            // set ocr and visual_convertor_
            ocr_->setImage( image );
            SegmentationPolicy<T>::initialize( visual_convertor_, image );
            
            if ( SegmentationPolicy<T>::k_perform_nm_suppresion )
            {
                return nonMaxSuppresion( letter_candidates );
            }

            // no nonmax suppresion performed, return all candidates
            std::vector<TranslationInfo> translations = SegmentOCRPolicy<OCR, MethodOutput>::translate(ocr_, letter_candidates);
            // every letter candidate is extracted because mask is set true for all 
            // of them, this means that we consider all of them to be maximal.
            std::vector<bool> mask( letter_candidates.size(), true );
            return extractMaximal( letter_candidates, translations, mask );
        }

    private:
        T *method_ptr_;
        VisualConvertor visual_convertor_;
        OCR * ocr_;



        std::vector<Letter> nonMaxSuppresion( const std::vector<MethodOutput> &objects ) 
        {
#if PRINT_TIME
            timeCounter<std::chrono::steady_clock, 
                std::chrono::milliseconds>  tC;
            auto begin = std::chrono::steady_clock::now();
#endif
            // std::vector<TranslationInfo> translations = translate( objects );
            std::vector<TranslationInfo> translations = SegmentOCRPolicy<OCR, MethodOutput>::translate(ocr_,
                    objects);
#if PRINT_TIME
            auto end = std::chrono::steady_clock::now();
            std::cout << "ocr phase takes: " << tC(begin,end).count() 
                << " ms" << std::endl;
#endif 
            std::vector<bool> mask = getMaximal(objects, translations);
            return extractMaximal(objects,translations,mask);
        }

        std::vector<TranslationInfo> translate( const std::vector<MethodOutput> &objects ) 
        {
            std::vector<TranslationInfo> translations;
            translations.reserve( objects.size() );
            for ( const auto &object : objects )
            {
                translations.push_back( SegmentationPolicy<T>::translate(ocr_, object) );
            }
            return translations;
        }

        std::vector<bool> getMaximal( const std::vector<MethodOutput> &objects, 
                const std::vector<TranslationInfo> &translations ) 
        {
            MaskCreator mask_creator( translations.size() );
            for ( size_t i = 0; i < translations.size(); ++i )
            {
                for ( size_t j = i + 1; j < translations.size(); ++j )
                {
                    if ( !SegmentationPolicy<T>::
                            haveSignificantOverlap( objects[i], objects[j] ) )
                    {
                        continue;
                    }
                    mask_creator.update( translations[i], i, translations[j], j );
                }
            }
            return mask_creator.getMask();
        }

        std::vector<Letter> extractMaximal( const std::vector<MethodOutput> &objects, 
                const std::vector<TranslationInfo> &translations, 
                const std::vector<bool> &mask ) 
        {
            std::vector<Letter> output;
            for ( size_t i = 0; i < mask.size(); ++i )
            {
                if ( mask[i] ) 
                {
                    output.push_back( SegmentationPolicy<T>::
                            convert( visual_convertor_, objects[i], translations[i] ) );
                }
            }

#if PRINT_TIME
            std::cout << "ocr max suppresion:" << translations.size() 
                << "/" << output.size() << std::endl;
#endif
            return output;
        }
};



#endif

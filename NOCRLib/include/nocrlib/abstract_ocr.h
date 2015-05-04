#ifndef NOCRLIB_ABSTRACT_OCR_H
#define NOCRLIB_ABSTRACT_OCR_H

#include "component.h"

/**
 * @brief Base abstract class for OCR with probabilities outputs
 */
class AbstractOCR 
{
    public:

        /**
         * @brief translate component to asci code in given alphabet
         *
         * @param c_ptr pointer to input component
         * @param probabilities vector, where probabilities output will be stored
         *
         * @return label from alphabent of the component
         *
         * This method performs character recognition on input component, it 
         * works with shared pointer c_ptr to input component.
         * Probability output will be stored in \p probabilities.
         */
        char translate( const std::shared_ptr<Component> &c_ptr, std::vector<double> &probabilities ) 
        {
            return translate( *c_ptr, probabilities );
        }

        /**
         * @brief translate component to asci code in given alphabet
         *
         * @param c input component
         * @param probabilities vector, where probabilities output will be stored
         *
         * @return label from alphabent of the component
         *
         * This method performs character recognition on input component c.
         * Probability output will be stored in \p probabilities.
         */
        virtual char translate( Component &c, std::vector<double> &probabilities ) = 0;

        virtual std::vector<char> translate(std::vector<Component> & components, std::vector<double> & probabilities)
        {
            std::vector<double> tmp;
            std::vector<char> letters;
            letters.reserve(components.size());

            for (auto & c : components)
            {
                letters.push_back(translate(c, tmp));
                probabilities.insert(probabilities.end(), tmp.begin(), tmp.end());
            }

            return letters;
        }

        virtual std::vector<char> translate(const std::vector< std::shared_ptr<Component> > & components, std::vector<double> & probabilities)
        {
            std::vector<double> tmp;
            std::vector<char> letters;
            letters.reserve(components.size());

            for (auto & c : components)
            {
                letters.push_back(translate(*c, tmp));
                probabilities.insert(probabilities.end(), tmp.begin(), tmp.end());
            }

            return letters;
        }

        /**
         * @brief set current image, where all following components will be extracted
         *
         * @param image domain of component extraction
         *
         * Set current image.
         */
        virtual void setImage( const cv::Mat &image ) { UNUSED(image); };
};

#endif /* ocr_interface.h */

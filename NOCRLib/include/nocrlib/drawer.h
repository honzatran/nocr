/**
 * @file drawer.h
 * @brief classes for displaying output words
 * @author Tran Tuan Hiep
 * @version 
 * @date 2014-09-14
 */


#ifndef NOCRLIB_DRAWER_H
#define NOCRLIB_DRAWER_H

#include <opencv2/core/core.hpp>
#include <vector>

#include "component.h"
#include "structures.h"

/**
 * @brief Interface for drawing Component,Letter or Word on Canvas
 */
class DrawerInterface
{
    public:
        /**
         * @brief initialize canvas
         *
         * @param image input image from which objects, that will be drawn, 
         * were extracted
         */
        virtual void init( const cv::Mat &image ) = 0;

        /**
         * @brief draw component on canvas
         *
         * @param c component to be drawn
         */
        virtual void draw( const Component &c ) = 0;

        /**
         * @brief draw letter on canvas
         *
         * @param l letter to be drawn
         */
        virtual void draw( const Letter &l ) = 0;


        /**
         * @brief draw word on canvas
         *
         * @param w word to be drawn 
         */
        virtual void draw( const Word &w ) = 0; 

        /**
         * @brief returns canvas
         *
         * @return canvas
         */
        virtual cv::Mat getImage() = 0;
    private:
};

/**
 * @brief Draw bounding boxes on the 
 * canvas. Canvas is a cloned copy of original image. 
 */
class RectangleDrawer : public DrawerInterface
{
    public:
        RectangleDrawer();

        void init( const cv::Mat &image ) override;

        void draw( const Component &c ) override;

        void draw( const Letter &l ) override;

        void draw( const Word &w ) override;

        cv::Mat getImage() override { return canvas_; } 

        void drawRectangle( const cv::Rect &rect);

        void setColor(const cv::Scalar & color); 
    private:
        cv::Mat canvas_;
        cv::Scalar color_;
};



/**
 * @brief Draw objects shape on canvas. 
 */
class BinaryDrawer : public DrawerInterface
{
    public:
        typedef std::shared_ptr<Component> CompPtr;

        /**
         * @brief 
         *
         * @param background
         * @param foreground
         */
        BinaryDrawer(uchar background = 0, uchar foreground = 255 ) 
            : background_(background), foreground_(foreground) { }

        ~BinaryDrawer() { }

        void init( const cv::Mat &image ) override;

        void draw( const Component &c ) override;

        void draw( const Letter &l ) override;

        void draw( const Word &w ) override;

        cv::Mat getImage() override { return canvas_; } 



    private:
        cv::Mat canvas_;
        uchar background_;
        uchar foreground_;
};

template <typename T>
inline void draw( 
        DrawerInterface * drawer_ptr,  
        const std::vector<T> &drawables )
{
    for ( const T & c: drawables)
    {
        drawer_ptr->draw(c);
    }
}

template <>
inline void draw(
        DrawerInterface * drawer_ptr,
        const std::vector<TranslatedWord> &translated_words )
{
    for ( const TranslatedWord & tw :translated_words )
    {
        drawer_ptr->draw(tw.visual_information_);
    }
}



#endif

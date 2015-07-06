/**
 * @file exception.h
 * @brief file containing exception
 * @author Tran Tuan Hiep
 * @version 
 * @date 2014-09-15
 */

#include <exception>
#include <string>
#include <iostream>

#ifndef NOCRLIB_EXCEPTION_H
#define NOCRLIB_EXCEPTION_H

#ifndef _MSC_VER
#define NOEXCEPT noexcept
#else 
#define NOEXCEPT
#endif


/**
 * @brief exception is thrown when unimplemented or 
 * unsupported method is called
 *
 */
class UnsupportedOperation : public std::exception
{
    public:
        UnsupportedOperation( const std::string &msg )
            : msg_(msg)
        {

        }

        virtual const char* what() const NOEXCEPT override
        {
            std::string final_msg = "UnsupportedOperation: " + msg_;
            return final_msg.c_str();
        }
    private:
        std::string msg_;
};

/**
 * @brief exception is thrown when file is not found
 */
class FileNotFoundException : public std::exception
{
    public:
        FileNotFoundException( const std::string &msg )
            : msg_(msg)
        {

        }

        virtual const char* what() const NOEXCEPT override
        {
            std::string final_msg = "FileNotFoundException: " + msg_;
            return final_msg.c_str();
        }
    private:
        std::string msg_;

};

/**
 * @brief exception is thrown when bad file formating occurs
 */
class BadFileFormatting : public std::exception
{
    public:
        BadFileFormatting( const std::string &msg )
            : msg_(msg)
        {

        }

        void appendMsg( const std::string &msg )
        {
            msg_ += ' ';
            msg_.append(msg);
        }

        virtual const char* what() const NOEXCEPT override
        {
            std::string final_msg = "BadFileFormatting:" + msg_;
            return final_msg.c_str();
        }
    private:
        std::string msg_;

};

template <typename T> 
class NocrException : public std::exception
{
    public:
        NocrException( const std::string &msg )
            : msg_(msg)
        {

        }

        void appendMsg( const std::string &msg )
        {
            msg_ += ' ';
            msg_.append(msg);
        }

        virtual const char* what() const NOEXCEPT override
        {
            std::string final_msg = "BadFileFormatting:" + msg_;
            return final_msg.c_str();
        }
    private:
        std::string msg_;
};


/**
 * @brief exception is thrown when bad file formating occurs
 */
class ActionError: public std::exception
{
    public:
        ActionError( const std::string &action)
            : action_(action)
        {

        }

        virtual const char* what() const NOEXCEPT override
        {
            std::string final_msg =  action_ + "cannot be done: "; 
            return final_msg.c_str();
        }
    private:
        std::string action_,msg_;

};

#endif /* exception.h */

/**
 * @file assert.h
 * @brief Assert for NOCR library 
 * @author Tran Tuan Hiep
 * @version 1.0
 * @date 2014-09-12
 */

/*
 * Compiler: g++ 4.8.3, 
 */

#ifndef NOCRLIB_ASSERT_H
#define NOCRLIB_ASSERT_H

#ifndef NDEBUG
#define NOCR_ASSERT(condition, message) \
    do \
    { \
        if (! (condition)) \
        { \
            std::cerr << "Assertion `" #condition "` failed in " << __FILE__ \
            << " line " << __LINE__ << ": " << message << std::endl; \
            std::exit(EXIT_FAILURE); \
        } \
} while (false)
#else
#   define NOCR_ASSERT(condition, message) do { } while (false)
#endif



#endif /* assert.h */





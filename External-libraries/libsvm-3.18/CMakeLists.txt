add_library( LibSVM SHARED ./libsvm/svm.h ./libsvm/svm.cpp )

add_definitions( -Wall -g -std=c++0x )

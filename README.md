NOCR
====

##Overview
NOCR is an open source C++ software package for text recognition in natural scenes, based on OpenCV. The package consists of 
a library, console program and GUI program for text recognition.

###Platforms
NOCR is for now only compatible with linux platform.

###External library used
  1.  [OpenCV](http://opencv.org/) - standard C++ library for computer vision, version 2.4.10, 
  2.  [Qt](http://www.qt.io/) - used for implementation of the GUI program, version 5.
  3.  [Boost](http://www.boost.org/) - used library for memory pool
  4.  [PugiXML](http://pugixml.org/) - xml library 
  5.  [LibSVM](http://www.csie.ntu.edu.tw/~cjlin/libsvm/) - support vector machine library 

The latter two are already a part of our package.


##Library
We have decomposed main problem into three distinct subproblems, letter localization, OCR with probability outputs and word formatting using located letters. 
We localize letters on the input bitmap in the first phase, afterwards words are detected from recognized letters using probality outputs from OCR and letter's location on image.

###Letter Localization
In our thesis we proposed to solve letter localization problem using 
[algorithm ER](http://cmp.felk.cvut.cz/~matas/papers/neumann-2012-rt_text-cvpr.pdf) proposed by Neumann and Matas.
The implementation was designed for research purposes, so it is easily upgratable and offers easy way to modifify the algorithm.
It's divided into two seperated phases. Look at source codes of [algorithm for first phase of ER](/nocr/NOCRLib/include/nocrlib/component_tree_builder.h) and 
header of second phase algorithm's implementation [header](/nocr/NOCRLib/include/nocrlib/extremal_region.h) and correspoding [cpp file](/nocr/NOCRLib/src/extremal_region.cpp).

###OCR
OCR with probability output is implemented using support vector machine from LibSVM and direction histogram.
For further details see [OCR interface](/nocr/NOCRLib/include/nocrlib/abstract_ocr.h), [OCR implementation](/nocr/NOCRLib/include/nocrlib/ocr.h) 
and [direction histogram](/nocr/NOCRLib/include/nocrlib/direction_histogram.h)

###Word formation
Second part of our thesis was focused on improving existing algorithm for word formatting from [Phan a co.](http://www.comp.nus.edu.sg/~tians/papers/ICCV2013_PerspectiveTextRecognition_Phan.pdf) See the [header file](/nocr/NOCRLib/include/nocrlib/word_generator.h)
for further details.

###Integration
Our library is providing easy integration of any algorithm for letter localization and any OCR with probability outputs and the modified algorithm for word formation
using *policy classes*. See [text recognition header](/nocr/NOCRLib/include/nocrlib/text_recognition.h) for further details.


##Console application
Our console application uses api provided from the nocr library for text recognition in natural scenes. 
It also supports XML and plane text output. For futher details use option *help*.

##GUI application
GUI application using the library provided api.


##Installation
If you don't have installed OpenCV, version 2.4.10 and Boost library, install them first.

    1.  git clone https://github.com/honzatran/nocr
    2.  mkdir release && cd release
    3.  cmake -DCMAKE_BUILD_TYPE=RELEASE ..
    4.  make

Directory *bin* contains console and GUI application binaries and directory *lib* contains shared library NOCR.

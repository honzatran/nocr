set ( HEADERS 
    ./include/nocrlib/structures.h
    ./include/nocrlib/drawer.h  
    ./include/nocrlib/ocr.h   
    ./include/nocrlib/swt.h
    ./include/nocrlib/utilities.h   
    ./include/nocrlib/component.h  
    ./include/nocrlib/segment.h
    ./include/nocrlib/features.h    
    ./include/nocrlib/iooper.h    
    ./include/nocrlib/train_data.h
    ./include/nocrlib/classifier_wrap.h  
    ./include/nocrlib/letter_equiv.h 
    ./include/nocrlib/feature_factory.h
    ./include/nocrlib/dictionary.h 
    ./include/nocrlib/exception.h 
    ./include/nocrlib/er_region.h
    ./include/nocrlib/word_generator.h 
    ./include/nocrlib/trie_node.h 
    ./include/nocrlib/levenstein_distance.h 
    ./include/nocrlib/text_recognition.h 
    ./include/nocrlib/bag_of_words.h 
    ./include/nocrlib/knn_ocr.h 
    ./include/nocrlib/direction_histogram.h 
    ./include/nocrlib/iksvm.h 
    ./include/nocrlib/component_tree_builder.h
    ./include/nocrlib/testing.h
    ./include/nocrlib/opencv_mser.h
    ./include/nocrlib/swt_segmentation.h
    ./include/nocrlib/ground_truth_impl.h
    )
  
set ( SOURCES 
    ./src/component.cpp  
    ./src/iooper.cpp
    ./src/swt.cpp 
    ./src/letter_equiv.cpp  
    ./src/segment.cpp 
    ./src/dictionary.cpp 
    ./src/features.cpp 
    ./src/ocr.cpp 
    ./src/er_region.cpp 
    ./src/classifier_wrap.cpp 
    ./src/word_generator.cpp  
    ./src/structures.cpp
    ./src/trie_node.cpp 
    ./src/levenstein_distance.cpp 
    ./src/text_recognition.cpp 
    ./src/bag_of_words.cpp 
    ./src/knn_ocr.cpp 
    ./src/direction_histogram.cpp 
    ./src/iksvm.cpp 
    ./src/extremal_region.cpp 
    ./src/drawer.cpp
    ./src/train_data.cpp
    ./src/testing.cpp
    ./src/opencv_mser.cpp
    ./src/swt_segmentation.cpp
    ./src/street_view_scene.cpp
    ./src/ground_truth_impl.cpp
    )
    
add_library( NOCRLib SHARED ${HEADERS} ${SOURCES} )
add_definitions( -DPRINT_INFO -DPRINT_TIME)



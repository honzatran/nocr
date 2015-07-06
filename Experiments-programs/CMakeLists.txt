set (TRAIN_BIN ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/experiment)

if(IS_DIRECTORY ${TRAIN_BIN})
    file(MAKE_DIRECTORY  ${TRAIN_BIN})
endif(IS_DIRECTORY ${TRAIN_BIN})

set( BIN_OUTPUT ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
set( CMAKE_RUNTIME_OUTPUT_DIRECTORY ${TRAIN_BIN})
add_subdirectory( ./icdar2013-evaluation)
add_subdirectory( ./result-drawer)
add_subdirectory( ./letter-segmentation)
add_subdirectory( ./modifikace-evaluation)
add_subdirectory( ./word-generator)
set ( CMAKE_RUNTIME_OUTPUT_DIRECTORY ${BIN_OUTPUT})





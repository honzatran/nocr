set (TRAIN_BIN ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/experiment)

if(IS_DIRECTORY ${TRAIN_BIN})
    file(MAKE_DIRECTORY  ${TRAIN_BIN})
endif(IS_DIRECTORY ${TRAIN_BIN})

set( BIN_OUTPUT ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
set( CMAKE_RUNTIME_OUTPUT_DIRECTORY ${TRAIN_BIN})
add_subdirectory( ./icdar2013-evaluation)
add_subdirectory( ./svt-evaluation)
add_subdirectory( ./1stage-letter-evaluation)
# add_subdirectory( ./gt-extractor)
# add_subdirectory( ./1-stage-scores)
# add_subdirectory( ./1-stage-test)
add_subdirectory( ./ocr-evaluation)
set ( CMAKE_RUNTIME_OUTPUT_DIRECTORY ${BIN_OUTPUT})





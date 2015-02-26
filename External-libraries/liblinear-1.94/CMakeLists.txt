set( LibLinearObjectFile 
    ./liblinear/blas/ddot.o 
    ./liblinear/blas/dscal.o  
    ./liblinear/blas/dnrm2.o
    ./liblinear/blas/daxpy.o )

add_library( LibLinear SHARED
    ./liblinear/linear.h 
    ./liblinear/linear.cpp 
    ./liblinear/tron.h
    ./liblinear/tron.cpp
    ${LibLinearObjectFile}
    )

# add_definitions( -Wall -g -std=c++0x -lblas -I${VLROOT} -L${VLROOT}/bin/glnx86 -lvl )
add_definitions( -Wall -g )

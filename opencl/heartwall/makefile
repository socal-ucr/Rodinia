include ../../common/make.config

CXX = g++
DEBUGFLAGS = -g
# You will need to modify the following 2 lines
# by inserting the correct path to the OpenCL
# lib and inc directories.
CXXFLAGS = -I$(OPENCL_INC) -c $(DEBUGFLAGS)
LINKFLAGS = -L$(OPENCL_LIB) -lOpenCL

# link objects(binaries) together
a.out: main.o OpenCL.o avilib.o avimod.o
	$(CXX) $(LINKFLAGS) main.o OpenCL.o avilib.o avimod.o -o heartwall

# compile main function file into object (binary)
main.o: main.cpp OpenCL.cpp kernels.cl avilib.c avimod.c
	$(CXX) $(CXXFLAGS) main.cpp OpenCL.cpp avilib.c avimod.c

clean:
	clear
	@rm -f *.o *.out *~ *\#

include ../../common/make.config

CXX = g++
DEBUGFLAGS = -g
# You will need to modify the following 2 lines
# by inserting the correct path to the OpenCL
# lib and inc directories.
CXXFLAGS = -I$(OPENCL_INC) $(DEBUGFLAGS)
LINKFLAGS = -L/$(OPENCL_LIB) -lOpenCL

ALLFLAGS = $(CXXFLAGS) $(LINKFLAGS)

all:
	$(CXX) $(ALLFLAGS) *.cpp -o srad

clean:
	clear
	@rm -f srad *.o *.out *~ *\#

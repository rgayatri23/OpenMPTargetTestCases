SHELL = /bin/sh

EXE = test.ex
SRC = TestHierarchical.cpp

CXX=clang++
CXXFLAGS += -std=c++11
CXXFLAGS += -fopenmp
CXXFLAGS += -fopenmp-targets=nvptx64-nvidia-cuda --cuda-path=${CUDA_PATH} -I${CUDA_PATH}/include -ffp-contract=fast
DEFINE =

#==========================
# Make the executable
#==========================
$(EXE): $(SRC)
	echo $(SRC)
	$(CXX) $(CXXFLAGS) $(DEFINE) $(SRC) -o $(EXE)

#==========================
#remove all objs
#==========================
clean:
	/bin/rm -f *.o $(EXE)

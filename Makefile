CXX=g++
NVCC=nvcc
CXXFLAGS=-O3
NVCCFLAGS=-O3 -arch=sm_61

all: nbody nbodygpu

nbody: nbody.cpp
		$(CXX) $(CXXFLAGS) nbody.cpp -o nbody

nbodygpu: nbodygpu.cu
		$(NVCC) $(NVCCFLAGS) nbodygpu.cu -o nbodygpu

clean:
		rm -f nbody nbodygpu *.out

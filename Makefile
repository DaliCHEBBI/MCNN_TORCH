PREFIX=/home/mohamedali/Documents/Evaluation_MCC-CNN_CODE/MC_CNN_CPP_PYTORCH/libtorch
PREFIXOPENCV=$(HOME)/miniconda3/envs/TorchEnv
CFLAGS=-I$(PREFIX)/include -I$(PREFIX)/include/torch/csrc/api/include -I$(PREFIXOPENCV)/include/opencv4
LDFLAGS_NVCC=-L$(PREFIX)/lib -Xlinker -rpath,$(PREFIX)/lib -lpng12
LDFLAGS_CPP=-L$(PREFIX)/lib -Wl,-rpath=$(PREFIX)/lib
LIBS_TORCH=-ltorch -lc10 -lc10_cuda -ltorch_cuda -lcuda -lnvrtc -lnvToolsExt -ltorch_cpu -lcudart
LOPENCV_FLAGS=-L$(PREFIXOPENCV)/lib -Wl,-rpath=$(PREFIXOPENCV)/lib
LIBS_OPENCV=-lopencv_contrib -lopencv_stitching -lopencv_nonfree -lopencv_gpu -lopencv_photo -lopencv_imgcodecs -lopencv_legacy -lopencv_calib3d -lopencv_features2d -lopencv_highgui -ljasper -ltiff -lpng -ljpeg -lopencv_imgproc -lopencv_flann -lopencv_core -lGLU -lGL -lz -lrt -lpthread -lm -ldl -lstdc++ 



all: cv.o Census.o TestMainCpuGpu.o TestMainCpuGpu 

cv.o: cv.cpp
	/usr/bin/g++-9 $(LOPENCV_FLAGS) $(CFLAGS) -o cv.o -c cv.cpp $(LIBS_OPENCV)

Census.o: Census.cu
	nvcc -arch sm_75 -O3 -DNDEBUG --compiler-options '-fPIC' -o Census.o -c Census.cu $(CFLAGS) $(LDFLAGS_NVCC)

TestMainCpuGpu.o:TestMainCpuGpu.cpp  
	/usr/bin/g++-9 $(LDFLAGS_CPP) $(CFLAGS) -o TestMainCpuGpu.o -c TestMainCpuGpu.cpp $(LIBS_TORCH)

TestMainCpuGpu: TestMainCpuGpu.o cv.o Census.o
	/usr/bin/g++-9 $(LDFLAGS_CPP) $(LOPENCV_FLAGS) -o TestMainCpuGpu TestMainCpuGpu.o cv.o Census.o $(LIBS_TORCH) $(LIBS_OPENCV)

clean:
	rm -f cv.o TestMainCpuGpu.o Census.o TestMainCpuGpu



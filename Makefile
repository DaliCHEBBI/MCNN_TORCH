PREFIX=$(HOME)/Documents/Evaluation_MCC-CNN_CODE/MC_CNN_CPP_PYTORCH/libtorch
PREFIXMicMac=$(HOME)/opt/micmac
PREFIXOPENCV=$(HOME)/miniconda3/envs/TorchEnv
CFLAGS=-I$(PREFIX)/include -I$(PREFIX)/include/torch/csrc/api/include -I$(PREFIXOPENCV)/include/opencv4 -I$(PREFIXMicMac)/include
LDFLAGS_NVCC=-L$(PREFIX)/lib -Xlinker -rpath,$(PREFIX)/lib -lpng16
LDFLAGS_CPP=-L$(PREFIX)/lib -Wl,-rpath=$(PREFIX)/lib
LIBS_TORCH=-ltorch -lc10 -lc10_cuda -ltorch_cuda -lcuda -lnvrtc -lnvToolsExt -ltorch_cpu -lcudart
LOPENCV_FLAGS=-L$(PREFIXOPENCV)/lib -Wl,-rpath=$(PREFIXOPENCV)/lib
LIBS_OPENCV=-lopencv_contrib -lopencv_stitching -lopencv_nonfree -lopencv_gpu -lopencv_photo -lopencv_imgcodecs -lopencv_legacy -lopencv_calib3d -lopencv_features2d -lopencv_highgui -ljasper -ltiff -lpng -ljpeg -lopencv_imgproc -lopencv_flann -lopencv_core -lGLU -lGL -lz -lrt -lpthread -lm -ldl -lstdc++ 
LMICMAC_FLAGS=-L$(PREFIXMicMac)/lib -Wl,-rpath=$(PREFIXMicMac)/lib
LIBS_MICMAC=-lelise -lANN 
LIBS_QT5_OPENGL=-lQt5Core -lQt5Gui -lGL -lX11 -lQt5Xml -lQt5Widgets -lQt5OpenGL -lglut

all: cv.o Census.o TestMainCpuGpu.o TestMainCpuGpu Image.o PrepareDataset.o PrepareDataset

cv.o: cv.cpp
	/usr/bin/g++-9 $(LOPENCV_FLAGS) $(CFLAGS) -o cv.o -c cv.cpp $(LIBS_OPENCV)

Census.o: Census.cu
	nvcc -arch sm_50 -O3 -DNDEBUG --compiler-options '-fPIC' -o Census.o -c Census.cu $(CFLAGS) $(LDFLAGS_NVCC)

TestMainCpuGpu.o:TestMainCpuGpu.cpp  
	/usr/bin/g++-9 $(LDFLAGS_CPP) $(CFLAGS) -o TestMainCpuGpu.o -c TestMainCpuGpu.cpp $(LIBS_TORCH)

TestMainCpuGpu: TestMainCpuGpu.o cv.o Census.o
	/usr/bin/g++-9 $(LDFLAGS_CPP) $(LOPENCV_FLAGS) -o TestMainCpuGpu TestMainCpuGpu.o cv.o Census.o $(LIBS_TORCH) $(LIBS_OPENCV)
	
Image.o: Image.cpp
	/usr/bin/g++-9 $(CFLAGS) $(LMICMAC_FLAGS) -o Image.o -c Image.cpp $(LIBS_MICMAC)

PrepareDataset.o: PrepareDataset.cpp 
	/usr/bin/g++-9 $(LDFLAGS_CPP) $(CFLAGS) $(LMICMAC_FLAGS) -o PrepareDataset.o -c PrepareDataset.cpp $(LIBS_TORCH) $(LIBS_MICMAC)

PrepareDataset: PrepareDataset.o 
	/usr/bin/g++-9 $(LDFLAGS_CPP) $(CFLAGS) $(LMICMAC_FLAGS) -o PrepareDataset PrepareDataset.o Image.o Census.o $(LIBS_TORCH) $(LIBS_MICMAC) $(LIBS_QT5_OPENGL) -lstdc++fs -lpng16

clean:
	rm -f cv.o TestMainCpuGpu.o Census.o TestMainCpuGpu Image.o PrepareDataset.o PrepareDataset



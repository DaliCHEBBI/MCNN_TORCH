PREFIX=/home/mohamedali/Documents/Evaluation_MCC-CNN_CODE/MC_CNN_CPP_PYTORCH/libtorch
PREFIXOPENCV=$(HOME)/miniconda3/envs/TorchEnv
CFLAGS=-I$(PREFIX)/include -I$(PREFIX)/include/torch/csrc/api/include -I$(PREFIXOPENCV)/include/opencv4
LDFLAGS_NVCC=-L$(PREFIX)/lib -Xlinker -rpath,$(PREFIX)/lib -lpng12
LDFLAGS_CPP=-L$(PREFIXOPENCV)/lib -L$(PREFIX)/lib -lopencv_contrib -lopencv_stitching -lopencv_nonfree -lopencv_superres -lopencv_ocl -lopencv_ts -lopencv_videostab -lopencv_gpu -lopencv_photo -lopencv_objdetect -lopencv_legacy -lopencv_video -lopencv_ml -lopencv_calib3d -lopencv_features2d -lopencv_highgui -ljasper -ltiff -lpng -ljpeg -lopencv_imgproc -lopencv_flann -lopencv_core -lGLU -lGL -lz -lrt -lpthread -lm -ldl -lstdc++

all: libcv.so testMain

libcv.so: cv.cpp
	g++ -fPIC -o libcv.so -shared cv.cpp $(CFLAGS) $(LDFLAGS_CPP)


testMain:TestMainCpuGpu.cpp
	#g++ -fPIC -o testMain -c TestMainCpuGpu.cpp $(CFLAGS) $(LDFLAGS_CPP)
	g++ -Wall -g TestMainCpuGpu.cpp -o testMain $(CFLAGS) $(LDFLAGS_CPP)

clean:
	rm -f libcv.so testMain

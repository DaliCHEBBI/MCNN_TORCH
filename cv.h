#include <stdio.h>
#include <assert.h>
#include <stdint.h>
#include <unistd.h>

#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <opencv2/core.hpp>
//#include <opencv2/imgcodecs.hpp>
#include <torch/torch.h>


void warp_affine(torch::Tensor *srcT, torch::Tensor *dstT, torch::Tensor *matT);
torch::Tensor ReadImage(std::string filename);


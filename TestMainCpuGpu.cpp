#include <torch/torch.h>
#include <ATen/ATen.h>
#include <iostream>
#include <stdio.h>
#include <string>
#include <fstream>
#include <vector>
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <opencv2/core.hpp>
//#include <cudnn.h>
//#include "Convnet_Fast.h"
#include "cv.h"
//#include "SampleCensus.cuh"

//using namespace std;


int main() {
  
  float d_brightness =0.5;
  torch::Tensor a= torch::rand({1});
  int trans=5;  
  torch::Tensor transT = torch::randint(-trans, trans,{2});  
  std::cout<<transT.accessor<float,1>()[0]<<"  "<<transT.accessor<float,1>()[1]<<std::endl; 
  std::cout<<-2*d_brightness*a.accessor<float,1>()[0]+d_brightness<<std::endl;
  
  // another issue with the dims
  int ws=5;
  int n_input_plane=1;
  torch::Tensor test_T=torch::empty({4, n_input_plane,ws,ws});
  // read an image using opencv 
  torch::Tensor ImageRead=ReadImage("./image2.png");
  torch::Tensor Transf=torch::tensor({0.3,0.1,1.0,0.2,0.5,0.7}); 
  torch::Tensor defor=torch::empty({ImageRead.size(0),ImageRead.size(1),ImageRead.size(2)});
  warp_affine(&ImageRead,&defor,&Transf);
  
  // Tranform tensor to image opencv and imshow
  float *res = reinterpret_cast<float*>(defor.data_ptr());
  //cv::Mat one_mat=cv::Mat(defor.size(1), defor.size(2), CV_8UC3,res);
  //std::memcpy(one_mat.data, defor.data_ptr(), sizeof(int) * defor.numel());
  //cv::imwrite("starry_night.png", one_mat);
 return 0;
}

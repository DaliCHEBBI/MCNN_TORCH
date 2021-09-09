#include <stdio.h>
#include <assert.h>
#include <stdint.h>
#include <unistd.h>
#include <opencv2/core.hpp>
//#include <opencv2/imgcodecs.hpp>
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <torch/torch.h>

void warp_affine(torch::Tensor *srcT, torch::Tensor *dstT, torch::Tensor *matT)
{
    float *src = reinterpret_cast<float*>(srcT->data_ptr());
    float *dst = reinterpret_cast<float*>(dstT->data_ptr());
    float *mat= reinterpret_cast<float*>(matT->data_ptr());
    int src_c = srcT->size(0);
    int src_h = srcT->size(1);
    int src_w = srcT->size(2);
    int dst_c = dstT->size(0);
    int dst_h = dstT->size(1);
    int dst_w = dstT->size(2);
    assert(matT->numel()>= 6);
    cv::Mat warp_mat = cv::Mat(2, 3, CV_32FC1, mat);
    const int sz[] = {src_c,src_h,src_w}; 
    cv::Mat srcMat,dstMat;
    if (src_c==3)
    {
    srcMat=cv::Mat(src_w,src_h, CV_8UC3, src);
    dstMat=cv::Mat(dst_w,dst_h, CV_8UC3, dst);
    }
    else
    {
    srcMat=cv::Mat(src_w,src_h, CV_8UC1, src);
    dstMat=cv::Mat(dst_w,dst_h, CV_8UC1, dst);
    }
    cv::warpAffine(srcMat, dstMat, warp_mat,cv::Size(1024,1024),CV_INTER_CUBIC,cv::BORDER_CONSTANT, 1); 
}


torch::Tensor ReadImage(std::string filename)
{
    cv::Mat img = cv::imread(filename, cv::IMREAD_COLOR);
    if(img.empty())
    {
        std::cout << "Could not read the image: " << filename << std::endl;
    }
    torch::Tensor img_tensor = torch::from_blob(img.data, {img.rows, img.cols, 3}, torch::kByte).clone();
    img_tensor = img_tensor.permute({2, 0, 1}); // convert to CxHxW
    return img_tensor;
}

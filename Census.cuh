#pragma once
#include <stdio.h>
#include <assert.h>
#include <math_constants.h>
#include <stdint.h>
#include <unistd.h>
#include <png++/image.hpp>
#include <torch/torch.h>


__device__ void sort(float *x, int n);
__global__ void ad(float *x0, float *x1, float *output, int size, int size2, int size3, int direction);
__global__ void census(float *x0, float *x1, float *output, int size, int num_channels, int size2, int size3, int direction);
__global__ void add_vol(float *vol, float *cnt, float *out, int size, int size1, int size2, int size3, float ratio);
__global__ void rho(float *x, int size, float lambda);
__global__ void spatial_argmin(float *input, float *output, int size, int size1, int size23);
__global__ void cross(float *x0, float *out, int size, int dim2, int dim3, int L1, float tau1);
__global__ void cbca(float *x0c, float *x1c, float *vol, float *out, int size, int dim2, int dim3, int direction);
__global__ void sgm(float *x0, float *x1, float *vol, float *tmp, float *out, int dim1, int dim2, int dim3, float pi1, float pi2, float tau_so, float alpha1, float sgm_q1, float sgm_q2, int sgm_direction, int direction);
template <int sgm_direction> __global__ void sgm2(float *x0, float *x1, float *input, float *output, float *tmp, float pi1, float pi2, float tau_so, float alpha1, float sgm_q1, float sgm_q2, int direction, int size1, int size2, int size3, int step);
template <int sgm_direction> __global__ void sgm3(float *x0, float *x1, float *input, float *output, float pi1, float pi2, float tau_so, float alpha1, float sgm_q1, float sgm_q2, int direction, int size1, int size2, int size3, int step);
__global__ void fliplr(float *in, float *out, int size, int dim3);
__global__ void outlier_detection(float *d0, float *d1, float *outlier, int size, int dim3, int disp_max);
__global__ void iterative_region_voting(float *d0, float *x0c, float *x1c, float *outlier, float *d0_out, float *outlier_out, int size, int dim2, int dim3, float tau_s, float tau_h, int disp_max);
__global__ void interpolate_mismatch(float *d0, float *outlier, float *out, int size, int dim2, int dim3);
__global__ void sobel(float *x, float *g1, float *g2, int size, int dim2, int dim3);
__global__ void depth_discontinuity_adjustment(float *d0, float *dg1, float *dg2, float *xg1, float *xg2, float *out, int size, int dim3, float tau_e);
__global__ void subpixel_enchancement(float *d0, float *c2, float *out, int size, int dim23, int disp_max);
__global__ void mean2d(float *img, float *kernel, float *out, int size, int kernel_radius, int dim2, int dim3, float alpha2);
__global__ void Normalize_get_norm_(float *input, float *norm, int size1, int size23, int size023);
__global__ void Normalize_forward_(float *input, float *norm, float *output, int size23, int size123, int size0123);
__global__ void Normalize_backward_input_(float *grad_output, float *input, float *norm, float *grad_input, int size1, int size23, int size0123);
template <class Op> __global__ void Margin2_(float *input, float *tmp, float *gradInput, float margin, Op op, int size);
__global__ void StereoJoin_(float *input_L, float *input_R, float *output_L, float *output_R, int size1_input, int size1, int size3, int size23);
__global__ void StereoL2R_(float *vol_L, float *vol_R, int size2, int size3, int size);
__global__ void bilateral_filter(float *img, float *out, int size, int dim2, int dim3, int kernel_radius, float sigma1, float sigma2);
__global__ void median2d(float *img, float *out, int size, int dim2, int dim3, int kernel_radius);
void readPNG16(torch::Tensor *imgT, const char * fname);   // See later how to make it a Float Tensor 
void readPNGIARPA(torch::Tensor *imgT, const char * fname);
void writePNG16(torch::Tensor imgT, int height, int width, const char * fname);
void writePFM(torch::Tensor imgT, const char * fname);
__global__ void remove_nonvisible(float *y, int size, int size3);
__global__ void remove_occluded(float *y, int size, int size3);
__global__ void remove_white(float *x, float *y, int size);
__global__ void copy_fill(float *in, float *out, int size, int in_size2, int in_size3, int out_size2, int out_size3);
void memcpy2d(float *dst, float *src, int x, int y, int win_radius, int height, int width);
double random_uniform();
void subset_dataset(torch::Tensor indexT, torch::Tensor inputT, torch::Tensor outputT );
void make_dataset2(torch::Tensor dispT, torch::Tensor nnzT, int img, int t);
void grey2jet(torch::Tensor *grey_img, torch::Tensor *col_img);

void sgm2(torch::Tensor x0, torch::Tensor x1, torch::Tensor input , torch::Tensor output, torch::Tensor tmp,
     float pi1,float pi2, float tau_so, float alpha1, float sgm_q1, float sgm_q2, int direction
        );
void CrBaCoAgg(torch::Tensor x0c, torch::Tensor x1c, torch::Tensor vol_in, torch::Tensor vol_out,  int direction);
void Cross(torch::Tensor x0, torch::Tensor out, int L1, float tau1);
void checkCudaError();
void outlier_detection (torch::Tensor d0, torch::Tensor d1, torch::Tensor outlier, int disp_max);
void interpolate_mismatch(torch::Tensor d0, torch::Tensor outlier, torch::Tensor out);
void interpolate_occlusion(torch::Tensor d0, torch::Tensor outlier,torch::Tensor out);
void subpixel_enchancement(torch::Tensor d0, torch::Tensor c2, torch::Tensor out, int disp_max) ;
void mean2d(torch::Tensor img, torch::Tensor kernel, torch::Tensor out, float alpha2);
 int StereoJoin(torch::Tensor input_L, torch::Tensor input_R, torch::Tensor output_L,torch::Tensor output_R);
void median2d(torch::Tensor img, torch::Tensor out, int kernel_size);



//void doVecAdd();
//__global__ void vecAdd(int *A,int *B,int *C,int N);











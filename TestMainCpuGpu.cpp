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
#include "StereoDataset.h"
//#include <cudnn.h>
//#include "Convnet_Fast.h"
#include "cv.h"
//#include "SampleCensus.cuh"

//using namespace std;
//**********************************************************************
// GLOBAL VARIABLES 
float hflip        =0.0 ;
float vflip        =0.0 ;
float Rotate       =0.0 ;
float hscale       =0.0 ;
float scale        =0.0 ;
int   trans        =0 ;
float hshear       =0.0 ;
float brightness   =0.0 ;
float contrast     =1.0 ;
int   d_vtrans     =0 ;
float d_rotate     =0.0 ;
float d_hscale     =0.0 ;
float d_hshear     =0.0 ;
float d_brightness =0.0;
float d_contrast   =0.0;
int  att           =0;
int    l1           =0;
int    fm           =0;
int    ks           =0;
int    l2           =0;
int    nh2          =0;
float  lr           =0.0;
int    bs           =0;
float  mom          =0.0;
int    true1        =2;
int    false1       =4;
int    false2       =10;
double L1           =0.0;
float  cbca_i1      =0.0;
float  cbca_i2      =0.0;
float tau1         =0.0;
float  pi1          =0.0;
float  pi2          =0.0;
float  sgm_i        =0.0;
float  sgm_q1       =0.0;
float  sgm_q2       =0.0;
float  alpha1       =0.0;
float  tau_so       =0.0;
float  blur_sigma   =0.0;
float  blur_t       =0.0;
float  m            =0.0;
float  Pow          =0.0;
float  ll1          =0.0;
int height          = 1024;
int width           = 1024;
int disp_max        = 392;
int n_te            = 0;
int n_input_plane   = 1;


void printArgs()
{
   std::cout<<ks - 1 * l1 + 1<<"  arch_patch_size"<<std::endl;
   std::cout<<l1 <<"  arch1_num_layers"<<std::endl;;
   std::cout<<fm <<"  arch1_num_feature_maps"<<std::endl;;
   std::cout<<ks <<"  arch1_kernel_size"<<std::endl;;
   std::cout<<l2 <<"  arch2_num_layers"<<std::endl;;
   std::cout<<nh2<<"  arch2_num_units_2"<<std::endl;;
   std::cout<<false1 <<"  dataset_neg_low"<<std::endl;;
   std::cout<<false2 <<"  dataset_neg_high"<<std::endl;;
   std::cout<<true1  <<"  dataset_pos_low"<<std::endl;;
   std::cout<<tau1   <<"  cbca_intensity"<<std::endl;;
   std::cout<<L1     <<"  cbca_distance"<<std::endl;;
   std::cout<<cbca_i1<<"  cbca_num_iterations_1"<<std::endl;;
   std::cout<<cbca_i2<<"  cbca_num_iterations_2"<<std::endl;;
   std::cout<<pi1    <<"  sgm_P1"<<std::endl;;
   std::cout<<pi1 * pi2 <<"  sgm_P2"<<std::endl;;
   std::cout<<sgm_q1 <<"  sgm_Q1"<<std::endl;;
   std::cout<<sgm_q1 * sgm_q2<<"  sgm_Q2"<<std::endl;;
   std::cout<<alpha1 <<"  sgm_V"<<std::endl;;
   std::cout<<tau_so <<"  sgm_intensity"<<std::endl;;
   std::cout<<blur_sigma <<"  blur_sigma"<<std::endl;;
   std::cout<<blur_t <<"  blur_threshold"<<std::endl;;
}

int main(int argc, char **argv) {
   torch::manual_seed(42);
  
   assert(argc>5);  // Dataset architecture Train|Test|Predict Datapath netpath 
  
  if (strcmp(argv[1],"SAT")==0)
      {
         // condition SAT
         hflip        =0;
         vflip        =0;
         Rotate       =7;
         hscale       =0.9;
         scale        =1;
         trans        =1;
         hshear       =0.1;
         brightness   =0.7;
         contrast     =1.3;
         d_vtrans     =1;
         d_rotate     =0.1;
         d_hscale     =1;
         d_hshear     =0;
         d_brightness =0.3;
         d_contrast   =1;
         height       = 1024;
         width        = 1024;
         disp_max     = 192;
         n_te         = 0;
         n_input_plane= 1;
	  }
  else if (strcmp(argv[1],"vahingen")==0)
      {
         // condition vahingen 
         hflip        =0;
         vflip        =0;
         Rotate       =7;
         hscale       =0.9;
         scale        =1;
         trans        =0;
         hshear       =0.1;
         brightness   =0.7;
         contrast     =1.3;
         d_vtrans     =0;
         d_rotate     =0;
         d_hscale     =1;
         d_hshear     =0;
         d_brightness =0.3;
         d_contrast   =1;
         height       = 1024;
         width        = 1024;
         disp_max     = 192;
         n_te         = 0;
         n_input_plane= 1;
      }
  else
      {
		  std::cout<<"  Le dataset n'est pas pris en charge "<<endl;
      }
  
  // ARCHITECTURE 
  if (strcmp(argv[2],"slow")==0)  
  {
	  if (strcmp(argv[1],"SAT")==0)
	     {
            att         = 0;
            ll1         = 4;
            fm         = 112;
            ks         = 3;
            l2         = 4;
            nh2        = 384;
            lr         = 0.003;
            bs         = 128;
            mom        = 0.9;
            true1      = 1;
            false1     = 4;
            false2     = 10;
            L1         = 9;
            cbca_i1    = 4;
            cbca_i2    = 6;
            tau1       = 0.03;
            pi1        = 1.32;
            pi2        = 18.0;
            sgm_i      = 1;
            sgm_q1     = 4.5;
            sgm_q2     = 9;
            alpha1     = 2;
            tau_so     = 0.13;
            blur_sigma = 3.0;
            blur_t     = 2.0;
         }
      else 
      {
            att         = 0;
            ll1         = 4;
            fm         = 112;
            ks         = 3;
            l2         = 4;
            nh2        = 384;
            lr         = 0.003;
            bs         = 128;
            mom        = 0.9;
            true1      = 1;
            false1     = 4;
            false2     = 10;
            L1         = 5;
            cbca_i1    = 2;
            cbca_i2    = 0;
            tau1       = 0.13;
            pi1        = 1.32;
            pi2        = 24.25;
            sgm_i      = 1;
            sgm_q1     = 3;
            sgm_q2     = 2;
            alpha1     = 2;
            tau_so     = 0.08;
            blur_sigma = 5.99;
            blur_t     = 6;
        }
	  
  }
  else if (strcmp(argv[2],"fast")==0)
  {
	  if (strcmp(argv[1],"SAT")==0)
	     {
            att         =  0;
            m          =  0.2;
            Pow        =  1;
            ll1         =  4;
            fm         =  64;
            ks         =  3;
            lr         =  0.002;
            bs         =  128;
            mom        =  0.9;
            true1      =  1;
            false1     =  4;
            false2     =  10;
            L1         =  0;
            cbca_i1    =  0;
            cbca_i2    =  0;
            tau1       =  0;
            pi1        =  4;
            pi2        =  55.72;
            sgm_i      =  1;
            sgm_q1     =  3;
            sgm_q2     =  2.5;
            alpha1     =  1.5;
            tau_so     =  0.02;
            blur_sigma =  7.74;
            blur_t     =  5;
         }
      else 
      {
            att         =  0;
            m          =  0.2;
            Pow        =  1;
            ll1         =  4;
            fm         =  64;
            ks         =  3;
            lr         =  0.002;
            bs         =  128;
            mom        =  0.9;
            true1      =  1;
            false1     =  4;
            false2     =  10;
            L1         =  0;
            cbca_i1    =  0;
            cbca_i2    =  0;
            tau1       =  0;
            pi1        =  4;
            pi2        =  55.72;
            sgm_i      =  1;
            sgm_q1     =  3;
            sgm_q2     =  2.5;
            alpha1     =  1.5;
            tau_so     =  0.02;
            blur_sigma =  7.74;
            blur_t     =  5;
         }
  }
  else
  {
     std::cout<<"  pas d'arcitecture connue  "<<endl;
  }
  //********************Reading Training and Testing Data**************\\
  // Get Tensors names : have been saved before for data preparation
  std::string Datapath="../DataExample/";
  std::string X0_left_Dataset=Datapath+"x0.bin";
  std::string X1_right_Dataset=Datapath+"x1.bin";
  std::string dispnoc_Data=Datapath+"dispnoc.bin";
  std::string metadata_File=Datapath+"metadata.bin";
  std::string tr_File=Datapath+"tr.bin";
  std::string te_File=Datapath+"te.bin";
  std::string nnztr_File=Datapath+"nnz_tr.bin";
  std::string nnzte_File=Datapath+"nnz_te.bin";
  
  // STEREO DATASET
  /********************************************************************/
  // Device
  auto cuda_available = torch::cuda::is_available();
  torch::Device device(cuda_available ? torch::kCUDA : torch::kCPU);
  std::cout << (cuda_available ? "CUDA available. Training on GPU." : "Training on CPU.") << '\n';
  /********************************************************************/ 
  printArgs();
  
  std::cout<<"=========> Loading the stored stereo dataset "<<std::endl;
  int Ws=9;
  auto IARPADataset = StereoDataset(
             X0_left_Dataset,X1_right_Dataset,dispnoc_Data,metadata_File,tr_File,te_File,nnztr_File,nnzte_File, 
             n_input_plane, Ws,trans, hscale, scale, hflip,vflip,brightness,true1, false1,false2,Rotate, 
             contrast,d_hscale, d_hshear, d_vtrans, d_brightness, d_contrast,d_rotate); 
             
  
  std::cout<<" Dataset has been read "<<std::endl;
  /**********************************************************************/
  // Data loader
  auto train_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
      std::move(IARPADataset), bs);
  
  // Training examples 
  
  int num_train_samples=IARPADataset.size().value();
  
  //Looping over data samples 
  for (auto& batch : *train_loader) 
  {
    // Transfer images and target labels to device
    auto data   = batch.data();  //to(device)
    
    std::cout<<"  batch data size and structure "<<data<<std::endl;
    
    //auto target = batch.target(); 
    //std::cout<<"  target data size and structure "<<target<<std::endl;
  }
 return 0;
}

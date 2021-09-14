#include <torch/torch.h>
#include <ATen/ATen.h>
#include <iostream>
#include <stdio.h>
#include <math.h>
#include <string>
#include <fstream>
#include <vector>
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <opencv2/core.hpp>
#include "StereoDataset.h"
//#include <cudnn.h>
#include "Convnet_Fast.h"
#include "cv.h"
//#include "SampleCensus.cuh"
namespace F = torch::nn::functional;


//**********************************************************************
bool isnan(string n)
{
   return n.compare("nan") == 0;
}


/**********************************************************************/
/**********************************************************************/
void ConvNet_FastImpl::createModel(int64_t mfeatureMaps, int64_t mNbHiddenLayers, int64_t mn_input_plane,int64_t mks)
{
    for (auto i=0; i<mNbHiddenLayers-1;i++)
    {
        if (i==0) // Initial image: it will take the number of channels of the patch$
        {
		  mFast->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(mn_input_plane, mfeatureMaps, mks).stride(1).padding(0)));  // we ll see types of padding later
          mFast->push_back(torch::nn::ReLU());
		}
		else 
		{
		mFast->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(mfeatureMaps, mfeatureMaps, mks).stride(1).padding(0)));
        mFast->push_back(torch::nn::ReLU());
	    }
	}
	mFast->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(mfeatureMaps, mfeatureMaps, mks).stride(1).padding(0)));
	
    mFast->push_back(NormL2());
    mFast->push_back(StereoJoin1());
}
/**********************************************************************/

torch::Tensor ConvNet_FastImpl::forward(torch::Tensor x)
{
	auto& model_ref = *mFast;
    for (auto module : model_ref)
    {
		x=module.forward(x);
	}
	
	//x=F::normalize(x,F::NormalizeFuncOptions().p(2).dim(2));
	return x;
}

/**********************************************************************/
/*torch::Tensor ConvNet_FastImpl::forward_twice(torch::Tensor X) // contains 4 tensors 2 positive correspondance and 2 negative correspondance 
{
  // auto outall=torch::empty({4, mninput_plane,mws,mws},torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
   auto output_left=forward_once(X.index({0}));
   //get sizes form here 
   auto outall=torch::empty({4, output_left.size(0),output_left.size(1),output_left.size(2)},torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
   auto output_right=forward_once(X.index({1}));
   outall.index_put_({0},output_left);
   outall.index_put_({1},output_right);
   output_left=forward_once(X.index({2}));
   output_right=forward_once(X.index({3}));
   outall.index_put_({2},output_left);
   outall.index_put_({3},output_right);
   return outall;  //  c'est à corriger c'est absolument faux !!!!!!!!!!!!!!!!
}*/

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


/***********************************************************************/
/**************Templates for training and testing***********************/
/***********************************************************************/
/*template <typename DataLoader>
void train(
    ConvNet_Fast& network,DataLoader& loader, torch::optim::Optimizer& optimizer,
    size_t epoch,size_t data_size) 
    {
      size_t index = 0;
      network->train();
      float Loss = 0, Acc = 0;
      
      for (auto& batch : loader) {
        auto data = batch.data()->data.to(device);     // custom dataset has been created 
        auto targets = batch.data()->target.to(device);// custom dataset has been created 
      
        auto output = network->forward(data);
        // Use a SiameseLoss
        auto loss= SiameseLoss(0.2);
        loss.forward_on_hinge(output,targets);
        //auto loss = torch::nll_loss(output, targets);
        assert(!std::isnan(loss.template item<float>()));
        auto acc = output.argmax(1).eq(targets).sum();
      
        optimizer.zero_grad();
        loss.backward();
        optimizer.step();
      
        Loss += loss.template item<float>();
        Acc += acc.template item<float>();
      
        if (index++ % options.log_interval == 0) {
          auto end = std::min(data_size, (index + 1) * options.train_batch_size);
      
          std::cout << "Train Epoch: " << epoch << " " << end << "/" << data_size
                    << "\tLoss: " << Loss / end << "\tAcc: " << Acc / end
                    << std::endl;
    }
  }
}*/
/*
template <typename DataLoader>
void test(ConvNet_Fast& network, DataLoader& loader, size_t data_size)
 {
  size_t index = 0;
  network->eval();
  torch::NoGradGuard no_grad;
  float Loss = 0, Acc = 0;

  for (const auto& batch : loader) 
  {
    auto data = batch.data.to(options.device);
    auto targets = batch.target.to(options.device).view({-1});

    auto output = network->forward(data);
    auto loss = torch::nll_loss(output, targets);
    assert(!std::isnan(loss.template item<float>()));
    auto acc = output.argmax(1).eq(targets).sum();

    Loss += loss.template item<float>();
    Acc += acc.template item<float>();
  }

  if (index++ % options.log_interval == 0)
    std::cout << "Test Loss: " << Loss / data_size
              << "\tAcc: " << Acc / data_size << std::endl;
}
*/
/***********************************************************************/
// compute the window size to allow reducing the dataset shape to N,C(features),1,1 as an ouptut of the convolutions 
// layers used without padding to make the stereojoin easy after going through the architecture

int GetWindowSize(ConvNet_Fast &Network)
{ 
	int ws=1;
	auto Fast=Network->getFastSequential();
	for (int i=0;i<Fast->size();i++)
    { 
		if (i%2==0) // even values where there is a convnet 2D
		{
		ws=ws+Network->getKernelSize()-1;
	    }
	}
	return ws-2;	
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
            fm         = 64;
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
  ConvNet_Fast Network(3,ll1);
  Network->createModel(fm,ll1,n_input_plane,3);
  
  int Ws=GetWindowSize(Network);

  std::cout<<"the window size is ===> "<<Ws<<std::endl;

  auto IARPADataset = StereoDataset(
             X0_left_Dataset,X1_right_Dataset,dispnoc_Data,metadata_File,tr_File,te_File,nnztr_File,nnzte_File, 
             n_input_plane, Ws,trans, hscale, scale, hflip,vflip,brightness,true1, false1,false2,Rotate, 
             contrast,d_hscale, d_hshear, d_vtrans, d_brightness, d_contrast,d_rotate); 
             
  
  std::cout<<" Dataset has been successfully read and processed  "<<std::endl;
  
  
  /**********************************************************************/
   // Hyper parameters
  const size_t num_epochs = 5;
  const double learning_rate = 0.001;
  /**********************************************************************/

   /************************DATA LOADER  ******************************/
   

  /**********************************************************************/
  // Data loader  ==> Training dataset  
  auto train_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
      std::move(IARPADataset), bs/2);

  // Data loader  ==> Testing dataset 
  /*auto test_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
      std::move(IARPADataset), bs);*/
  /**********************************************************************/


  // Training examples 
  
  int num_train_samples=IARPADataset.size().value();
  

  torch::optim::SGD optimizer(
      Network->parameters(), torch::optim::SGDOptions(0.001).momentum(0.5)); 
  
 for (int epoch=0;epoch<1;epoch++)
 {
  float Loss = 0;
  //Looping over data samples 
  for (auto& batch : *train_loader) 
  {
	  //std::cout<<"  batch size "<<batch.size()<<std::endl;
	  /*for (int i=0;i<batch.size();i++)
	  {
		  std::cout<<batch.at(i).data.sizes()<<std::endl;
	  }*/
	  //Create a custom batch out of the vector that has been generated by  the data loader 
	  int size2=batch.at(0).data.size(1);
	  int size3=batch.at(0).data.size(2);
	  int size4=batch.at(0).data.size(3);
	  torch::Tensor BatchData=torch::empty({2*bs, size2,size3,size4},torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
	  torch::Tensor BatchTarget=torch::empty({bs},torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU));
	  for (int j=0;j<bs/2;j++)
	  {
		  BatchData.index_put_({4*j},batch.at(j).data.index({0}));
		  BatchData.index_put_({4*j+1},batch.at(j).data.index({1}));
		  BatchData.index_put_({4*j+2},batch.at(j).data.index({2}));
		  BatchData.index_put_({4*j+3},batch.at(j).data.index({3}));
		  BatchTarget.index_put_({2*j},1);
		  BatchTarget.index_put_({2*j+1},0);
      }
    
    
    //std::cout<<"Batched data sizes "<<BatchData.sizes()<<std::endl;
    //std::cout<<"Targets sizes      "<<BatchTarget.sizes()<<std::endl;
	  
    // Transfer images and target labels to device
    //auto data   = batch.data()->data.to(device);
    
    //auto targets = batch.data()->target.to(device);    
    //std::cout<<"  batch data size and structure "<<data.sizes()<<std::endl;
    //make
    //auto target = batch.target(); 
    ////std::cout<<"  target data size and structure "<<target<<std::endl;
    auto out=Network->forward(BatchData);
    std::cout<<"  output size "<<out.index({0})<<std::endl;
    // Use a SiameseLoss
    //****auto loss= SiameseLoss(0.2);
    //****auto outpt=loss.forward(out);
    
    //auto loss = torch::nll_loss(output, targets);
    //assert(!std::isnan(loss.template item<float>()));
    //auto acc = output.argmax(1).eq(targets).sum();
    //****torch::Tensor allsum=at::sum(outpt);
    //****Loss+=allsum.accessor<float,1>()[0];
    //****optimizer.zero_grad();
    //****loss.backward();
    //****optimizer.step();
    //understanding data struture 
    //Later see the ouput of the network to define the normalization step   
    // Now That batched data has been created forward pass needs to be defined and Siamese Loss accordingl 
  }
  //****std::cout<<"=============> LOSS ==="<<Loss<<std::endl;
}
  
 return 0;
}

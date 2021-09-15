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
/***********************SimilarityLearner *****************************/
class SimilarityLearner
{
public:
		SimilarityLearner()
		{};
		template <typename DataLoader> void  train (StereoDataset Dataset, ConvNet_Fast Network, DataLoader& loader,
		torch::optim::Optimizer& optimizer,size_t epoch,size_t data_size, int interval,torch::Device device);
		template <typename DataLoader> void  test  (StereoDataset Dataset, ConvNet_Fast Network, DataLoader& loader,
		size_t data_size, size_t batch_size, int interval, torch::Device device);
		torch::Tensor predict(torch::Tensor X_batch, ConvNet_Fast Network,int ind,const int disp_max,torch::Device device, 
                              ptions opt);
		// All postprocessing Steps Stereo-Method
		// Cross-based cost aggregation 
		//***+++++void StereoJoin(torch::Tensor left, torch::Tensor right, torch::Tensor *VolLeft, torch::Tensor *volRight);
		//***+++++void Cross(torch::Tensor x_batch, torch::Tensor * x0c, float L1, float tau1);
		//***+++++void CrBaCoAgg(torch::Tensor x0c, torch::Tensor x1c, torch::Tensor vol, torch::Tensor tmp_cbca,int  direction);
        //***+++++void sgm2(torch::Tensor x_batch1, torch::Tensor x_batch2, torch::Tensor vol,torch::Tensor out,torch::Tensor tmp,
        //***+++++          float pi1, float pi2, float tau_so, float alpha1, float sgm_q1, float sgm_q2, int direction);
        //***+++++void sgm3(torch::Tensor x_batch1,torch::Tensor x_batch2,torch::Tensor vol,torch::Tensor out,float pi1,
        //***+++++          float pi2,float tau_so,float alpha1,float sgm_q1,float sgm_q2,float direction);
        //***+++++void outlier_detection(torch::Tensor disp2,torch::Tensor disp1,torch::Tensor outlier,int disp_max);
        //***+++++torch::Tensor interpolate_occlusion(torch::Tensor disp2, torch::Tensor outlier);
        //***+++++torch::Tensor interpolate_mismatch(torch::Tensor disp2,torch::Tensor outlier);
        //***+++++torch::Tensor subpixel_enchancement(torch::Tensor disp2,torch::Tensor vol,int disp_max);
        //***+++++torch::Tensor median2d(torch::Tensor disp2, int filersize);
        //***+++++torch::Tensor mean2d(torch::Tensor disp2, torch::Tensor GaussKern, float blur_t);
        void Save_Network(ConvNet_Fast Network, std::string fileName);
        torch::Tensor gaussian(float blur_sigma);
};


/***********************************************************************/
template <typename DataLoader> void SimilarityLearner::train(StereoDataset Dataset, ConvNet_Fast Network, DataLoader& loader,
    torch::optim::Optimizer& optimizer, size_t epoch,size_t data_size,int batch_size, int interval, torch::Device device)
{ 
   size_t index = 0;
   float Loss = 0, Acc = 0;
   
   for (auto& batch : loader) 
   {
	  int size2=batch.at(0).data.size(1);
	  int size3=batch.at(0).data.size(2);
	  int size4=batch.at(0).data.size(3);
	  torch::Tensor BatchData=torch::empty({2*batch_size, size2,size3,size4},torch::TensorOptions().dtype(torch::kFloat32).device(device));
	  torch::Tensor BatchTarget=torch::empty({batch_size},torch::TensorOptions().dtype(torch::kInt32).device(device));
	  for (int j=0;j<batch_size/2;j++)
	  {
		BatchData.index_put_({4*j},batch.at(j).data.index({0}));
		BatchData.index_put_({4*j+1},batch.at(j).data.index({1}));
		BatchData.index_put_({4*j+2},batch.at(j).data.index({2}));
		BatchData.index_put_({4*j+3},batch.at(j).data.index({3}));
		BatchTarget.index_put_({2*j},1);
		BatchTarget.index_put_({2*j+1},0);
      }
      auto out=Network->forward(BatchData);
      // Use a SiameseLoss
      auto loss= SiameseLoss(0.2);
      auto outpt=loss.forward(out,BatchTarget);
      assert(!std::isnan(outpt.template item<float>()));
      auto acc = outpt.argmax(1).eq(BatchTarget).sum();  //to see later !!!!!!!
      optimizer.zero_grad();
      outpt.backward();
      optimizer.step();
      
      Loss+=outpt.accessor<float,1>()[0];
      Acc += acc.template item<float>();
	  
      if (index++ % interval == 0)
         {
           auto end = std::min(data_size, (index + 1) * batch_size);
         
           std::cout << "Train Epoch: " << epoch << " " << end << "/" << data_size
                     << "\tLoss: " << Loss / end << "\tAcc: " << Acc / end
                     << std::endl;
         
         }
   }
}
/***********************************************************************/ // pas de besoin de batching pour le test
template <typename DataLoader> void  SimilarityLearner::test (StereoDataset Dataset, ConvNet_Fast Network, DataLoader& loader,
size_t data_size, size_t batch_size, int interval, torch::Device device)

// NEED TO ADD PREDICTION AND COMPARISON WITH RESPECT TO GROUND TRUTH !!!!!!!!!!!!!!!!!!!!!!!!!!!!!
{
  size_t index = 0;
  Network->eval();
  torch::NoGradGuard no_grad;
  float Loss = 0, Acc = 0;
  size_t index;
  for (const auto& batch : loader) 
  { 
	  // PREPARE BATCH DATA AND TARGET  
	int size2=batch.at(0).data.size(1);
	int size3=batch.at(0).data.size(2);
	int size4=batch.at(0).data.size(3);
	torch::Tensor BatchData=torch::empty({2*batch_size, size2,size3,size4},torch::TensorOptions().dtype(torch::kFloat32).device(device));
	torch::Tensor BatchTarget=torch::empty({batch_size},torch::TensorOptions().dtype(torch::kInt32).device(device));
	for (int j=0;j<batch_size/2;j++)
	{
	 BatchData.index_put_({4*j},batch.at(j).data.index({0}));
	 BatchData.index_put_({4*j+1},batch.at(j).data.index({1}));
	 BatchData.index_put_({4*j+2},batch.at(j).data.index({2}));
	 BatchData.index_put_({4*j+3},batch.at(j).data.index({3}));
	 BatchTarget.index_put_({2*j},1);
	 BatchTarget.index_put_({2*j+1},0);
    }
    // Use a SiameseLoss
    auto Loss= SiameseLoss(0.2);
    auto output = network->forward(BatchData);
    auto loss = Loss->forward(output, BatchTarget);
    assert(!std::isnan(loss.template item<float>()));
    auto acc = output.argmax(1).eq(BatchTarget).sum();
    Loss += loss.template item<float>();
    Acc += acc.template item<float>();
  }

  if (index++ % interval == 0)
    std::cout << "Test Loss: " << Loss / data_size
              << "\tAcc: " << Acc / data_size << std::endl;
}
/**********************************************************************/
/************************INFERENCE OU STEREO_PREDICT*******************/
torch::Tensor SimilarityLearner::predict(torch::Tensor X_batch, ConvNet_Fast Network,int ind,const int disp_max,torch::Device device, 
                                         Options opt)
{
  // CHECK X_batch shape !!!!!!!!!!!!!!!!!!!!
  auto output=Network->forward(X_batch);
  torch::Tensor vols = torch::empty({2, disp_max, X_batch.size(2), X_batch.size(3)},torch::TensorOptions().dtype(torch::kFloat32).device(device));
  this->StereoJoin(output.index({0}), output.index({1}), vols.index({0}), vols.index({1}));
  this->fix_border(Network,  vols.index({0}), -1);          // fix_border to implement !!!!!!!!!
  this->fix_border(Network,  vols.index({1}), 1);           // fix_border to implement !!!!!!!!!*
  
  /**********************/
   torch::Tensor disp;
   int mb_directions = {1,-1};
   for (auto direction : mb_directions)
   {
	  torch::Tensor vol = vols.index({direction == -1 and 0 or 1});
      //cross based cost aggregation CBCA
      torch::Tensor x0c=torch::empty({1, 4, vol.size(2), vol.size(3)},torch::TensorOptions().dtype(torch::kFloat32).device(device));
      torch::Tensor x1c=torch::empty({1, 4, vol.size(2), vol.size(3)},torch::TensorOptions().dtype(torch::kFloat32).device(device));
      this->Cross(X_batch.index({0}), &x0c, opt.L1, opt.tau1); // Need to take care of Options struct creation !!!!!!!!!!!
      this->Cross(X_batch.index({1}), &x1c, opt.L1, opt.tau1); // Need to take care of Options struct creation !!!!!!!!!!!
      torch::Tensor tmp_cbca = torch::empty({1, disp_max, vol.size(2), vol.size(3)},torch::TensorOptions().dtype(torch::kFloat32).device(device));
      for  (int i=0,i<opt.cbca_i1,i++)
        {
         this->CrBaCoAgg(x0c,x1c,vol,tmp_cbca,direction);
         vol.copy(tmp_cbca);
	    }
	  // SGM 
      vol = vol.transpose(1, 2).transpose(2, 3).clone(); // see it later !!!!!!!! it is absolutely  not going to work !!!!!!!!!
      torch::Tensor out = torch::empty({1, vol.size(1), vol.size(2), vol.size(3)},torch::TensorOptions().dtype(torch::kFloat32).device(device));
      torch::Tensor tmp = torch::empty({vol.size(2), vol.size(3)},torch::TensorOptions().dtype(torch::kFloat32).device(device));
      for (int i=0;i<opt.sgm_i;i++)
        {
             this->sgm2(x_batch[1], x_batch[2], vol, out, tmp, opt.pi1, opt.pi2, opt.tau_so,
                opt.alpha1, opt.sgm_q1, opt.sgm_q2, direction);
             vol.copy(out).div(3);
	    }
      vol.resize_({1, disp_max, X_batch.size(2), X_batch.size(3)});
      vol.copy(out.transpose(2, 3).transpose(1, 2)).div(3);
      
      //  ANOTHER CBCA 2
      for (int i=0;i<opt.cbca_i2;i++)
         {
           this->CrBaCoAgg(x0c, x1c, vol, tmp_cbca, direction);
           vol.copy(tmp_cbca);
         }
       // Get the min disparity from the cost volume 
      torch::Tensor d = torch::min(vol, 1);
      disp.index_put_({direction == 1 and 0 or 1}, d.add(-1)); // Make sure it is correct and see what it gives as a result !!!!!!!!!!!! 
   }
      // All Subsequent steps that allow to handle filtering and interpolation 
      torch::Tensor outlier = torch::empty(disp.index({1}).sizes(),torch::TensorOptions().dtype(torch::kFloat32).device(device));
      this->outlier_detection(disp.index({1}), disp.index({0}), outlier, disp_max);
      disp.index({1}) = this->interpolate_occlusion(disp.index({1}), outlier);       // CHECK THIS UP !!!!!!!!!!!!!!!
      disp.index({1}) = this->interpolate_mismatch(disp.index({1}), outlier);        // CHECK THIS UP !!!!!!!!!!!!!!!
      disp.index({1}) = this->subpixel_enchancement(disp.index({1}), vol, disp_max); // CHECK THIS UO !!!!!!!!!!!!!!!
      disp.index({1}) = this->median2d(disp.index({1}), 5);                          // CHECK THIS UO !!!!!!!!!!!!!!!
      disp.index({1}) = this->mean2d(disp.index({1}), gaussian(opt.blur_sigma), opt.blur_t);  // CHECK THIS UO !!!!!!!!!!!!!!!  GAUSSIAN
   return disp.index({1});
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
	return x;
}

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
  
 Network->to(device);
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
	  torch::Tensor BatchData=torch::empty({2*bs, size2,size3,size4},torch::TensorOptions().dtype(torch::kFloat32).device(device));
	  torch::Tensor BatchTarget=torch::empty({bs},torch::TensorOptions().dtype(torch::kInt32).device(device));
	  for (int j=0;j<bs/2;j++)
	  {
		BatchData.index_put_({4*j},batch.at(j).data.index({0}));
		BatchData.index_put_({4*j+1},batch.at(j).data.index({1}));
		BatchData.index_put_({4*j+2},batch.at(j).data.index({2}));
		BatchData.index_put_({4*j+3},batch.at(j).data.index({3}));
		BatchTarget.index_put_({2*j},1);
		BatchTarget.index_put_({2*j+1},0);
      }
    auto out=Network->forward(BatchData);
    // Use a SiameseLoss
    auto loss= SiameseLoss(0.2);
    auto outpt=loss.forward(out,BatchTarget);

    Loss+=outpt.accessor<float,1>()[0];
    optimizer.zero_grad();
    outpt.backward();
    optimizer.step();
  }
  std::cout<<"=============> LOSS ==="<<Loss<<std::endl;
}
  
 return 0;
}

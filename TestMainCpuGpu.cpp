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
#include <cuda_runtime.h>
#include "StereoDataset.h"
#include "Convnet_Fast.h"
#include "cv.h"
#include "Census.cuh"

namespace F = torch::nn::functional;



//**********************************************************************
bool isnan(string n)
{
   return n.compare("nan") == 0;
}
/**********************************************************************/
struct Options {
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
    int err_at          = 1;
};
/***********************SimilarityLearner *****************************/
class SimilarityLearner
{
public:
		SimilarityLearner()
		{};
		template <typename DataLoader> void  train (ConvNet_Fast Network, DataLoader& loader,
		torch::optim::Optimizer& optimizer,size_t epoch,size_t data_size, int batch_size,int interval,torch::Device device);
		template <typename DataLoader> void  test  (StereoDataset Dataset,ConvNet_Fast Network,size_t data_size, torch::Device device, Options opt);
		torch::Tensor predict(torch::Tensor X_batch, ConvNet_Fast Network,int disp_max,torch::Device device, 
                              Options opt);

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
        int GetWindowSize(ConvNet_Fast &Network);
        void Load_Network(ConvNet_Fast Network, std::string filename);
        void fix_border(ConvNet_Fast net,torch::Tensor vol,int  direction);
};


/***********************************************************************/
torch::Tensor SimilarityLearner::gaussian(float sigma)
{
   int kr=ceil(sigma/3);
   int ks=kr*2+1;
   torch::Tensor K=torch::empty({ks,ks},torch::TensorOptions().dtype(torch::kFloat32));
   for (int i=0;i<ks;i++)
   {
     for (int j=0;j<ks;j++)
        {
			float y=i-kr;
			float x=j-kr;
			float val=exp(-(x * x + y * y) / (2 * sigma * sigma));
            K.index_put_({i,j},val);
        }
   }
   return K;
}
/***********************************************************************/
void SimilarityLearner::Save_Network(ConvNet_Fast Network, std::string filename)
{
	auto Fast=Network->getFastSequential();
	// make sure the model ends with .pt to apply torch::load (model, "model.pt") later 
	torch::save(Fast, filename.c_str());
}
/***********************************************************************/
void SimilarityLearner::Load_Network(ConvNet_Fast Network, std::string filename)
{
	auto Fast=Network->getFastSequential();
	// make sure the model ends with .pt to apply torch::load (model, "model.pt") later 
	torch::load(Fast, filename.c_str());
}
/***********************************************************************/
int SimilarityLearner::GetWindowSize(ConvNet_Fast &Network)
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
/***********************************************************************/
void SimilarityLearner::fix_border(ConvNet_Fast net,torch::Tensor vol,int  direction)
{
   int n = (this->GetWindowSize(net) - 1) / 2;
   for (int i=0;i<n;i++)
    {
		using namespace torch::indexing;
		vol.index_put_({"...","...","...",direction*i},vol.index({"...","...","...",direction*n}));     // a verifier plus tard !!!!!!!!!!!!!!!!!!!!!!!!!
		//vol.index_put_({,direction*i},vol.index({,direction*n}));     // a verifier plus tard !!!!!!!!!!!!!!!!!!!!!!!!!
    }
      //vol[{{},{},{},direction * i}]:copy(vol[{{},{},{},direction * (n + 1)}])
}
/***********************************************************************/
template <typename DataLoader> void SimilarityLearner::train(ConvNet_Fast Network, DataLoader& loader,
    torch::optim::Optimizer& optimizer, size_t epoch,size_t data_size,int batch_size, int interval, torch::Device device)
{ 
   size_t index = 0;
   float Loss = 0, Acc = 0;
   for (auto& batch : *loader) 
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
template <typename DataLoader> void  SimilarityLearner::test (StereoDataset Dataset,ConvNet_Fast NetworkTe,
size_t data_size, torch::Device device,Options opt)

// NEED TO ADD PREDICTION AND COMPARISON WITH RESPECT TO GROUND TRUTH !!!!!!!!!!!!!!!!!!!!!!!!!!!!!
{
  //Network->eval();
  torch::NoGradGuard no_grad;
  float Loss = 0, Acc = 0;
  float err_sum=0.0;
  torch::Tensor X_batch=torch::empty({2,1,opt.height,opt.width},torch::TensorOptions().dtype(torch::kFloat32).device(device));
  torch::Tensor pred_good, pred_bad, pred;  // SIZE IS TO BE DEFINED !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  for (int i=0; i<Dataset.mte.size(0); i++) // the indexes of test images are in "te.bin" of dimension n 
  {
	int indexIm=(int)(Dataset.mte.accessor<float,1>()[i]);
	torch::Tensor currentMetadata=Dataset.mmetadata.index({indexIm});
	int img_heigth=(int)(currentMetadata.accessor<float,1>()[0]);
	int img_width =(int)(currentMetadata.accessor<float,1>()[1]);
	int id  = (int)(currentMetadata.accessor<float,1>()[2]);
	X_batch.index_put_({0},Dataset.mX0_left_Dataset.index({indexIm})); // May ll add img_width to avoid "out_of_range"   ^^^^^^^^^
	X_batch.index_put_({1},Dataset.mX1_right_Dataset.index({indexIm})); // May ll add img_width to avoid "out_of_range"  ^^^^^^^^^
	//*******+++++++++ I think i need to synchronize with GPU Later 
	pred=this->predict(X_batch,NetworkTe,opt.disp_max,device,opt);   // This will bug no size defined for the tensor          ^^^^^^^^^
	torch::Tensor actualGT=Dataset.mdispnoc.index({indexIm});            // May ll add img_width to avoid "out_of_range" ^^^^^^^^^
	at::resize_as_(pred_good,actualGT);
	at::resize_as_(pred_bad,actualGT);
	torch::Tensor mask;
	at::resize_as_(mask,actualGT);
	mask=actualGT.ne(0);              // To check accordingly !!!!!!!!!!!!!!!!!!
	actualGT.sub(pred).abs();
	pred_bad=actualGT.gt(opt.err_at).mul(mask);                                        // To check accordingly !!!!!!!!!!!!!!!!!! 
	pred_good=actualGT.le(opt.err_at).mul(mask);
	float err=pred_bad.sum()/mask.sum();
	err_sum+=err;
	std::cout<<"  Error for the image id  "<<indexIm<<" is  : "<<err<<std::endl;
   }
   std::cout<<" Overall error value for the test dataset is : "<<err_sum/Dataset.mte.size(0)<<std::endl;
}
/**********************************************************************/
/************************INFERENCE OU STEREO_PREDICT*******************/
torch::Tensor SimilarityLearner::predict(torch::Tensor X_batch, ConvNet_Fast Network,int disp_max,torch::Device device, 
                                         Options opt)
{
   // CHECK X_batch shape !!!!!!!!!!!!!!!!!!!!
   auto output=Network->forward(X_batch);
   torch::Tensor vols = torch::empty({2, disp_max, X_batch.size(2), X_batch.size(3)},torch::TensorOptions().dtype(torch::kFloat32).device(device));
   StereoJoin(output.index({0}), output.index({1}), vols.index({0}), vols.index({1}));  // implemented in cuda !!!!!!!+++++++
   this->fix_border(Network,  vols.index({0}), -1);             // fix_border to implement !!!!!!!!!
   this->fix_border(Network,  vols.index({1}), 1);              // fix_border to implement !!!!!!!!!*
   
   /**********************/
   torch::Tensor disp,vol;
   int mb_directions[2] = {1,-1};
   for (auto direction : mb_directions)
   {
	  vol = vols.index({direction == -1 and 0 or 1});
      //cross based cost aggregation CBCA
      torch::Tensor x0c=torch::empty({1, 4, vol.size(2), vol.size(3)},torch::TensorOptions().dtype(torch::kFloat32).device(device));
      torch::Tensor x1c=torch::empty({1, 4, vol.size(2), vol.size(3)},torch::TensorOptions().dtype(torch::kFloat32).device(device));
      Cross(X_batch.index({0}), x0c, opt.L1, opt.tau1); 
      Cross(X_batch.index({1}), x1c, opt.L1, opt.tau1); 
      torch::Tensor tmp_cbca = torch::empty({1, disp_max, vol.size(2), vol.size(3)},torch::TensorOptions().dtype(torch::kFloat32).device(device));
      for  (int i=0;i<opt.cbca_i1;i++)
        {
         CrBaCoAgg(x0c,x1c,vol,tmp_cbca,direction);
         vol.copy_(tmp_cbca);
	    }
	  // SGM 
      vol = vol.transpose(1, 2).transpose(2, 3).clone(); // see it later !!!!!!!! it is absolutely  not going to work !!!!!!!!!
      torch::Tensor out = torch::empty({1, vol.size(1), vol.size(2), vol.size(3)},torch::TensorOptions().dtype(torch::kFloat32).device(device));
      torch::Tensor tmp = torch::empty({vol.size(2), vol.size(3)},torch::TensorOptions().dtype(torch::kFloat32).device(device));
      for (int i=0;i<opt.sgm_i;i++)
        {
             sgm2(X_batch.index({0}), X_batch.index({1}), vol, out, tmp, opt.pi1, opt.pi2, opt.tau_so,
                opt.alpha1, opt.sgm_q1, opt.sgm_q2, direction);
             vol.copy_(out).div(3);
	    }
      vol.resize_({1, disp_max, X_batch.size(2), X_batch.size(3)});
      vol.copy_(out.transpose(2, 3).transpose(1, 2)).div(3);
      
      //  ANOTHER CBCA 2
      for (int i=0;i<opt.cbca_i2;i++)
         {
           CrBaCoAgg(x0c, x1c, vol, tmp_cbca, direction);
           vol.copy_(tmp_cbca);
         }
       // Get the min disparity from the cost volume 
      std::tuple<torch::Tensor, torch::Tensor> d_Tpl = torch::min(vol,1);
      torch::Tensor d=std::get<0>(d_Tpl);
      at::reshape(d, {1, X_batch.size(2),X_batch.size(3)});
      disp.index_put_({direction == 1 and 0 or 1}, d.add(-1)); // Make sure it is correct and see what it gives as a result !!!!!!!!!!!! 
   }
      // All Subsequent steps that allow to handle filtering and interpolation 
      torch::Tensor outlier = torch::empty(disp.index({1}).sizes(),torch::TensorOptions().dtype(torch::kFloat32).device(device));
      torch::Tensor out = torch::empty(disp.index({1}).sizes(),torch::TensorOptions().dtype(torch::kFloat32).device(device));
      torch::Tensor out2 = torch::empty(disp.index({1}).sizes(),torch::TensorOptions().dtype(torch::kFloat32).device(device));
      torch::Tensor out3 = torch::empty(disp.index({1}).sizes(),torch::TensorOptions().dtype(torch::kFloat32).device(device));
      torch::Tensor out4 = torch::empty(disp.index({1}).sizes(),torch::TensorOptions().dtype(torch::kFloat32).device(device));
      torch::Tensor out5 = torch::empty(disp.index({1}).sizes(),torch::TensorOptions().dtype(torch::kFloat32).device(device));
      outlier_detection(disp.index({1}), disp.index({0}), outlier, disp_max);
      interpolate_occlusion(disp.index({1}), outlier,out);       // CHECK THIS UP !!!!!!!!!!!!!!!
      interpolate_mismatch(out, outlier,out2);                        // CHECK THIS UP !!!!!!!!!!!!!!!
      subpixel_enchancement(out2, vol, out3, disp_max);                 // CHECK THIS UO !!!!!!!!!!!!!!!
      median2d(out3,out4,5);                                          // CHECK THIS UO !!!!!!!!!!!!!!!
      mean2d(out4, gaussian(opt.blur_sigma), out5, opt.blur_t);         // CHECK THIS UO !!!!!!!!!!!!!!!  GAUSSIAN
   return out5;
}
/**********************************************************************/
/***********************************************************************/

void printArgs(Options opt)
{
   std::cout<<opt.ks - 1 * opt.l1 + 1<<"  arch_patch_size"<<std::endl;
   std::cout<<opt.l1 <<"  arch1_num_layers"<<std::endl;;
   std::cout<<opt.fm <<"  arch1_num_feature_maps"<<std::endl;;
   std::cout<<opt.ks <<"  arch1_kernel_size"<<std::endl;;
   std::cout<<opt.l2 <<"  arch2_num_layers"<<std::endl;;
   std::cout<<opt.nh2<<"  arch2_num_units_2"<<std::endl;;
   std::cout<<opt.false1 <<"  dataset_neg_low"<<std::endl;;
   std::cout<<opt.false2 <<"  dataset_neg_high"<<std::endl;;
   std::cout<<opt.true1  <<"  dataset_pos_low"<<std::endl;;
   std::cout<<opt.tau1   <<"  cbca_intensity"<<std::endl;;
   std::cout<<opt.L1     <<"  cbca_distance"<<std::endl;;
   std::cout<<opt.cbca_i1<<"  cbca_num_iterations_1"<<std::endl;;
   std::cout<<opt.cbca_i2<<"  cbca_num_iterations_2"<<std::endl;;
   std::cout<<opt.pi1    <<"  sgm_P1"<<std::endl;;
   std::cout<<opt.pi1 * opt.pi2 <<"  sgm_P2"<<std::endl;;
   std::cout<<opt.sgm_q1 <<"  sgm_Q1"<<std::endl;;
   std::cout<<opt.sgm_q1 * opt.sgm_q2<<"  sgm_Q2"<<std::endl;;
   std::cout<<opt.alpha1 <<"  sgm_V"<<std::endl;;
   std::cout<<opt.tau_so <<"  sgm_intensity"<<std::endl;;
   std::cout<<opt.blur_sigma <<"  blur_sigma"<<std::endl;;
   std::cout<<opt.blur_t <<"  blur_threshold"<<std::endl;;
}

/***********************************************************************/
// compute the window size to allow reducing the dataset shape to N,C(features),1,1 as an ouptut of the convolutions 
// layers used without padding to make the stereojoin easy after going through the architecture

int main(int argc, char **argv) {
   torch::manual_seed(42);
   //Instance of Options
   Options opt;
   assert(argc>5);  // Dataset architecture Train|Test|Predict Datapath netpath 
  
  if (strcmp(argv[1],"SAT")==0)
      {
         // condition SAT
         opt.hflip        =0;
         opt.vflip        =0;
         opt.Rotate       =7;
         opt.hscale       =0.9;
         opt.scale        =1;
         opt.trans        =1;
         opt.hshear       =0.1;
         opt.brightness   =0.7;
         opt.contrast     =1.3;
         opt.d_vtrans     =1;
         opt.d_rotate     =0.1;
         opt.d_hscale     =1;
         opt.d_hshear     =0;
         opt.d_brightness =0.3;
         opt.d_contrast   =1;
         opt.height       = 1024;
         opt.width        = 1024;
         opt.disp_max     = 192;
         opt.n_te         = 0;
         opt.n_input_plane= 1;
	  }
  else if (strcmp(argv[1],"vahingen")==0)
      {
         // condition vahingen 
         opt.hflip        =0;
         opt.vflip        =0;
         opt.Rotate       =7;
         opt.hscale       =0.9;
         opt.scale        =1;
         opt.trans        =0;
         opt.hshear       =0.1;
         opt.brightness   =0.7;
         opt.contrast     =1.3;
         opt.d_vtrans     =0;
         opt.d_rotate     =0;
         opt.d_hscale     =1;
         opt.d_hshear     =0;
         opt.d_brightness =0.3;
         opt.d_contrast   =1;
         opt.height       = 1024;
         opt.width        = 1024;
         opt.disp_max     = 192;
         opt.n_te         = 0;
         opt.n_input_plane= 1;
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
            opt.att         = 0;
            opt.ll1         = 4;
            opt.fm         = 64;
            opt.ks         = 3;
            opt.l2         = 4;
            opt.nh2        = 384;
            opt.lr         = 0.003;
            opt.bs         = 128;
            opt.mom        = 0.9;
            opt.true1      = 1;
            opt.false1     = 4;
            opt.false2     = 10;
            opt.L1         = 9;
            opt.cbca_i1    = 4;
            opt.cbca_i2    = 6;
            opt.tau1       = 0.03;
            opt.pi1        = 1.32;
            opt.pi2        = 18.0;
            opt.sgm_i      = 1;
            opt.sgm_q1     = 4.5;
            opt.sgm_q2     = 9;
            opt.alpha1     = 2;
            opt.tau_so     = 0.13;
            opt.blur_sigma = 3.0;
            opt.blur_t     = 2.0;
         }
      else 
      {
            opt.att         = 0;
            opt.ll1         = 4;
            opt.fm         = 112;
            opt.ks         = 3;
            opt.l2         = 4;
            opt.nh2        = 384;
            opt.lr         = 0.003;
            opt.bs         = 128;
            opt.mom        = 0.9;
            opt.true1      = 1;
            opt.false1     = 4;
            opt.false2     = 10;
            opt.L1         = 5;
            opt.cbca_i1    = 2;
            opt.cbca_i2    = 0;
            opt.tau1       = 0.13;
            opt.pi1        = 1.32;
            opt.pi2        = 24.25;
            opt.sgm_i      = 1;
            opt.sgm_q1     = 3;
            opt.sgm_q2     = 2;
            opt.alpha1     = 2;
            opt.tau_so     = 0.08;
            opt.blur_sigma = 5.99;
            opt.blur_t     = 6;
        }
	  
  }
  else if (strcmp(argv[2],"fast")==0)
  {
	  if (strcmp(argv[1],"SAT")==0)
	     {
            opt.att         =  0;
            opt.m          =  0.2;
            opt.Pow        =  1;
            opt.ll1         =  4;
            opt.fm         =  64;
            opt.ks         =  3;
            opt.lr         =  0.002;
            opt.bs         =  128;
            opt.mom        =  0.9;
            opt.true1      =  1;
            opt.false1     =  4;
            opt.false2     =  10;
            opt.L1         =  0;
            opt.cbca_i1    =  0;
            opt.cbca_i2    =  0;
            opt.tau1       =  0;
            opt.pi1        =  4;
            opt.pi2        =  55.72;
            opt.sgm_i      =  1;
            opt.sgm_q1     =  3;
            opt.sgm_q2     =  2.5;
            opt.alpha1     =  1.5;
            opt.tau_so     =  0.02;
            opt.blur_sigma =  7.74;
            opt.blur_t     =  5;
         }
      else 
      {
            opt.att         =  0;
            opt.m          =  0.2;
            opt.Pow        =  1;
            opt.ll1         =  4;
            opt.fm         =  64;
            opt.ks         =  3;
            opt.lr         =  0.002;
            opt.bs         =  128;
            opt.mom        =  0.9;
            opt.true1      =  1;
            opt.false1     =  4;
            opt.false2     =  10;
            opt.L1         =  0;
            opt.cbca_i1    =  0;
            opt.cbca_i2    =  0;
            opt.tau1       =  0;
            opt.pi1        =  4;
            opt.pi2        =  55.72;
            opt.sgm_i      =  1;
            opt.sgm_q1     =  3;
            opt.sgm_q2     =  2.5;
            opt.alpha1     =  1.5;
            opt.tau_so     =  0.02;
            opt.blur_sigma =  7.74;
            opt.blur_t     =  5;
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
  printArgs(opt);
  
  ConvNet_Fast Network(3,opt.ll1);
  Network->createModel(opt.fm,opt.ll1,opt.n_input_plane,3);
  
  SimilarityLearner SimilarityLearn;
  int Ws=SimilarityLearn.GetWindowSize(Network);
  std::cout<<"=========> Loading the stored stereo dataset "<<std::endl;
  //Train Dataset 
  auto IARPADatasetTr = StereoDataset(
             X0_left_Dataset,X1_right_Dataset,dispnoc_Data,metadata_File,tr_File,te_File,nnztr_File,nnzte_File,
             opt.n_input_plane, Ws,opt.trans, opt.hscale, opt.scale, opt.hflip,opt.vflip,opt.brightness,opt.true1,opt.false1,opt.false2,opt.Rotate, 
             opt.contrast,opt.d_hscale, opt.d_hshear, opt.d_vtrans, opt.d_brightness, opt.d_contrast,opt.d_rotate); 

  std::cout<<" Dataset Training has been successfully read and processed  "<<std::endl;
  // Test Dataset   // BASED ON "StereoDataset2.h"
  /*auto IARPADatasetTe = StereoDataset(
             X0_left_Dataset,X1_right_Dataset,dispnoc_Data,metadata_File,tr_File,te_File,nnzte_File, 
             opt.n_input_plane, Ws,opt.trans, opt.hscale, opt.scale, opt.hflip,opt.vflip,opt.brightness,opt.true1,opt.false1,opt.false2,opt.Rotate, 
             opt.contrast,opt.d_hscale, opt.d_hshear, opt.d_vtrans, opt.d_brightness, opt.d_contrast,opt.d_rotate); */ 
             
  std::cout<<" Dataset Testing has been successfully read and processed  "<<std::endl;
/**********************************************************************/
// Training On the IARPA DATASET 
   // Hyper parameters
  const size_t num_epochs = 5;
  const double learning_rate = 0.001;
  // Data loader  ==> Training dataset  
  auto train_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
      std::move(IARPADatasetTr), opt.bs/2);
  
  int num_train_samples=IARPADatasetTr.size().value();
  

  torch::optim::SGD optimizer(
      Network->parameters(), torch::optim::SGDOptions(0.001).momentum(0.5)); 
  
 Network->to(device);
 for (int epoch=0;epoch<num_epochs;epoch++)
 {
    SimilarityLearn.train(Network, train_loader,optimizer, epoch,num_train_samples,opt.bs, 20,device); // 20 means a display of results in a periodic fashion
 }
 
 std::string fileName=std::string(argv[5])+std::string("/net_")+std::string(argv[1])+std::string("_")+std::string(argv[2])+std::string("_")+std::to_string(num_epochs)+std::string(".pt");
 SimilarityLearn.Save_Network(Network,fileName);
 
/**********************************************************************/
 // Testing on an Unseen chunk of the IARPA DATASET 
 
 std::string outputfileName=std::string(argv[5])+std::string("/net_")+std::string(argv[1])+std::string("_")+std::string(argv[2])+std::string("_")+std::to_string(num_epochs)+std::string(".pt");
 
 ConvNet_Fast TestNetwork;
 TestNetwork->createModel(opt.fm,opt.ll1,opt.n_input_plane,3);
 //SimilarityLearn.Load_Network(TestNetwork,outputfileName);
 
 // Need to change padding to value 1 so output image will keep the same size as the input 
 /*auto Fast=TestNetwork->getFastSequential(); 
 size_t Sz=Fast->size();
 size_t cc=0;
 for (cc=0;cc<Sz;cc++)
  {
    if (Fast->named_children()[cc].key()==std::string("conv")+std::to_string(cc))
        { Fast->named_children()[cc].options.padding=1;}
  } */
 // Now Testing routine on test dataset : No need for a dataloader because we ll be using the whole pair of left and right tile 
 
/**********************************************************************/
 return 0;
}

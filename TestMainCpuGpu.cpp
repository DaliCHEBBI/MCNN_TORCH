#include <torch/torch.h>
#include <torch/script.h>
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

template <typename T> void showTensor(torch::Tensor a,int dim1, int dim2)
{
	cv::Mat Output=cv::Mat(a.size(dim1),a.size(dim2),CV_32FC1,a.data_ptr<T>());
	cv::imshow("see",Output);
	cv::waitKey(0);
}

void showTensorMask(torch::Tensor a,int dim1, int dim2)
{

	cv::Mat Output=cv::Mat(a.size(dim1),a.size(dim2),CV_32FC1,a.data_ptr<bool>());
	cv::imshow("Mask",Output);
	cv::waitKey(0);
}

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
		void  test  (StereoDataset Dataset,ConvNet_Fast Network,size_t data_size, torch::Device device, Options opt);
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
        void fix_border(ConvNet_Fast Network,torch::Tensor vol,int  direction);
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
	torch::save(Fast, filename.c_str());
}
/***********************************************************************/
void SimilarityLearner::Load_Network(ConvNet_Fast Network, std::string filename)
{
	auto Fast=Network->getFastSequential();
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
       if (direction>0)
       {
        vol.index_put_({Slice(0,None,1),Slice(0,None,1),Slice(0,None,1),Slice(direction*i,direction*(i+1),1)},vol.slice(3,direction*n,direction*(n+1),1));
       }
       else
       {
        vol.index_put_({Slice(0,None,1),Slice(0,None,1),Slice(0,None,1),Slice(direction*(i+1),direction*i,1)},vol.slice(3,direction*(n+1),direction*n,1));
       }
    }
}
/***********************************************************************/
template <typename DataLoader> void SimilarityLearner::train(ConvNet_Fast Network, DataLoader& loader,
    torch::optim::Optimizer& optimizer, size_t epoch,size_t data_size,int batch_size, int interval, torch::Device device)
{ 
   size_t index = 0;
   float Loss = 0, err_tr=0;
   int err_tr_count=0;
   for (auto& batch : *loader) 
   {
	  //std::cout<<"indexes at batch   =========="<<std::endl;
	if (batch.size()==batch_size/2)
	 {
	  int size2=batch.at(0).data.size(1);
	  int size3=batch.at(0).data.size(2);
	  int size4=batch.at(0).data.size(3);
	   //std::cout<<"indexes at batch   ================="<<std::endl;
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
	   //std::cout<<"indexes at batch   ========================="<<std::endl;
      auto out=Network->forward(BatchData);
      // Use a SiameseLoss
      auto loss= SiameseLoss(0.2);
      auto outpt=loss.forward(out,BatchTarget);
      //std::cout<<"output after forward to loss "<<outpt.sizes()<<std::endl;
      assert(!std::isnan(outpt.template item<float>()));
      //std::cout<<"out od netwok size and dimension "<<out.sizes()<<"target size "<<BatchTarget.sizes()<<std::endl;
      
      err_tr=outpt.accessor<float,1>()[0];
      if (err_tr>=0 && err_tr<100)
      {
		  Loss+=err_tr;
		  err_tr_count+=1;
	  }
	  else
	  {
		  std::cout<<"be careful divergence : Error ==> "<<err_tr<<std::endl;
	  }
      //auto acc = outpt.argmax(1).eq(BatchTarget).sum();  //to see later !!!!!!!
      optimizer.zero_grad();
      outpt.backward();
      optimizer.step();
      
      if (index++ % interval == 0)
         {
           auto end = std::min(data_size, (index + 1) * batch_size);
         
           std::cout << "Train Epoch: " << epoch << " " << end << "/" << data_size
                     << "\tLoss: " << Loss << std::endl;
         
         }
     }
   }
   // Overall error 
   std::cout<<" Overall training error =====> "<<"Epoch: "<<epoch<<"Loss :"<<Loss/err_tr_count<<std::endl;
}
/***********************************************************************/ // pas de besoin de batching pour le test
void  SimilarityLearner::test (StereoDataset Dataset,ConvNet_Fast NetworkTe,
size_t data_size, torch::Device device,Options opt)

// NEED TO ADD PREDICTION AND COMPARISON WITH RESPECT TO GROUND TRUTH !!!!!!!!!!!!!!!!!!!!!!!!!!!!!
{
  //Network->eval();
  torch::NoGradGuard no_grad;
  float Loss = 0;
  float err_sum=0.0;
  torch::Tensor X_batch=torch::empty({2,1,opt.height,opt.width},torch::TensorOptions().dtype(torch::kFloat32).device(device));
  torch::Tensor pred_good, pred_bad, pred;  // SIZE IS TO BE DEFINED !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  for (int i=0; i<Dataset.mte.size(0); i++) // the indexes of test images are in "te.bin" of dimension n 
  {
	std::cout<<"testing code at level test data accessor "<<std::endl;
	auto indexIm=Dataset.mte.accessor<int64_t,1>()[i];    // toujours >=1 (créé par torch lua)
	std::cout<<"testing code at level test data accessor auto"<<std::endl;
	std::cout<<"index value "<<indexIm<<std::endl;
	torch::Tensor currentMetadata=Dataset.mmetadata.index({indexIm-1}); // toujours >=1 (créé par torch lua)
	std::cout<<"testing code at level test data accessor auto +"<<std::endl;
	auto img_heigth=currentMetadata.accessor<int32_t,1>()[0];
	auto img_width =currentMetadata.accessor<int32_t,1>()[1];
	auto id  =currentMetadata.accessor<int32_t,1>()[2];
	std::cout<<"testing code at level after test data accessor "<<std::endl;
	X_batch.index_put_({0},Dataset.mX0_left_Dataset.index({indexIm-1})); // May ll add img_width to avoid "out_of_range"   ^^^^^^^^^
	X_batch.index_put_({1},Dataset.mX1_right_Dataset.index({indexIm-1})); // May ll add img_width to avoid "out_of_range"  ^^^^^^^^^
	//*******+++++++++ I think i need to synchronize with GPU Later 
	// SHOW IMAGES  
	//showTensor<float> (X_batch.slice(0,0,1,1),2,3);
	//showTensor<float> (X_batch.slice(0,1,2,1),2,3);
	std::cout<<"testing code at level predict  "<<std::endl;
	pred=this->predict(X_batch,NetworkTe,opt.disp_max,device,opt);   // This will bug no size defined for the tensor          ^^^^^^^^^
	std::cout<<"testing code at level after predict presque impossible "<<std::endl;
	showTensor<float> (pred,1,2);
    std::cout<<"Predicted disparity map "<<pred.sizes()<<std::endl;
	torch::Tensor actualGT=Dataset.mdispnoc.index({indexIm-1});            // May ll add img_width to avoid "out_of_range" ^^^^^^^^^
	showTensor<float> (actualGT,1,2);
	std::cout<<"actual gt size "<<actualGT.sizes()<<std::endl;
	pred_good=torch::empty(actualGT.sizes(),torch::TensorOptions().dtype(torch::kFloat32).device(device));
	pred_bad=torch::empty(actualGT.sizes(),torch::TensorOptions().dtype(torch::kFloat32).device(device));
	torch::Tensor mask=torch::zeros(actualGT.sizes(),torch::TensorOptions().dtype(torch::kBool).device(device));
	std::cout<<"mask declaration "<<std::endl;
	mask=actualGT.ne(0.0);              // To check accordingly !!!!!!!!!!!!!!!!!!
	showTensorMask(mask,1,2);
	actualGT.sub(pred).abs();
	pred_bad=actualGT.gt(opt.err_at).mul(mask);                                        // To check accordingly !!!!!!!!!!!!!!!!!! 
	pred_good=actualGT.le(opt.err_at).mul(mask);
	auto errT=pred_bad.sum().div(mask.sum());
	std::cout<<errT.item<float>()<<std::endl;
	float err=errT.item<float>();
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
   auto output=Network->forward_but_Last(X_batch);
   std::cout<<"ouput of network sizes "<<output.sizes()<<std::endl;
   torch::Tensor vols = torch::empty({2, disp_max, X_batch.size(2), X_batch.size(3)},torch::TensorOptions().dtype(torch::kFloat32).device(device));
	std::cout<<"testing code at level before stereo join "<<std::endl;
   StereoJoin(output.index({0}), output.index({1}), vols.index({0}), vols.index({1}));  // implemented in cuda !!!!!!!+++++++
	std::cout<<"testing code at level after stereo join  "<<std::endl;
	std::cout<<"volumes sizes of slices "<<vols.slice(0,1,2,1).sizes()<<std::endl;
   this->fix_border(Network,  vols.slice(0,0,1,1), -1);             // fix_border to implement !!!!!!!!!
   std::cout<<"dir 1 done"<<std::endl;
   this->fix_border(Network,  vols.slice(0,1,2,1), 1);              // fix_border to implement !!!!!!!!!*
	std::cout<<"testing code at level after fix border  "<<std::endl;
   /*******************************************************************/
   torch::Tensor vol;
   torch::Tensor disp=torch::empty({2, X_batch.size(2),X_batch.size(3)},torch::TensorOptions().dtype(torch::kFloat32).device(device));
   int mb_directions[2] = {1,-1};
   std::cout<<"Cross based cost aggregation to be done !"<<std::endl;
   for (auto direction : mb_directions)
   {
	  vol=vols.slice(0,direction == -1 ? 0 : 1, direction == -1 ? 1 : 2,1);
	  std::cout<<"volume size "<<vol.sizes()<<std::endl;
      //cross based cost aggregation CBCA
      torch::Tensor x0c=torch::empty({1, 4, vol.size(2), vol.size(3)},torch::TensorOptions().dtype(torch::kFloat32).device(device));
      torch::Tensor x1c=torch::empty({1, 4, vol.size(2), vol.size(3)},torch::TensorOptions().dtype(torch::kFloat32).device(device));
      std::cout<<"tempo tensor sizes "<<x0c.sizes()<<"   "<<x1c.sizes()<<std::endl;
      Cross(X_batch.slice(0,0,1,1), x0c, opt.L1, opt.tau1); 
      Cross(X_batch.slice(0,1,2,1), x1c, opt.L1, opt.tau1); 
      torch::Tensor tmp_cbca = torch::empty({1, disp_max, vol.size(2), vol.size(3)},torch::TensorOptions().dtype(torch::kFloat32).device(device));
      for  (int i=0;i<opt.cbca_i1;i++)
        {
         CrBaCoAgg(x0c,x1c,vol,tmp_cbca,direction);
         vol.copy_(tmp_cbca);
	    }
	  // SGM 
	  std::cout<<"Cross based cost aggregation has been done !"<<std::endl;
      vol = vol.transpose(1, 2).transpose(2, 3).clone(); // see it later !!!!!!!! it is absolutely  not going to work !!!!!!!!!
      std::cout<<"volume  transposed "<<vol.sizes()<<std::endl;
      torch::Tensor out = torch::empty({1, vol.size(1), vol.size(2), vol.size(3)},torch::TensorOptions().dtype(torch::kFloat32).device(device));
      torch::Tensor tmp = torch::empty({vol.size(2), vol.size(3)},torch::TensorOptions().dtype(torch::kFloat32).device(device));
      for (int i=0;i<opt.sgm_i;i++)
        {
             sgm2(X_batch.slice(0,0,1,1), X_batch.slice(0,1,2,1), vol, out, tmp, opt.pi1, opt.pi2, opt.tau_so,
                opt.alpha1, opt.sgm_q1, opt.sgm_q2, direction);
             vol.copy_(out).div(3);
	    }
      vol.resize_({1, disp_max, X_batch.size(2), X_batch.size(3)});
      vol.copy_(out.transpose(2, 3).transpose(1, 2)).div(3);
      std::cout<<"after sgm 2 has been done !"<<std::endl;
      //  ANOTHER CBCA 2
      for (int i=0;i<opt.cbca_i2;i++)
         {
           CrBaCoAgg(x0c, x1c, vol, tmp_cbca, direction);
           vol.copy_(tmp_cbca);
         }
      std::cout<<"cross based cost aggregation 2 "<<std::endl;
       // Get the min disparity from the cost volume 
      std::tuple<torch::Tensor, torch::Tensor> d_Tpl = torch::min(vol,1);
      torch::Tensor d=std::get<0>(d_Tpl);
      at::reshape(d, {1, X_batch.size(2),X_batch.size(3)});
      disp.index_put_({direction == 1 ? 0 : 1}, d.add(-1)); // Make sure it is correct and see what it gives as a result !!!!!!!!!!!! 
   }
   std::cout<<"stereo process completed"<<std::endl;
      // All Subsequent steps that allow to handle filtering and interpolation 
      torch::Tensor outlier = torch::empty(disp.slice(0,1,2,1).sizes(),torch::TensorOptions().dtype(torch::kFloat32).device(device));
      torch::Tensor out = torch::empty(disp.slice(0,1,2,1).sizes(),torch::TensorOptions().dtype(torch::kFloat32).device(device));
      torch::Tensor out2 = torch::empty(disp.slice(0,1,2,1).sizes(),torch::TensorOptions().dtype(torch::kFloat32).device(device));
      torch::Tensor out3 = torch::empty(disp.slice(0,1,2,1).sizes(),torch::TensorOptions().dtype(torch::kFloat32).device(device));
      torch::Tensor out4 = torch::empty(disp.slice(0,1,2,1).sizes(),torch::TensorOptions().dtype(torch::kFloat32).device(device));
      torch::Tensor out5 = torch::empty(disp.slice(0,1,2,1).sizes(),torch::TensorOptions().dtype(torch::kFloat32).device(device));
      std::cout<<"disparity map dimensions "<<disp.sizes()<<std::endl;
      outlier_detection(disp.slice(0,1,2,1), disp.slice(0,1,1,1), outlier, disp_max);
      std::cout<<"outlier detection done !"<<std::endl;
      interpolate_occlusion(disp.slice(0,1,2,1), outlier,out);       // CHECK THIS UP !!!!!!!!!!!!!!!
      std::cout<<"intepolate occlusion done !"<<std::endl;
      interpolate_mismatch(out, outlier,out2);                        // CHECK THIS UP !!!!!!!!!!!!!!!
      std::cout<<" Mismatch interpolation done !"<<std::endl;
      subpixel_enchancement(out2, vol, out3, disp_max);                 // CHECK THIS UO !!!!!!!!!!!!!!!
      std::cout<<"sub pixel enhhancement done !"<<std::endl;
      median2d(out3,out4,5);            
      std::cout<<"Median 2 done !"<<std::endl;                              // CHECK THIS UO !!!!!!!!!!!!!!!
      mean2d(out4, gaussian(opt.blur_sigma), out5, opt.blur_t);         // CHECK THIS UO !!!!!!!!!!!!!!!  GAUSSIAN
   std::cout<<"  filtering has beeen realized "<<std::endl;
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
  std::string Datapath="./DataExample/";
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

/**********************************************************************/
// Training On the IARPA DATASET 
   // Hyper parameters
  const size_t num_epochs = 1;
  const double learning_rate = 0.001;
  // Data loader  ==> Training dataset  
  auto train_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
      std::move(IARPADatasetTr), opt.bs/2);
  
  int num_train_samples=IARPADatasetTr.size().value();
  std::cout<<"dataset size "<<std::endl;
  

  torch::optim::SGD optimizer(
      Network->parameters(), torch::optim::SGDOptions(0.001).momentum(0.5)); 
  
 Network->to(device);
 for (int epoch=0;epoch<num_epochs;epoch++)
 {
    SimilarityLearn.train(Network, train_loader,optimizer, epoch,num_train_samples,opt.bs, 20,device); // 20 means a display of results in a periodic fashion
 }
 
 std::string fileName=std::string(argv[5])+std::string("/net_")+std::string(argv[1])+std::string("_")+std::string(argv[2])+std::string("_")+std::to_string(num_epochs)+std::string(".pt");
 //SimilarityLearn.Save_Network(Network,fileName);
 torch::save(Network,fileName);
/**********************************************************************/
 // Testing on an Unseen chunk of the IARPA DATASET 
 
 std::string outputfileName=std::string(argv[5])+std::string("/net_")+std::string(argv[1])+std::string("_")+std::string(argv[2])+std::string("_")+std::to_string(num_epochs)+std::string(".pt");
 
 //ConvNet_Fast TestNetwork(3,opt.ll1);
 
 /**********************************************************************/
 
 /********* *HOW TO READ PICKLED MODEL ==> FIND A SOLUTION *************/
 
 /**********************************************************************/
 
 // Need to copy the learnt model to the test model and do testing 
 //auto copy =Network->clone();
 
 //TestNetwork= std::dynamic_pointer_cast<ConvNet_Fast>(copy);
 //TestNetwork= Network->clone();
 //TestNetwork->createModel(opt.fm,opt.ll1,opt.n_input_plane,3);
 //SimilarityLearn.Load_Network(TestNetwork,outputfileName);
  
  //torch::load(TestNetwork,outputfileName);
 
 // Need to change padding to value 1 so output image will keep the same size as the input 
 auto Fast=Network->getFastSequential(); 
 size_t Sz=Fast->size();
 size_t cc=0;
 for (cc=0;cc<Sz;cc++)
  {
	std::cout<<"Name of layer "<<Fast->named_children()[cc].key()<<std::endl;
	std::string LayerName=Fast->named_children()[cc].key();
    if (LayerName.rfind(std::string("conv"),0)==0)
        
        {   //torch::nn::Conv2dImpl *mod=Fast->named_children()[cc].value().get()->as<torch::nn::Conv2dImpl>();
			std::cout<<"condition verified on name of convolution "<<std::endl;
        	Fast->named_children()[cc].value().get()->as<torch::nn::Conv2dImpl>()->options.padding()=1;
        }
  }
 // Now Testing routine on test dataset : No need for a dataloader because we ll be using the whole pair of left and right tile 

/*********************LOADING TEST DATASET*****************************/
  // Test Dataset 
 auto IARPADatasetTe = StereoDataset(
             X0_left_Dataset,X1_right_Dataset,dispnoc_Data,metadata_File,tr_File,te_File,nnztr_File,nnzte_File, 
             opt.n_input_plane, Ws,opt.trans, opt.hscale, opt.scale, opt.hflip,opt.vflip,opt.brightness,opt.true1,opt.false1,opt.false2,opt.Rotate, 
             opt.contrast,opt.d_hscale, opt.d_hshear, opt.d_vtrans, opt.d_brightness, opt.d_contrast,opt.d_rotate);
 
  int num_test_samples=IARPADatasetTe.size().value();
  std::cout<<" Test dataset size "<<std::endl;
  
 std::cout<<" Dataset Testing has been successfully read and processed  "<<std::endl;
 SimilarityLearn.test(IARPADatasetTe,Network,num_test_samples, device,opt);
 
 
 
 /***********************DEBUGGAGE *************************************
 //torch::Tensor a = torch::empty({2, 192, 1024, 1024},torch::TensorOptions().dtype(torch::kFloat32).device(device));
 //torch::Tensor b = torch::empty({2, 192, 1024, 1024},torch::TensorOptions().dtype(torch::kFloat32).device(device));

 //torch::Tensor b = torch::ones({10,10},torch::TensorOptions().dtype(torch::kFloat32).device(device));
 /*int n=9;
 int direction=-1;
 //auto b=a.slice(0,0,1,1);
 std::cout<<b.sizes()<<std::endl;
 auto temp=a.slice(3,direction*(n+1),direction*n,1);
 std::cout<<"out size temp "<<temp.sizes()<<std::endl;
 for (int i=0;i<n;i++)
 {
  using namespace torch::indexing;
  a.index_put_({Slice(0,None,1),Slice(0,None,1),Slice(0,None,1),Slice(direction*(i+1),direction*i,1)},a.slice(3,direction*(n+1),direction*n,1));
 }
 //auto sliced_a= a.slice(1,0,1,1); // dim, start,end, step 
 //a.index_put_({"...",0},b.slice(1,0,1,1));
 //a.slice(1,0,4)=b.slice(1,0,4);
 
 //a.index_put_({Slice(0,None,1),Slice(0,4,1)},b.slice(1,0,4));
 //std::cout<<sliced_a<<std::endl;
 //std::cout<<Slice(0,1,1)<<std::endl;
 **********************************************************************/
/**********************************************************************/
 return 0;
}

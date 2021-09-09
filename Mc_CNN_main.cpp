#include <torch/torch.h>
#include <ATen/ATen.h>
#include <iostream>
#include <stdio.h>
#include <string>
#include <fstream>
#include <vector>
#include <cudnn.h>
#include "Convnet_Fast.h"
#include "cv.h"
#include "Census.cuh"
#include "StereoDataset.h"


using namespace std;
// Seed for Random number


// TRAINING 
//mainAbsoluteDirData.lua SAT slow -a train_tr -datapath /tmp/IARPA_100-50 -netpath /tmp/net/I
//ARPA_100-50_Epochs -err_at 2"


//TESTING AND VALIDATION 
//./mainAbsoluteDirData.lua vahingen slow -a test_te -net_fname /tmp/net/IARPA_100-50/net_SAT_slow_-a_train_tr_14.t7 -datapath /tmp/TrainingAerial -netpath /tmp/net -err_at $error > 
//tmp/net/IARPA_100-50/Slow_Test_process_err_$error.txt

// INFERENCE
//./mainAbsoluteDirData.lua SAT slow -a predict -net_fname /tmp/net/IARPA_500-200/net_SAT_slow_-a_train_tr.t7 -left /tmp/BsurH_0.23_sat/$b1 -right /tmp/BsurH_0.23_sat/$b2 -disp_max $Disp_Pos -datapath /tmp/BsurH_0.23_sat -netpath . -err_at 1


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
		  mFast->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(mn_input_plane, mfeatureMaps, mks).stride(1).padding(1)));  // we ll see types of padding later
          mFast->push_back(torch::nn::ReLU());
		}
		else 
		{
		mFast->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(mfeatureMaps, mfeatureMaps, mks).stride(1).padding(1)));
        mFast->push_back(torch::nn::ReLU());
	    }
	}
	// add margin layer compute the similarity between the layers 
	//this->mFast->push_back(torch::nn::()); // to check later 
	//this->mFast->push_back(StereoJoin1());
}
/**********************************************************************/

torch::Tensor ConvNet_FastImpl::forward_once(torch::Tensor x)
{
    /*for (auto i = this->mFast->begin(); i != this->mFast->end(); ++i)
    {
	     x=mFast->at(i).forward(x);
	}*/
	auto& model_ref = *mFast;
    for (auto module : model_ref)
    {
		x=module.forward();
	}
	return x;
}

/**********************************************************************/
std::vector<torch::Tensor> ConvNet_FastImpl::forward(torch::Tensor x_left, torch::Tensor x_right)
{
   std::vector <torch::Tensor> outall;
   auto output_left=forward_once(x_left);
   auto output_right=forward_once(x_right);
   outall.push_back(output_left);
   outall.push_back(output_right);
   return outall;
}
/**********************************************************************/
torch::Tensor SeameseLoss::forward_on(torch::Tensor x1,torch::Tensor x2, int label)
{
	//torch::nn::CosineSimilarity CosineSim=torch::nn::CosineSimilarity();// to check later !!!!!!!!!
	torch::Tensor similarity=F::cosine_similarity(x1, x2, F::CosineSimilarityFuncOptions().dim(1));
	torch::Tensor clmp=at::mul(at::sub(similarity,this->getMargin()),-1);
	torch::Tensor loss_c=at::mean((1-label)*at::pow(2,similarity)+(label)*at::pow(2,at::clamp(clmp,0.0)));
	return loss_c;
}
/**********************************************************************/

class Mandatory_Tr{
    friend void FuncMandatory_Tr(Mandatory_Tr p);
public:
    Mandatory_Tr(string dataset,string arch,string mode,string datapath, string netpath, int err_at):
    dataset_(dataset), // dataset on which to train
    architecture_(arch), // architecture FAST or SLOW
    mode_(mode),         //  'train_tr | train_all | test_te | test_all | submit | time | predict', 'train'
    datapath_(datapath), // DATAPATH WHERE to find the dataset for training and testing 
    netpath_(netpath),  // where to store the network 
    err_at_(err_at)    // the n pixel error 
    {
       }

    /*Mandatory_Tr& x(int i)
     {
       optional_x_=i;
       eturn *this;
     }*/
private:
    string  dataset_;
    string  architecture_;
    string  mode_;
    string  datapath_;
    string  netpath_;
    int     err_at_;
};

//**********************************************************************
class Mandatory_Test{
    friend void FuncMandatory_Test(Mandatory_Test p);
public:
    Mandatory_Test(string dataset,string arch,string mode,string datapath, string net_name, int err_at):
    dataset_(dataset), // dataset on which to train
    architecture_(arch), // architecture FAST or SLOW
    mode_(mode),         //  'train_tr | train_all | test_te | test_all | submit | time | predict', 'train'
    datapath_(datapath), // DATAPATH WHERE to find the dataset for training and testing 
    net_name_(net_name),  // the network name that we are testing on 
    err_at_(err_at)  // the n pixel error 
    {}

    /*Mandatory_Test& x(int i)
     {
       optional_x_=i;
       eturn *this;
     }*/
private:
    string  dataset_;
    string  architecture_;
    string  mode_;
    string  datapath_;
    string  net_name_;
    int  err_at_;
};

//**********************************************************************
class Mandatory_Inference{
    friend void FuncMandatory_Infer(Mandatory_Inference p);
public:
    Mandatory_Inference(string dataset,string left,string right,string arch,string mode, string net_name, int err_at,int disp_max):
    dataset_(dataset), // dataset on which to work
    left_(left),   // left tile
    right_(right),   // right tile
    architecture_(arch), // architecture FAST or SLOW
    mode_(mode),         //  'train_tr | train_all | test_te | test_all | submit | time | predict', 'train'
    net_name_(net_name),  // the network name that we need to infer based on
    err_at_(err_at), 
    disp_max_(disp_max)   // the n pixel error 
    {}

    /*Mandatory_Test& x(int i)
     {
       optional_x_=i;
       eturn *this;
     }*/
private:
    string  dataset_;
    string  left_;
    string  right_;
    string  architecture_;
    string  mode_;
    string  net_name_;
    int     err_at_;
    int     disp_max_;
};


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
float contrast     =0.0 ;
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
int    true1        =0;
int    false1       =0;
int    false2       =0;
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
   std::cout<<ks - 1 * l1 + 1<<'  arch_patch_size';
   std::cout<<l1 <<'  arch1_num_layers';
   std::cout<<fm <<'  arch1_num_feature_maps';
   std::cout<<ks <<'  arch1_kernel_size';
   std::cout<<l2 <<'  arch2_num_layers';
   std::cout<<nh2<<'  arch2_num_units_2';
   std::cout<<false1 <<'  dataset_neg_low';
   std::cout<<false2 <<'  dataset_neg_high';
   std::cout<<true1  <<'  dataset_pos_low';
   std::cout<<tau1   <<'  cbca_intensity';
   std::cout<<L1     <<'  cbca_distance';
   std::cout<<cbca_i1<<'  cbca_num_iterations_1';
   std::cout<<cbca_i2<<'  cbca_num_iterations_2';
   std::cout<<pi1    <<'  sgm_P1';
   std::cout<<pi1 * pi2 << '  sgm_P2';
   std::cout<<sgm_q1 <<'  sgm_Q1';
   std::cout<<sgm_q1 * sgm_q2<< '  sgm_Q2';
   std::cout<<alpha1 <<'  sgm_V';
   std::cout<<tau_so <<'  sgm_intensity';
   std::cout<<blur_sigma <<'  blur_sigma';
   std::cout<<blur_t <<'  blur_threshold';
}
//**********************************************************************

/**********************************************************************/
/***                                                                  */
/***                  StereoJoin1                                     */
/***                                                                  */
/**********************************************************************/
 /*       CONSTRUCTEUR */

/*StereoJoin1::StereoJoin1(torch::Tensor gradIn, torch::Tensor tmpp) :
    gradInput (gradIn),
    tmp  (tmpp)
{
}
*/

/*StereoJoin1::StereoJoin1() :
{
}
*/
/****** Methods declaration *******/ // to(torch::kCUDA)

std::vector<torch::Tensor> StereoJoin1::slice_input(torch::Tensor input)
{
   std::vector <torch::Tensor> out;
   //out=out.to(torch::kCUDA);
   std::vector<int64_t> sizes={input.size(1)/2,input.size(2),input.size(3),input.size(4)};
   std::vector<int64_t> strides={input.stride(1)*2,input.stride(2),input.stride(3),input.stride(4)};
   torch::Tensor input_L = input.slice(1,0,1,1);
   torch::Tensor input_R = input.slice(1,1,1,1);
   out.push_back(input_L.to(torch::kCUDA));
   out.push_back(input_R.to(torch::kCUDA));
   return out;
}


/*****updateOutput*****/

torch::Tensor StereoJoin1::updateOutput (torch::Tensor input)
{
  std::vector <torch::Tensor> out = slice_input(input);
  tmp.to(torch::kCUDA);
  tmp.reshape_as(out.at(0)); // changing private attributes
  tmp=out.at(0).mul(out.at(1)); //multilying both left and right tensors
  torch::Tensor output;
  output.to(torch::kCUDA);
  output=torch::sum(tmp,2);            // Summing for each row 
  return output;
}



/*******updateGradInput*******/

void StereoJoin1::updateGradInput(torch::Tensor input, torch::Tensor gradOutput)
{
  gradInput.to(torch::kCUDA);
  gradInput.reshape_as(input);
  std::vector <torch::Tensor> out = slice_input(input);
  //std::vector <torch::Tensor> outGrad = slice_input(gradInput);
  torch::Tensor gradInput_L=out.at(1).mul(gradOutput.expand_as(out.at(1)));
  torch::Tensor gradInput_R=out.at(0).mul(gradOutput.expand_as(out.at(0)));
  //return gradInput;
}


/**********************************************************************/
/***                                                                  */
/***                  StereoJoin                                      */
/***                                                                  */
/**********************************************************************/

/********updateOutput********/

/*torch::Tensor StereoJoin::updateOutput (torch::Tensor input)
{
   assert (input.size(1)==2);
   torch::Tensor input_L=input[{{1}}];     //  to check later !!!!!!!!!!!!!
   torch::Tensor input_R = input[{{2}}];   //  to check later !!!!!!!!!!!!!
   output_L.to(torch::kCUDA);
   
   // Reshaping the new tensor 
   std::vector<int64_t> dims={1,mdisp_max_, input_L.size(3), input_L.size(4)};
   c10::ArrayRef<int64_t> dims2=c10::ArrayRef<int64_t>(dims);
   output_L.reshape(dims2);
   StereoJoin_alternative(input_L, input_R, output_L);
   return output_L;
}*/

//**********************************************************************
/*void FuncMandatory_Tr(Mandatory_Tr p) 
    {
      // declaring the function that takes the class of Mandatory Training parameters
    }
    
void FuncMandatory_Test(Mandatory_Test p) 
    {
      // declaring the function that takes the class of Mandatory Training parameters
    }
    
void FuncMandatory_Infer(Mandatory_Inference p) 
    {
      // declaring the function that takes the class of Mandatory Training parameters
    }

*/

//************************THE NETWORK IN PYTORCH************************



// Create both architectures FAST and SLOW 
/**********************************************************************/
int getWindowSize(ConvNet_Fast Net)
{
  int window=1;  const auto& p : net.parameters()
  for (auto module =Net.begin();module!=Net.end();module++)
  {
     window+=module.kernel_size()-1;    // am not sure that this works  !!!!!!!!!
   }
  /*for (const auto& p : Net.parameters())
  {
     window+=p.key()["Kernel"]-1;
   }*/
   return window;
} 
/**********************************************************************/

/**********************************************************************/
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
  std::string Datapath=argv[4];
  std::string X0_left_Dataset=Datapath.append("x0.bin");
  std::string X1_right_Dataset=Datapath.append("x1.bin");
  std::string dispnoc_Data=Datapath.append("dispnoc.bin");
  std::string metadata_File=Datapath.append("metadata.bin");
  std::string tr_File=Datapath.append("tr.bin");
  std::string te_File=Datapath.append("te.bin");
  std::string nnztr_File=Datapath.append("nnz_tr.bin");
  std::string nnzte_File=Datapath.append("nnz_te.bin");
  std::string nnzte_File=Datapath.append("nnz_te.bin");
  
  /********************************************************************/
  // Device
  auto cuda_available = torch::cuda::is_available();
  torch::Device device(cuda_available ? torch::kCUDA : torch::kCPU);
  std::cout << (cuda_available ? "CUDA available. Training on GPU." : "Training on CPU.") << '\n';
  /********************************************************************/ 
  printArgs();
  /********************************************************************/
  /************************ MODEL ARCHITECTURE    *********************/
  /********************************************************************/
  
  ConvNet_Fast model(3);
  model->createModel(fm,l1,n_input_plane,ks);
  model->to(device);
  // Get window size for the patch size 
  int Ws = getWindowSize(model);
  
  std::cout<<"=========> Loading the stored stereo dataset "<<std::endl;
  
  StereoDataset IARPADataset(
             X0_left_Dataset,X1_right_Dataset,dispnoc_Data,metadata_File,tr_File,te_File,nnztr_File,nnzte_File, 
             n_input_plane, Ws,trans, v_trans, hscale, scale, hflip,vflip,brightness,true1, false1,false2,rotate, 
             contrast,d_hscale, d_hshear, d_vtrans, d_brightness, d_contrast,d_rotate);
             
  std::cout<<"=========> The stereo dataset has been successfully loaded "<<std::endl;

  // Hyper parameters
  const size_t num_epochs = 5;
  const double learning_rate = 0.001;

/**********************************************************************/
  // Data loader
  auto train_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
      std::move(IARPADataset), bs);

  auto test_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
      std::move(IARPADataset), bs);
/**********************************************************************/

  // Optimizer
  torch::optim::Adam optimizer(model->parameters(), torch::optim::AdamOptions(learning_rate));

  // Set floating point output precision
  std::cout << std::fixed << std::setprecision(4);

  std::cout << "=========> Training...\n";

  // Train the model
  for (size_t epoch = 0; epoch != num_epochs; ++epoch) {
      // Initialize running metrics
      double running_loss = 0.0;
      size_t num_correct = 0;
      for (auto& batch : *train_loader) {
          // Transfer images and target labels to device
          auto data = batch.data.to(device);
          auto target = batch.target.to(device);

          // Forward pass
          auto output = model->forward(data);

          //Define the loss metric 
          lossF=SeameseLoss(0.2);

          // Calculate loss
          //auto loss = torch::nn::functional::cross_entropy(output, target);
          auto loss=lossF.forward_on(output.at(0),output.at(1),target);    // See later !!!!!!!!!!!!!
          // Update running loss
          running_loss += loss.item<double>() * data.size(0);              // See later !!!!!!!!!!!!!

          // Calculate prediction
          auto prediction = output.argmax(1);

          // Update number of correctly classified samples
          num_correct += prediction.eq(target).sum().item<int64_t>();

          // Backward pass and optimize
          optimizer.zero_grad();
          lossF.backward();
          optimizer.step();
      }

      auto sample_mean_loss = running_loss / num_train_samples;
      auto accuracy = static_cast<double>(num_correct) / num_train_samples;

      std::cout << "Epoch [" << (epoch + 1) << "/" << num_epochs << "], Trainset - Loss: "
          << sample_mean_loss << ", Accuracy: " << accuracy << '\n';
  }

  std::cout << "Training finished!\n\n";
  std::cout << "Testing...\n";

  // Test the model
  model->eval();
  torch::NoGradGuard no_grad;

  double running_loss = 0.0;
  size_t num_correct = 0;

  for (const auto& batch : *test_loader) {
      auto data = batch.data.to(device);
      auto target = batch.target.to(device);

      auto output = model->forward(data);

      auto loss = torch::nn::functional::cross_entropy(output, target);
      running_loss += loss.item<double>() * data.size(0);

      auto prediction = output.argmax(1);
      num_correct += prediction.eq(target).sum().item<int64_t>();
  }

  std::cout << "Testing finished!\n";

  auto test_accuracy = static_cast<double>(num_correct) / num_test_samples;
  auto test_sample_mean_loss = running_loss / num_test_samples;

  std::cout << "Testset - Loss: " << test_sample_mean_loss << ", Accuracy: " << test_accuracy << '\n';
  
  
  // Then the most interesting part 
  // 1. Read the data saved in binary files 
  // 2. Launch the training on that dataset using available learning framework 
  // 3. Save the model 
  // 4. launch testing 
  // 5. Data Augmentation Routine ( do not forget about it !!!!!)
   
 return 0;
}


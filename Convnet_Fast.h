#pragma once
#include <torch/torch.h>
namespace F = torch::nn::functional;

/**********************************************************************/
class StereoJoin1 : public torch::nn::Module {
public:
     StereoJoin1 ()
     {
	 };
      // constructeur
     // objects
     //Members functions 
     std::vector <torch::Tensor> slice_input(torch::Tensor input); // the vector contains input_L, et input_R
     torch::Tensor updateOutput(torch::Tensor input);
     void updateGradInput(torch::Tensor input, torch::Tensor gradOutput);
private:
     torch::Tensor gradInput; // to check later
     torch::Tensor tmp;
	};
/**********************************************************************/

/*class StereoJoin : public torch::nn::Module 

{
public:
     StereoJoin (int64_t disp_max):mdisp_max_(disp_max){}; // constructeur
     // objects
     //Members functions 
     torch::Tensor updateOutput (torch::Tensor input);
private:
	int mdisp_max_;
	int mdirection =-1;
	torch::Tensor output_L;
	};
*/
/**********************************************************************/
/**********************************************************************/
/*                HINGE LOSS FUNCTION                                 */
/**********************************************************************/
class SeameseLoss: public torch::nn::Module 
{
public:
      SeameseLoss (double margin):mMargin(margin){};
      torch::Tensor forward_on (torch::Tensor x1, torch::Tensor x2, int label);
      double getMargin()
      {
		  return this->mMargin;
	   };
	  void setMargin (double Margin)
	  {
		  this->mMargin=Margin;
	   };
	   
private:
    double mMargin=0.2;
};

/************************CLASS STEREO DATASET**************************/
class StereoDataset: public torch::data::datasets
{
public:


private:
}
;
/**********************************************************************/
class ConvNet_FastImpl : public torch::nn::Module {
 public:
    explicit ConvNet_FastImpl(int64_t kern):mkernel(kern){};
    torch::Tensor forward_once(torch::Tensor x);
    void createModel(int64_t featureMaps, int64_t NbHiddenLayers, int64_t n_input_plane,int64_t ks);
    std::vector<torch::Tensor> forward(torch::Tensor x_left, torch::Tensor x_right);

 private:
   int mkernel;
   // add hidden layers to sequential staff
    torch::nn::Sequential mFast;
};


TORCH_MODULE(ConvNet_Fast);




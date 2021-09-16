#pragma once
#include <torch/torch.h>
namespace F = torch::nn::functional;





/**********************************************************************/
class NormL2:public torch::nn::Module{
	
public:
      NormL2()
      {};
      torch::Tensor forward (torch::Tensor input)
      {
		  //torch::Tensor out=F::normalize(input, F::NormalizeFuncOptions().p(2).dim(1).eps(1e-8));
		  //std::cout<<"effect of normalization "<<out.index({0})<<std::endl;
          return F::normalize(input, F::NormalizeFuncOptions().p(2).dim(1).eps(1e-8));
	   };
};

/**********************************************************************/
/**********************************************************************/
class StereoJoin1 : public torch::nn::Module {
public:
     StereoJoin1 ()
     {
	 };
     std::vector <torch::Tensor> slice_input(torch::Tensor input) // the vector contains input_L, et input_R
     {
        std::vector <torch::Tensor> out;
        torch::Tensor input_L = torch::empty({input.size(0)/2, input.size(1),input.size(2),input.size(3)},torch::TensorOptions().dtype(torch::kFloat32));
        torch::Tensor input_R = torch::empty({input.size(0)/2, input.size(1),input.size(2),input.size(3)},torch::TensorOptions().dtype(torch::kFloat32));
        // fill tensors left and right 
        for (int i=0;i<input.size(0)/2;i++)
        {
			//std::cout<<"sizes for checkoing "<<input.index({2*i}).sizes()<<std::endl;
           input_L.index_put_({i},input.index({2*i}));
           input_R.index_put_({i},input.index({2*i+1}));
         }
        return {input_L,input_R};
	  };
     /****updateOutput*****/
     
     torch::Tensor forward (torch::Tensor input)
     {
		//std::cout<<"sizes for checkoing "<<input.sizes()<<std::endl;
       std::vector <torch::Tensor> out = this->slice_input(input);
       torch::Tensor tmp=torch::empty({input.size(0)/2, input.size(1),input.size(2),input.size(3)},torch::TensorOptions().dtype(torch::kFloat32));
       tmp.reshape_as(out.at(0));     // changing private attributes
       tmp=out.at(0).mul(out.at(1));  //multilying both left and right tensors
       return torch::sum(tmp,1);            // Summing for each row 
     };
     //torch::Tensor updateOutput(torch::Tensor input);
     //void updateGradInput(torch::Tensor input, torch::Tensor gradOutput);
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
class SiameseLoss: public torch::nn::Module {
public:
      SiameseLoss (double margin):mMargin(margin){};
      //torch::Tensor forward_on_contrastive (torch::Tensor x1, torch::Tensor x2, int label);
      //torch::Tensor forward_on_hinge (torch::Tensor x);
      //torch::Tensor forward(torch::Tensor x);
      double getMargin()
      {
		  return this->mMargin;
	   };
	  void setMargin (double Margin)
	  {
         this->mMargin=Margin;
	  };
      /**********************************************************************/
      torch::Tensor forward_on_contrastive (torch::Tensor x1,torch::Tensor x2, int label)
      {
      	//torch::nn::CosineSimilarity CosineSim=torch::nn::CosineSimilarity();// to check later !!!!!!!!!
      	torch::Tensor similarity=F::cosine_similarity(x1, x2, F::CosineSimilarityFuncOptions().dim(1));
      	torch::Tensor clmp=at::mul(at::sub(similarity,this->getMargin()),-1);
      	torch::Tensor loss_c=at::mean(at::mul(at::pow(2,similarity),1-label)+at::mul(at::pow(2,at::clamp(clmp,0.0)),label));
      	return loss_c;
      };
      
      /*torch::Tensor SiameseLoss::forward_on_hinge (torch::Tensor Sample)
      {
          torch::Tensor similarity_plus=F::cosine_similarity(Sample.index({0}), Sample.index({1}), F::CosineSimilarityFuncOptions().dim(1));
          torch::Tensor similarity_moins=F::cosine_similarity(Sample.index({2}), Sample.index({3}), F::CosineSimilarityFuncOptions().dim(1));
          auto metric=at::sub(at::add(similarity_moins,this->getMargin()),similarity_plus.accessor<float,1>()[0]);
          auto maxvalue=fmax(0, metric.accessor<float,1>()[0]);
          
          torch::Tensor loss_hinge=torch::tensor({maxvalue});
          return loss_hinge;
      };*/
      
      
      torch::Tensor forward(torch::Tensor input, torch::Tensor target)
      {
        // get hinge loss for each couple of data 
        torch::Tensor pair=torch::empty({input.size(0)/2},torch::TensorOptions().dtype(torch::kInt32));
        torch::Tensor Impair=torch::empty({input.size(0)/2},torch::TensorOptions().dtype(torch::kInt32));
        
        for (int i=0;i<input.size(0)/2;i++)
        {
			pair.index_put_({i},2*i);
			Impair.index_put_({i},2*i+1);
		}
        torch::Tensor similarity_plus=torch::index_select(input,0,pair);
        torch::Tensor similarity_minus=torch::index_select(input,0,Impair);
        similarity_plus=torch::squeeze(similarity_plus);
        similarity_minus=torch::squeeze(similarity_minus);
        //std::cout<<"similarity_plus  "<<similarity_plus.sizes()<<std::endl;
        //std::cout<<"similarity_minus  "<<similarity_minus.sizes()<<std::endl;
        auto metric=similarity_minus.add(this->getMargin()).sub(similarity_plus);
        //std::cout<<"metric      "<<metric.sizes()<<std::endl;
        //auto metric=at::sub(at::add(similarity_minus,this->getMargin()),similarity_plus);
        metric=torch::fmax(metric,torch::zeros({input.size(0)/2}));
        //std::cout<<"metric      "<<metric.sizes()<<std::endl;
        metric=torch::mean(metric,0,1);
        //std::cout<<"metric      "<<metric.sizes()<<std::endl;
        return metric;
       } ;
       
private:
    double mMargin=0.2;
};

/**********************************************************************/
class ConvNet_FastImpl : public torch::nn::Module {
 public:
    explicit ConvNet_FastImpl(int64_t kern, int64_t nbHidden):mkernel(kern),mnbHidden(nbHidden){};
    torch::Tensor forward(torch::Tensor x);
    torch::Tensor forward_but_Last(torch::Tensor x);
    void createModel(int64_t featureMaps, int64_t NbHiddenLayers, int64_t n_input_plane,int64_t ks);
    //torch::Tensor forward_twice(torch::Tensor x);
    torch::nn::Sequential getFastSequential()
    {
		return this->mFast;
	};
	int64_t getKernelSize()
	{ return this->mkernel;};

 private:
   int64_t mkernel;
   int64_t mnbHidden;
   // add hidden layers to sequential staff
    torch::nn::Sequential mFast;
};


TORCH_MODULE(ConvNet_Fast);




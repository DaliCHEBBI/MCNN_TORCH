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

/***********************************************************************/

bool savePFM(const cv::Mat image, const std::string filePath)
{
    //Open the file as binary!
    ofstream imageFile(filePath.c_str(), ios::out | ios::trunc | ios::binary);

    if(imageFile)
    {
        int width(image.cols), height(image.rows);
        int numberOfComponents(image.channels());

        //Write the type of the PFM file and ends by a line return
        char type[3];
        type[0] = 'P';
        type[2] = 0x0a;

        if(numberOfComponents == 3)
        {
            type[1] = 'F';
        }
        else if(numberOfComponents == 1)
        {
            type[1] = 'f';
        }

        imageFile << type[0] << type[1] << type[2];

        //Write the width and height and ends by a line return
        imageFile << width << " " << height << type[2];

        //Assumes little endian storage and ends with a line return 0x0a
        //Stores the type
        char byteOrder[10];
        byteOrder[0] = '-'; byteOrder[1] = '1'; byteOrder[2] = '.'; byteOrder[3] = '0';
        byteOrder[4] = '0'; byteOrder[5] = '0'; byteOrder[6] = '0'; byteOrder[7] = '0';
        byteOrder[8] = '0'; byteOrder[9] = 0x0a;

        for(int i = 0 ; i<10 ; ++i)
        {
            imageFile << byteOrder[i];
        }

        //Store the floating points RGB color upside down, left to right
        float* buffer = new float[numberOfComponents];

        for(int i = 0 ; i<height ; ++i)
        {
            for(int j = 0 ; j<width ; ++j)
            {
                if(numberOfComponents == 1)
                {
                    buffer[0] = image.at<float>(height-1-i,j);
                }
                else
                {
                    cv::Vec3f color = image.at<cv::Vec3f>(height-1-i,j);

                   //OpenCV stores as BGR
                    buffer[0] = color.val[2];
                    buffer[1] = color.val[1];
                    buffer[2] = color.val[0];
                }

                //Write the values
                imageFile.write((char *) buffer, numberOfComponents*sizeof(float));

            }
        }

        delete[] buffer;

        imageFile.close();
    }
    else
    {
        cerr << "Could not open the file : " << filePath << endl;
        return false;
    }

    return true;
}

/***********************************************************************/

template <typename T> void showTensor(torch::Tensor a,int dim1, int dim2,string name)
{
	cv::Mat Output=cv::Mat(a.size(dim1),a.size(dim2),CV_32FC1,a.data_ptr<T>());
	//cv::Mat op2 ;
	//cv::normalize(Output, op2, 0,255, cv::NORM_MINMAX, -1, cv::noArray());
	std::cout<<"mat data "<<name<<Output.at<float>(10,10)<<std::endl;
	cv::imshow("see "+name,Output);
	savePFM(Output,"./see "+name+".pfm");
	cv::waitKey(0);
}

torch::Tensor ReadBinaryFile(std::string filename, torch::Tensor Host)
{
  	int fd;
  	float *TensorContent;
  	fd = open(filename.c_str(), O_RDONLY);
  	TensorContent = static_cast<float*>(mmap(NULL, Host.numel() * sizeof(float), PROT_READ, MAP_SHARED, fd, 0));
  	/*for (int i=0;i<64;i++)
  	{
		std::cout<<TensorContent[i]<<std::endl;
	}*/
	std::cout<<"host sizes "<<std::endl;
	torch::Tensor Temp=torch::from_blob(TensorContent, Host.sizes(), torch::TensorOptions().dtype(torch::kFloat32));
  	//showTensor<float>(Host,2,3);
  	close(fd);
  	return Temp;
}


void CompareTwoTensors(torch::Tensor a1,int dim, int idSlice)
{
	//Read the reference tensor and compare it with the computed tensor 
	torch::Tensor volume=torch::empty({1,50,1024,1024},torch::TensorOptions().dtype(torch::kFloat32));
	//ReadBinaryFile("./vol1.bin",volume);
	showTensor<float>(volume.slice(1,5,6,1).abs(),2,3,"volume lua");
	std::cout<<volume.slice(dim,idSlice,idSlice+1,1).slice(1,0,1,1).sizes()<<std::endl;
	std::cout<<volume.slice(dim,idSlice,idSlice+1,1).slice(2,0,20,1).slice(3,0,20,1)<<std::endl;
	auto diff=a1.sub(volume.slice(dim,idSlice,idSlice+1,1).slice(2,0,20,1).slice(3,0,20,1));
	//std::cout<<"difference de tenseurs "<<diff<<std::endl;
}

torch::Tensor TranslateTileTest(torch::Tensor Tile, int TransOffsetX)
{
	torch::Tensor Res=torch::zeros(Tile.sizes(),torch::TensorOptions().dtype(torch::kFloat32));
	float * datahandler=Tile.data_ptr<float>();
	for (int i=0;i<Tile.size(2);i++)
	{
		for (int j=0;j<Tile.size(1)-TransOffsetX;j++)
		{
			Res.data_ptr<float>()[j+i*Tile.size(1)]=datahandler[j+TransOffsetX+i*Tile.size(2)];
		}
	}
	return Res;
}
void showTensorDisparity(torch::Tensor a,int dim1, int dim2)
{


	long * datahandler=a.data_ptr<long>();
	cv::Mat Output=cv::Mat::zeros(a.size(dim1),a.size(dim2),CV_8UC1);
	for (int i=0;i<1024;i++)
	{
		for (int j=0;j<1024;j++)
		{
			Output.at<uchar>(i,j)=(uchar)datahandler[j+i*1024];
		}
	}
	
	cv::imshow("DISPARITY",Output);
	cv::imwrite("./disp_out.png",Output);
	//writePNG16(a, 1024, 1024,"./disp_out.png");
	//savePFM(Output,"./disp.pfm");
	cv::waitKey(0);
}

void showTensorDisparity2(torch::Tensor a,int dim1, int dim2)
{


	float * datahandler=a.data_ptr<float>();
	if (a.size(1)==1)
	{
	cv::Mat Output=cv::Mat::zeros(a.size(dim1),a.size(dim2),CV_8UC1);
	for (int i=0;i<1024;i++)
	{
		for (int j=0;j<1024;j++)
		{
			Output.at<uchar>(i,j)=(uchar)datahandler[j+i*1024];
		}
	}
	
	cv::imshow("outocclusion",Output);
	cv::imwrite("./occlusioninterp.png",Output);
	//writePNG16(a, 1024, 1024,"./disp_out.png");
	//savePFM(Output,"./disp.pfm");
	cv::waitKey(0);
   }
   else
   { 
	   std::vector<cv::Mat> AllChannels;
	   for (int cc=0;cc<a.size(1);cc++)
	   {
			cv::Mat Output=cv::Mat::zeros(a.size(dim1),a.size(dim2),CV_8UC1);
			for (int i=0;i<1024;i++)
			{
				for (int j=0;j<1024;j++)
				{
					Output.at<uchar>(i,j)=(uchar)datahandler[j+i*1024+cc*1024*1024];
				}
			}
			AllChannels.push_back(Output); 
		}
		cv::Mat AllChannelsImage;
		cv::merge(AllChannels,AllChannelsImage);
	    cv::imshow("rgb disp",AllChannelsImage);
	    cv::imwrite("./rgb_disp.png",AllChannelsImage);
	    //writePNG16(a, 1024, 1024,"./disp_out.png");
	    //savePFM(Output,"./disp.pfm");
	    cv::waitKey(0);
   }
}

void showTensorMask(torch::Tensor a,int dim1, int dim2)
{	
	bool * datahandler =a.data_ptr<bool>();
	cv::Mat Output=cv::Mat::zeros(a.size(dim1),a.size(dim2),CV_8UC1);
	
	for (int i=0;i<1024;i++)
	{
		for (int j=0;j<1024;j++)
		{
			Output.at<uchar>(i,j)=datahandler[j+i*1024]== 1 ? 255:0;
		}
	}
	
	cv::imshow("Mask",Output);
	//savePFM(Output,"./mask.pfm");
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
    int    L1           =0;
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
        void Save_Network(ConvNet_Fast Network, std::string fileName, Options opt);
        torch::Tensor gaussian(float blur_sigma);
        int GetWindowSize(ConvNet_Fast &Network);
        void Load_Network(ConvNet_Fast Network, std::string filename);
        void fix_border(ConvNet_Fast Network,torch::Tensor vol,int  direction);
        void PopulateModelFromBinary(ConvNet_Fast Network,std::vector<std::pair<std::string,std::string>> Names);
};


/***********************************************************************/
void SimilarityLearner::PopulateModelFromBinary(ConvNet_Fast Network,std::vector<std::pair<std::string,std::string>> Names)
{
	auto Fast=Network->getFastSequential();
	int countFiles=0;
	for (int i=0;i<Fast->size();i++)
    { 
		if (i%2==0 && countFiles<4) // even values where there is a convnet 2D
		{
			std::cout<<"entered  "<<i<<std::endl;
			auto weight1=std::get<0>(Names.at(countFiles));
			auto bias1=std::get<1>(Names.at(countFiles));
			//READ BINARY FILES AND STORE TENSORS IN RELEVANT Weights and biases of the model structure 
			torch::Tensor Weights=torch::empty(Fast->named_children()[i].value().get()->as<torch::nn::Conv2dImpl>()->weight.data().sizes(),torch::TensorOptions().dtype(torch::kFloat32));
			Weights=ReadBinaryFile(weight1,Fast->named_children()[i].value().get()->as<torch::nn::Conv2dImpl>()->weight.data());
			//memcpy to copy content into conv2D weight 
			std::memcpy(Fast->named_children()[i].value().get()->as<torch::nn::Conv2dImpl>()->weight.data().data_ptr<float>(),Weights.data_ptr<float>(),sizeof(float)*Weights.numel());
			std::cout<<"inner weight sizes "<<Fast->named_children()[i].value().get()->as<torch::nn::Conv2dImpl>()->weight.data().sizes()<<std::endl;
			showTensor<float>(Fast->named_children()[i].value().get()->as<torch::nn::Conv2dImpl>()->weight.data().slice(0,0,1,1).slice(1,0,1,1),2,3,"weights");
			//std::cout<<Fast->named_children()[i].value().get()->as<torch::nn::Conv2dImpl>()->weight.data()<<std::endl;
			//std::cout<<Fast->named_children()[i].value().get()->as<torch::nn::Conv2dImpl>()->weight.data().slice(0,1,2,1)<<std::endl;
			//std::cout<<Fast->named_children()[i].value().get()->as<torch::nn::Conv2dImpl>()->weight.data().slice(0,2,3,1)<<std::endl;
			//std::cout<<Fast->named_children()[i].value().get()->as<torch::nn::Conv2dImpl>()->weight.data().slice(0,3,4,1)<<std::endl;
			::cout<<"WEIGHTSSSS "<<Fast->named_children()[i].value().get()->as<torch::nn::Conv2dImpl>()->weight.data().slice(0,0,5,1).slice(1,0,1,1)<<std::endl;
			torch::Tensor Biases=torch::empty(Fast->named_children()[i].value().get()->as<torch::nn::Conv2dImpl>()->bias.data().sizes(),torch::TensorOptions().dtype(torch::kFloat32));
			Biases=ReadBinaryFile(bias1,Fast->named_children()[i].value().get()->as<torch::nn::Conv2dImpl>()->bias.data());
			std::memcpy(Fast->named_children()[i].value().get()->as<torch::nn::Conv2dImpl>()->bias.data().data_ptr<float>(),Biases.data_ptr<float>(),sizeof(float)*Biases.numel());

			std::cout<<"BIASSS  "<<Fast->named_children()[i].value().get()->as<torch::nn::Conv2dImpl>()->bias.data()<<std::endl;
			// Another way to read info and store it in Weights 
			countFiles++;
	    }
	}
}
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
void SimilarityLearner::Save_Network(ConvNet_Fast Network, std::string filename,Options opt)
{
	auto Fast=Network->getFastSequential();
	int countFiles=0;
	for (int i=0;i<Fast->size();i++)
    { 
		if (i%2==0 && countFiles<opt.ll1)
		{ 
			int numwght =Fast->named_children()[i].value().get()->as<torch::nn::Conv2dImpl>()->weight.data().numel();
			int numwbias=Fast->named_children()[i].value().get()->as<torch::nn::Conv2dImpl>()->bias.data().numel();
			float* weight=Fast->named_children()[i].value().get()->as<torch::nn::Conv2dImpl>()->weight.data().data_ptr<float>();
			float * bias =Fast->named_children()[i].value().get()->as<torch::nn::Conv2dImpl>()->bias.data().data_ptr<float>();
			std::string FileWght=filename+"/"+"Weights_"+std::to_string(i+1)+"_cudnn.SpatialConvolution.bin";
			std::string FileBias=filename+"/"+"Biases_"+std::to_string(i+1)+"_cudnn.SpatialConvolution.bin";
			FILE *Wght = fopen(FileWght.c_str(), "wb");
			FILE *Bias = fopen(FileBias.c_str(), "wb");
			fwrite(weight, sizeof(float), numwght, Wght);
			fwrite(bias, sizeof(float), numwbias, Bias);
			countFiles++;
	    }
	}
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
  NetworkTe->eval();
  float Loss = 0;
  float err_sum=0.0;
  torch::Tensor X_batch=torch::empty({2,opt.n_input_plane,opt.height,opt.width},torch::TensorOptions().dtype(torch::kFloat32).device(device));
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
	std::cout<<"testing code at level after test data accessor "<<id<<std::endl;
	X_batch.index_put_({0},Dataset.mX0_left_Dataset.index({indexIm-1})); // May ll add img_width to avoid "out_of_range"   ^^^^^^^^^
	X_batch.index_put_({1},Dataset.mX1_right_Dataset.index({indexIm-1})); // May ll add img_width to avoid "out_of_range"  ^^^^^^^^^
	
	//*******+++++++++ I think i need to synchronize with GPU Later 
	// SHOW IMAGES  
	//Check read data 
	std::cout<<"first image "<<X_batch.slice(0,0,1,1).slice(2,0,10,1).slice(3,0,10,1)<<std::endl;
	std::cout<<"second image "<<X_batch.slice(0,1,2,1).slice(2,0,10,1).slice(3,0,10,1)<<std::endl;
	showTensor<float> (X_batch.slice(0,0,1,1),2,3,"image 1");
	showTensor<float> (X_batch.slice(0,1,2,1),2,3,"image 2");
	std::cout<<"testing code at level predict  "<<std::endl;
	pred=this->predict(X_batch,NetworkTe,opt.disp_max,device,opt);   // This will bug no size defined for the tensor          ^^^^^^^^^
	std::string filenameprediction="./Estimated_disp_BH_"+std::to_string(i)+".bin";
    FILE *file_prediction_disp = fopen(filenameprediction.c_str(), "wb");
    fwrite(pred.data_ptr<float>(), sizeof(float), pred.numel(), file_prediction_disp);
	std::cout<<"testing code at level after predict presque impossible "<<std::endl;
	//store it as rgb color 
	torch::Tensor colorPred=torch::empty({1,3,pred.size(2),pred.size(3)},torch::TensorOptions().dtype(torch::kFloat32)).contiguous();
	grey2jet(pred.add(1).div(opt.disp_max),colorPred);
	showTensorDisparity2(colorPred,2,3);
    std::cout<<"Predicted disparity map "<<pred.sizes()<<std::endl;
	torch::Tensor actualGT=Dataset.mdispnoc.index({indexIm-1});            // May ll add img_width to avoid "out_of_range" ^^^^^^^^^
	showTensor<float> (actualGT,1,2,"ground truth");
	std::string actualGtname="./actual_groundTruth_"+std::to_string(i)+".bin";
    FILE *actualgt = fopen(actualGtname.c_str(), "wb");
    fwrite(actualGT.data_ptr<float>(), sizeof(float), actualGT.numel(), actualgt);
	//std::cout<<"actual ground truth "<<actualGT<<std::endl;
	std::cout<<"actual gt size "<<actualGT.sizes()<<std::endl;
	pred_good=torch::empty(actualGT.sizes(),torch::TensorOptions().dtype(torch::kFloat32).device(device));
	pred_bad=torch::empty(actualGT.sizes(),torch::TensorOptions().dtype(torch::kFloat32).device(device));
	torch::Tensor mask=torch::zeros(actualGT.sizes(),torch::TensorOptions().dtype(torch::kBool).device(device));
	std::cout<<"mask declaration "<<std::endl;
	mask=actualGT.ne(0.0);              // To check accordingly !!!!!!!!!!!!!!!!!!
	//std::cout<<"mask"<<mask<<std::endl;
	showTensorMask(mask,1,2);
	actualGT=actualGT.sub(pred.index({0}));
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
   //std::cout<<"output 1"<<output.slice(0,0,1,1).slice(1,20,21,1).slice(2,0,10,1).slice(3,0,10,1)<<std::endl;
   //std::cout<<"output 2"<<output.slice(0,1,2,1).slice(1,20,21,1).slice(2,0,10,1).slice(3,0,10,1)<<std::endl;
   //std::cout<<"ouput of network sizes "<<output.sizes()<<std::endl;
   torch::Tensor vols = torch::ones({2, disp_max, X_batch.size(2), X_batch.size(3)},torch::TensorOptions().dtype(torch::kFloat32).device(device));
   vols=vols.mul(NAN);
   std::cout<<"testing code at level before stereo join "<<std::endl;
   StereoJoin(output.slice(0,0,1,1), output.slice(0,1,2,1), vols.slice(0,0,1,1), vols.slice(0,1,2,1));  // implemented in cuda !!!!!!!+++++++
   cudaDeviceSynchronize();
   //check consistency of results 
   std::cout<<"cost volume left after sj "<<vols.slice(0,0,1,1).slice(1,50,51,1).slice(2,50,60,1).slice(3,50,60,1)<<std::endl;
   std::cout<<"cost volume right aftersj "<<vols.slice(0,1,2,1).slice(1,50,51,1).slice(2,50,60,1).slice(3,50,60,1)<<std::endl;
   //std::cout<<output.slice(0,0,1,1).slice(1,20,21,1)<<std::endl;
   showTensor<float> (output.slice(0,0,1,1).slice(1,20,21,1),2,3,"stereo join");
   //CompareTwoTensors(vols.slice(0,0,1,1).slice(1,5,6,1).slice(2,0,20,1).slice(3,0,20,1),1,5);
   showTensor<float> (vols.slice(0,0,1,1).slice(1,5,6,1),2,3, "computed Tensor to compare");
   //showTensor<float> (output.slice(1,1,2,1),2,3,"stereo join");
   std::cout<<"testing code at level after stereo join  "<<std::endl;
   std::cout<<"volumes sizes of slices "<<vols.slice(0,1,2,1).sizes()<<std::endl;
   this->fix_border(Network,  vols.slice(0,0,1,1), -1);             // fix_border to implement !!!!!!!!!
   std::cout<<"dir 1 done"<<std::endl;
   this->fix_border(Network,  vols.slice(0,1,2,1), 1);              // fix_border to implement !!!!!!!!!*
   std::cout<<"testing code at level after fix border  "<<std::endl;
   std::cout<<"cost volume left after fix border "<<vols.slice(0,0,1,1).slice(1,5,6,1).slice(2,0,10,1).slice(3,0,10,1)<<std::endl;
   std::cout<<"cost volume right after fix border "<<vols.slice(0,1,2,1).slice(1,5,6,1).slice(2,0,10,1).slice(3,0,10,1)<<std::endl;
   /*******************************************************************/
   //std::cout<<"volume stereo join "<<vols.slice(0,0,1,1).slice(1,50,51,1)<<std::endl;
   showTensor<float> (vols.slice(0,0,1,1).slice(1,50,51,1),2,3, "stereo after fix border");
   torch::Tensor vol;
   torch::Tensor disp=torch::empty({2,1, X_batch.size(2),X_batch.size(3)},torch::TensorOptions().dtype(torch::kFloat32).device(device));
   int mb_directions[2] = {1,-1};
   std::cout<<"Cross based cost aggregation to be done !"<<std::endl;
   for (auto direction : mb_directions)
   {
      vol=vols.slice(0,direction == -1 ? 0 : 1, direction == -1 ? 1 : 2,1);
      torch::Tensor x0c=torch::empty({1, 4, vol.size(2), vol.size(3)},torch::TensorOptions().dtype(torch::kFloat32).device(device));
      torch::Tensor x1c=torch::empty({1, 4, vol.size(2), vol.size(3)},torch::TensorOptions().dtype(torch::kFloat32).device(device));
      std::cout<<"tempo tensor sizes "<<x0c.sizes()<<"   "<<x1c.sizes()<<std::endl;
      cudaDeviceSynchronize();
      Cross(X_batch.slice(0,0,1,1), x0c, opt.L1, opt.tau1); 
      cudaDeviceSynchronize();
      Cross(X_batch.slice(0,1,2,1), x1c, opt.L1, opt.tau1); 
      cudaDeviceSynchronize();
      torch::Tensor tmp_cbca = torch::empty({1, disp_max, vol.size(2), vol.size(3)},torch::TensorOptions().dtype(torch::kFloat32).device(device));
      for (int i=0;i<opt.cbca_i1;i++)
        {
	     std::cout<<"++++++++++++++++++++pppppppppppppppppppppp"<<std::endl;
         CrBaCoAgg(x0c,x1c,vol,tmp_cbca,direction);
	     showTensor<float>(tmp_cbca.slice(1,20,21,1),2,3,"after first cbca volume");
         vol.copy_(tmp_cbca);
	    }
	  tmp_cbca=tmp_cbca.mul(0);
	  cudaDeviceSynchronize();
	  //showTensor<float>(vol.slice(1,9,10,1).abs(),2,3,"volume at disp 9");
	  //showTensor<float>(vol.slice(1,10,11,1).abs(),2,3,"volume at disp 10");
	  // SGM 
      auto voll =at::transpose(at::transpose(vol,1,2),2,3).contiguous();
      //auto voll = vol.view({vol.size(0),vol.size(2),vol.size(3),vol.size(1)}); 
      std::cout<<"cost volume transpose direction  "<<direction<<"    "<<voll.slice(1,0,10,1).slice(2,0,10,1).slice(3,5,6,1).sizes()<<std::endl;
      std::cout<<"cost volume transpose direction  "<<direction<<"    "<<voll.slice(3,5,6,1).slice(1,0,10,1).slice(2,0,10,1)<<std::endl;
      std::cout<<"volume  transposed "<<voll.sizes()<<std::endl;
      torch::Tensor out = torch::zeros({1, voll.size(1), voll.size(2), voll.size(3)},torch::TensorOptions().dtype(torch::kFloat32).device(device));
      torch::Tensor tmp = torch::zeros({voll.size(2), voll.size(3)},torch::TensorOptions().dtype(torch::kFloat32).device(device));
      std::cout<<"temp tensor size   ============>>>>>>  "<<tmp.sizes()<<std::endl;;
      for (int i=0;i<opt.sgm_i;i++)
        {
             std::cout<<"++++++++++++++++++++SGM SGM SHGM SGM SGM "<<std::endl;
             //out=out.mul(0);
             sgm2(X_batch.slice(0,0,1,1), X_batch.slice(0,1,2,1), voll, out, tmp, opt.pi1, opt.pi2, opt.tau_so,
                opt.alpha1, opt.sgm_q1, opt.sgm_q2, direction);
             //cudaDeviceSynchronize();
             //vol=vol.copy_(out).div(4);
             voll=out.div(4);
	    }
	  //cudaDeviceSynchronize();
      //vol.resize_({1, disp_max, X_batch.size(2), X_batch.size(3)});
      std::cout<<"out computed transpose direction  "<<direction<<"    "<<out.slice(1,0,10,1).slice(2,0,10,1).slice(3,5,6,1)<<std::endl;
      //auto outt=out.transpose(3, 2).transpose(2, 1);
      auto outt=at::transpose(at::transpose(out,3,2),2,1).contiguous();
      //auto outt=out.view({1, disp_max, X_batch.size(2), X_batch.size(3)});
      vol=outt.div(4);
      std::cout<<"after sgm 2 has been done !"<<std::endl;
      
      //compare volume results after sgm2
      std::cout<<"cost volume direction "<<direction<<" 0-10        : "<<vol.slice(1,5,6,1).slice(2,0,10,1).slice(3,0,10,1)<<std::endl;
      std::cout<<"cost volume direction "<<direction<<" 100-110     : "<<vol.slice(1,60,61,1).slice(2,100,110,1).slice(3,100,110,1)<<std::endl;
      //std::cout<<"cost volume right after fix border "<<vols.slice(0,1,2,1).slice(1,5,6,1).slice(2,0,10,1).slice(3,0,10,1)<<std::endl;
	  //showTensor<float>(vol.slice(1,10,11,1),2,3,"after sgm2 applied volume");
      //  ANOTHER CBCA 2
      for (int i=0;i<opt.cbca_i2;i++)
         {
	       std::cout<<"+++++++++++++++++++++ppppppppppppppppppppppp"<<std::endl;
           CrBaCoAgg(x0c, x1c, vol, tmp_cbca, direction);
           cudaDeviceSynchronize();
           vol.copy_(tmp_cbca);
         }
      //cudaDeviceSynchronize();
      showTensor<float>(vol.slice(1,49,50,1),2,3,"after cbca2 volume");
      std::cout<<"cross based cost aggregation 2 "<<std::endl;
       // Get the min disparity from the cost volume 
      
      std::tuple<torch::Tensor, torch::Tensor> d_Tpl = at::min(vol,1);
      torch::Tensor d=std::get<0>(d_Tpl);
      torch::Tensor indexes=std::get<1>(d_Tpl);
      std::cout<<"volume sizes "<<vol.sizes()<<std::endl;
      //std::cout<<indexes.slice(1,100,200,1).slice(2,100,200,1)<<std::endl;
      std::cout<<"disparity indexes"<<indexes.sizes()<<std::endl;
      //std::cout<<"Disparity tensor values "<<d<<std::endl;
      at::reshape(d, {1,1, X_batch.size(2),X_batch.size(3)});
	  showTensorDisparity(indexes,1,2); 
	  std::cout<<"indexes values and sizes  ++++++"<<indexes.numel()<<std::endl;
      disp.index_put_({direction == 1 ? 0 : 1},indexes); // Make sure it is correct and see what it gives as a result !!!!!!!!!!!! 
   }
      std::cout<<"stereo process completed"<<std::endl;
      std::cout<<"disparity before outlier detection 1 "<<disp.slice(0,0,1,1).slice(2,0,15,1).slice(3,0,15,1)<<std::endl;
      std::cout<<"disparity before outlier detection -1"<<disp.slice(0,1,2,1).slice(2,0,15,1).slice(3,70,85,1)<<std::endl;
      // All Subsequent steps that allow to handle filtering and interpolation 
      torch::Tensor outlier = torch::zeros(disp.slice(0,1,2,1).sizes(),torch::TensorOptions().dtype(torch::kFloat32).device(device));
      torch::Tensor out = torch::zeros(disp.slice(0,1,2,1).sizes(),torch::TensorOptions().dtype(torch::kFloat32).device(device));
      torch::Tensor out2 = torch::zeros(disp.slice(0,1,2,1).sizes(),torch::TensorOptions().dtype(torch::kFloat32).device(device));
      torch::Tensor out3 = torch::zeros(disp.slice(0,1,2,1).sizes(),torch::TensorOptions().dtype(torch::kFloat32).device(device));
      torch::Tensor out4 = torch::zeros(disp.slice(0,1,2,1).sizes(),torch::TensorOptions().dtype(torch::kFloat32).device(device));
      torch::Tensor out5 = torch::zeros(disp.slice(0,1,2,1).sizes(),torch::TensorOptions().dtype(torch::kFloat32).device(device));
      std::cout<<"disparity map dimensions "<<disp.sizes()<<std::endl;
      outlier_detection(disp.slice(0,1,2,1), disp.slice(0,0,1,1), outlier, disp_max);
      //std::cout<<"disparity after outlier detection 1 "<<disp.slice(0,0,1,1).slice(2,0,15,1).slice(3,0,15,1)<<std::endl;
      //std::cout<<"disparity after outlier detection -1"<<disp.slice(0,1,2,1).slice(2,0,15,1).slice(3,0,15,1)<<std::endl;
      std::cout<<"outlier detection done !"<<std::endl;
      interpolate_occlusion(disp.slice(0,1,2,1), outlier,out);       // CHECK THIS UP !!!!!!!!!!!!!!!
      //std::cout<<"interpolate occlusion   "<<out<<std::endl;
      showTensorDisparity2(out,2,3); 
      FILE *file = fopen("./outocclusioncpp.bin", "wb");
      fwrite(out.data_ptr<float>(), sizeof(float), out.numel(), file);
      fclose (file);
      // check at this level 
      std::cout<<"intepolate occlusion done !"<<std::endl;
      interpolate_mismatch(out, outlier,out2);                        // CHECK THIS UP !!!!!!!!!!!!!!!
      FILE *filemis = fopen("./mismatched.bin", "wb");
      fwrite(out2.data_ptr<float>(), sizeof(float), out2.numel(), filemis);
      fclose(filemis);
      std::cout<<" Mismatch interpolation done !"<<std::endl;
      subpixel_enchancement(out2, vol, out3, disp_max);                 // CHECK THIS UO !!!!!!!!!!!!!!!
      FILE *fileenhance = fopen("./enhanced.bin", "wb");
      fwrite(out3.data_ptr<float>(), sizeof(float), out3.numel(), fileenhance);
      fclose(fileenhance);
      std::cout<<"sub pixel enhhancement done !"<<std::endl;
      median2d(out3,out4,5);   
      FILE *filemedian = fopen("./median.bin", "wb");
      fwrite(out4.data_ptr<float>(), sizeof(float), out4.numel(), filemedian);         
      fclose(filemedian); 
      std::cout<<"Median 2 done !"<<std::endl;                              // CHECK THIS UO !!!!!!!!!!!!!!!
      std::cout<<"gaussian blur "<<gaussian(opt.blur_sigma)<<std::endl;
      mean2d(out4, gaussian(opt.blur_sigma), out5, opt.blur_t);         // CHECK THIS UO !!!!!!!!!!!!!!!  GAUSSIAN
      std::cout<<"  filtering has beeen realized "<<std::endl;
      std::cout<<"disparity end"<<out5.slice(2,0,15,1).slice(3,0,15,1)<<std::endl;
      FILE *file2 = fopen("./end.bin", "wb");
      fwrite(out5.data_ptr<float>(), sizeof(float), out5.numel(), file2);
      fclose(file2);
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
         opt.disp_max     = 150;
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
            opt.lr         = 0.002;
            opt.bs         = 128;
            opt.mom        = 0.9;
            opt.true1      = 1;
            opt.false1     = 4;
            opt.false2     = 10;
            opt.L1         = 0;
            opt.cbca_i1    = 0;
            opt.cbca_i2    = 0;
            opt.tau1       = 0.0;
            opt.pi1        = 4;
            opt.pi2        = 55.72;
            opt.sgm_i      = 1;
            opt.sgm_q1     = 3;
            opt.sgm_q2     = 2.5;
            opt.alpha1     = 1.5;
            opt.tau_so     = 0.02;
            opt.blur_sigma = 7.74;
            opt.blur_t     = 5.0;
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
            opt.cbca_i1    =  4;
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
  std::string Datapath="/home/mohamedali/Documents/BH_STUDY/TestVahingNoBH058/";
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
  //torch::cuda::Tensor aaa=torch::zeros(2);at::cuda::is_available()
  auto cuda_available = torch::cuda::is_available();
  torch::Device device(cuda_available ? torch::kCUDA : torch::kCPU);
  std::cout << (cuda_available ? "CUDA available. Training on GPU." : "Training on CPU.") << '\n';
  /********************************************************************/ 
  //device=torch::kCUDA ;
  printArgs(opt);
  SimilarityLearner SimilarityLearn;
  if (strcmp(argv[3],"Train")==0)
  {
     ConvNet_Fast Network(3,opt.ll1);
     Network->createModel(opt.fm,opt.ll1,opt.n_input_plane,3);
     
     
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
       const size_t num_epochs =50;
       const double learning_rate = 0.001;
       const double learning_rate_decay=10.0;
       // Data loader  ==> Training dataset  
       auto train_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
           std::move(IARPADatasetTr), opt.bs/2);
       
       int num_train_samples=IARPADatasetTr.size().value();
       std::cout<<"dataset size "<<num_train_samples<<std::endl;
       
     
       torch::optim::SGD optimizer(
           Network->parameters(), torch::optim::SGDOptions(learning_rate).momentum(0.5)); 
       
      Network->to(device);
      for (int epoch=0;epoch<num_epochs;epoch++)
      {
         SimilarityLearn.train(Network, train_loader,optimizer, epoch,num_train_samples,opt.bs, 20,device); // 20 means a display of results in a periodic fashion
      }
      
      std::string fileName="./SavedNetwork";
      SimilarityLearn.Save_Network(Network,fileName,opt);
      //torch::save(Network,fileName); 
      
      /*******************SAVING WEIGHTS AND BIASES AFTER TRAINING ************/
  }
  else if (strcmp(argv[3],"Test")==0)
  {
     /**********************************************************************/
      // Testing on an Unseen chunk of the IARPA DATASET 
      
      //std::string outputfileName=std::string(argv[5])+std::string("/net_")+std::string(argv[1])+std::string("_")+std::string(argv[2])+std::string("_")+std::to_string(num_epochs)+std::string(".pt");
      
      ConvNet_Fast TestNetwork(3,opt.ll1);
      
      /**********************************************************************/
      
      /********* *HOW TO READ PICKLED MODEL ==> FIND A SOLUTION *************/
      
      /**********************************************************************/
      
      // Need to copy the learnt model to the test model and do testing 
      //auto copy =Network->clone();
      
      //TestNetwork= std::dynamic_pointer_cast<ConvNet_Fast>(copy);
      //TestNetwork= Network->clone();
      TestNetwork->createModel(opt.fm,opt.ll1,opt.n_input_plane,3);
      int Ws=SimilarityLearn.GetWindowSize(TestNetwork);
      /**********************************************************************/
      
      /***CODE SNIPPET TO LOAD WEIGHTS AND BIAS STORED IN BINARY Files ******/
      std::vector<std::pair<std::string,std::string>> Container;
      
      auto p1 = std::make_pair("/home/mohamedali/Documents/BH_STUDY/WB_No_BH_Ep_50/Weights_1_cudnn.SpatialConvolution.bin","/home/mohamedali/Documents/BH_STUDY/WB_No_BH_Ep_50/Biases_1_cudnn.SpatialConvolution.bin");
      Container.push_back(p1);
      auto p2 = std::make_pair("/home/mohamedali/Documents/BH_STUDY/WB_No_BH_Ep_50/Weights_3_cudnn.SpatialConvolution.bin","/home/mohamedali/Documents/BH_STUDY/WB_No_BH_Ep_50/Biases_3_cudnn.SpatialConvolution.bin");
      Container.push_back(p2);
      auto p3 = std::make_pair("/home/mohamedali/Documents/BH_STUDY/WB_No_BH_Ep_50/Weights_5_cudnn.SpatialConvolution.bin","/home/mohamedali/Documents/BH_STUDY/WB_No_BH_Ep_50/Biases_5_cudnn.SpatialConvolution.bin");
      Container.push_back(p3);
      auto p4 = std::make_pair("/home/mohamedali/Documents/BH_STUDY/WB_No_BH_Ep_50/Weights_7_cudnn.SpatialConvolution.bin","/home/mohamedali/Documents/BH_STUDY/WB_No_BH_Ep_50/Biases_7_cudnn.SpatialConvolution.bin");
      Container.push_back(p4);
      SimilarityLearn.PopulateModelFromBinary(TestNetwork,Container);
      /**********************************************************************/ 
      
      //****++++++auto FastNetworkTrained=TestNetwork->getFastSequential();
       //auto model =torch::load(FastNetworkTrained,"net_vahingen_fast_-a_train_tr_14.t7");
     
      //std::cout<<" Trained network >>>>> "<<FastNetworkTrained<<std::endl;
      //SimilarityLearn.Load_Network(TestNetwork,outputfileName);
       
       //torch::load(TestNetwork,outputfileName);  net_vahingen_fast_-a_train_tr_14
      
      
      // Need to change padding to value 1 so output image will keep the same size as the input 
      auto Fast=TestNetwork->getFastSequential(); 
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
      SimilarityLearn.test(IARPADatasetTe,TestNetwork,num_test_samples, device,opt);
  }
  else
   {
	  std::cout<<"no other option "<<std::endl; 
   }
  


 
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

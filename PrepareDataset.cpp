#include <torch/torch.h>
#include <torch/script.h>
#include <ATen/ATen.h>
#include <iostream>
#include <StdAfx.h>
#include <stdio.h>
#include <math.h>
#include <string>
#include <fstream>
#include <vector>
//#include <opencv/cv.h>
//#include <opencv/highgui.h>
//#include <opencv2/core.hpp>
#include <cuda_runtime.h>
//#include "StereoDataset.h"
//#include "Convnet_Fast.h"
//#include "cv.h"
#include "Census.cuh"
#include <experimental/filesystem>
#include "Image.h"

/***********************************************************************/
    /*std::string homols_path = "../IMG_5568.tif.dat";
    ElPackHomologue aPackIn1 =  ElPackHomologue::FromFile(homols_path);
    std::cout<<"Found "<<aPackIn1.size()<<" points.\n";*/
/**********************************************************************/
    
namespace F = torch::nn::functional;
namespace fs = std::experimental::filesystem;


template <typename T>  void Tensor2File(torch::Tensor a, std::string fname)
{
   //Store Tensor 
   T * TensorContent=a.data_ptr<T>();
   FILE *finaldestination = fopen(fname.c_str(), "wb");
   fwrite(TensorContent, sizeof(T), a.numel(), finaldestination);
   
   //Store dimensions 
   std::ofstream finaldestinationDim(fname.append(".dim").c_str());
   
   for (int dd=0;dd<a.dim();dd++)
   {
      finaldestinationDim<<a.size(dd)<<std::endl;
   }
   finaldestinationDim.close();
   
   //Store data type 
   std::ofstream datatypetensor(fname.append(".type").c_str());
   
   if (std::is_floating_point<T>::value)
   {
      datatypetensor<<"float32"<<std::endl;
   }
   else
   {
      datatypetensor<<"int32"<<std::endl;
   }
   datatypetensor.close();
}

int main(int argc, char **argv) {
   torch::manual_seed(42);
	// Call Elise Librairy We will be using because classes that handle 
	// These transformations are already computed 

	//std::string Datapath;
	//std::string NTraining,NTesting;
	//std::string Option;
		
	/***********************************************************************
	 Initilialize ElInitArgMain which as i understood captures arguments entered 
	 by the operator 
	/**********************************************************************/
		
	/*(
		argc,argv,
		LArgMain()  << EAMC (Datapath,"Path to the dataset where triplets of folders are lying",eSAM_IsExistFile)
					<< EAMC (NTraining, "Number of training couples pf Tiles ")
					<< EAMC (NTesting, "Number of Testing couples of Tiles"),
		LArgMain()  << EAM(Option,"Option",true,"FOR NOW, DON'T DO ANYTHING")
	);
	
	if (MMVisualMode) return EXIT_SUCCESS;*/
	std::string Datapath(argv[1]);
    std::string NTraining(argv[2]);
    std::string NTesting(argv[3]);
    std::string Dataset(argv[3]);
    
    int Ntr=std::stoi(NTraining);
    int Nte=std::stoi(NTesting);
    
    //std::cout<<"Ntraining "<<Ntr<<"  Testing "<<Nte<<"  Datapath "<<Datapath<<std::endl;
    /*******************************************************************/
    std::string image0="colored_0";
    std::string image1="colored_1";
    std::string dispocc="disp_occ";
    
    int height=1024;
    int width=1024;
    int nbChannels=1;
    
    //Initialize tensors X0 and X1
    torch::Tensor X0=torch::zeros({Ntr+Nte,nbChannels,height,width},torch::TensorOptions().dtype(torch::kFloat32));
    torch::Tensor X1=torch::zeros({Ntr+Nte,nbChannels,height,width},torch::TensorOptions().dtype(torch::kFloat32));
    torch::Tensor dispnoc=torch::zeros({Ntr+Nte,1,height,width},torch::TensorOptions().dtype(torch::kFloat32));
    torch::Tensor metadata=torch::zeros({Ntr+Nte,3},torch::TensorOptions().dtype(torch::kInt32));
    
    
    // Parse the DataPath folder and get all possible triplets of left | right | disparity Tiles 
    std::vector<std::pair<std::string,std::string>> Examples;
    
    
    const fs::path pathToShow{ Datapath};
    
    for (const auto& entry : fs::directory_iterator(pathToShow))
       {
          const auto filenameStr = entry.path().filename().string();
          std::cout<<" File name STr "<<filenameStr<<std::endl;
          if (is_directory(status(entry.path())))
           {  
              // under this directory there should be colored_0 colored_1 and disp_occ directories 
              const fs::path pathToImage{Datapath+"/"+filenameStr+"/"+image0};
              for (const auto& entry : fs::directory_iterator(pathToImage))
                {
                   const auto imageName=entry.path().filename().string();
                   
                   if (is_regular_file(status(entry.path())))
                      {
                        // Check if it is png extension 
                        if(imageName.find_last_of(".") != std::string::npos)
                           {
							   std::size_t posPoint=imageName.find_last_of(".");
							  if (strcmp(imageName.substr(posPoint+1).c_str(),"png")==0) // is a png file
                               {
								std::size_t posUnderScore= imageName.find_last_of("_");
								//get the id of the image 
								std::string IdImage=imageName.substr(posUnderScore+1,posPoint-posUnderScore-1);
								//make a pair of folder and IdImage 
								auto pair=std::make_pair(filenameStr,IdImage);
								Examples.push_back(pair);
                               } 
                           }
                       }
                }
          }
       }
       
    /*************DEBUG PRUPOSESS *********************************/
   /*for (int i=0;i<Examples.size();i++)
   {
    std::cout<<"Examples dataset Names "<<std::get<0>(Examples.at(i))<<"  "<<std::get<1>(Examples.at(i))<<std::endl;
   }*/
    /*************DEBUG PRUPOSESS *********************************/
    
    //CREATE DATASET 
    int AlldataNb=0;
    for (int cc =0;cc<Examples.size();cc)
    {
       //IMAGE NAMES 
       std::string Image0FullPath=Datapath+"/"+std::get<0>(Examples.at(cc))+"/"+image0+"/"+std::get<0>(Examples.at(cc))+"_"+std::get<1>(Examples.at(cc))+".png";
       std::string Image1FullPath=Datapath+"/"+std::get<0>(Examples.at(cc))+"/"+image1+"/"+std::get<0>(Examples.at(cc))+"_"+std::get<1>(Examples.at(cc))+".png";
       std::string DisparityFullPath=Datapath+"/"+std::get<0>(Examples.at(cc))+"/"+dispocc+"/"+std::get<0>(Examples.at(cc))+"_"+std::get<1>(Examples.at(cc))+".png";
       
       // Read Images 
       ColorImg img0(Image0FullPath);
       ColorImg img1(Image1FullPath);
       
       //Tiff_Im img0=Tiff_Im::UnivConvStd(Image0FullPath);
       //Tiff_Im img1=Tiff_Im::UnivConvStd(Image0FullPath);
       
       //Tiff_Im Disp=Tiff_Im::UnivConvStd(Image0FullPath);
       
       //sizes 
       int img_height=img0.sz().y;
       int img_width=img0.sz().x;
       
       //narrow image if needed 
       torch::Tensor img0InTensor =torch::empty({1,img0.getChannnels(),img_height,img_width},torch::TensorOptions().dtype(torch::kFloat32));       
       torch::Tensor img1InTensor =torch::empty({1,img1.getChannnels(),1,img_height,img_width},torch::TensorOptions().dtype(torch::kFloat32));       
       torch::Tensor DispInTensor =torch::empty({1,1,img_height,img_width},torch::TensorOptions().dtype(torch::kFloat32));  
       
       //Copy image content in tensors 
       if (img0.getChannnels()==1) // Gray image 
       {
		   //Tile Left
		   std::memcpy(img0InTensor.data_ptr<float>(),img0.getRedChannel()->data(),sizeof(U_INT2)*img0InTensor.numel()); //, dim, torch::kFloat32)
		   //Tile Right
		   std::memcpy(img1InTensor.data_ptr<float>(),img1.getRedChannel()->data(),sizeof(U_INT2)*img1InTensor.numel()); //, dim, torch::kFloat32)
       }
       else
       {
           //Tile Left
           using namespace torch::indexing;
           torch::Tensor channelImage=torch::from_blob(img0.getRedChannel()->data(),{1,img_height,img_width},torch::kFloat32);
           img0InTensor.index_put_({Slice(0,None,1),Slice(0,1,1),Slice(0,None,1),Slice(0,None,1)},channelImage.slice(0,0,1,1));
           channelImage=torch::from_blob(img0.getGreenChannel()->data(),{img_height,img_width},torch::kFloat32);
           img0InTensor.index_put_({Slice(0,None,1),Slice(1,2,1),Slice(0,None,1),Slice(0,None,1)},channelImage.slice(0,0,1,1));
           channelImage=torch::from_blob(img0.getBlueChannel()->data(),{img_height,img_width},torch::kFloat32);
           img0InTensor.index_put_({Slice(0,None,1),Slice(2,3,1),Slice(0,None,1),Slice(0,None,1)},channelImage.slice(0,0,1,1));
           //Tile Right 
           channelImage=torch::from_blob(img1.getRedChannel()->data(),{1,img_height,img_width},torch::kFloat32);
           img1InTensor.index_put_({Slice(0,None,1),Slice(0,1,1),Slice(0,None,1),Slice(0,None,1)},channelImage.slice(0,0,1,1));
           channelImage=torch::from_blob(img1.getGreenChannel()->data(),{img_height,img_width},torch::kFloat32);
           img1InTensor.index_put_({Slice(0,None,1),Slice(1,2,1),Slice(0,None,1),Slice(0,None,1)},channelImage.slice(0,0,1,1));
           channelImage=torch::from_blob(img1.getBlueChannel()->data(),{img_height,img_width},torch::kFloat32);
           img1InTensor.index_put_({Slice(0,None,1),Slice(2,3,1),Slice(0,None,1),Slice(0,None,1)},channelImage.slice(0,0,1,1));
       }
       
       //Normalize images 
       img0InTensor=img0InTensor.add(img0InTensor.mean().mul(-1.0)).div(img0InTensor.std());
       img1InTensor=img1InTensor.add(img1InTensor.mean().mul(-1.0)).div(img1InTensor.std());
       
       // Push Tensor images in relevant X0 and X1 Container Tensors 
       X0.index_put_({cc},img0InTensor);
       X1.index_put_({cc},img1InTensor);
       
       
       // Read disparity image 
       if (strcmp(Dataset.c_str(),"Vahingen")==0)
       {
       readPNG16(DispInTensor, DisparityFullPath.c_str());
       }
       else if (strcmp(Dataset.c_str(),"IARPA")==0)
       {
       readPNGIARPA(DispInTensor, DisparityFullPath.c_str());
       }
       dispnoc.index_put_({cc},DispInTensor);
       
       // Fill Metadata 
       metadata.index_put_({cc},torch::tensor({img_height,img_width,cc}));
     AlldataNb++;
     
     // IF TRaining ans Testing requirements have been met, no need to continue looping 
     if (cc>=Ntr+Nte-1)
     {
        break;
     }
    }
    //Generating a random split between Training and Testing datasets 
    
    torch::Tensor Randomizer=torch::randperm(Ntr+Nte);
    torch::Tensor te=Randomizer.slice(0,0,Nte,1);
    torch::Tensor tr=Randomizer.slice(0,Nte+1,Nte+Ntr,1);
    
    torch::Tensor nnz_tr=torch::empty({1000*1000*1000,4},torch::TensorOptions().dtype(torch::kFloat32));  
    torch::Tensor nnz_te=torch::empty({500*1000*1000,4},torch::TensorOptions().dtype(torch::kFloat32));  
    
    int nnz_tr_t=0;
    int nnz_te_t=0;
    for (int cc=0;cc<Ntr+Nte;cc++)
    {
       auto disp=dispnoc.slice(0,cc,cc+1,1);
       remove_nonvisible(disp);
       remove_occluded(disp);
       remove_white(X0.slice(0,cc,cc+1,1), disp);
       
       bool is_te = false;
       for (int j=0;j<te.numel();j++)   //.accessor<int32_t,1>()[0]
       {
         if (cc==te.accessor<int,1>()[j])
         {is_te=true;}
       }
       
       bool is_tr = false;
       for (int j=0;j<tr.numel();j++)
       {
         if (cc==tr.accessor<int,1>()[j])
         {is_tr=true;}
       }
       
       if (is_te)
       {make_dataset2(disp, nnz_te, cc, nnz_te_t);}
       
       if (is_tr)
       {make_dataset2(disp, nnz_tr, cc, nnz_tr_t);}
    }
    
    // Clean up tensors and get filled data 
    nnz_te=nnz_te.slice(0,0,nnz_te_t,1);
    nnz_tr=nnz_tr.slice(0,0,nnz_tr_t,1);
    
    
    //Resmove last files 
    #ifdef _WIN32
    //system("rm -f %s/*.{bin,dim,type}");
    std::cout<<"windows os "<<std::endl;
    #endif
    #ifdef __unix__
    std::string command="rm -f "+Datapath+"/*.{bin,dim,type}";
    system(command.c_str());
    #endif

    
    // Store tensors in bin files by data type
    std::string X0Name=Datapath+"/x0.bin";
    Tensor2File<float>(X0,X0Name);
    std::string X1Name=Datapath+"/x1.bin";
    Tensor2File<float>(X1,X1Name);
    std::string dispnocc=Datapath+"/dispnoc.bin";
    Tensor2File<float>(dispnoc,dispnocc);
    std::string metadataa=Datapath+"/metadata.bin";
    Tensor2File<int32_t>(metadata,metadataa);
    std::string trr=Datapath+"/tr.bin";
    Tensor2File<int32_t>(tr,trr);
    
    std::string tee=Datapath+"/te.bin";
    Tensor2File<int32_t>(te,tee);
    
    std::string nnztr=Datapath+"/nnz_tr.bin";
    Tensor2File<float>(nnz_tr,nnztr);
    
    std::string nnzte=Datapath+"/nnz_te.bin";
    Tensor2File<float>(nnz_te,nnzte);

    return EXIT_SUCCESS;
}


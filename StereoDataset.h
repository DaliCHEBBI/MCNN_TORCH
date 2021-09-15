#pragma once
#include <vector>
#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <ATen/ATen.h>
#include <iostream>
#include <stdio.h>
#include <string>
#include <fstream>
#include "cv.h"
#include <math.h>

//imports to read binary file 
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>


using namespace torch::indexing;
using namespace std;


class StereoDataset : public torch::data::Dataset<StereoDataset>
{
    private:

        //std::vector<std::tuple<std::string /*file location*/, int64_t /*label*/>> csv_;
         torch::Tensor mX0_left_Dataset;
         torch::Tensor mX1_right_Dataset;
         torch::Tensor mdispnoc;
         torch::Tensor mmetadata;
         torch::Tensor mtr;
         torch::Tensor mte;
         torch::Tensor mnnztr;
         torch::Tensor mnnzte;
         int mninput_plane;
         int mws;
         int trans;
         float hscale;
         float scale;
         float hflip;
         float vflip;
         float brightness;
         int true1;
         int false1;
         int false2;
         float rotate;
         float contrast;
         float d_hscale;
         float d_hshear;
         int d_vtrans;
         float d_brightness;
         float d_contrast;
         float d_rotate;
         
    public:
        explicit StereoDataset(std::string X0_left_Dataset,std::string X1_right_Dataset,std::string dispnoc_Data,
             std::string metadata_File,std::string tr_File,std::string te_File,std::string nnztr_File,std::string nnzte_File, 
             int input_plane, int ws, int transs, float hscalee, float scalee, float hflipp,
             float vflipp, float bright, int tr1, int flse1, int flse2, float rotatee, 
             float cntrast, float d_hscle, float d_shearh, int d_vtrns, float d_bright, float d_cntrast,
             float d_rotatee):
            // Load csv file with file locations and labels.
              mX0_left_Dataset(fromfile(X0_left_Dataset)),
              mX1_right_Dataset(fromfile(X1_right_Dataset)),
              mdispnoc(fromfile(dispnoc_Data)),
              mmetadata(fromfile(metadata_File)),
              mtr(fromfile(tr_File)),
              mte(fromfile(te_File)),
              mnnztr(fromfile(nnztr_File)),
              mnnzte(fromfile(nnzte_File)),
              mninput_plane(input_plane),
              mws(ws),
              trans  (transs),
              hscale (hscalee),
              scale  (scalee),
              hflip  (hflipp),
              vflip  (vflipp),
              brightness(bright),
              true1  (tr1),
              false1 (flse1),
              false2 (flse2),
              rotate (rotatee),
              contrast(cntrast),
              d_hscale(d_hscle),
              d_hshear(d_shearh),
              d_vtrans(d_vtrns),
              d_brightness(d_bright),
              d_contrast  (d_cntrast),
              d_rotate    (d_rotatee)
        {
        };
/**********************************************************************/
/**********************READING SAVED TENSORS***************************/
/**********************************************************************/
        // From file methods 
        torch::Tensor fromfile(std::string fname)
        {
           string dimFile=fname;
           std::cout<<"  file name "<<fname<<std::endl; 
           dimFile.append(".dim");
           ifstream BinFile (dimFile);  // to check it is not sure !!!
           std::vector<int64_t> dim;
           string line;
           if (BinFile.is_open())
           {
             while ( getline (BinFile,line) )
             {
               dim.push_back(stoi(line));
             }
             BinFile.close();
           }
           //std::cout<<"  file dim read "<<dim.size()<<dim.at(0)<<std::endl;
           
           if (dim.size()==1 and dim.at(0)==0)
           {
             return torch::full({-1},1);                       // to check later  !!!!
           }
           string typeFile=fname;
           typeFile.append(".type");
           ifstream TypeFile (typeFile);
           string type;
           
           if (TypeFile.is_open())
           {
             while ( getline (TypeFile,line) )
             {
               type=line;
             }
           std::cout<<"  type of data "<<type<<std::endl;
             TypeFile.close();
           }
           // get the sizes of the tensor 
           
           torch::Tensor x;//=torch::empty({dim},torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
           //local size = io.open(fname):seek('end')  // commencer d'ici
           
           if (type == "float32")
              // Change the type of tensor data kFloat32
              {
			     int alldim=1;
			     for (int i=0;i<dim.size();i++)
			     {
                   alldim*=dim.at(i);
                 }
                 //float * data;
                 x=torch::empty({dim},torch::TensorOptions().dtype(at::kFloat).device(torch::kCPU));
                 std::cout<<"  before "<<std::endl;
                 FILE *file = fopen(fname.c_str(), "rb");
                 //fseek(file, 0, SEEK_END);
                 //int size = ftell(file);
                 //std::cout<<"   size vs alldim of tensor "<<size<<"  "<<alldim<<std::endl;
                 //fseek(file, 0, SEEK_SET);
                 float *data = (float *)malloc(sizeof(float)*alldim);
                 for (int i=0;i<alldim;i++)
                 {
                    fread(&data[i],sizeof(data[i]),1,file); 
				  }
                 /*for (int i=0;i<1000;i++)
                 {
					 std::cout<<"  data "<<data[i]<<std::endl; }*/
                 x = torch::from_blob(data, dim, torch::kFloat32).clone();
                 //std::cout<<x.index({0})<<std::endl;
                 data=NULL;
                 //std::cout<<fname<<std::endl;
			     /*int fd = open(fname.c_str(), O_RDONLY);
			     std::cout<<"  dfdfdfdfdfdf  "<<fd<<std::endl;
			     int alldim;
			     for (int i=0;i<dim.size();i++)
			     {
                   alldim*=dim.at(i);
                 }
			     auto * data = static_cast<float*>(mmap(NULL,alldim * sizeof(float), PROT_READ, MAP_SHARED, fd, 0));
                 std::cout<<"  after static_cast "<<"  data size "<<x.sizes()<<std::endl;*/
			     
			     // insert elements of data one by one in tensor 
			     /*for (int i=0;i<alldim;i++) // all elements 
			     {
					 torch.index_put_()
                 }*/
			     /*close(fd);
			     //c10::ArrayRef<int64_t> dim2=c10::ArrayRef<int64_t>(dim);
			     std::cout<<"  torch assignement before "<<std::endl;   
			     x = torch::from_blob(data, dim, at::kFloat).clone();
			     std::cout<<"  torch assignement after  "<<std::endl;
			     data=NULL;
                 std::cout<<"  after delete "<<"  data size "<<x.sizes()<<std::endl;
                 //std::cout<<x.index({0})<<std::endl;*/
              }
           else if (type == "int32")
              // Change the type of tensor data KInt32
              {
                 x=torch::empty({dim},torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU));
			     /*int fd = open(fname.c_str(), O_RDONLY);
			     int alldim;
			     for (int i=0;i<dim.size();i++)
			     {
                   alldim*=dim.at(i);
                 }
			     auto * data =  static_cast<int32_t*>(mmap(NULL,alldim * sizeof(int32_t), PROT_READ, MAP_SHARED, fd, 0));
			     close(fd);
			     x = torch::from_blob(data, dim, torch::kInt32).clone();*/
			     int alldim=1;
			     for (int i=0;i<dim.size();i++)
			     {
                   alldim*=dim.at(i);
                 }
                 
                 FILE *file = fopen(fname.c_str(), "rb");
                 //fseek(file, 0, SEEK_END);
                 //int size = ftell(file);
                 //std::cout<<"   size vs alldim of tensor "<<size<<"  "<<alldim<<std::endl;
                 //fseek(file, 0, SEEK_SET);
                 int32_t *data = (int32_t *)malloc(sizeof(int32_t)*alldim);
                 for (int i=0;i<alldim;i++)
                 {
                    fread(&data[i],sizeof(data[i]),1,file); 
				  }
                 /*for (int i=0;i<1000;i++)
                 {
					 std::cout<<"  data "<<data[i]<<std::endl; }*/
                 x = torch::from_blob(data, dim, torch::kInt32).clone();
                 //std::cout<<x.index({0})<<std::endl;
                 data=NULL; 
                     
              }
           else if (type == "int64")
              // Change the type of tensor data KInt64
              {
                 x=torch::empty({dim},torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU));
			     /*int fd = open(fname.c_str(), O_RDONLY);
			     int alldim;
			     for (int i=0;i<dim.size();i++)
			     {
                   alldim*=dim.at(i);
                 }
			     auto * data =  static_cast<int64_t*>(mmap(NULL,alldim * sizeof(int64_t), PROT_READ, MAP_SHARED, fd, 0));
			     close(fd);
			     x = torch::from_blob(data, dim, torch::kInt64).clone();  
			     */
			     int alldim=1;
			     for (int i=0;i<dim.size();i++)
			     {
                   alldim*=dim.at(i);
                 }
                 
                 FILE *file = fopen(fname.c_str(), "rb");
                 //fseek(file, 0, SEEK_END);
                 //int size = ftell(file);
                 //std::cout<<"   size vs alldim of tensor "<<size<<"  "<<alldim<<std::endl;
                 //fseek(file, 0, SEEK_SET);
                 int64_t *data = (int64_t *)malloc(sizeof(int64_t)*alldim);
                 for (int i=0;i<alldim;i++)
                 {
                    fread(&data[i],sizeof(data[i]),1,file); 
				  }
                 /*for (int i=0;i<1000;i++)
                 {
					 std::cout<<"  data "<<data[i]<<std::endl; }*/
                 x = torch::from_blob(data, dim, torch::kInt64).clone();
                 //std::cout<<x.index({0})<<std::endl;
                 data=NULL;  
              }  
           else
              {
                 std::cout<<" binary file name "<<fname<<"    "<<type<<endl;
                 //assert(false);
              }
           std::cout<<"  check tensor has been correctly read "<<x.sizes()<<std::endl;
           //c10::ArrayRef<int64_t> dim2=c10::ArrayRef<int64_t>(dim);
           //auto x_out = x.reshape(dim2);
           return x;
        }
/**********************************************************************/
        // Override the get method to load custom data.
        torch::data::Example<> get(size_t ind) override {
            //get left image 
            //torch::Tensor tile_left=this->mX0_left_Dataset.accessor<int64_t,index>(); //later in data access !!!!!
            // index means:: nnz indexes 
            torch::Tensor tmp=mnnztr.index({(int)ind});
            int img  = tmp.accessor<float,1>()[0];
            //std::cout<<" img "<<img<<std::endl;
            int dim3 = tmp.accessor<float,1>()[1];
            //std::cout<<" dim3 "<<dim3<<std::endl;
            int dim4 = tmp.accessor<float,1>()[2];
            //std::cout<<" dim4 "<<dim4<<std::endl;
            float       d = tmp.accessor<float,1>()[3];
            //std::cout<<" dddd "<<d<<std::endl;
            
            //Tile left tile right
            torch::Tensor tile_left  = this-> mX0_left_Dataset.index({img-1});    //!!!!!!!!!!!!!!!
            torch::Tensor tile_right = this->mX1_right_Dataset.index({img-1});    //!!!!!!!!!!!!!!!
            
            //std::cout<<tile_left.sizes()<<std::endl;
            // Random transforms applied in order to get patches
            
            /*************************************************************************************************************/
            // Apply random transformations to the left patches that are different from those applied to the right patches 
            /*************************************************************************************************************/
            torch::Tensor d_posT = torch::randint(-true1, true1,{1});   
            int d_pos = (int)d_posT.accessor<float,1>()[0]; 
            //std::cout<<"  check points "<<std::endl;
            torch::Tensor d_negT = torch::randint(false1, false2,{1});   
            int d_neg = (int)d_negT.accessor<float,1>()[0];
            //std::cout<<"  check points ====="<<std::endl;
            torch::Tensor rr=torch::rand({1});
            if (rr.accessor<float,1>()[0]<0.5)
            {
               d_neg = - d_neg;
            }
            assert(hscale <= 1 && scale <= 1);   // these are given variables entered by the user !!!
            torch::Tensor scaleT =torch::rand({1});
            //std::cout<<"  check points ========"<<std::endl;
            // Scaling 
            float hscale =(hscale-1)*scaleT.accessor<float,1>()[0]+1;     // loi uniforme entre [hscale, 1]
            rr=torch::rand({1});
            float s =(scale-1)*rr.accessor<float,1>()[0]+1; // loi uniforme entre [scale,1]
            torch::Tensor scale = torch::tensor ({s * hscale, s});
            rr=torch::rand({1});
            if ((hflip == 1) && (rr.accessor<float,1>()[0] < 0.5))
               {
                  scale.index_put_({0}, - scale.accessor<float,1>()[0]);
               }
            rr=torch::rand({1});
            if ((vflip == 1) && (rr.accessor<float,1>()[0] < 0.5))
               {
                  scale.index_put_({1}, - scale.accessor<float,1>()[1]);
               }
            //std::cout<<"  check points =============="<<std::endl;
            // Shear
            torch::Tensor hshearT = torch::rand({1});
            float hshear = -2*hshear*hshearT.accessor<float,1>()[0]+hshear; // loi uniforme entre [-hshear,hshear]
            
            
            // Translation 
            torch::Tensor transT = torch::randint(-trans, trans,{2});  
            
            // Rotation 
            torch::Tensor phiT =torch::rand({1});
            float phi=-2*(rotate*M_PI / 180.0)*phiT.accessor<float,1>()[0]+(rotate*M_PI / 180.0); //loi uniforme entre [-rotate,rotate]
            
            // Brightness 
            torch::Tensor brightnessT =torch::rand({1});
            float brightness =-2*brightness*brightnessT.accessor<float,1>()[0]+brightness;  // loi uniform entre [-bright,bright]
            
            // Contrast
            assert(contrast >= 1 and d_contrast >= 1);
            torch::Tensor contrastT =torch::rand({1});   
            float contrast = (1/contrast-contrast)*contrastT.accessor<float,1>()[0]+contrast; // [1/contrast, contrast]

            // Another shuffling of the parameters for the rights patches 
            
            //scale 2D  
            rr=torch::rand({1});
            float scaleP=(d_hscale-1) * rr.accessor<float,1>()[0]+1; // loi uniforme 
            torch::Tensor scale_ = torch::tensor({scale.accessor<float,1>()[0] * scaleP, scale.accessor<float,1>()[1]});
            
            // Cisaillement 
            rr=torch::rand({1});
            float hshear_add =-2*d_hshear* rr.accessor<float,1>()[0] + d_hshear;
            float hshear_ = hshear+hshear_add; 
            
            // Translation 2D 
            rr=torch::rand({1});
            float trans_add=-2*d_vtrans+rr.accessor<float,1>()[0]+d_vtrans;
            torch::Tensor trans_=torch::tensor({transT.accessor<float,1>()[0],transT.accessor<float,1>()[1]+trans_add});
            
            // Rotation .clamp_min_(- d_rotate * pi / 180).clamp_max_(d_rotate * pi / 180)
            rr=torch::rand({1});
            float phi_add=-(2*d_rotate*M_PI/180.0) * rr.accessor<float,1>()[0]+ (d_rotate*M_PI/180.0);
            float phi_= phi + phi_add;
            
            // Brightness 
            rr=torch::rand({1});
            float brightness_add= -2*d_brightness*rr.accessor<float,1>()[0]+d_brightness; // loi uniforme 
            float brightness_= brightness + brightness_add; 

            // Contrast 
            rr=torch::rand({1});
            float contrast_add=((1/contrast)-contrast) * rr.accessor<float,1>()[0] + d_contrast; //loi uniforme
            float contrast_ = contrast * contrast_add;
            //std::cout<<"  check points ==================================="<<std::endl;
            // Make 2 pairs of patches: 1. True and 1. False matches with data augmentation 
            auto DoublePairOfPatches=torch::empty({4, mninput_plane,mws,mws},torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
            torch::Tensor PatchDestination =torch::empty({mninput_plane,mws,mws},torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
            Make_Patch(tile_left, PatchDestination,dim3,dim4,scale,phi,transT,hshear,brightness,contrast);
            //put into DoublePairOfPatches 
            DoublePairOfPatches.index_put_({0},PatchDestination);
            //std::cout<<DoublePairOfPatches.index({0});
            Make_Patch(tile_right, PatchDestination,dim3,dim4-d+d+(float)d_pos,scale_,phi_,trans_,hshear_,brightness_,contrast_);
            //put into DoublePairOfPatches 
            DoublePairOfPatches.index_put_({1},PatchDestination);
            //std::cout<<DoublePairOfPatches.index({1});
            Make_Patch(tile_left, PatchDestination,dim3,dim4,scale,phi,transT,hshear,brightness,contrast);
            //put into DoublePairOfPatches 
            DoublePairOfPatches.index_put_({2},PatchDestination);
            //std::cout<<DoublePairOfPatches.index({2});
            Make_Patch(tile_right, PatchDestination,dim3,dim4-d+(float)d_neg,scale_,phi_,trans_,hshear_,brightness_,contrast_);
            //put into DoublePairOfPatches 
            DoublePairOfPatches.index_put_({3},PatchDestination);
            //std::cout<<DoublePairOfPatches.index({3});
            
            //torch::Tensor label_tensor = torch::full({1}, label);
            auto label_tensor = torch::tensor({1});

            return {DoublePairOfPatches, label_tensor};
        };
/**********************************************************************/
        void Make_Patch(torch::Tensor src, torch::Tensor dst, int dim3, int dim4, torch::Tensor scale, float phi, 
             torch::Tensor trans, float hshear, float brightness, float contrast)
        {
           //std::cout<<"  check points ***************"<<std::endl;
           torch::Tensor m=torch::tensor({1.0, 0.0, -(double)dim4, 0.0, 1.0, -(double)dim3});
           //std::cout<<"  check points *********************"<<std::endl;
           m = this->mul32(torch::tensor({1.0, 0.0, (double)trans.accessor<float,1>()[0], 0.0, 1.0, (double)trans.accessor<float,1>()[1]}), m); //translate   !!!!!!
           //std::cout<<"  check points ***************************"<<std::endl;
           m = this->mul32(torch::tensor({(double)scale.accessor<float,1>()[0], 0.0, 0.0, 0.0, (double)scale.accessor<float,1>()[1], 0.0}), m);       //scale !!!!!!
           double c=cos(phi);
           double s=sin(phi);
           //std::cout<<"  check points ********************************"<<std::endl;
           m = this->mul32(torch::tensor({c, s, 0.0, -s, c, 0.0}), m);     //rotate
           m = this->mul32(torch::tensor({1.0, (double)hshear, 0.0, 0.0, 1.0, 0.0}), m); //shear
           m = this->mul32(torch::tensor({1.0, 0.0, (double)(this->mws - 1) / 2, 0.0, 1.0, (double)(this->mws - 1) / 2}), m);
           //std::cout<<"  check points *************************************"<<std::endl;
           // Need to interface torch Tensors with opencv Mat // Later interface with MicMac 
           warp_affine(&src, &dst, &m);
                                               // this will potentially lead to errors (will not cause errors) 
           dst.mul(contrast).add(brightness);
        };
/**********************************************************************/
        torch::Tensor mul32 (torch::Tensor a, torch::Tensor b)
        {
          return torch::tensor({a.accessor<float,1>()[0]*b.accessor<float,1>()[0]+a.accessor<float,1>()[1]*b.accessor<float,1>()[3], 
			  a.accessor<float,1>()[0]*b.accessor<float,1>()[1]+a.accessor<float,1>()[1]*b.accessor<float,1>()[4], 
			  a.accessor<float,1>()[0]*b.accessor<float,1>()[2]+a.accessor<float,1>()[1]*b.accessor<float,1>()[5]+a.accessor<float,1>()[2],
			  a.accessor<float,1>()[3]*b.accessor<float,1>()[0]+a.accessor<float,1>()[4]*b.accessor<float,1>()[3], 
			  a.accessor<float,1>()[3]*b.accessor<float,1>()[1]+a.accessor<float,1>()[4]*b.accessor<float,1>()[4], 
			  a.accessor<float,1>()[3]*b.accessor<float,1>()[2]+a.accessor<float,1>()[4]*b.accessor<float,1>()[5]+a.accessor<float,1>()[5]});
        };
/**********************************************************************/

        // Override the size method to infer the size of the data set.
        torch::optional<size_t> size() const override 
        {
            //std::cout<<" mnnztr size of data "<<
            return this->mnnztr.size(0);
        };

/**********************************************************************/
/********************CHECKING TILE LIMITS FOR PADDING******************/
/**********************************************************************/
        
};

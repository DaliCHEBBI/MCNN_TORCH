#pragma once

#include <vector>
#include <tuple>
#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <ATen/ATen.h>
#include <iostream>
#include <stdio.h>
#include <string>
#include <fstream>
#include "cv.h"
using namespace torch::indexing;

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
         int v_trans;
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
        explicit StereoDataset(string X0_left_Dataset,string X1_right_Dataset,string dispnoc_Data,
             string metadata_File,string tr_File,string te_File,string nnztr_File,string nnzte_File, 
             int input_plane, int ws, int transs, int v_transs, float hscalee, float scalee, float hflipp,
             float vflipp, float bright, int tr1, int flse1, int flse2, float rotatee, 
             float cntrast, float d_hscle, float d_shearh, int d_vtrns, float d_bright, float d_cntrast,
             float d_rotatee)
            // Load csv file with file locations and labels.
            : mX0_left_Dataset(fromfile(X0_left_Dataset)),
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
              v_trans(v_transs),
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
        // From file methods 
        torch::Tensor fromfile(char *fname)
        {
           ifstream BinFile (strcat(fname,".dim"));  // to check it is not sure !!!
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
           
           if (dim.size()==1 and dim.at(1)==0)
           {
             return torch::Tensor() ;                       // to check later  !!!!
           }
        
           ifstream TypeFile (strcat(fname,".type"));
           string type;
           if (TypeFile.is_open())
           {
             while ( getline (TypeFile,line) )
             {
               type=line;
             }
             TypeFile.close();
           }
              
           torch::Tensor x;
           //local size = io.open(fname):seek('end')  // commencer d'ici
           
           if (type == "float32")
              // Change the type of tensor data kFloat32
              {
                 x = x.to(torch::kFloat32);
                 torch::load(x, fname);
              }
           else if (type == "int32")
              // Change the type of tensor data KInt32
              {
                 x = x.to(torch::kInt32);
                 torch::load(x, fname);   
              }
           else if (type == "int64")
              // Change the type of tensor data KInt64
              {
                 x = x.to(torch::kInt64);
                 torch::load(x, fname);     
              }  
           else
              {
                 std::cout<<" binary file name "<<fname<<"    "<<type<<endl;
                 assert(false);
              }
           // reshape the tensor 
           /*for (auto i = dim.begin(); i!= dim.end(); ++i)
           {
             
            }*/
           c10::ArrayRef<int64_t> dim2=c10::ArrayRef<int64_t>(dim);
           auto x_out = x.reshape(dim2);
           return x_out;
        }

/**********************************************************************/
        // Override the get method to load custom data.
        torch::data::Example<> get(size_t index) override {
            //get left image 
            //torch::Tensor tile_left=this->mX0_left_Dataset.accessor<int64_t,index>(); //later in data access !!!!!
            // index means:: nnz indexes 
            int img  = this->mnnztr.index({index,0}) ;
            int dim3 = this->mnnztr.index({index, 1});
            int dim4 = this->mnnztr.index({index, 2});
            float  d = this->mnnztr.index({index, 3});
            
            //Tile left tile right
            torch::Tensor tile_left  = this-> mX0_left_Dataset.index({img});    //!!!!!!!!!!!!!!!
            torch::Tensor tile_right = this->mX1_right_Dataset.index({img});    //!!!!!!!!!!!!!!!
            
            
            // Random transforms applied in order to get patches
            
            /*************************************************************************************************************/
            // Apply random transformations to the left patches that are different from those applied to the right patches 
            /*************************************************************************************************************/
            torch::Tensor d_posT = torch::randint(-true1, true1,{1});   
            int d_pos = (int)d_posT.accessor<float,1>()[0]; 
            torch::Tensor d_negT = torch::randint(false1, false2,{1});   
            int d_neg = (int)d_negT.accessor<float,1>()[0];
            
            if (torch::rand({1}).accessor<float,1>()[0]<0.5)
            {
               d_neg = - d_neg;
            }
            assert(hscale <= 1 && scale <= 1);   // these are given variables entered by the user !!!
            torch::Tensor scaleT =torch::rand({1});
            
            // Scaling 
            float hscale =(hscale-1)*scaleT.accessor<float,1>()[0]+1;     // loi uniforme entre [hscale, 1]
            float s =(scale-1)*torch::rand({1}).accessor<float,1>()[0]+1; // loi uniforme entre [scale,1]
            torch::Tensor scale = torch::Tensor ({s * hscale, s});
            
            if ((hflip == 1) && (torch::rand({1}).accessor<float,1>[0] < 0.5))
               {
                  scale.index_put_({0}, - scale.accessor<float,1>()[0]);
               }
            if ((vflip == 1) && (torch::rand({1}).accessor<float,1>()[0] < 0.5))
               {
                  scale.index_put_({1}, - scale.accessor<float,1>()[1]);
               }
            
            // Shear
            torch::Tensor hshearT = torch::rand({1});
            float hshear = -2*hshear*hshearT.accessor<float,1>()[0]+hshear; // loi uniforme entre [-hshear,hshear]
            
            
            // Translation 
            torch::Tensor transT = torch::randint(-trans, trans,{2});  
            
            // Rotation 
            torch::Tensor phiT =torch::rand({1});
            float phi=-2*(rotate*math.pi / 180.0)*phiT.accessor<float,1>()[0]+(rotate*math.pi / 180.0); //loi uniforme entre [-rotate,rotate]
            
            // Brightness 
            torch::Tensor brightnessT =torch::rand({1});
            float brightness =-2*brightness*brightnessT.accessor<float,1>()[0]+brightness;  // loi uniform entre [-bright,bright]
            
            // Contrast
            assert(contrast >= 1 and d_contrast >= 1);
            torch::Tensor contrastT =torch::rand({1});   
            float contrast = (1/constrast-contrast)*contrastT.accessor<float,1>()[0]+contrast; // [1/contrast, contrast]

            // Another shuffling of the parameters for the rights patches 
            
            //scale 2D  
            float scaleP=(d_hscale-1) * torch::rand({1}).accessor<float,1>()[0]+1; // loi uniforme 
            torch::Tensor scale_ = torch::Tensor({scale.accessor<float,1>[0] * scaleP, scale.accessor<float,1>[1]});
            
            // Cisaillement 
            float hshear_add =-2*d_hshear* torch::rand({1}).accessor<float,1>()[0] + d_hshear;
            float hshear_ = hshear+hshear_add; 
            
            // Translation 2D 
            float trans_add=-2*d_vtrans+torch::rand({1}).accessor<float,1>()[0]+d_vtrans;
            torch::Tensor trans_=torch::Tensor({transT.accessor<float,1>()[0],transT.accessor<float,1>()[1]+trans_add});
            
            // Rotation .clamp_min_(- d_rotate * pi / 180).clamp_max_(d_rotate * pi / 180)
            float phi_add=-(2*d_rotate*pi/180.0) * torch::rand({1}).accessor<float,1>()[0]+ (d_rotate*pi/180.0);
            float phi_= phi + phi_add;
            
            // Brightness 
            float brightness_add= -2*d_brightness*torch::rand({1}).accessor<float,1>()[0]+d_brightness; // loi uniforme 
            float brightness_= brightness + brightness_add; 

            // Contrast 
            float contrast_add=((1/contrast)-contrast) * torch::rand({1}).accessor<float,1>()[0] + d_contrast; //loi uniforme
            float contrast_ = contrast * contrast_add;
             
            // Make 2 pairs of patches: 1. True and 1. False matches with data augmentation 
            
            torch::Tensor DoublePairOfPatches=torch::empty({4, n_input_plane,ws,ws});
            torch::Tensor PatchDestination =torch::empty({n_input_plane,ws,ws});
            Make_Patch(tile_left, PatchDestination,dim3,dim4,scale,phi,trans,hshear,brightness,contrast);
            //put into DoublePairOfPatches 
            DoublePairOfPatches.index_put_({0},PatchDestination);
            Make_Patch(tile_left, PatchDestination,dim3,dim4,scale_,phi_,trans_,hshear_,brightness_,contrast_);
            //put into DoublePairOfPatches 
            DoublePairOfPatches.index_put_({1},PatchDestination);
            Make_Patch(tile_right, PatchDestination,dim3,dim4,scale,phi,trans,hshear,brightness,contrast);
            //put into DoublePairOfPatches 
            DoublePairOfPatches.index_put_({2},PatchDestination);
            Make_Patch(tile_right, PatchDestination,dim3,dim4,scale_,phi_,trans_,hshear_,brightness_,contrast_);
            //put into DoublePairOfPatches 
            DoublePairOfPatches.index_put_({3},PatchDestination);
            
            
            torch::Tensor label_tensor = torch::full({1}, label);

            return {DoublePairOfPatches, label_tensor};
        };
/**********************************************************************/
        void Make_Patch(torch::Tensor src, torch::Tensor dst, int dim3, int dim4, float scale, float phi, 
             float trans, float hshear, float brightness, float contrast);
        {
           //
           torch::Tensor m=torch::Tensor({1, 0, -dim4, 0, 1, -dim3});
           m = this->mul32(torch::Tensor({1, 0, trans.accessor<float,1>()[0], 0, 1, trans.accessor<float,1>()[1]}), m); //translate   !!!!!!
           m = this->mul32(torch::Tensor({scale.accessor<float,1>()[0], 0, 0, 0, scale.accessor<float,1>()[1], 0}), m);       //scale !!!!!!
           float c=cos(phi);
           float s=sin(phi);
           m = this->mul32(torch::Tensor({c, s, 0, -s, c, 0}), m);     //rotate
           m = this->mul32(torch::Tensor({1, hshear, 0, 0, 1, 0}), m); //shear
           m = this->mul32(torch::Tensor({1, 0, (this->mws - 1) / 2, 0, 1, (this->mws - 1) / 2}), m);
           
           // Need to interface torch Tensors with opencv Mat // Later interface with MicMac 
           
           warp_affine(&src, &dst, &m);
                                               // this will potentially lead to errors (will not cause errors) 
           dst.mul(contrast).add(brightness);
        };
/**********************************************************************/
        torch::Tensor mul32 (torch::Tensor a, torch::Tensor b)
        {
          return torch::Tensor({a.accessor<float,1>()[0]*b.accessor<float,1>()[0]+a.accessor<float,1>()[1]*b.accessor<float,1>()[3], 
			  a.accessor<float,1>()[0]*b.accessor<float,1>()[1]+a.accessor<float,1>()[1]*b.accessor<float,1>()[4], 
			  a.accessor<float,1>()[0]*b.accessor<float,1>()[2]+a.accessor<float,1>()[1]*b.accessor<float,1>()[5]+a.accessor<float,1>()[2],
			  a.accessor<float,1>()[3]*b.accessor<float,1>()[0]+a.accessor<float,1>()[4]*b.accessor<float,1>()[3], 
			  a.accessor<float,1>()[3]*b.accessor<float,1>()[1]+a.accessor<float,1>()[4]*b.accessor<float,1>()[4], 
			  a.accessor<float,1>()[3]*b.accessor<float,1>()[2]+a.accessor<float,1>()[4]*b.accessor<float,1>()[5]+a.accessor<float,1>()[5]});
        };
/**********************************************************************/

        // Override the size method to infer the size of the data set.
        torch::optional<size_t> size() const override {

            return csv_.size();
        };
};

/*Demosaicing Kernel 4  

6x8 Image Demonstration for G R, B G Bayer CFA pattern (- = data, + = threadIdx ) 
GreenEvent = Populates green, each thread populates green, it stays on

- - - - - - - -
- - - - - - - -
- - - + - + - -
- - - - - - - -
- - - + - + - -
- - - - - - - -

RedBlueEvent = Populates red and blue, thread located on top-left corner of 2x2 data

- - - - - - - -
- - - - - - - -
- - + - + - - -
- - - - - - - -
- - + - + - - -
- - - - - - - -


*/

#include "../include/image.hpp"
#include <CL/sycl.hpp>
#include <iostream>
#include <algorithm>
#include <limits>
#include <limits.h>

using namespace hipsycl::sycl;
using namespace cv;

//T -> image type, LT -> local type
template <typename T, typename LT=int16_t, typename idxT=unsigned int>
void PopulateParallel4(Image<T,idxT>& image){
    gpu_selector gpu;                
    queue Q(gpu);                   

    idxT height = image.getHeight();
    idxT width = image.getWidth();
    
    {
        buffer<T,3> imgBuffer {image.getPointer(),range<3>(image.getHeight(),image.getWidth(),3),{property::buffer::use_host_ptr{}}};

        /*
        event -> populating all channels --- total green pixels on width and height = h/2,w/2 
        since left, right, top and bottom pixels are not used for algo h/2 - 2, w/2 - 2 thread domain has to be defined
        (w/2 -> total thread count in one row, w/2-2 exluding boundaries)  
        */
        
       

        
        

        event GreenEvent = Q.submit([&](handler& h){
            accessor<T,3> imgAcc(imgBuffer,h,read_write);
           
            h.parallel_for(range<2>{height/2-2,width/2-2},[=](id<2> idx){
                //calculating red pixel's green color
                idxT i = 2*idx[0]+2;
                idxT j = 2*idx[1]+3;

                //red colors
                LT A1 = imgAcc[i-2][j][2];
                LT A3 = imgAcc[i][j-2][2];
                LT A5 = imgAcc[i][j][2];
                LT A7 = imgAcc[i][j+2][2];
                LT A9 = imgAcc[i+2][j][2];
                //green colors
                LT G2 = imgAcc[i-1][j][1];
                LT G4 = imgAcc[i][j-1][1];
                LT G6 = imgAcc[i][j+1][1];
                LT G8 = imgAcc[i+1][j][1];

                LT alpha = std::abs(-A3 + 2*A5 - A7) + std::abs(G4 - G6);
                LT beta = std::abs(-A1 + 2*A5 - A9) + std::abs(G2 - G8);

                LT G5;
                if(alpha<beta)
                    G5 = (G4 + G6 - A3 + 2*A5 - A7) / 2;
                else if(alpha>beta)
                    G5 = (G2 + G8 - A1 + 2*A5 - A9) / 2;
                else if(alpha==beta)
                    G5 = (2*(G2 + G4 + G6 + G8) - A1 - A3 + 4*A5 - A7 - A9) / 8;

                imgAcc[i][j][1] = std::clamp<LT>(G5,0,UCHAR_MAX);
                

                
                //calculating blue pixel's green color
                i = 2*idx[0]+3;
                j = 2*idx[1]+2;
                //blue colors
                A1 = imgAcc[i-2][j][0];
                A3 = imgAcc[i][j-2][0];
                A5 = imgAcc[i][j][0];
                A7 = imgAcc[i][j+2][0];
                A9 = imgAcc[i+2][j][0];
                //green colors
                G2 = imgAcc[i-1][j][1];
                G4 = imgAcc[i][j-1][1];
                G6 = imgAcc[i][j+1][1];
                G8 = imgAcc[i+1][j][1];

                alpha = std::abs(-A3 + 2*A5 - A7) + std::abs(G4 - G6);
                beta = std::abs(-A1 + 2*A5 - A9) + std::abs(G2 - G8);


                if(alpha<beta)
                    G5 = (G4 + G6 - A3 + 2*A5 - A7) / 2;
                if(alpha>beta)
                    G5 = (G2 + G8 - A1 + 2*A5 - A9) / 2;
                if(alpha==beta)
                    G5 = (2*(G2 + G4 + G6 + G8) - A1 - A3 + 4*A5 - A7 - A9) / 8;

                
                imgAcc[i][j][1] = std::clamp<LT>(G5,0,UCHAR_MAX);
            
                
                
            });

        });
        
        event RedBlueEvent = Q.submit([&](handler& h){
            accessor<T,3> imgAcc(imgBuffer,h,read_write);
           
            h.parallel_for(range<2>{height/2-2,width/2-2},[=](id<2> idx){

                idxT ix = 2*idx[0]+2;  //thread index x
                idxT iy = 2*idx[1]+2;  //thread index y
                idxT i,j;

                LT A1,A2,A3,A4,A7,A9;
                LT G1,G2,G3,G4,G5,G7,G9;
                LT alpha,beta;
                

                //Populating top-left green pixel's red and blue color
                //Vertical Neighbour (blue color)
                i = ix; 
                j = iy; 

                //blue colors
                A1 = imgAcc[i-1][j][0];
                A7 = imgAcc[i+1][j][0];
                
                //green colors
                G1 = imgAcc[i-1][j][1];
                G4 = imgAcc[i][j][1];
                G7 = imgAcc[i+1][j][1];
        
                A4 = (A1 + A7 - G1 + 2*G4 - G7) / 2;

                imgAcc[i][j][0] = std::clamp<LT>(A4,0,UCHAR_MAX);

                //Horizontal Neighbour (red color)
                //red colors
                A1 = imgAcc[i][j-1][2];
                A3 = imgAcc[i][j+1][2];
                
                //green colors
                G1 = imgAcc[i][j-1][1];
                G2 = imgAcc[i][j][1];
                G3 = imgAcc[i][j+1][1];
        
                A2 = (A1 + A3 - G1 + 2*G2 - G3) / 2;

                imgAcc[i][j][2] = std::clamp<LT>(A2,0,UCHAR_MAX);

                
                //Populating bottom-right green pixel's red and blue color
                //Vertical Neighbour (red color)
                i = ix+1;
                j = iy+1;

                //red colors
                A1 = imgAcc[i-1][j][2];
                A7 = imgAcc[i+1][j][2];
                
                //green colors
                G1 = imgAcc[i-1][j][1];
                G4 = imgAcc[i][j][1];
                G7 = imgAcc[i+1][j][1];
        
                A4 = (A1 + A7 - G1 + 2*G4 - G7) / 2;

                imgAcc[i][j][2] = std::clamp<LT>(A4,0,UCHAR_MAX);


                //Horizontal Neighbour (blue color)
                //blue colors
                A1 = imgAcc[i][j-1][0];
                A3 = imgAcc[i][j+1][0];
                
                //green colors
                G1 = imgAcc[i][j-1][1];
                G2 = imgAcc[i][j][1];
                G3 = imgAcc[i][j+1][1];
        
                A2 = (A1 + A3 - G1 + 2*G2 - G3) / 2;

                imgAcc[i][j][0] = std::clamp<LT>(A2,0,UCHAR_MAX);
                
            
                //Populating red pixel's blue color 
                //Diagonal Neigbours
                i = ix;
                j = iy+1;

                //blue colors
                A1 = imgAcc[i-1][j-1][0];
                A3 = imgAcc[i-1][j+1][0];
                A7 = imgAcc[i+1][j-1][0];
                A9 = imgAcc[i+1][j+1][0];

                
                //green colors
                G1 = imgAcc[i-1][j-1][1];   

                LT C5;
                G3 = imgAcc[i-1][j+1][1];
                G5 = imgAcc[i][j][1];
                G7 = imgAcc[i+1][j-1][1];
                G9 = imgAcc[i+1][j+1][1];

                alpha = std::abs(-G3 + 2*G5 - G7) + std::abs(A3 - A7);
                beta = std::abs(-G1 + 2*G5 - G9) + std::abs(A1 - A9);
                
                if(alpha<beta)
                    C5 = (A3 + A7 -G3 +2*G5 - G7) / 2;
                else if(alpha>beta)
                    C5 = (A1 + A9 -G1 +2*G5 - G9) / 2;
                else if(alpha==beta)
                    C5 = (2*(A1 + A3 + A7 + A9) - G1 - G3 + 4*G5 - G7 - G9) / 8;


                imgAcc[i][j][0] = std::clamp<LT>(C5,0,UCHAR_MAX);


                //Populating blue pixel's red color 
                //Diagonal Neigbours
                i = ix+1;
                j = iy;

                //red colors
                A1 = imgAcc[i-1][j-1][2];
                A3 = imgAcc[i-1][j+1][2];
                A7 = imgAcc[i+1][j-1][2];
                A9 = imgAcc[i+1][j+1][2];

                
                //green colors
                G1 = imgAcc[i-1][j-1][1];
                G3 = imgAcc[i-1][j+1][1];
                G5 = imgAcc[i][j][1];
                G7 = imgAcc[i+1][j-1][1];
                G9 = imgAcc[i+1][j+1][1];

                alpha = std::abs(-G3 + 2*G5 - G7) + std::abs(A3 - A7);
                beta = std::abs(-G1 + 2*G5 - G9)+ std::abs(A1 - A9);
             
                if(alpha<beta)
                    C5 = (A3 + A7 -G3 +2*G5 - G7) / 2;
                else if(alpha>beta)
                    C5 = (A1 + A9 -G1 +2*G5 - G9) / 2;
                else if(alpha==beta)
                    C5 = (2*(A1 + A3 + A7 + A9) - G1 - G3 +4*G5 - G7 - G9) / 8;


                imgAcc[i][j][2] = std::clamp<LT>(C5,0,UCHAR_MAX);
                
            });
        });
        
    }
}
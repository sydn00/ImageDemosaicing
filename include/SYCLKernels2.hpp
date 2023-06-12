/*Demosaicing SYCL Algorithm 2

6x8 Image Demonstration for G R, B G Bayer CFA pattern (- = data, + = threadIdx ) 
First event = Populating Green (Each threads calculates only the green it stays on)
- - - - - - - -
- - - - - - - -
- - - + - + - -
- - - - - - - -
- - - + - + - -
- - - - - - - -

BlueHorizontalEvent = Populating Blue for horizontal neighbours (Each threads calculates only the blue it stays on)
- - - - - - - -
- + - + - + - -
- - - - - - - -
- + - + - + - -
- - - - - - - -
- + - + - + - -

BlueVerticalEvent = Populating Blue for vertical neigbours (Each threads calculates only the blue it stays on)
- - - - - - - -
- - - - - - - -
+ - + - + - + -
- - - - - - - -
+ - + - + - + -
- - - - - - - -

BlueDiagonalEvent event = Populating Blue for diagonal neigbours (Each threads calculates only the blue it stays on)
- - - - - - - -
- - - - - - - -
- + - + - + - -
- - - - - - - -
- + - + - + - -
- - - - - - - -

RedHorizontalEvent event = Populating Red for horizontal neighbours (Each threads calculates only the red it stays on)
- - + - + - + -
- - - - - - - -
- - + - + - + -
- - - - - - - -
- - + - + - + -
- - - - - - - -

RedVerticalEvent = Populating Red for vertical neighbours (Each threads calculates only the red it stays on)
- - - - - - - -
- + - + - + - +
- - - - - - - -
- + - + - + - +
- - - - - - - -
- - - - - - - -

RedDiagonalEvent event = Populating Red for diagonal neighbours (Each threads calculates only the red it stays on)
- - - - - - - -
- - + - + - + -
- - - - - - - -
- - + - + - + -
- - - - - - - -
- - - - - - - -


    Code Notes
    - queue object(Q) implicitly defined out_of_order so that each event execution expected asynchronously.
    - Since the 2..7 events depend on the first event explicit barriers are placed but still there are implicit dependences because accessors access the same buffer. So that each event will run synchronously.
    - Each population could have been done in one event to fix that issue in next algorithm. (Multiple parallel_for in one command group) but it
      is not sycl conformant so it is also avoided after try.)
    - For next kernel, single event will be used  

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
void PopulateParallel2(Image<T,idxT>& image){
    gpu_selector gpu;                
    queue Q(gpu);                    //out-of-order queue

    idxT height = image.getHeight();
    idxT width = image.getWidth();
    {
        buffer<T,3> imgBuffer {image.getPointer(),range<3>(image.getHeight(),image.getWidth(),3),{property::buffer::use_host_ptr{}}};

        /*
        1th event -> populating luminance(green) channel --- total red pixels on width and height = h/2,w/2 
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

        /*
        2nd event -> populating horizontal blue channel --- total green pixels on width and height = h/2,w/2 
        since right pixels are not used for algo h/2, w/2-1 thread domain has to be defined
        (w/2 -> total thread count in one row, w/2-1 exluding boundaries)  
        */
        event BlueHorizontalEvent = Q.submit([&](handler& h){
            h.depends_on(GreenEvent);
            accessor<T,3> imgAcc(imgBuffer,h,read_write);
           
            h.parallel_for(range<2>{height/2,width/2-1},[=](id<2> idx){
                
                idxT i = 2*idx[0]+1;
                idxT j = 2*idx[1]+1;

                //blue colors
                LT A1 = imgAcc[i][j-1][0];
                LT A3 = imgAcc[i][j+1][0];
                
                //green colors
                LT G1 = imgAcc[i][j-1][1];
                LT G2 = imgAcc[i][j][1];
                LT G3 = imgAcc[i][j+1][1];
                
                LT A2 = (A1 + A3 - G1 + 2*G2 - G3) / 2;

                imgAcc[i][j][0] = std::clamp<LT>(A2,0,UCHAR_MAX);
                
                
            });

        });
        
        /*
        3rd event -> populating vertical blue channel --- total green pixels on width and height = h/2,w/2 
        since top pixels are not used for algo h/2-1, w/2 thread domain has to be defined
        (h/2 -> total thread count in one column, h/2-1 exluding boundaries)  
        */
        event BlueVerticalEvent = Q.submit([&](handler& h){
            h.depends_on(GreenEvent);
            accessor<T,3> imgAcc(imgBuffer,h,read_write);
           
            h.parallel_for(range<2>{height/2-1,width/2},[=](id<2> idx){
                
                idxT i = 2*idx[0]+2;
                idxT j = 2*idx[1];

                //blue colors
                LT A1 = imgAcc[i-1][j][0];
                LT A7 = imgAcc[i+1][j][0];
                
                //green colors
                LT G1 = imgAcc[i-1][j][1];
                LT G4 = imgAcc[i][j][1];
                LT G7 = imgAcc[i+1][j][1];
                
                LT A4 = (A1 + A7 - G1 + 2*G4 - G7) / 2;

                imgAcc[i][j][0] = std::clamp<LT>(A4,0,UCHAR_MAX);
                
                
            });

        });

        /*
        4rd event -> populating diagonal blue channel --- total red pixels on width and height = h/2,w/2 
        since top, right pixels are not used for algo h/2-1, w/2-1 thread domain has to be defined
        (h/2 -> total thread count in one column, h/2-1 exluding boundaries are needed.)  
        */
        event BlueDiagonalEvent = Q.submit([&](handler& h){
            h.depends_on(GreenEvent);
            accessor<T,3> imgAcc(imgBuffer,h,read_write);
           
            h.parallel_for(range<2>{height/2-1,width/2-1},[=](id<2> idx){
                
                idxT i = 2*idx[0]+2;
                idxT j = 2*idx[1]+1;

                //blue colors
                LT A1 = imgAcc[i-1][j-1][0];
                LT A3 = imgAcc[i-1][j+1][0];
                LT A7 = imgAcc[i+1][j-1][0];
                LT A9 = imgAcc[i+1][j+1][0];

                
                //green colors
                LT G1 = imgAcc[i-1][j-1][1];
                LT G3 = imgAcc[i-1][j+1][1];
                LT G5 = imgAcc[i][j][1];
                LT G7 = imgAcc[i+1][j-1][1];
                LT G9 = imgAcc[i+1][j+1][1];

                LT alpha = std::abs(-G3 + 2*G5 - G7) + std::abs(A3 - A7);
                LT beta = std::abs(-G1 + 2*G5 - G9) + std::abs(A1 - A9);
                
                LT C5;
                if(alpha<beta)
                    C5 = (A3 + A7 -G3 +2*G5 - G7) / 2;
                else if(alpha>beta)
                    C5 = (A1 + A9 -G1 +2*G5 - G9) / 2;
                else if(alpha==beta)
                    C5 = (2*(A1 + A3 + A7 + A9) - G1 - G3 + 4*G5 - G7 - G9) / 8;


                imgAcc[i][j][0] = std::clamp<LT>(C5,0,UCHAR_MAX);
                
            });

        });

        /*
        5th event -> populating horizontal red channel --- total green pixels on width and height = h/2,w/2 
        since left pixels are not used for algo h/2, w/2-1 thread domain has to be defined
        (w/2 -> total thread count in one row, w/2-1 exluding boundaries)  
        */
        event RedHorizontalEvent = Q.submit([&](handler& h){
            h.depends_on(GreenEvent);
            accessor<T,3> imgAcc(imgBuffer,h,read_write);
           
            h.parallel_for(range<2>{height/2,width/2-1},[=](id<2> idx){
                
                idxT i = 2*idx[0];
                idxT j = 2*idx[1]+2;

                //red colors
                LT A1 = imgAcc[i][j-1][2];
                LT A3 = imgAcc[i][j+1][2];
                
                //green colors
                LT G1 = imgAcc[i][j-1][1];
                LT G2 = imgAcc[i][j][1];
                LT G3 = imgAcc[i][j+1][1];
                
                LT A2 = (A1 + A3 - G1 + 2*G2 - G3) / 2;

                imgAcc[i][j][2] = std::clamp<LT>(A2,0,UCHAR_MAX);
                
                
            });

        });
        
        /*
        6th event -> populating vertical red channel --- total green pixels on width and height = h/2,w/2 
        since bottom pixels are not used for algo h/2-1, w/2 thread domain has to be defined
        (h/2 -> total thread count in one column, h/2-1 exluding boundaries)  
        */
        event RedVerticalEvent = Q.submit([&](handler& h){
            h.depends_on(GreenEvent);
            accessor<T,3> imgAcc(imgBuffer,h,read_write);
           
            h.parallel_for(range<2>{height/2-1,width/2},[=](id<2> idx){
                
                idxT i = 2*idx[0]+1;
                idxT j = 2*idx[1]+1;

                //red colors
                LT A1 = imgAcc[i-1][j][2];
                LT A7 = imgAcc[i+1][j][2];
                
                //green colors
                LT G1 = imgAcc[i-1][j][1];
                LT G4 = imgAcc[i][j][1];
                LT G7 = imgAcc[i+1][j][1];
                
                LT A4 = (A1 + A7 - G1 + 2*G4 - G7) / 2;

                imgAcc[i][j][2] = std::clamp<LT>(A4,0,UCHAR_MAX);
                
                
            });

        });

        /*
        7th event -> populating diagonal red channel --- total red pixels on width and height = h/2,w/2 
        since bottom, left pixels are not used for algo h/2-1, w/2-1 thread domain has to be defined
        (h/2 -> total thread count in one column, h/2-1 exluding boundaries are needed.)  
        */
        event RedDiagonalEvent = Q.submit([&](handler& h){
            h.depends_on(GreenEvent);
            accessor<T,3> imgAcc(imgBuffer,h,read_write);
           
            h.parallel_for(range<2>{height/2-1,width/2-1},[=](id<2> idx){
                
                idxT i = 2*idx[0]+1;
                idxT j = 2*idx[1]+2;

                //red colors
                LT A1 = imgAcc[i-1][j-1][2];
                LT A3 = imgAcc[i-1][j+1][2];
                LT A7 = imgAcc[i+1][j-1][2];
                LT A9 = imgAcc[i+1][j+1][2];

                
                //green colors
                LT G1 = imgAcc[i-1][j-1][1];
                LT G3 = imgAcc[i-1][j+1][1];
                LT G5 = imgAcc[i][j][1];
                LT G7 = imgAcc[i+1][j-1][1];
                LT G9 = imgAcc[i+1][j+1][1];

                LT alpha = std::abs(-G3 + 2*G5 - G7) + std::abs(A3 - A7);
                LT beta = std::abs(-G1 + 2*G5 - G9 )+ std::abs(A1 - A9);
                
                LT C5;
                if(alpha<beta)
                    C5 = (A3 + A7 -G3 +2*G5 - G7) / 2;
                else if(alpha>beta)
                    C5 = (A1 + A9 -G1 +2*G5 - G9) / 2;
                else if(alpha==beta)//Q.wait();
                    C5 = (2*(A1 + A3 + A7 + A9) - G1 - G3 +4*G5 - G7 - G9) / 8;


                imgAcc[i][j][2] = std::clamp<LT>(C5,0,UCHAR_MAX);
                
            });

        });
    }
    
}
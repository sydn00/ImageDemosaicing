#pragma once
#include "image.hpp"
#include <CL/sycl.hpp>
#include <iostream>
//syclcc -o sycl_main SYCLmain.cpp -O3 --hipsycl-targets="cuda-nvcxx" -l opencv_imgcodecs -l opencv_core -l opencv_highgui
//Data Parallel C++ -> p.66 image buffer class  p.184->image accessor
using namespace hipsycl::sycl;
using namespace cv;

template <typename T, typename idxT=unsigned int>
void PopulateParallel(Image<T,idxT>& image){
    host_selector host;
    cpu_selector cpu;
    gpu_selector gpu;                
    queue Q(gpu);                    //out-of-order queue

    std::cout << "Selected device is: " <<
    Q.get_device().get_info<info::device::name>() << "\n";    

    idxT height = image.getHeight();
    idxT width = image.getWidth();
    {
        buffer<T,3> imgBuffer {image.getPointer(),range<3>(image.getHeight(),image.getWidth(),3),{property::buffer::use_host_ptr{}}};

        ////1th run -> populating luminance(green) channel --- total red pixels on width and height = h/2,w/2 
        //since left, right, top and bottom pixels are not used for algo h/2 - 2, w/2 - 2 threads
        //are needed.  
        Q.submit([&](handler& h){
            accessor<T,3> imgAcc(imgBuffer,h,read_write);
            local_accessor<T,2> imgLocalAcc(range(10,10),h);          

            //h.parallel_for(nd_range<2>{range<2>{height,width},range<2>{15,15}},[=](nd_item<2> item){ //num_group = global_size / local_size check constructor
            h.parallel_for(range<2>{(height-2)/2,(width-2)/2},[=](id<2> idx){
                
                //calculating red pixel's green color
                idxT i = 2*idx[0]+2;
                idxT j = 2*idx[1]+3;

                //red colors
                T A1 = imgAcc[i-2][j][2];
                T A3 = imgAcc[i][j-2][2];
                T A5 = imgAcc[i][j][2];
                T A7 = imgAcc[i][j+2][2];
                T A9 = imgAcc[i+2][j][2];
                //green colors
                T G2 = imgAcc[i-1][j][1];
                T G4 = imgAcc[i][j-1][1];
                T G6 = imgAcc[i][j+1][1];
                T G8 = imgAcc[i+1][j][1];

                T alpha = std::abs(-A3 + 2*A5 - A7) + std::abs(G4 - G6);
                T beta = std::abs(-A1 + 2*A5 - A9) + std::abs(G2 - G8);
                T G5;
                
                if(alpha<beta)
                    G5 = (G4 + G6 - A3 + 2*A5 - A7) / 2;
                if(alpha>beta)
                    G5 = (G2 + G8 - A1 + 2*A5 - A9) / 2;
                if(alpha==beta)
                    G5 = (2*(G2 + G4 + G6 + G8) - A1 - A3 + 4*A5 - A7 - A9) / 8;

                imgAcc[i][j][1] = G5;
                


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

                imgAcc[i][j][1] = G5;
            
            
                
            });

        });
        Q.wait();
        
            
        
        //2nd run-> populating blue channel
        Q.submit([&](handler& h){
            accessor<T,3> imgAcc(imgBuffer,h,read_write);

            h.parallel_for(range<2>{(height-2)/2,(width-2)/2},[=](id<2> idx){
                idxT i = 2*idx[0]+2;
                idxT j = 2*idx[1]+1;
                
                T A1 = imgAcc[i-1][j-1][0];
                T A3 = imgAcc[i-1][j+1][0];
                T A7 = imgAcc[i+1][j-1][0];
                T A9 = imgAcc[i+1][j+1][0];

                T G1 = imgAcc[i-1][j-1][1];
                T G2 = imgAcc[i-1][j][1];
                T G3 = imgAcc[i-1][j+1][1];
                T G4 = imgAcc[i][j-1][1];
                T G5 = imgAcc[i][j][1];
                T G6 = imgAcc[i][j+1][1];
                T G7 = imgAcc[i+1][j-1][1];
                T G8 = imgAcc[i+1][j][1];
                T G9 = imgAcc[i+1][j+1][1];

                //horizontal neighbours
                T A2 = (A1 + A3 - G1 + 2*G2 - G3) / 2;
                T A8 = (A7 + A9 - G7 + 2*G8 - G9) / 2;
                imgAcc[i-1][j][0] = A2;
                imgAcc[i+1][j][0] = A8;

                //vertical neighbours
                T A4 = (A1 + A7 -G1 + 2*G4 - G7) / 2;
                T A6 = (A3 + A9 - G3 + 2*G6 - G9) / 2;
                imgAcc[i][j-1][0] = A4;
                imgAcc[i][j+1][0] = A6;

                //diagonal neighbours
                T alpha = std::abs(-G3 + 2*G5 - G7) + std::abs(A3 - A7);
                T beta = std::abs(-G1 + 2*G5 - G9) + std::abs(A1 - A9);
                T C5;
                if(alpha<beta)
                    C5 = (A3 + A7 - G3 + 2*G5 - G7) / 2;

                else if(alpha>beta)
                    C5 = (A1 + A9 - G1 + 2*G5 - G9) / 2;

                else
                    C5 = (2*(A1 + A3 + A7 + A9) + (-G1 - G3 + 4*G5 - G7 - G9)) / 8;

                imgAcc[i][j][0] = C5;

            });
        });
        Q.wait();
        


        //3rd run -> populating red channel
        Q.submit([&](handler& h){
            accessor<T,3> imgAcc(imgBuffer,h,read_write);

            h.parallel_for(range<2>{(height-2)/2,(width-2)/2},[=](id<2> idx){
                
                idxT i = 2*idx[0]+1;
                idxT j = 2*idx[1]+2;
                
                T A1 = imgAcc[i-1][j-1][2];
                T A3 = imgAcc[i-1][j+1][2];
                T A7 = imgAcc[i+1][j-1][2];
                T A9 = imgAcc[i+1][j+1][2];

                T G1 = imgAcc[i-1][j-1][1];
                T G2 = imgAcc[i-1][j][1];
                T G3 = imgAcc[i-1][j+1][1];
                T G4 = imgAcc[i][j-1][1];
                T G5 = imgAcc[i][j][1];
                T G6 = imgAcc[i][j+1][1];
                T G7 = imgAcc[i+1][j-1][1];
                T G8 = imgAcc[i+1][j][1];
                T G9 = imgAcc[i+1][j+1][1];

                //horizontal neighbours
                T A2 = (A1 + A3 - G1 + 2*G2 - G3) / 2;
                T A8 = (A7 + A9 - G7 + 2*G8 - G9) / 2;
                imgAcc[i-1][j][2] = A2;
                imgAcc[i+1][j][2] = A8;

                //vertical neighbours
                T A4 = (A1 + A7 -G1 + 2*G4 - G7) / 2;
                T A6 = (A3 + A9 - G3 + 2*G6 - G9) / 2;
                imgAcc[i][j-1][2] = A4;
                imgAcc[i][j+1][2] = A6;

                //diagonal neighbours
                T alpha = std::abs(-G3 + 2*G5 - G7) + std::abs(A3 - A7);
                T beta = std::abs(-G1 + 2*G5 - G9) + std::abs(A1 - A9);
                T C5;
                if(alpha<beta)
                    C5 = (A3 + A7 - G3 + 2*G5 - G7) / 2;

                else if(alpha>beta)
                    C5 = (A1 + A9 - G1 + 2*G5 - G9) / 2;

                else
                    C5 = (2*(A1 + A3 + A7 + A9) + (-G1 - G3 + 4*G5 - G7 - G9)) / 8;

                imgAcc[i][j][2] = C5;
            });
        });
        
        Q.wait();
        
    }
       


}


int main(){
    //Demosaicing Created Shape Image
    Image<short> RAWShapeImage("../assets/Shapes/RAWshapes.ppm");
    PopulateParallel<short,uint32_t>(RAWShapeImage);

    Image<short> SerialShapeImage = {"../assets/Shapes/SerialDemosaicedShapes.ppm"};
    
    RAWShapeImage.isEqual(SerialShapeImage);
    RAWShapeImage.printPixel(50,51);
    SerialShapeImage.printPixel(50,51);

    uint32_t height=RAWShapeImage.getHeight();
    uint32_t width=RAWShapeImage.getWidth();

    Image<short,uint32_t> diff(height,width);
    RAWShapeImage.diff(SerialShapeImage,diff);

    diff.writeImage("../assets/Shapes/diff.ppm");
    RAWShapeImage.writeImage("../assets/Shapes/SYCLDemosaicedShapes.ppm");


        
    //Demosaicing Landscape of Monschau
    Image<short,uint32_t> RAWMonschauImage("../assets/Landscape/RAWMonschau.ppm");
    PopulateParallel<short,uint32_t>(RAWMonschauImage);

    Image<short,uint32_t> SerialMonschauImage = {"../assets/Landscape/SerialDemosaicedMonschau.ppm"};
    
    SerialMonschauImage.isEqual(SerialShapeImage);
    uint32_t h = SerialMonschauImage.getHeight();
    uint32_t w = SerialMonschauImage.getWidth();

    Image<short,uint32_t> diff2(h,w);
    RAWMonschauImage.diff(SerialMonschauImage,diff2);

    diff2.writeImage("../assets/Landscape/diff.ppm");
    RAWMonschauImage.writeImage("../assets/Landscape/SYCLDemosaicedMonschau.ppm");
    
    return 0;
}
#include "../include/SYCLKernels1.hpp"
#include "../include/SYCLKernels2.hpp"
#include "../include/SYCLKernels3_1.hpp"
#include "../include/SYCLKernels3_2.hpp"
#include "../include/SYCLKernels4.hpp"
#include <chrono>




/*
compilation
syclcc -o ../bin/test1_kernel4 SYCLmain.cpp -O3 --opensycl-targets="cuda-nvcxx::ccnative" -l opencv_imgcodecs -l opencv_core -l opencv_highgui
*/



int main(){
    Image<uint8_t,uint32_t> img("../assets/Car/rear/RAW_frame_0.ppm");
    auto t0 = std::chrono::high_resolution_clock::now();
    for(int i=0;i<1;++i){
        PopulateParallel4<uint8_t>(img);
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    img.writeImage("../assets/Car/rear/demosaiced_frame_0.ppm");
    auto t2 = std::chrono::high_resolution_clock::now();

    std::cout << "image demosaicing took: " << std::chrono::duration_cast<std::chrono::milliseconds>(t1-t0).count() << " miliseconds\n";
    std::cout << "image write from main memory to the harddrive took: " << std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count() << " miliseconds\n";

    return 0;
}
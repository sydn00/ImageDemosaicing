#include <iostream>
#include <image.hpp>
#include <SerialKernels.hpp>
#include <chrono>


using namespace std;
int main() {
    /*
    //Demosaicing Shape Image
    Image<short> RAWshapes("../assets/Shapes/RAWshapes.ppm");

    auto t1 = std::chrono::high_resolution_clock::now();
    populateGreen<short>(RAWshapes);
    populateBlue<short>(RAWshapes);
    populateRed<short>(RAWshapes);
    auto t2 = std::chrono::high_resolution_clock::now();

    RAWshapes.writeImage("../assets/Shapes/SerialDemosaicedShapes.ppm");
    */

    //Demosaicing Landscape Image
    Image<short> RAWMonschau("../assets/Landscape/RAWMonschau.ppm");

    auto t3 = std::chrono::high_resolution_clock::now();
    for(size_t i=0;i<30;++i){
    populateGreen<short>(RAWMonschau);
    populateBlue<short>(RAWMonschau);
    populateRed<short>(RAWMonschau);
    }
    auto t4 = std::chrono::high_resolution_clock::now();
    /*
    RAWMonschau.writeImage("../assets/Landscape/SerialDemosaicedMonschau.ppm");
    */

    //std::cout << "RAW shapes = " << std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count() << " miliseconds\n";
    std::cout << "Monschau = " << std::chrono::duration_cast<std::chrono::milliseconds>(t4-t3).count() << " miliseconds\n";
}
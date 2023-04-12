#include "../include/SYCLKernels1.hpp"
#include "../include/SYCLKernels2.hpp"
#include "../include/SYCLKernels3.hpp"
#include "../include/SYCLKernels4.hpp"
#include <chrono>




/*
compilation
syclcc -o sycl SYCLmain.cpp -O3 --opensycl-targets="cuda-nvcxx::ccnative" -l opencv_imgcodecs -l opencv_core -l opencv_highgui
*/

// Takes demosaiced images and write diff image (serial-sycl image) and output image to the specified directory.
// Postfix has to be match with the demosaiced kernel. (ie. diff1.ppm for kernels1, diff2.ppm for kernels2) 
template <typename T, typename idxT=unsigned int>
void GenerateImages(Image<T,idxT>& ShapeImg,Image<T,idxT>& MonschauImg,std::string postfix){
    //Shape Image
    
    Image<short> SerialShapeImage = {"../assets/Shapes/SerialDemosaicedShapes.ppm"};
    
    ShapeImg.isEqual(SerialShapeImage);
 
    Image<short> diff(ShapeImg.getHeight(),ShapeImg.getWidth());
    ShapeImg.diff(SerialShapeImage,diff);

    std::string diffShape = "../assets/Shapes/diff" + postfix + ".ppm";
    diff.writeImage(diffShape);

    std::string writeShape = "../assets/Shapes/SYCLDemosaicedShapes" + postfix + ".ppm";
    ShapeImg.writeImage(writeShape);
    

    //Landscape of Monschau Image
    Image<short> SerialMonschauImage = {"../assets/Landscape/SerialDemosaicedMonschau.ppm"};
    
    SerialMonschauImage.isEqual(MonschauImg);
 
    Image<short> diffM(MonschauImg.getHeight(),MonschauImg.getWidth());
    MonschauImg.diff(SerialMonschauImage,diffM);

    std::string diffMonschau = "../assets/Landscape/diff" + postfix + ".ppm";
    diffM.writeImage(diffMonschau);

    std::string writeMonschau = "../assets/Landscape/SYCLDemosaicedMonschau" + postfix + ".ppm";
    MonschauImg.writeImage(writeMonschau);
}



int main(){
    /*
    //Demosaicing Created Shape Image
    Image<short> RAWShapeImage("../assets/Shapes/RAWshapes.ppm");
    auto t1 = std::chrono::high_resolution_clock::now();
    PopulateParallel4<short>(RAWShapeImage);
    auto t2 = std::chrono::high_resolution_clock::now();
    std::cout << "RAW shapes = " << std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count() << " miliseconds\n";
    */

    //Demosaicing Landscape Monschau
    Image<short,uint32_t> RAWMonschauImage("../assets/Landscape/RAWMonschau.ppm");
    auto t3 = std::chrono::high_resolution_clock::now();
    for(size_t i=0;i<30;++i){
        PopulateParallel4<short>(RAWMonschauImage);
    }
    auto t4 = std::chrono::high_resolution_clock::now();
    std::cout << "Monschau = " << std::chrono::duration_cast<std::chrono::milliseconds>(t4-t3).count() << " miliseconds\n";

    
    //GenerateImages<short>(RAWShapeImage,RAWMonschauImage,"4");

    return 0;
}
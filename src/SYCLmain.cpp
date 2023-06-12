#include "../include/SYCLKernels1.hpp"
#include "../include/SYCLKernels2.hpp"
#include "../include/SYCLKernels3_1.hpp"
#include "../include/SYCLKernels3_2.hpp"
#include "../include/SYCLKernels4.hpp"
#include <chrono>




/*
compilation
syclcc -o sycl SYCLmain.cpp -O3 --opensycl-targets="cuda-nvcxx::ccnative" -l opencv_imgcodecs -l opencv_core -l opencv_highgui
*/

// Takes demosaiced images and write diff image (serial-sycl image) and output image to the assets directory.
// Postfix has to be match with the demosaiced kernel. (ie. diff1.ppm for kernels1, diff2.ppm for kernels2) 
template <typename T, typename idxT=unsigned int>
void GenerateImages(Image<T,idxT>& ShapeImg,Image<T,idxT>& MonschauImg,std::string postfix){
    //Shape Image
    
    Image<T> SerialShapeImage = {"../assets/Shapes/SerialDemosaicedShapes.ppm"};
    
    //ShapeImg.isEqual(SerialShapeImage);
 
    Image<T> diff(ShapeImg.getHeight(),ShapeImg.getWidth());
    ShapeImg.diff(SerialShapeImage,diff);

    std::string diffShape = "../assets/Shapes/diff" + postfix + ".ppm";
    diff.writeImage(diffShape);

    std::string writeShape = "../assets/Shapes/SYCLDemosaicedShapes" + postfix + ".ppm";
    ShapeImg.writeImage(writeShape);
    

    //Landscape of Monschau Image
    Image<T> SerialMonschauImage = {"../assets/Landscape/SerialDemosaicedMonschau.ppm"};
    
    SerialMonschauImage.isEqual(MonschauImg);
 
    Image<T> diffM(MonschauImg.getHeight(),MonschauImg.getWidth());
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
    auto t2 = std::chrono::high_resolution_clock::now();./s 
    std::cout << "RAW shapes = " << std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count() << " miliseconds\n";
    */

    //Demosaicing Landscape Monschau
    Image<uint8_t,uint32_t> RAWMonschauImage("../assets/Landscape/RAWMonschau.ppm");
    auto t3 = std::chrono::high_resolution_clock::now();
    //for(int i=0;i<30;++i){
        PopulateParallel2<uint8_t>(RAWMonschauImage);
    //}
    auto t4 = std::chrono::high_resolution_clock::now();
    std::cout << "Monschau = " << std::chrono::duration_cast<std::chrono::milliseconds>(t4-t3).count() << " miliseconds\n";
    
    Image<uint8_t> RAWShapeImage("../assets/Shapes/RAWshapes.ppm");
    GenerateImages<uint8_t>(RAWShapeImage,RAWMonschauImage,"2");

    Image<uint8_t,int> RAW("../assets/Full_Intensity_RAW/RAW.ppm");
    PopulateParallel2<uint8_t>(RAW);
    RAW.writeImage("../assets/Full_Intensity_RAW/SYCLDemosaicedRAW.ppm");

    return 0;
}
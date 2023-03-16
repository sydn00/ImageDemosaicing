#include "../include/SYCLKernels1.hpp"
#include "../include/SYCLKernels2.hpp"


/*
compilation
syclcc -o sycl SYCLmain.cpp -O3 --opensycl-targets="cuda-nvcxx::ccnative" -l opencv_imgcodecs -l opencv_core -l opencv_highgui
*/

int main(){

    //-----------------SYCLKernels1---------------------
    
    /*
    //Demosaicing Created Shape Image
    Image<short> RAWShapeImage("../assets/Shapes/RAWshapes.ppm");
    PopulateParallel1<short,uint32_t>(RAWShapeImage);
    

    Image<short> SerialShapeImage = {"../assets/Shapes/SerialDemosaicedShapes.ppm"};
    
    RAWShapeImage.isEqual(SerialShapeImage);
 
    uint32_t height=RAWShapeImage.getHeight();
    uint32_t width=RAWShapeImage.getWidth();

    Image<short,uint32_t> diff(height,width);
    RAWShapeImage.diff(SerialShapeImage,diff);

    diff.writeImage("../assets/Shapes/diff1.ppm");
    RAWShapeImage.writeImage("../assets/Shapes/SYCLDemosaicedShapes1.ppm");


    //Demosaicing Landscape of Monschau
    Image<short,uint32_t> RAWMonschauImage("../assets/Landscape/RAWMonschau.ppm");
    PopulateParallel1<short,uint32_t>(RAWMonschauImage);

    Image<short,uint32_t> SerialMonschauImage = {"../assets/Landscape/SerialDemosaicedMonschau.ppm"};
    
    SerialMonschauImage.isEqual(SerialShapeImage);
    uint32_t h = SerialMonschauImage.getHeight();
    uint32_t w = SerialMonschauImage.getWidth();

    Image<short,uint32_t> diffM(h,w);
    RAWMonschauImage.diff(SerialMonschauImage,diffM);

    diffM.writeImage("../assets/Landscape/diff1.ppm");
    RAWMonschauImage.writeImage("../assets/Landscape/SYCLDemosaicedMonschau1.ppm");
    
    */
    


    //-----------------SYCLKernels2---------------------
    //Demosaicing Created Shape Image
    Image<short> RAWShapeImage("../assets/Shapes/RAWshapes.ppm");
    PopulateParallel2<short,uint32_t>(RAWShapeImage);
    

    Image<short> SerialShapeImage = {"../assets/Shapes/SerialDemosaicedShapes.ppm"};
    
    RAWShapeImage.isEqual(SerialShapeImage);
 
    uint32_t height=RAWShapeImage.getHeight();
    uint32_t width=RAWShapeImage.getWidth();

    Image<short,uint32_t> diff(height,width);
    RAWShapeImage.diff(SerialShapeImage,diff);

    diff.writeImage("../assets/Shapes/diff2.ppm");
    RAWShapeImage.writeImage("../assets/Shapes/SYCLDemosaicedShapes2.ppm");


    //Demosaicing Landscape of Monschau
    Image<short,uint32_t> RAWMonschauImage("../assets/Landscape/RAWMonschau.ppm");
    PopulateParallel2<short,uint32_t>(RAWMonschauImage);

    Image<short,uint32_t> SerialMonschauImage = {"../assets/Landscape/SerialDemosaicedMonschau.ppm"};
    
    SerialMonschauImage.isEqual(SerialShapeImage);
    uint32_t h = SerialMonschauImage.getHeight();
    uint32_t w = SerialMonschauImage.getWidth();

    Image<short,uint32_t> diffM(h,w);
    RAWMonschauImage.diff(SerialMonschauImage,diffM);

    diffM.writeImage("../assets/Landscape/diff2.ppm");
    RAWMonschauImage.writeImage("../assets/Landscape/SYCLDemosaicedMonschau2.ppm");
    return 0;
}
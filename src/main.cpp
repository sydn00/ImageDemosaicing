#include <iostream>
#include "image.hpp"
#include "SerialKernels.hpp"


using namespace std;
int main() {

    //Demosaicing Shape Image
    Image<short> RAWshapes("../assets/Shapes/RAWshapes.ppm");

    populateGreen<short>(RAWshapes);
    populateBlue<short>(RAWshapes);
    populateRed<short>(RAWshapes);

    RAWshapes.writeImage("../assets/Shapes/SerialDemosaicedShapes.ppm");


    //Demosaicing Landscape Image
    Image<short> RAWMonschau("../assets/Landscape/RAWMonschau.ppm");

    populateGreen<short>(RAWMonschau);
    populateBlue<short>(RAWMonschau);
    populateRed<short>(RAWMonschau);

    RAWMonschau.writeImage("../assets/Landscape/SerialDemosaicedMonschau.ppm");


}
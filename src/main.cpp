#include <iostream>
#include "image.hpp"
#include "SerialKernels.hpp"

using namespace std;
int main() {
    cout << "hello\n";
    Image<short> image(6, 6);
    cout << image.getRed(0, 0) << image.getGreen(0, 0) << image.getBlue(0, 0) << endl;
    cout << image(0,0,0) << image(1,0,0) << image(2,0,0) <<endl;

    Image<short> RAWshapes("../assets/RAWshapes.ppm");



    populateGreen<short>(RAWshapes);
    populateBlue<short>(RAWshapes);
    //populateRed<short>(RAWshapes);

    RAWshapes.showImage();
    RAWshapes.writeImage("../assets/DemosaicedShapes.ppm");
}
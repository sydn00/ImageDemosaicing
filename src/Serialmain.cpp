#include <iostream>
#include <image.hpp>
#include <SerialKernels.hpp>
#include <chrono>

#define N 3
using namespace std;
int main() {
    Image<uchar> img {"../assets/Landscape/DemosaicedRAWMonschau.ppm"};

    populateGreen<uchar,short>(img);
    populateBlue<uchar,short>(img);
    populateRed<uchar,short>(img);

    img.showImage();

    return 0;
}
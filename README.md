# ImageDemosaicing

ImageDemosaicing repository is written in c++ to implement and optimize image demosaicing, targeting heterogeneous systems with SYCL. 


## Method and Implementation
Adams-Hamilton Method is used as an algorithm and compilation is done by hipSYCL(OpenSYCL) implementation. Details can be found on;
- https://github.com/OpenSYCL/OpenSYCL
- https://patents.google.com/patent/US5652621A/en

## Algorithm
Three step process:
 1. populate green channel (luminance)
 2. populate red or blue (chrominance)
 3. populate remaining chrominance channel

### First pass: calculate green luminance data for red and blue positions

We calculate the green value for the image position A5, where Ax is either red for all positions or blue.

```
      A1
      G2
A3 G4 A5 G6 A7
      G8
      A9
```

First: calculate the classifiers `alpha` and `beta`

```
alpha = abs(-A3 + 2*A5 - A7) + abs(G4 - G6)
beta  = abs(-A1 + 2*A5 - A9) + abs(G2 - G8)
```

Then, if alpha < beta:

```
G5 = (G4 + G6) / 2 + (-A3 + 2*A5 - A7) / 2
```

else if alpha > beta:

```
G5 = (G2 + G8) / 2 + (-A1 + 2*A5 - A9) / 2
```

else // alpha == beta

```
G5 = (G2 + G4 + G6 + G8) / 4 + (-A1 - A3 + 4*A5 - A7 - A9) / 8
```

Note that the signed versions of alpha and beta are reused in each of those calculations. The alpha == beta case is almost the average of the other cases.

### Second pass: calculate blue chromaticity channel

#### Overview and cases

First we calculate the missing blue values. Here we use the following pattern, where C5 is red, Gx is green and Ax is blue.

```
A1 G2 A3
G4 C5 G6
A7 G8 A9
```

We have blue values for Ax and need to calculate it for the remaining 5 positions. There are three cases:
 * G2 and G8, where we have blue values to the right and left (A1, A3 and A7, A9)
 * G4 and G6, where we have blue values to the top and bottom (A1, A7 and A3, A9)
 * C5, where we have blue values on the diagonal (A1, A3, A7, A9)

#### Case 1: Horizontal neighbours

We want to calculate the blue values for the positions labeled G2 and G8 in our pattern, which would be apropriately named A2 and A8. Here we have horizontal neighbours A1, A3 and A7, A9 of the same color. These are calculated as follows:

```
A2 = (A1 + A3) / 2 + (-G1 + 2*G2 - G3) / 2
A8 = (A7 + A9) / 2 + (-G7 + 2*G8 - G9) / 2
```

#### Case 2: Vertical neighbours

We want to calculate the blue values for the positions labeled G4 and G6 in our pattern, which would be apropriately named A4 and A6. Here we have vertical neighbours A1, A7 and A3, A9 of the same color. These are calculated as follows:

```
A4 = (A1 + A7) / 2 + (-G1 + 2*G4 - G7) / 2
A6 = (A3 + A9) / 2 + (-G3 + 2*G6 - G9) / 2
```

#### Case 3: Diagonals

To calculate the blue value for C5, we first calculate the classifiers `alpha` and `beta`:

```
alpha = abs(-G3 + 2*G5 - G7) + abs(A3 - A7)
beta  = abs(-G1 + 2*G5 - G9) + abs(A1 - A9) 
```

Then, if alpha < beta:

```
C5 = (A3 + A7) / 2 + (-G3 + 2*G5 - G7) / 2
```

else if alpha > beta:

```
C5 = (A1 + A9) / 2 + (-G1 + 2*G5 - G9) / 2
```

else // alpha == beta

```
C5 = (A1 + A3 + A7 + A9) / 4 + (-G1 - G3 + 4*G5 - G7 - G9) / 8
```

Again, note the similarities arround alpha, beta and the alpha == beta case.

### Third pass: repeat the second pass for the remaining color

## Folder Structure
All the c++ headers and source files are located under **src** folder. Image class is created in image.hpp header file which uses OpenCV for image read-write operations. Serial codes can be found in main.cpp and SerialKernels.cpp compilation units. SYCL kernels are written in SYCLmain.cpp file. 

Folder **python** is used to store testimages.ipynb file to create image and converting it to bayer filter.

All image outputs are stored in **assets** folder.

## Compilation models and Requirements
Proper SYCL compilers and OpenCV libraries has to be installed before build the script. Since cmake dependencies hasn't been prepared yet serial and SYCL code has to be compiled separately. Compilation of SYCL with hipSYCL(OpenSYCL) implementation can be done with ***nvc++, openmp, clang*** backends;

***
- syclcc -o sycl_main SYCLmain.cpp -O3 --hipsycl-targets="cuda-nvcxx" -l opencv_imgcodecs -l opencv_core -l opencv_highgui    

- syclcc -o sycl_main SYCLmain.cpp -O3 --hipsycl-targets="omp" -l opencv_imgcodecs -l opencv_core -l opencv_highgui

- syclcc -o sycl_main  SYCLmain.cpp -O3 --hipsycl-targets="omp;cuda:sm_75" -l opencv_imgcodecs -l opencv_core -l opencv_highgui

***


Serial part can be compiled with cmake;

***
cmake . -B build --fresh --warn-uninitialized -DCMAKE_BUILD_TYPE=Debug
cmake --build build
./build/main

cmake . -B build --fresh --warn-uninitialized -DCMAKE_BUILD_TYPE=Debug && cmake --build build && ./build/main
***


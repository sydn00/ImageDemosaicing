#include "image.hpp"
#include <algorithm>
#include <limits>
#include <limits.h>


//1th run -> populating luminance(green) channel
//T -> image type, LT -> local type
template <typename T, typename LT=int16_t, typename idxT=unsigned int>
void populateGreen(Image<T,idxT>& image){
    idxT height = image.getHeight();
    idxT width = image.getWidth();

    LT alpha = 0, beta = 0;
    LT A1, A3, A5, A7, A9;
    LT G2, G4, G6, G8;
    LT G5;


    //calculating red pixel's green color
    for(idxT i=2;i<height-2;i+=2){
        for(idxT j=3;j<width-1;j+=2){

            //red colors
            A1 = image(2,i-2,j);
            A3 = image(2,i,j-2);
            A5 = image(2,i,j);
            A7 = image(2,i,j+2);
            A9 = image(2,i+2,j);

            //neigbor green colors
            G2 = image(1,i-1,j);
            G4 = image(1,i,j-1);
            G6 = image(1,i,j+1);
            G8 = image(1,i+1,j); 
            alpha = std::abs(-A3 + 2 * A5 - A7) + std::abs(G4 - G6);
            beta = std::abs(-A1 + 2 * A5 - A9) + std::abs(G2 - G8);

            //calculating G5
            if (alpha < beta)
                G5 = (G4 + G6 - A3 + 2*A5 - A7) / 2;
            else if(alpha > beta)
                G5 = (G2 + G8 - A1 + 2*A5 - A9) / 2;
            
            else if(alpha == beta)
                G5 = (2*(G2 + G4 + G6 + G8) - A1 - A3 + 4*A5 - A7 - A9)/8;
            
            image(1,i,j) = std::clamp<LT>(G5,0,UCHAR_MAX);
            
        }
    }


    //calculating blue pixel's green color
    for(idxT i=3;i<height-1;i+=2){
        for(idxT j=2;j<width-2;j+=2){

            //blue colors
            A1 = image(0,i-2,j);
            A3 = image(0,i,j-2);
            A5 = image(0,i,j);
            A7 = image(0,i,j+2);
            A9 = image(0,i+2,j);
            //neigbor green colors
            G2 = image(1,i-1,j);
            G4 = image(1,i,j-1);
            G6 = image(1,i,j+1);
            G8 = image(1,i+1,j); 
            alpha = std::abs(-A3 + 2 * A5 - A7) + std::abs(G4 - G6);
            beta = std::abs(-A1 + 2 * A5 - A9) + std::abs(G2 - G8);

            //finding G5
            if (alpha < beta)
                G5 = (G4 + G6 - A3 + 2*A5 - A7) / 2;
            else if(alpha > beta)
                G5 = (G2 + G8 - A1 + 2*A5 - A9) / 2;
            
            else if(alpha == beta)
                G5 = (2*(G2 + G4 + G6 + G8) - A1 - A3 + 4*A5 - A7 - A9)/8;
            
            image(1,i,j) = std::clamp<LT>(G5,0,UCHAR_MAX);
            
        }
    }


}


//2nd run -> populating chrominance(blue) channel
//T -> image type, LT -> local type
template <typename T, typename LT=int16_t, typename idxT=unsigned int>
void populateBlue(Image<T,idxT>& image){
    idxT height = image.getHeight();
    idxT width = image.getWidth();

    //horizontal neighbours
    for(idxT i=1;i<height;i+=2){
        for(idxT j=1;j<width-1;j+=2){
            //blue values
            LT A1 = image(0,i,j-1);
            LT A3 = image(0,i,j+1);
            //green values
            LT G1 = image(1,i,j-1);
            LT G2 = image(1,i,j);
            LT G3 = image(1,i,j+1);
            
            LT A2 = (A1 + A3 - G1 + 2*G2 - G3) / 2;

            image(0,i,j) = std::clamp<LT>(A2,0,UCHAR_MAX);
        }
    }

    //vertical neighbours  
    for(idxT i=2;i<height-1;i+=2){
        for(idxT j=0;j<width-1;j+=2){
            //blue values
            LT A1 = image(0,i-1,j);
            LT A7 = image(0,i+1,j);
            //green values
            LT G1 = image(1,i-1,j);
            LT G4 = image(1,i,j);
            LT G7 = image(1,i+1,j);

            LT A4 = (A1 + A7 - G1 + 2*G4 -G7) / 2;

            image(0,i,j) = std::clamp<LT>(A4,0,UCHAR_MAX);
        }
    }

    //diagonal neighbours
    for(idxT i=2;i<height;i+=2){
        for(idxT j=1;j<width-1;j+=2){
            //blue colors
            LT A1 = image(0,i-1,j-1);
            LT A3 = image(0,i-1,j+1);
            LT A7 = image(0,i+1,j-1);
            LT A9 = image(0,i+1,j+1);

            //green colors
            LT G1 = image(1,i-1,j-1);
            LT G3 = image(1,i-1,j+1);
            LT G5 = image(1,i,j);
            LT G7 = image(1,i+1,j-1);
            LT G9 = image(1,i+1,j+1);

            LT alpha = std::abs(-G3 + 2*G5 - G7) + std::abs(A3 - A7);
            LT beta = std::abs(-G1 + 2*G5 - G9) + std::abs(A1 - A9);

            LT C5;
            if(alpha<beta)
                C5 = (A3 + A7 - G3 + 2*G5 - G7) / 2;
            if(alpha>beta)
                C5 = (A1 + A9 - G1 + 2*G5 - G9) / 2;
            if(alpha==beta)
                C5 = (2*(A1 + A3 + A7 + A9) - G1 - G3 + 4*G5 - G7 - G9) / 8;
            
            image(0,i,j) = std::clamp<LT>(C5,0,UCHAR_MAX);

            
        }
    }
    
}


//3rd run -> populating chrominance(red) channel
//T -> image type, LT -> local type
template <typename T, typename LT=int16_t, typename idxT=unsigned int>
void populateRed(Image<T,idxT>& image){
    idxT height = image.getHeight();
    idxT width = image.getWidth();

    //horizontal neighbours
    for(idxT i=0;i<height;i+=2){
        for(idxT j=2;j<width;j+=2){
            //red values
            LT A1 = image(2,i,j-1);
            LT A3 = image(2,i,j+1);
            //green values
            LT G1 = image(1,i,j-1);
            LT G2 = image(1,i,j);
            LT G3 = image(1,i,j+1);
            
            LT A2 = (A1 + A3 - G1 + 2*G2 - G3) / 2;

            image(2,i,j) = std::clamp<LT>(A2,0,UCHAR_MAX);
        }
    }

    //vertical neighbours  
    for(idxT i=1;i<height-1;i+=2){
        for(idxT j=1;j<width;j+=2){
            //red values
            LT A1 = image(2,i-1,j);
            LT A7 = image(2,i+1,j);
            //green values
            LT G1 = image(1,i-1,j);
            LT G4 = image(1,i,j);
            LT G7 = image(1,i+1,j);

            LT A4 = (A1 + A7 - G1 + 2*G4 -G7) / 2;

            image(2,i,j) = std::clamp<LT>(A4,0,UCHAR_MAX);
        }
    }

    //diagonal neighbours
    for(idxT i=1;i<height-1;i+=2){
        for(idxT j=2;j<width;j+=2){
            //green colors
            LT G1 = image(1,i-1,j-1);
            LT G3 = image(1,i-1,j+1);
            LT G5 = image(1,i,j);
            LT G7 = image(1,i+1,j-1);
            LT G9 = image(1,i+1,j+1);
            //red colors
            LT A1 = image(2,i-1,j-1);
            LT A3 = image(2,i-1,j+1);
            LT A7 = image(2,i+1,j-1);
            LT A9 = image(2,i+1,j+1);

            LT alpha = std::abs(-G3 + 2*G5 - G7) + std::abs(A3 - A7);
            LT beta = std::abs(-G1 + 2*G5 - G9) + std::abs(A1 - A9);

            LT C5;
            if(alpha<beta)
                C5 = (A3 + A7 - G3 + 2*G5 - G7) / 2;
            if(alpha>beta)
                C5 = (A1 + A9 - G1 + 2*G5 - G9) / 2;
            if(alpha==beta)
                C5 = (2*(A1 + A3 + A7 + A9) - G1 - G3 + 4*G5 - G7 - G9) / 8;
            
            image(2,i,j) = std::clamp<LT>(C5,0,UCHAR_MAX);
        }
    }
    
}

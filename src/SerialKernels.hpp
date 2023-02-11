#include "image.hpp"

//Algorithm approach 1 -> Each block of 9 pixel(3x3) can be iterated but
//                        right edge of the one block will be left of the next one(same for 
//                        top and bottom). That will make redundant operation for same pixel 
//                        thus it is avoided.

//Algorithm approach 2 -> Implemented algorithm. Green, blue, red values were populated respectively. For chrominance channels(red, blue)
// first horizontal then vertical and diagonal neighbor cases were calculated. This header consists of 3 kernels 1-PopulateGreen, 2-PopulateBlue 3-PopulateRed


//1th run -> populating luminance(green) channel
template <typename T, typename idxT=unsigned int>
void populateGreen(Image<T,idxT>& image){
    idxT height = image.getHeight();
    idxT width = image.getWidth();

    T alpha = 0, beta = 0;
    T A1, A3, A5, A7, A9;
    T G2, G4, G6, G8;
    T G5;


    //calculating red pixel's green color
    for(idxT i=2;i<height-2;i+=2){
        for(idxT j=3;j<width-2;j+=2){

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
            
            image(1,i,j) = G5;
            
        }
    }


    //calculating blue pixel's green color
    for(idxT i=3;i<height-2;i+=2){
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
            
            image(1,i,j) = G5;
            
        }
    }


}


//2nd run -> populating chrominance(blue) channel
template <typename T, typename idxT=unsigned int>
void populateBlue(Image<T,idxT>& image){
    idxT height = image.getHeight();
    idxT width = image.getWidth();

    //horizontal neighbours
    for(idxT i=1;i<height;i+=2){
        for(idxT j=1;j<width-1;j+=2){
            //blue values
            T A1 = image(0,i,j-1);
            T A3 = image(0,i,j+1);
            //green values
            T G1 = image(1,i,j-1);
            T G2 = image(1,i,j);
            T G3 = image(1,i,j+1);
            
            T A2 = (A1 + A3 - G1 + 2*G2 - G3) / 2;
            image(0,i,j) = A2;
        }
    }

    //vertical neighbours  
    for(idxT i=2;i<height-1;i+=2){
        for(idxT j=0;j<width-1;j+=2){
            //blue values
            T A1 = image(0,i-1,j);
            T A7 = image(0,i+1,j);
            //green values
            T G1 = image(1,i-1,j);
            T G4 = image(1,i,j);
            T G7 = image(1,i+1,j);

            T A4 = (A1 + A7 - G1 + 2*G4 -G7) / 2;
            image(0,i,j) = A4;
        }
    }

    //diagonal neighbours
    for(idxT i=2;i<height-1;i+=2){
        for(idxT j=1;j<width-2;j+=2){
            //blue colors
            T A1 = image(0,i-1,j-1);
            T A3 = image(0,i-1,j+1);
            T A7 = image(0,i+1,j-1);
            T A9 = image(0,i+1,j+1);

            //green colors
            T G1 = image(1,i-1,j-1);
            T G3 = image(1,i-1,j+1);
            T G5 = image(1,i,j);
            T G7 = image(1,i+1,j-1);
            T G9 = image(1,i+1,j+1);

            T alpha = std::abs(-G3 + 2*G5 - G7) + std::abs(A3 - A7);
            T beta = std::abs(-G1 + 2*G5 - G9) + std::abs(A1 - A9);

            T C5;
            if(alpha<beta)
                C5 = (A3 + A7 - G3 + 2*G5 - G7) / 2;
            if(alpha>beta)
                C5 = (A1 + A9 - G1 + 2*G5 - G9) / 2;
            if(alpha==beta)
                C5 = (2*(A1 + A3 + A7 + A9) - G1 - G3 + 4*G5 - G7 - G9) / 8;
            
            image(0,i,j) = C5;

            
        }
    }
    
}


//3rd run -> populating chrominance(red) channel
template <typename T, typename idxT=unsigned int>
void populateRed(Image<T,idxT>& image){
    idxT height = image.getHeight();
    idxT width = image.getWidth();

    //horizontal neighbours
    for(idxT i=0;i<height;i+=2){
        for(idxT j=2;j<width-1;j+=2){
            //red values
            T A1 = image(2,i,j-1);
            T A3 = image(2,i,j+1);
            //green values
            T G1 = image(1,i,j-1);
            T G2 = image(1,i,j);
            T G3 = image(1,i,j+1);
            
            T A2 = (A1 + A3 - G1 + 2*G2 - G3) / 2;
            image(2,i,j) = A2;
        }
    }

    //vertical neighbours  
    for(idxT i=1;i<height-2;i+=2){
        for(idxT j=1;j<width;j+=2){
            //red values
            T A1 = image(2,i-1,j);
            T A7 = image(2,i+1,j);
            //green values
            T G1 = image(1,i-1,j);
            T G4 = image(1,i,j);
            T G7 = image(1,i+1,j);

            T A4 = (A1 + A7 - G1 + 2*G4 -G7) / 2;
            image(2,i,j) = A4;
        }
    }

    //diagonal neighbours
    for(idxT i=1;i<height-2;i+=2){
        for(idxT j=2;j<width-1;j+=2){
            //green colors
            T G1 = image(1,i-1,j-1);
            T G3 = image(1,i-1,j+1);
            T G5 = image(1,i,j);
            T G7 = image(1,i+1,j-1);
            T G9 = image(1,i+1,j+1);
            //red colors
            T A1 = image(2,i-1,j-1);
            T A3 = image(2,i-1,j+1);
            T A7 = image(2,i+1,j-1);
            T A9 = image(2,i+1,j+1);

            T alpha = std::abs(-G3 + 2*G5 - G7);
            T beta = std::abs(-G1 + 2*G5 - G9);

            T C5;
            if(alpha<beta)
                C5 = (A3 + A7 - G3 + 2*G5 - G7) / 2;
            if(alpha>beta)
                C5 = (A1 + A9 - G1 + 2*G5 - G9) / 2;
            if(alpha==beta)
                C5 = (2*(A1 + A3 + A7 + A9) - G1 - G3 + 4*G5 - G7 - G9) / 8;
            
            image(2,i,j) = C5;
        }
    }
    
}

#pragma once
#include <iostream>
#include <cassert>
#include <string>
#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
template <typename T, typename idxT=unsigned int>
class Image {
    public:
        Image(): height(0), width(0), data(nullptr) {}

        Image(idxT height, idxT width):
            height(height), width(width),
            data(new T[height * width * 3]) {}

        Image(std::string imagePath){
            cv::Mat Image = cv::imread(imagePath);
            height = (idxT)Image.rows;
            width = (idxT)Image.cols;
            data = new T[height*width*3];

            uchar* ptr = Image.data;
            for(idxT i=0;i<height*width*3;++i){
                data[i] = ptr[i];
            }
        }

        Image(Image & other): height(other.height), width(other.width),
            data((height>0 && width>0)?
                new T[height * width * 3]:nullptr) {
            for (idxT i=0; i<height*width*3; ++i) {
                data[i] = other.data[i];
            }
        }

        friend void swap(Image& i1, Image& i2) {
            using std::swap;
            swap(i1.height, i2.height);
            swap(i1.width, i2.width);
            swap(i1.data, i2.data);
        }

        Image(Image && other) noexcept: Image() {
            swap(*this, other);
        }

        ~Image() {delete[] data;}

        T & getRed(idxT i, idxT j) {
            assert(i<height && j<width);
            return data[(i*width + j)*3 + 2];
        }
        T & getGreen(idxT i, idxT j) {
            assert(i<height && j<width);
            return data[(i*width + j)*3 + 1];
        }
        T & getBlue(idxT i, idxT j) {
            assert(i<height && j<width);
            return data[(i*width + j)*3 + 0];
        }
        const T & getRed(idxT i, idxT j) const {
            assert(i<height && j<width);
            return data[(i*width + j)*3 + 2];
        }
        const T & getGreen(idxT i, idxT j) const {
            assert(i<height && j<width);
            return data[(i*width + j)*3 + 1];
        }
        const T & getBlue(idxT i, idxT j) const {
            assert(i<height && j<width);
            return data[(i*width + j)*3 + 0];
        }

        
        //Color ====>   0->Blue 1->Green 2->Red (can be enum)
        T& operator()(uchar Color, idxT i, idxT j){
            assert(Color<3 && i<height && j<width );
            if(Color==0)
                return data[(i*width + j)*3 + 0];
            if(Color==1)
                return data[(i*width + j)*3 + 1];
            
            return data[(i*width + j)*3 + 2]; 
        }

        T* getPointer(){
            return data;
        }

        idxT total(){
            return height*width;
        }

        idxT getHeight() const{
            return height;
        }

        idxT getWidth() const{
            return width;
        }

        void printPixel(idxT i, idxT j){
            for(int k=0;k<3;++k){
                std::cout << (long)operator()(k,i,j) << " ";
            }
            std::cout << std::endl;
        }
        bool isEqual(Image& other){
            long* diff = new long[3];
            for(idxT i=0;i<3;++i) diff[i] = 0;
            
            for(idxT i=0;i<height;++i){
                for(idxT j=0;j<width;++j){
                    if(std::abs(other(0,i,j) == operator()(0,i,j)) &&
                       std::abs(other(1,i,j) == operator()(1,i,j)) &&
                       std::abs(other(2,i,j) == operator()(2,i,j))) continue;
                    else { 
                        std::cout << "pixel colors doesn't match in location " 
                        << i << " " << j << std::endl;
                        return false;
                    }
                }
            }
            return true;
        }
        //abs(src - other) = dst
        void diff(Image& other, Image& dst){
            assert(height==other.height && width == other.width &&
                   height==dst.height && width == dst.width);
                   
            for(idxT i=0;i<height;++i){
                for(idxT j=0;j<width;++j){
                    dst(0,i,j) = std::abs(operator()(0,i,j) - other(0,i,j));
                    dst(1,i,j) = std::abs(operator()(1,i,j) - other(1,i,j));
                    dst(2,i,j) = std::abs(operator()(2,i,j) - other(2,i,j));
                }
            }
        }
        
        void showImage(std::string WindowName="Image"){
            cv::Mat Image(height,width,CV_8UC3,cv::Scalar(0,0,0));
            assert(Image.isContinuous());
            uchar* ptr = Image.data;
            for(idxT i=0;i<height*width*3;++i){
                ptr[i] = data[i];
            }
            cv::imshow(WindowName,Image);
            cv::waitKey(0);
            cv::destroyWindow(WindowName);
        }

        void writeImage(std::string fileName){
            cv::Mat Image(height,width,CV_8UC3,cv::Scalar(0,0,0));
            assert(Image.isContinuous());
            uchar* ptr = Image.data;
            for(idxT i=0;i<height*width*3;++i){
                ptr[i] = data[i];
            }
            cv::imwrite(fileName, Image);
        }

    private:
        idxT height, width;
        T* data;
};

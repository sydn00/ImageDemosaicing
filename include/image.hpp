#pragma once
#include <iostream>
#include <vector>
#include <cassert>
#include <string>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>

template <typename T, typename idxT=unsigned int>
class Image {
    public:
        Image(): height(0), width(0), data(nullptr) {}

        Image(idxT height, idxT width):
            height(height), width(width),
            data(new T[height * width * 3]) {}


        Image(cv::Mat matrix){
            height = matrix.rows;
            width = matrix.cols;

            data = new T[height * width * 3];
            uchar* ptr = matrix.data;
            for(idxT i=0;i<height*width*3;++i){
                data[i] = ptr[i];
            }
        }

        Image(std::string imagePath){
            auto t0 = std::chrono::high_resolution_clock::now();
            cv::Mat src = cv::imread(imagePath);
            cv::Mat dst;
            auto t1 = std::chrono::high_resolution_clock::now();
            copyMakeBorder(src, dst, 2, 2, 2, 2, cv::BORDER_REFLECT101, 0);
            auto t2 = std::chrono::high_resolution_clock::now();

            std::cout << "Image tranfer from harddrive to main memory took: = " << std::chrono::duration_cast<std::chrono::milliseconds>(t1-t0).count() << " miliseconds\n";
            std::cout << "Border creation took: = " << std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count() << " miliseconds\n";

            height = (idxT)dst.rows;
            width = (idxT)dst.cols;

            data = new T[height*width*3];

            uchar* ptr = dst.data;
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
        //gray-scale read (1byte per pixel)
        Image(std::string imagePath, bool is_grayscale){
            cv::Mat gray = cv::imread(imagePath);

            idxT h = gray.rows;
            idxT w = gray.cols;
            cv::Mat colored(h, w, CV_8UC3, cv::Scalar(0,0,0));

            //RG to GR
            for(idxT i=0; i<h; i+=2){
                for(idxT j=0; j<w; j+=2){
                        colored.at<cv::Vec3b>(i,j)[2] = gray.at<cv::Vec3b>(i,j)[0];
                        T red = colored.at<cv::Vec3b>(i,j)[2];   colored.at<cv::Vec3b>(i,j)[2]=0;
                        colored.at<cv::Vec3b>(i,j+1)[1] = gray.at<cv::Vec3b>(i,j+1)[0];
                        T g1 = colored.at<cv::Vec3b>(i,j+1)[1];  colored.at<cv::Vec3b>(i,j+1)[1]=0;
                        colored.at<cv::Vec3b>(i+1,j)[1] = gray.at<cv::Vec3b>(i+1,j)[0];
                        T g2 = colored.at<cv::Vec3b>(i+1,j)[1];  colored.at<cv::Vec3b>(i+1,j)[1]=0;
                        colored.at<cv::Vec3b>(i+1,j+1)[0] = gray.at<cv::Vec3b>(i,j)[0];
                        T blue = colored.at<cv::Vec3b>(i+1,j+1)[0];  colored.at<cv::Vec3b>(i+1,j+1)[0]=0;

                        colored.at<cv::Vec3b>(i,j)[1] = g1;
                        colored.at<cv::Vec3b>(i,j+1)[2] = red;
                        colored.at<cv::Vec3b>(i+1,j)[0] = blue;
                        colored.at<cv::Vec3b>(i+1,j+1)[1] = g2;


                }
            }
            
            cv::Mat dst;
            copyMakeBorder(colored, dst, 2, 2, 2, 2, cv::BORDER_REFLECT101, 0);
            height = (idxT)dst.rows;
            width = (idxT)dst.cols;
            data = new T[height*width*3];

            uchar* ptr = dst.data;
            for(idxT i=0;i<height*width*3;++i){
                data[i] = ptr[i];
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

        
        //Color ====>   0->Blue 1->Green 2->Red 
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
            
            for(idxT i=2;i<height-4;++i){
                for(idxT j=2;j<width-4;++j){
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
            std::cout << "Images are same" << std::endl;
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
            cv::Mat img(height,width,CV_8UC3,cv::Scalar(0,0,0));
            assert(img.isContinuous());
            uchar* ptr = img.data;
            for(idxT i=0;i<height*width*3;++i){
                ptr[i] = data[i];
            }

            //crop image a,b,w,h (a,b top left coordinates)
            cv::Rect crop_region(2, 2, width-4, height-4 );
            cv::Mat croppedImg = img(crop_region);
            cv::imshow(WindowName,croppedImg);
            cv::waitKey(0);
            cv::destroyWindow(WindowName);
        }

        void writeImage(std::string fileName){
            cv::Mat img(height,width,CV_8UC3,cv::Scalar(0,0,0));
            assert(img.isContinuous());
            uchar* ptr = img.data;
            for(idxT i=0;i<height*width*3;++i){
                ptr[i] = data[i];
            }

             //crop image a,b,w,h (a,b top left coordinates)
            cv::Rect crop_region(2, 2, width-4, height-4);
            cv::Mat croppedImg = img(crop_region);

            cv::imwrite(fileName, croppedImg);
        }

    private:
        idxT height, width;
        T* data;
};

#include<cassert>
#include<string>

template <typename T, typename idxT=unsigned int>
class Image {
    public:
        Image(): height(0), width(0), data(nullptr) {}

        Image(idxT height, idxT width):
            height(height), width(width),
            data(new T[height * width * 3]) {}

        Image(std::string imagePath); // TODO: load image from file

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
            return data[(i + j*height)*3 + 0];
        }
        T & getGreen(idxT i, idxT j) {
            assert(i<height && j<width);
            return data[(i + j*height)*3 + 1];
        }
        T & getBlue(idxT i, idxT j) {
            assert(i<height && j<width);
            return data[(i + j*height)*3 + 2];
        }
        const T & getRed(idxT i, idxT j) const {
            assert(i<height && j<width);
            return data[(i + j*height)*3 + 0];
        }
        const T & getGreen(idxT i, idxT j) const {
            assert(i<height && j<width);
            return data[(i + j*height)*3 + 1];
        }
        const T & getBlue(idxT i, idxT j) const {
            assert(i<height && j<width);
            return data[(i + j*height)*3 + 2];
        }

    bool writeImage(std::string path); // TODO

    private:
        idxT height, width;
        T* data;
};

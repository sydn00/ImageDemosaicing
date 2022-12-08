#include <iostream>
#include "image.hpp"

using namespace std;
int main() {
    cout << "hello\n";
    Image<short> A(5, 5);
    cout << A.getRed(0, 0) << A.getGreen(0, 0) << A.getBlue(0, 0) << endl;

}
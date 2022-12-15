# image-class

cmake . -B build --fresh --warn-uninitialized -DCMAKE_BUILD_TYPE=Debug
cmake --build build
./build/main

cmake . -B build --fresh --warn-uninitialized -DCMAKE_BUILD_TYPE=Debug && cmake --build build && ./build/main

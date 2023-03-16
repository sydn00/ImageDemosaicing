cmake_minimum_required(VERSION 3.21)
PROJECT(image-class)

set (CMAKE_CXX_STANDARD 17)    #std::clamp 

# Froned upon way to globally set options on all targets
if(MSVC)
    add_compile_options(/W4)
    # VS2019 and later, not tested!
    add_compile_options(/fsanitize=address)
else()
    add_compile_options(-Wall -Wextra -Wpedantic)
    add_compile_options(-fsanitize=address)
    add_link_options(-fsanitize=address)
endif()

include_directories(${CMAKE_SOURCE_DIR}/include)

#OpenCV 4.7.0 (older versions hasn't been tested)
find_package(OpenCV 4.7.0 REQUIRED COMPONENTS opencv_imgcodecs opencv_highgui opencv_core)

include_directories(${OpenCV_INCLUDE_DIRS})
#executable for serial-part
add_executable(serial src/Serialmain.cpp)
target_link_libraries(serial PUBLIC ${OpenCV_LIBS})



#test stage
#executable for SYCL-part - OpenSYCL config file is hipSYCL - https://github.com/OpenSYCL/OpenSYCL/blob/develop/doc/using-hipsycl.md
#find_package(hipSYCL 0.9.4 REQUIRED)           #cmake .. -DHIPSYCL_TARGETS="omp;cuda-nvcxx"


#add_executable(syclmain src/SYCLmain.cpp src/image.hpp)
#add_sycl_to_target(TARGET syclmain
#                   SOURCES src/SYCLmain.cpp)
#target_link_libraries(syclmain PUBLIC ${OpenCV_LIBS})

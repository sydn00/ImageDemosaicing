cmake_minimum_required(VERSION 3.21)
PROJECT(image-class)

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

add_executable(main src/main.cpp src/image.hpp src/kernels.hpp)
target_link_libraries(main opencv_imgcodecs opencv_core opencv_highgui)

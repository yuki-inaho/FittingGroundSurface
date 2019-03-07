
cmake_minimum_required(VERSION 3.5)
project(DetailGroundFitting)


set(CMAKE_CXX_COMPILER "/usr/bin/g++")
set(CMAKE_CXX_FLAGS "-O0 -std=gnu++11 -fopenmp -fexceptions")

include_directories(
  include
)

add_executable(DetailGroundFitting src/main.cpp src/ConvertFunction.cpp src/FilterFunction.cpp src/GroundFittingManager.cpp)

find_package(PkgConfig REQUIRED)
pkg_check_modules(eigen3 REQUIRED)
find_package(PCL REQUIRED)
find_package(VTK REQUIRED)

pkg_check_modules(OpenCV opencv REQUIRED)
include_directories(${OpenCV_CFLAGS})
target_link_libraries(DetailGroundFitting ${OpenCV_LDFLAGS})

#pkg_check_modules(Armadillo armadillo REQUIRED)
#include_directories(${Armadillo_INCLUDE_DIRS})
#target_link_libraries(TestArma ${Armadillo_LIBRARIES})

pkg_check_modules(realsense2 realsense2 REQUIRED)
include_directories(${realsense2_CFLAGS})
target_link_libraries(DetailGroundFitting ${realsense2_LDFLAGS})


pkg_check_modules(mlpack mlpack REQUIRED)

include_directories(${eigen3_INCLUDE_DIRS})
target_link_libraries(DetailGroundFitting ${eigen3_LIBRARIES})
include_directories(${PCL_INCLUDE_DIRS})
target_link_libraries(DetailGroundFitting ${PCL_LIBRARIES})
include_directories(${VTK_INCLUDE_DIRS})
target_link_libraries(DetailGroundFitting ${VTK_LIBRARIES})
include_directories(${mlpack_INCLUDE_DIRS})
target_link_libraries(DetailGroundFitting ${mlpack_LIBRARIES})


cmake_minimum_required(VERSION 2.8.3)

project(mapping)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Og -fopenmp -march=native -DEIGEN_DONT_PARALLELIZE")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Og -fopenmp -march=native -DEIGEN_DONT_PARALLELIZE")
add_definitions(-DBOOST_ERROR_CODE_HEADER_ONLY)

#include(/usr/local/share/OpenCV/OpenCVConfig.cmake)

find_package(Boost COMPONENTS thread serialization filesystem REQUIRED)
find_package(OpenCV 3.4 REQUIRED)
find_package(Ceres REQUIRED)
#CUDA
find_package(CUDA 9 REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})
#set(CUDA_ARCH "-arch=sm_50" CACHE STRING "Value of the NVCC -arch option.")
#SET(CUDA_VERBOSE_BUILD ON CACHE BOOL "nvcc verbose" FORCE)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "-Xcompiler;-fPIC;" -std=c++11 --generate-code arch=compute_75,code=sm_75)
FILE(GLOB MOTIONSTEREO_SRC motion_stereo/*.*)
CUDA_COMPILE(MOTIONSTEREO ${MOTIONSTEREO_SRC})
aux_source_directory(./data_process DATAPROCESS_SRC)
aux_source_directory(./SurfelFilter SURFELFILTER_SRC)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c++11")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

include_directories(
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/SurfelFilter
    )

set(LIBS
    ${OpenCV_LIBS}
    ${CUDA_LIBRARIES}
#    ${CUDA_nppi_LIBRARY}
    ${CERES_LIBRARIES}
)

add_executable(evaluate evaluate.cc ${MOTIONSTEREO} ${MOTIONSTEREO_SRC} ${DATAPROCESS_SRC} ${SURFELFILTER_SRC})
target_link_libraries(evaluate ${LIBS})
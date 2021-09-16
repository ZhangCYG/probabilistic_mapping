#ifndef ELEMENTS_H
#define ELEMENTS_H

#include <Eigen/Dense>
struct surfel_gn
{
    // basic attributes
    float px=0.0f,py=0.0f;
    int pixel_num=0;
    float color_r=0.0f, color_g=0.0f, color_b=0.0f;
    float a,b,c;
    // semantics
    int semantic_label=0;
    // parameters
    float nx, ny, nz;
    float center_depth;
    // gauss distribution parameters
    float sigma = 0.0f;
    Eigen::Matrix3f std_mat = Eigen::Matrix3f::Zero();  // 3*3
    Eigen::Vector3f mean_mat = Eigen::Vector3f::Zero();  // 3*1
    bool inlier=false;
    // beta parameters
    float alpha=10.0f, beta=10.0f;
};
// 1/d = a x +b y + c;
// (a,b,c) -> Guass
// MLE 
struct surfel_vmf
{
    // basic attributes
    float px=0.0f,py=0.0f;
    int pixel_num=0;
    float color_r=0.0f, color_g=0.0f, color_b=0.0f;
    float a,b,c;
    // plane parameters
    float nx, ny, nz;
    float center_depth;
    // distribution parameters
    float nx_sd, ny_sd, nz_sd;
};

struct plain_point{
    int x;
    int y;
};
#ifdef MODE_VMF
typedef struct surfel_vmf surfel;
#else
typedef struct surfel_gn surfel;
#endif

#endif

#ifndef UTILS_H
#define UTILS_H
#include <Eigen/Dense>

namespace paras{
    static int IMG_WIDTH=640, IMG_HEIGHT=480;
    static float MIN_DEPTH=0.1, MAX_DEPTH=10.0;
    static float MIN_DISP = 0.1, MAX_DISP = 10.0;

    enum filter_mode {VMF, GAUSS};
    const static filter_mode mode=GAUSS;

    static int MAX_SURFELS=4000, SEARCH_RANGE=50;

    // use for calculate probability
    const float k1_color = 1.0f, k2_color = 0.04f;
    const float k1_semantic = 1.0f, k2_semantic = 3.0f;
    const float k1_depth = 1.0f, k2_depth = 0.004f;
    const float k1_distance = 1.0f, k2_distance = 0.03f;

    const float prob_threshold = 0.01f;

    // camara para
    const float fx = 525.0 ;
    const float fy = 525.0 ;
    const float cx = 319.5 ;
    const float cy = 239.5 ;

    // control flags
    const float diff_threshold=1.0;
    const float fuse_threshold = 0.3;
    const bool semantic_on = false;
    // distribution parameters
    const float uni_value = 0.1;
    const float color_thres = 100.0;
    const float simple_semantic_penlty = 0.5;

    // surfel filters
    // TODO: need to be adjusted
    const int valid_point_num = 10;
    const float std_threshold = 2.0f;
    const float vert_direc_cos = 0.95f;
}

#endif
#ifndef FILTER_H
#define FILTER_H
#include <iostream>
#include <vector>
#include <string>
#include <iterator>
#include <type_traits>

#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>
#include "utils.h"
#include "elements.h"

#include<opencv2/opencv.hpp>
#include <opencv2/ximgproc/slic.hpp>
#include "math.h"
#include <string.h>
#include "data_process/data_process.h"

typedef std::pair<Eigen::Matrix3f,Eigen::Vector3f> trans_f;
typedef std::pair<Eigen::Matrix3d,Eigen::Vector3d> trans_d;
typedef std::conditional<paras::mode==paras::VMF, surfel_vmf, surfel_gn>::type surfel;

class BaseFilter{
public:
    inline void set_img(cv::Mat & img){
        this->cur_img = img.clone();
    }
    inline void set_depth(cv::Mat & depth){
        this->cur_depth = depth.clone();
    }
    inline void set_semantic(cv::Mat & semantic){
        this->cur_semantic = semantic.clone();
    }
    inline void iter(){
        this->ID++;
    }
    inline void reset(){
        this->ID = 0;
        this->is_first = true;
    }

protected:
    cv::Mat cur_img, cur_depth, cur_semantic;
    int ID=0;
    bool is_first = true;
};

//template<class surfel>
class SurfelFilter:public BaseFilter{
public:
    SurfelFilter();
    ~SurfelFilter();
    // surfel-level operation
    void filter_once(cv::Mat& img, cv::Mat& depth, cv::Mat & semantic, trans_f & cam_pose);
    void filter_once(cv::Mat& img, cv::Mat& depth, trans_f & cam_pose);
    void state_init(cv::Mat& img, cv::Mat& depth, cv::Mat & semantic);
    void state_init_no_sem(cv::Mat& img, cv::Mat& depth);
    void state_filter();
    void state_transform(trans_f & cam_pose);

    float calc_prob_surf(surfel& s1, surfel& s2, bool use_semantic);

    void save_depth(int depth_case);
    void swap_and_reset();
    // element-level operation
    float calc_prob_color(const int& px, const int& py, const int& sur_x, const int& sur_y);
    float calc_prob_distance(const int& px, const int& py, const int& sur_x, const int& sur_y);
    float calc_prob_semantic(const int& px, const int& py, const int& sur_x, const int& sur_y);

    void fuse(surfel& pre_surfel, surfel& post_surfel);
    void decide_region(int sur_x, int sur_y, int region_x[], int region_y[]);
    // TODO: calc_prob_depth

    // surfel filter
    bool init_valid(int surfel_id);
    bool direction_valid(int surfel_id);
    cv::Mat get_depth_init() {return this->depth_init.clone();}
    cv::Mat get_depth_res() {return this->depth_res.clone();}

private:
    float * prob_map=NULL; // MAX_SURFEL_NUM*MAX_REGION_PIXEL_NUM
    int * search_map=NULL;
    std::vector<surfel> surfels,pre_surfels;
    int surfel_num = 0, pre_surfel_num = 0;
    int surfel_pixels = 0;
    // record pixels in individial surfel
    std::vector<std::vector<plain_point> > pixel_in_surfel;
    cv::Mat cur_spx_seg;
    cv::Mat depth_surfel;
    cv::Mat depth_res;
    cv::Mat depth_init;
};
# endif
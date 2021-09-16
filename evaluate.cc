#include<chrono>
#include<mutex>
// #include<ros/ros.h>
// #include<rosbag/bag.h>
// #include<rosbag/chunked_file.h>
// #include<rosbag/view.h>
// #include<rosbag/query.h>
#include <stdio.h>
#include "eigen3/Eigen/Dense"
#include "opencv2/opencv.hpp"
#include <boost/filesystem.hpp>
#include "sys/types.h"
#include "sys/sysinfo.h"
//#include "sensor_msgs/Imu.h"
//#include "sensor_msgs/Image.h"
#include "src/tools/transport_util.h"
//#include "geometry_msgs/PoseStamped.h"
#include "motion_stereo/stereo_mapper.h"
#include "src/tools/tic_toc.h"
#include "glob.h"
#include <dirent.h>
#include "data_process/data_process.h"
#include "SurfelFilter/filter.h"

using namespace std;


const int downSampleTimes = 0 ;

const int aggNum = 3 ;
const int interval = 3 ;
const int candidateNum = aggNum*interval + 10 ;
float maxDepth = 20.0 ;
float minDepth = 0.2 ;

//TUM
float fx = 525.0 ;
float fy = 525.0 ;
float cx = 319.5 ;
float cy = 239.5 ;

//ICL-NUIM
// float fx = 481.2017 ;
// float fy = -480.0002 ;
// float cx = 319.5 ;
// float cy = 239.5 ;

std::list<ImageMeasurement> imageBuf;
std::mutex mMutexImg;
std::list<PoseElement> poseBuf;
std::mutex mMutexPose;
std::list<cv::Mat> semanticBuf;
std::mutex mMutexSem;
StereoMapper motion_stereo_mapper;
int tImage = 0; // count for Image Num
int tPose = 0;
int lastSize = 0;

struct meshingFrame
{
    cv::Mat img ;
    Eigen::Matrix3d R;
    Eigen::Vector3d t;
    cv::Mat sem;
};

static std::vector<meshingFrame> toAggNonKFs ;

void depthUp(cv::Mat& in, cv::Mat& out)
{
    cv::Mat tmp = cv::Mat(in.rows*2, in.cols*2, in.type() ) ;
    for( int i = 0 ; i < in.rows; i++ )
    {
        for( int j = 0; j < in.cols; j++ )
        {
            tmp.at<float>((i<<1), (j<<1)) = in.at<float>(i, j) ;
            tmp.at<float>((i<<1)+1, (j<<1)) = in.at<float>(i, j) ;
            tmp.at<float>((i<<1), (j<<1)+1) = in.at<float>(i, j) ;
            tmp.at<float>((i<<1)+1, (j<<1)+1) = in.at<float>(i, j) ;
        }
    }
    out = tmp.clone() ;
}


// every 3 key frames interval = 6; interval: only use key frames
int processImg_withsem(const int& idx)
{
    std::list<ImageMeasurement>::iterator iterImg, iterImg_pre ;
    std::list<PoseElement>::iterator iterPose, iterPose_pre ;
    std::list<cv::Mat>::iterator iterSem, iterSem_pre;
    cv::Mat curImg, img, curSem ;
    Eigen::Quaterniond q_R ;
    Eigen::Matrix3d cur_R ;
    Eigen::Vector3d cur_t ;
    cv::Mat cv_R_l, cv_T_l;
    Eigen::Matrix3d R ;
    Eigen::Vector3d t ;
    cv::Mat cv_R_r, cv_T_r;
    SurfelFilter surfel_filter;

    {
        // unique_lock<mutex> lock_0(mMutexImg);
        // unique_lock<mutex> lock_1(mMutexPose);
        if(imageBuf.size() <= lastSize){
            LOG(INFO)<<"program ended: imageBuf Size = "<<int(imageBuf.size())<<" lastSize = "<< int(lastSize);
            return 0;
        }
        iterImg = imageBuf.begin();
        iterPose = poseBuf.begin();
        iterSem = semanticBuf.begin();
        if(iterPose->t != iterImg->t){
            LOG(INFO)<<"program ended: iterPose t = "<< iterPose->t <<" iterImg t = "<<iterImg->t;
            return -1;
        }

        curImg = iterImg->image.clone();
        for (int i = 0; i < downSampleTimes; i++) {
            cv::pyrDown(curImg, curImg, cv::Size(curImg.cols / 2, curImg.rows / 2));
        }
        q_R.x() = iterPose->qx;
        q_R.y() = iterPose->qy;
        q_R.z() = iterPose->qz;
        q_R.w() = iterPose->qw;
        cur_R = q_R.toRotationMatrix();
        cur_t << iterPose->tx, iterPose->ty, iterPose->tz;
        curSem = iterSem->clone();

        iterImg_pre = iterImg;  //current frame
        iterPose_pre = iterPose;
        iterSem_pre = iterSem;

        for (int i = 0; i < aggNum; i++) {
            iterImg = std::next(iterImg, interval);
            iterPose = std::next(iterPose, interval);
            iterSem = std::next(iterSem, interval);

            q_R.x() = iterPose->qx;
            q_R.y() = iterPose->qy;
            q_R.z() = iterPose->qz;
            q_R.w() = iterPose->qw;
            R = q_R.toRotationMatrix();
            t << iterPose->tx, iterPose->ty, iterPose->tz;

            toAggNonKFs[i].img = iterImg->image.clone();
            for (int j = 0; j < downSampleTimes; j++) {
                cv::pyrDown(toAggNonKFs[i].img, toAggNonKFs[i].img,
                            cv::Size(toAggNonKFs[i].img.cols / 2, toAggNonKFs[i].img.rows / 2));
            }
            //cv::imshow(std::to_string(i), toAggNonKFs[i].img ) ;
            toAggNonKFs[i].sem = iterSem->clone();
            toAggNonKFs[i].R = R;
            toAggNonKFs[i].t = t;
        }
        lastSize++;

        imageBuf.pop_front();
        poseBuf.pop_front();
        semanticBuf.pop_front();
    }
    // cv::imshow("curImg", curImg ) ;
    //    cv::waitKey(0) ;

    TicToc tc_sgm ;

    cv::eigen2cv(cur_R, cv_R_l);
    cv::eigen2cv(cur_t, cv_T_l);
    motion_stereo_mapper.initReference(curImg);

    for( int i = 0 ; i < aggNum; i++ )
    {
        img = toAggNonKFs[i].img.clone() ;
        R = toAggNonKFs[i].R ;
        t = toAggNonKFs[i].t ;
        cv::eigen2cv(R, cv_R_r);
        cv::eigen2cv(t, cv_T_r);
        motion_stereo_mapper.update(img, cv_R_l, cv_T_l, cv_R_r, cv_T_r);
    }

    cv::Mat curDepth ;
    motion_stereo_mapper.outputFusionCPU(curDepth, cur_R, cur_t, fx, fy, cx, cy) ;

    // use surfel filter if CASE = 2
    if(CASE == 2){
        trans_f pose = std::make_pair(cur_R.cast<float>(),cur_t.cast<float>());
        surfel_filter.filter_once(curImg, curDepth, curSem, pose);
    }

    for( int j = 0 ; j < downSampleTimes; j++ )
    {
        depthUp(curDepth, curDepth) ;
        //cv::pyrUp(curDepth, curDepth, cv::Size(curDepth.cols*2, curDepth.rows*2) ) ;
    }

    static double sum_time = 0 ;
    static int sum_cnt = 0 ;
    sum_time += tc_sgm.toc() ;
    sum_cnt++ ;
    LOG(INFO)<<"AVERAE CAL TIME "<< sum_time/sum_cnt;

    // use opencv store to file

    for( int i = 0 ; i < curDepth.rows; i++ )
    {
        for( int j=0; j < curDepth.cols; j++ )
        {
            if ( curDepth.at<float>(i, j) < minDepth ) {
                curDepth.at<float>(i, j) = 0 ;
            }
            if ( curDepth.at<float>(i, j) > maxDepth ){
                curDepth.at<float>(i, j) = 0 ;
            }
        }
    }

    static cv::Mat color_disp, disp_depth;
    //disp_depth = curDepth/maxDepth*255;
    //disp_depth.convertTo(disp_depth, CV_8U);
    cv::normalize(curDepth, disp_depth, 0, 255, CV_MINMAX, CV_8U);
    cv::applyColorMap(disp_depth, color_disp, cv::COLORMAP_RAINBOW);
    for( int i = 0 ; i < curDepth.rows; i++ )
    {
        for( int j=0; j < curDepth.cols; j++ )
        {
            if ( curDepth.at<float>(i, j) < 0.001 ) {
                color_disp.at<cv::Vec3b>(i, j)[0] = 0 ;
                color_disp.at<cv::Vec3b>(i, j)[1] = 0 ;
                color_disp.at<cv::Vec3b>(i, j)[2] = 0 ;
            }
        }
    }
    //cv::imshow("Current depth", color_disp ) ;
    // store file
    char file_name[200];
    sprintf(file_name,  "/home/vslam/rgbd_dataset_freiburg2_desk/result/%d.png",  idx);
    cv::imwrite(file_name, color_disp);
    cv::waitKey(1) ;
    return 0;
}

int processImg(const int& idx)
{
    std::list<ImageMeasurement>::iterator iterImg, iterImg_pre ;
    std::list<PoseElement>::iterator iterPose, iterPose_pre ;
    cv::Mat curImg, img;
    Eigen::Quaterniond q_R ;
    Eigen::Matrix3d cur_R ;
    Eigen::Vector3d cur_t ;
    cv::Mat cv_R_l, cv_T_l;
    Eigen::Matrix3d R ;
    Eigen::Vector3d t ;
    cv::Mat cv_R_r, cv_T_r;
    SurfelFilter surfel_filter;

    {
        // unique_lock<mutex> lock_0(mMutexImg);
        // unique_lock<mutex> lock_1(mMutexPose);
        if(imageBuf.size() <= lastSize){
            LOG(INFO)<<"program ended: imageBuf Size = "<<int(imageBuf.size())<<" lastSize = "<< int(lastSize);
            return 0;
        }
        iterImg = imageBuf.begin();
        iterPose = poseBuf.begin();
        if(iterPose->t != iterImg->t){
            LOG(INFO)<<"program ended: iterPose t = "<< iterPose->t <<" iterImg t = "<<iterImg->t;
            return -1;
        }

        curImg = iterImg->image.clone();
        for (int i = 0; i < downSampleTimes; i++) {
            cv::pyrDown(curImg, curImg, cv::Size(curImg.cols / 2, curImg.rows / 2));
        }
        q_R.x() = iterPose->qx;
        q_R.y() = iterPose->qy;
        q_R.z() = iterPose->qz;
        q_R.w() = iterPose->qw;
        cur_R = q_R.toRotationMatrix();
        cur_t << iterPose->tx, iterPose->ty, iterPose->tz;

        iterImg_pre = iterImg;  //current frame
        iterPose_pre = iterPose;

        for (int i = 0; i < aggNum; i++) {
            iterImg = std::next(iterImg, interval);
            iterPose = std::next(iterPose, interval);

            q_R.x() = iterPose->qx;
            q_R.y() = iterPose->qy;
            q_R.z() = iterPose->qz;
            q_R.w() = iterPose->qw;
            R = q_R.toRotationMatrix();
            t << iterPose->tx, iterPose->ty, iterPose->tz;

            toAggNonKFs[i].img = iterImg->image.clone();
            for (int j = 0; j < downSampleTimes; j++) {
                cv::pyrDown(toAggNonKFs[i].img, toAggNonKFs[i].img,
                            cv::Size(toAggNonKFs[i].img.cols / 2, toAggNonKFs[i].img.rows / 2));
            }
            //cv::imshow(std::to_string(i), toAggNonKFs[i].img ) ;
            toAggNonKFs[i].R = R;
            toAggNonKFs[i].t = t;
        }
        lastSize++;

        imageBuf.pop_front();
        poseBuf.pop_front();
    }
    // cv::imshow("curImg", curImg ) ;
    //    cv::waitKey(0) ;

    TicToc tc_sgm ;

    cv::eigen2cv(cur_R, cv_R_l);
    cv::eigen2cv(cur_t, cv_T_l);
    motion_stereo_mapper.initReference(curImg);

    for( int i = 0 ; i < aggNum; i++ )
    {
        img = toAggNonKFs[i].img.clone() ;
        R = toAggNonKFs[i].R ;
        t = toAggNonKFs[i].t ;
        cv::eigen2cv(R, cv_R_r);
        cv::eigen2cv(t, cv_T_r);
        motion_stereo_mapper.update(img, cv_R_l, cv_T_l, cv_R_r, cv_T_r);
    }

    cv::Mat curDepth ;
    motion_stereo_mapper.outputFusionCPU(curDepth, cur_R, cur_t, fx, fy, cx, cy) ;

    LOG(INFO)<<"output middle result by motion stereo";
    char mid_file_name[200];
    sprintf(mid_file_name,  "/home/vslam/rgbd_dataset_freiburg2_desk/mid_result/%d.png",  idx);
    cv::imwrite(mid_file_name, curDepth);
    cv::waitKey(1);

    // LOG(INFO)<<"output ori_img";
    // char ori_name[200];
    // sprintf(ori_name,  "/home/vslam/rgbd_dataset_freiburg2_desk/img/%d.png",  idx);
    // cv::imwrite(ori_name, curImg);
    // cv::waitKey(1);

    // use surfel filter if CASE = 2
    if(CASE == 2){
        trans_f pose = std::make_pair(cur_R.cast<float>(),cur_t.cast<float>());
        for( int i = 0 ; i < curDepth.rows; i++ )
        {
            for( int j = 0 ; j < curDepth.cols; j++ )
            {
                if ( curDepth.at<float>(i, j) > paras::MAX_DEPTH ){
                    curDepth.at<float>(i, j) = 0.0 ;
                }
                if ( curDepth.at<float>(i, j) < paras::MIN_DEPTH){
                    curDepth.at<float>(i, j) = 0.0 ;
                }
            }
        }
        surfel_filter.filter_once(curImg, curDepth, pose);
        surfel_filter.save_depth(0);
        curDepth = surfel_filter.get_depth_res();
        cv::Mat depth_init = surfel_filter.get_depth_init();
        LOG(INFO)<<"output depth_init";
        char init_name[200];
        sprintf(init_name,  "/home/vslam/rgbd_dataset_freiburg2_desk/init_dep/%d.png",  idx);
        cv::imwrite(init_name, depth_init);
        cv::waitKey(1);
    }

    for( int j = 0 ; j < downSampleTimes; j++ )
    {
        depthUp(curDepth, curDepth) ;
        //cv::pyrUp(curDepth, curDepth, cv::Size(curDepth.cols*2, curDepth.rows*2) ) ;
    }

    static double sum_time = 0 ;
    static int sum_cnt = 0 ;
    sum_time += tc_sgm.toc() ;
    sum_cnt++ ;
    LOG(INFO)<<"AVERAE CAL TIME "<< sum_time/sum_cnt;

    // use opencv store to file

    for( int i = 0 ; i < curDepth.rows; i++ )
    {
        for( int j=0; j < curDepth.cols; j++ )
        {
            if ( curDepth.at<float>(i, j) < minDepth ) {
                curDepth.at<float>(i, j) = 0 ;
            }
            if ( curDepth.at<float>(i, j) > maxDepth ){
                curDepth.at<float>(i, j) = 0 ;
            }
        }
    }

    static cv::Mat color_disp, disp_depth;
    //disp_depth = curDepth/maxDepth*255;
    //disp_depth.convertTo(disp_depth, CV_8U);
    cv::normalize(curDepth, disp_depth, 0, 255, CV_MINMAX, CV_8U);
    cv::applyColorMap(disp_depth, color_disp, cv::COLORMAP_RAINBOW);
    for( int i = 0 ; i < curDepth.rows; i++ )
    {
        for( int j=0; j < curDepth.cols; j++ )
        {
            if ( curDepth.at<float>(i, j) < 0.001 ) {
                color_disp.at<cv::Vec3b>(i, j)[0] = 0 ;
                color_disp.at<cv::Vec3b>(i, j)[1] = 0 ;
                color_disp.at<cv::Vec3b>(i, j)[2] = 0 ;
            }
        }
    }
    //cv::imshow("Current depth", color_disp ) ;
    // store file
    char file_name[200];
    sprintf(file_name,  "/home/vslam/rgbd_dataset_freiburg2_desk/result/%d.png",  idx);
    cv::imwrite(file_name, color_disp);
    cv::waitKey(1) ;
    return 0;
}

int main(int argc, char **argv)
{

    LogCfg log(argv[0]);
    DataProcess dp("/data2/vslam/data/rgbd_dataset_freiburg2_desk/rgb",
                    "/data2/vslam/data/rgbd_dataset_freiburg2_desk/align_pose.txt");

    if ( argc < 2 ){
        CASE = 2;
        SEM_CASE = 0; // default: no semantic
    }
    else {
        sscanf(argv[1], "%d", &CASE );
        if(argc > 2){
            sscanf(argv[2], "%d", &SEM_CASE );
        }
        else{
            SEM_CASE = 0; // default: no semantic
        }
    }
    LOG(INFO)<<"CASE = "<<CASE;
    LOG(INFO)<<"SEM_CASE = "<<SEM_CASE;

    LOG(INFO)<<"Ready to read Image and Pose";
    if (opendir(dp.get_file_root().c_str()) == NULL){
        LOG(ERROR)<<"Invalid File Root!";
        exit(1);
   }
    // read Image and pose
    std::vector<cv::String> ImageFn;
    cv::glob(dp.get_file_root(),  ImageFn);
//    int img_total_count = ImageFn.size();
    int img_total_count = 40;
    LOG(INFO)<<"We have "<<img_total_count<<" images to treat with.";
    dp.grab_image_batch(ImageFn, &imageBuf, &mMutexImg, &tImage, img_total_count);
    LOG(INFO)<<"Read Images Finished!";

    std::ifstream PoseFp;
    PoseFp.open(dp.get_pose_root(), std::ios::in);
    for(int i = 0; i < img_total_count; i++){
        std::string tempStr;
        getline(PoseFp, tempStr, '\n');
        dp.grab_pose(tempStr, &poseBuf, &mMutexPose, &tPose);
    }
    LOG(INFO)<<"Read Poses Finished!";

    // read Semantic
    if(SEM_CASE != 0){
        std::vector<cv::String> semFn;
        cv::glob(dp.get_file_root(),  semFn);
        dp.grab_semantic_batch(&semanticBuf, &mMutexSem);
        LOG(INFO)<<"Read Sematics Finished!";
    }

    for( int i = 0 ; i < downSampleTimes; i++ ){
        fx /= 2 ;
        fy /= 2 ;
        cx = (cx+0.5)/2.0 - 0.5;
        cy = (cy+0.5)/2.0 - 0.5;
    }

    cv::Mat K(3, 3, CV_64F) ;
    K.setTo(0.0) ;
    K.at<double>(0, 0) = fx ;
    K.at<double>(1, 1) = fy ;
    K.at<double>(0, 2) = cx ;
    K.at<double>(1, 2) = cy ;
    K.at<double>(2, 2) = 1.0 ;
    float bf = 0.02*fx ;
    float dep_sample = 1.0f / (0.15 * 160.0);
    motion_stereo_mapper.initIntrinsic( K, bf, dep_sample );

    lastSize = aggNum*interval ;
    toAggNonKFs.clear();
    toAggNonKFs.resize(aggNum+5);

    // use idx to browse all
    LOG(INFO)<<"Start to Process!";
    int process_flg = 0;
    for( int i = 0; i < img_total_count; i++)
    {
        if(SEM_CASE != 0){
            process_flg = processImg_withsem(i);
        }
        else{
            process_flg = processImg(i);
        }
        if(process_flg != 0){
            LOG(ERROR)<<"Process Failed with code "<<process_flg;
            return -1;
        }
        if(i % 20 == 0){
            LOG(INFO)<<"Processing " << i << " / " << img_total_count << " ...";
        }
    }

    return 0;
}

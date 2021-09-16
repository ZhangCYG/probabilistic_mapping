#ifndef DATA_PROCESS_H
#define DATA_PROCESS_H
#include "opencv2/opencv.hpp"
#include <mutex>
#include <stdio.h>
#include "glog/logging.h"
#include "glog/raw_logging.h"

// demand:
// 1. A class to read and preprocess different types of datasets.
// 2. Get aligned semantic(if available) pose, depth and rgb images. 
// 3. Support tum rgb-d format and kitti format
// 4. Support Ros (optional)


class ImageMeasurement
        {
        public:
            int t;
            cv::Mat   image;

            ImageMeasurement(const int& _t, const cv::Mat& _image)
            {
                t     = _t;
                image = _image.clone();
            }

            ImageMeasurement(const ImageMeasurement& i)
            {
                t     = i.t;
                image = i.image.clone();
            }

            ~ImageMeasurement() { ;}
        };

struct PoseElement
{
    int t;
    double tx,ty,tz;
    double qx,qy,qz,qw;
};


class DataProcess{
public:
    DataProcess(){;}
    DataProcess(const cv::String& file_root, const cv::String& pose_root, const cv::String& sem_root);
    DataProcess(const cv::String& file_root, const cv::String& pose_root);
    ~DataProcess(){;}
    
    void grab_image(const cv::String& path, std::list<ImageMeasurement>* imageBuf, std::mutex* mMutexImg, int* tImage);
    void grab_pose(const std::string& pose, std::list<PoseElement>* poseBuf, std::mutex* mMutexPose, int* tPose);
    void grab_image_batch(const std::vector<cv::String>& imageFn, std::list<ImageMeasurement>* imageBuf, std::mutex* mMutexImg, int* tImage, const int & img_total_count);
    
    void grab_semantic_batch(std::list<cv::Mat>* semBuf, std::mutex* mMutexSem);
    cv::Mat* process_sem_pic(const cv::Mat& semPic);
    int determine_class(const cv::Vec3b& tempColor);
    void grab_sem_map();

    const cv::String& get_file_root() { return this->m_file_root;}
    const cv::String& get_pose_root() { return this->m_pose_root;}
    const cv::String& get_sem_root() { return this->m_sem_root;}

private:
    std::unordered_multimap<std::string, cv::Vec3b> sem_map;
    cv::String m_file_root;
    cv::String m_pose_root;
    cv::String m_sem_root;
};

void SignalHandle(const char* data, int size);

class LogCfg{
public:
    LogCfg(char* program);
    ~LogCfg();
};
#endif
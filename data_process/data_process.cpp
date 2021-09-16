#include "data_process.h"
// config log dir
#define LOGDIR "log"
#define MKDIR "mkdir -p " LOGDIR

void DataProcess::grab_image(const cv::String& path, std::list<ImageMeasurement>* imageBuf, std::mutex* mMutexImg, int* tImage) {
    cv::Mat img ;
    img = cv::imread(path, 0);   // TODO: maybe no need: change it into opencv from road, as gray scale
    std::unique_lock<std::mutex> lock(*mMutexImg);
    imageBuf->push_front(ImageMeasurement(*tImage, img));
    *tImage ++;   // graph_num global variable, try to use counting, it also could be time stamp
}

void DataProcess::grab_pose(const std::string& pose, std::list<PoseElement>* poseBuf, std::mutex* mMutexPose, int* tPose){
    PoseElement tt ;
    std::vector<std::string> tempstr;
    std::istringstream in(pose);
    std::string str;
    while (getline(in, str, ' ')) {
	    tempstr.push_back(str);
    }
    tt.tx = atof(tempstr[1].c_str());
    tt.ty = atof(tempstr[2].c_str());
    tt.tz = atof(tempstr[3].c_str());
    tt.qx = atof(tempstr[4].c_str());
    tt.qy = atof(tempstr[5].c_str());
    tt.qz = atof(tempstr[6].c_str());
    tt.qw = atof(tempstr[7].c_str());
    tt.t = *tPose;
    *tPose++;

    std::unique_lock<std::mutex> lock(*mMutexPose) ;
    poseBuf->push_front(tt);
}

void DataProcess::grab_image_batch(const std::vector<cv::String>& imageFn, std::list<ImageMeasurement>* imageBuf, std::mutex* mMutexImg, int* tImage, const int & img_total_count){
    
    for(int i = 0; i < img_total_count; i++){
        grab_image(imageFn[i], imageBuf, mMutexImg, tImage);
    }

}

// Virtual Kitti Format, translate it into 1-14
// 1-14 means: Building, Car, GuardRail, Misc, Pole, Road, Sky, Terrain, TrafficLight, TrafficSign, Tree, Truck, Van, Vegetation 
void DataProcess::grab_semantic_batch(std::list<cv::Mat>* semBuf, std::mutex* mMutexSem){
    
    std::vector<cv::String> semFn;
    cv::glob(this->m_sem_root,  semFn);
    for(int i = 0; i < semFn.size(); i++){
        cv::Mat img ;
        img = cv::imread(semFn[i]);
        std::unique_lock<std::mutex> lock(*mMutexSem);
        semBuf->push_front(*process_sem_pic(img));
    }

}

// translate RGB semantic picture to Mat with code
// code format: 1-14 for catogories
cv::Mat* DataProcess::process_sem_pic(const cv::Mat& semPic){

    cv::Mat* res = new cv::Mat(semPic.rows, semPic.cols, CV_8UC1, cv::Scalar::all(0));
    for(int i = 0; i < semPic.rows; i++){
        uchar* ptr = res->ptr<uchar>(i); 
        for(int j = 0; j < semPic.cols; j++){
            cv::Vec3b tempColor = semPic.at<cv::Vec3b>(i, j);
            ptr[j] = this->determine_class(tempColor);
        }
    }

    return res;
}

// translate rgb into code using semantic map
// code format: 1-14 for catogories
int DataProcess::determine_class(const cv::Vec3b& tempColor){

    for(auto iter = this->sem_map.begin(); iter != this->sem_map.end(); iter++){
        if(tempColor == iter->second){
            if(iter->first == "Building"){
                return 1;
            }
            else if(iter->first == "Car"){
                return 2;
            }
            else if(iter->first == "GuardRail"){
                return 3;
            }
            else if(iter->first == "Misc"){
                return 4;
            }
            else if(iter->first == "Pole"){
                return 5;
            }
            else if(iter->first == "Road"){
                return 6;
            }
            else if(iter->first == "Sky"){
                return 7;
            }
            else if(iter->first == "Terrain"){
                return 8;
            }
            else if(iter->first == "TrafficLight"){
                return 9;
            }
            else if(iter->first == "TrafficSign"){
                return 10;
            }
            else if(iter->first == "Tree"){
                return 11;
            }
            else if(iter->first == "Truck"){
                return 12;
            }
            else if(iter->first == "Van"){
                return 13;
            }
            else if(iter->first == "Vegetation"){
                return 14;
            }
        }
    }

}

void DataProcess::grab_sem_map(){
  
    std::ifstream semFp;
    semFp.open(this->m_sem_root, std::ios::in);
    std::string lineStr;
    int i = 0;
    while(getline(semFp, lineStr, '\n')){
        // skip 1st line
        if(i == 0){
            continue;
        }
        std::vector<std::string> tempStr;
        std::istringstream in(lineStr);
        std::string str;
        while (getline(in, str, ' ')) {
            tempStr.push_back(str);
        }
        cv::Vec3b tempColor;
        tempColor[0] = uchar(atoi(tempStr[3].c_str()));
        tempColor[1] = uchar(atoi(tempStr[2].c_str()));
        tempColor[2] = uchar(atoi(tempStr[1].c_str()));
        // handle instance
        if(tempStr[0].substr(0, 3) == "Car"){
            tempStr[0] = "Car";
        }
        if(tempStr[0].substr(0, 3) == "Van"){
            tempStr[0] = "Van";
        }
        this->sem_map.insert({tempStr[0], tempColor});
        i++;
    }

}

DataProcess::DataProcess(const cv::String& file_root, const cv::String& pose_root, const cv::String& sem_root) {
    this->m_file_root = file_root;
    this->m_pose_root = pose_root;
    this->m_sem_root = sem_root;
    grab_sem_map();
}

DataProcess::DataProcess(const cv::String& file_root, const cv::String& pose_root) {
    this->m_file_root = file_root;
    this->m_pose_root = pose_root;
}

LogCfg::LogCfg(char* program) {
    
    system(MKDIR);
    google::InitGoogleLogging(program);
    // when level is higher than google::INFO, put onto screen
    google::SetStderrLogging(google::INFO); 
    FLAGS_colorlogtostderr=true;
    google::SetLogDestination(google::ERROR,LOGDIR"/ERROR_");
    google::SetLogDestination(google::INFO,LOGDIR"/INFO_");
    google::SetLogDestination(google::WARNING,LOGDIR"/WARNING_");
    google::SetLogDestination(google::ERROR,LOGDIR"/ERROR_");
    // output at instance
    FLAGS_logbufsecs = 0;     
    FLAGS_max_log_size = 128;
    FLAGS_stop_logging_if_full_disk = true;
    // handle core dumped
    google::InstallFailureSignalHandler();
    google::InstallFailureWriter(&SignalHandle);
}

// handle core dumped by LOG(ERROR)
void SignalHandle(const char* data, int size)
{
    std::ofstream fs("glog_dump.log",std::ios::app);
    std::string str = std::string(data,size);
    fs<<str;
    fs.close();
    LOG(ERROR)<<str;
}

LogCfg::~LogCfg(){
    google::ShutdownGoogleLogging();
}
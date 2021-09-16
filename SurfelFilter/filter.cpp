#include "filter.h"
// Assignments:
// lzq:
//  1. state_filter
// zcyg:
//  2. others

SurfelFilter::SurfelFilter(){
    this->surfel_pixels = paras::SEARCH_RANGE * paras::SEARCH_RANGE;
    this->prob_map = new float[paras::MAX_SURFELS*this->surfel_pixels]; // row-> surfel, 
    this->search_map = new int[paras::IMG_HEIGHT*paras::IMG_WIDTH];
    memset(this->search_map,0,sizeof(int)*paras::IMG_HEIGHT*paras::IMG_WIDTH);
    this->surfels.resize(paras::MAX_SURFELS);
    this->pre_surfels.resize(paras::MAX_SURFELS);
    this->pixel_in_surfel.resize(paras::MAX_SURFELS);
    this->depth_init = cv::Mat(paras::IMG_HEIGHT, paras::IMG_WIDTH, CV_32FC1, cv::Scalar::all(0));
    this->depth_surfel = cv::Mat(paras::IMG_HEIGHT, paras::IMG_WIDTH, CV_32FC1, cv::Scalar::all(0));
}
SurfelFilter::~SurfelFilter(){
    if(this->prob_map!=NULL){
        delete this->prob_map;
    }
}
void SurfelFilter::swap_and_reset(){
    std::swap(this->surfels,this->pre_surfels);
    for(auto it=this->surfels.begin();it!=this->surfels.end();it++){
        it->px =0.0f; it->py=0.0f;
        it->pixel_num = 0; 
        it->color_r = it->color_g = it->color_b = 0.0;
        it->semantic_label = 0;
        it->inlier = false;
        it->alpha = 10.0; it->beta = 10.0;
    }
}
void SurfelFilter::filter_once(cv::Mat& img, cv::Mat& depth, cv::Mat & semantic, trans_f & cam_pose){
    if(this->is_first){
        this->is_first=false;
        this->state_init(img,depth,semantic);
    }else{
        this->swap_and_reset();
        this->state_init(img,depth,semantic);
        this->state_transform(cam_pose);
        this->state_filter();
    }
    this->iter();
}

void SurfelFilter::filter_once(cv::Mat& img, cv::Mat& depth, trans_f & cam_pose){
    this->state_init_no_sem(img,depth);
    if(this->is_first){
        this->is_first=false;
    }else{
        this->state_transform(cam_pose);
        this->state_filter();
    }
    this->iter();
}

void SurfelFilter::state_init(cv::Mat& img, cv::Mat& depth, cv::Mat & semantic){
    // set image and depth
    this->set_img(img);
    this->set_depth(depth);
    this->set_semantic(semantic);
    // superpixel segmentation
    cv::Ptr<cv::ximgproc::SuperpixelSLIC> slicer =
            cv::ximgproc::createSuperpixelSLIC(img,cv::ximgproc::SLIC::SLICO,15);
    slicer->iterate(4);
    slicer->enforceLabelConnectivity(25);
    slicer->getLabels(this->cur_spx_seg);
    this->surfel_num = slicer->getNumberOfSuperpixels();
    // assign superpixel centers
    for(int y=0;y<paras::IMG_WIDTH;y++){
        for(int x=0;x<paras::IMG_HEIGHT;x++){
            int seg_id = this->cur_spx_seg.at<int>(x,y);
            surfel & cur_surfel = this->surfels[seg_id];
            cur_surfel.px+=x;
            cur_surfel.py+=y;
            cur_surfel.pixel_num++;
//            LOG(INFO)<<"cur surfel pixel num is "<<cur_surfel.pixel_num<<" id is "<<seg_id;
            plain_point temp_point;
            temp_point.x = x;
            temp_point.y = y;
            this->pixel_in_surfel[seg_id].push_back(temp_point);
//            LOG(INFO)<<"cur y,x is "<<y<<", "<<x;
        }
    }
    auto end = this->surfels.begin()+this->surfel_num;
    for(auto it = this->surfels.begin();it!=end;it++){
        if(it->pixel_num>0){
            it->px = it->px/it->pixel_num;
            it->py = it->py/it->pixel_num;
        }
    }
    // initialize probability map
    for(int i = 0; i < this->surfel_num; i++){
        int sur_x = round(this->surfels[i].px);
        int sur_y = round(this->surfels[i].py);
        int num = 0;
        for(int px = sur_x - ceil(paras::SEARCH_RANGE / 2.0f) + 1; px <= sur_x + floor(paras::SEARCH_RANGE / 2.0f); px++){
            for(int py = sur_y - ceil(paras::SEARCH_RANGE / 2.0f) + 1; py <= sur_y + floor(paras::SEARCH_RANGE / 2.0f); py++){
                if(py < 0 || py >= paras::IMG_WIDTH || px < 0 || px >= paras::IMG_HEIGHT){
                    // invalid search, designate it to -1
                    this->prob_map[i * this->surfel_pixels + num] = -1.0f;
                    num++;
                    continue;
                }
                float prob_color = this->calc_prob_color(px, py, sur_x, sur_y);
                float prob_distance = this->calc_prob_distance(px, py, sur_x, sur_y);
                float prob_semantic = this->calc_prob_semantic(px, py, sur_x, sur_y);
                this->prob_map[i * this->surfel_pixels + num] = prob_color * prob_distance * prob_semantic;
                num++;
            }
        }
    }
    // initialize surfel para
    int invalid_sur_num = 0;
    for(int i = 0; i < this->surfel_num; i++){
        Eigen::Matrix3f sxx = Eigen::Matrix3f::Zero();
        Eigen::Vector3f sxy = Eigen::Vector3f::Zero();
        for(auto it = this->pixel_in_surfel[i].begin(); it != this->pixel_in_surfel[i].end(); it++){
//            LOG(INFO)<<"cur y,x is "<<it->y<<", "<<it->x;
            float temp_depth = (float)this->cur_depth.at<uchar>(it->x, it->y);
//            LOG(INFO)<<"cur depth is "<<temp_depth;
            if(temp_depth != 0){
                Eigen::Vector3f pixel_ori_xy_homo; // pixel coor
                float depth_inv = 1.0 / temp_depth;
                // homo_coords << y, x, 1
                pixel_ori_xy_homo << float(it->y), float(it->x), 1.0;
                // project using camara intrinsic para
                Eigen::Vector3f pixel_xy_homo;  // world coor
                Eigen::Matrix3f k_camara;
                k_camara << paras::fx,0,paras::cx,0,paras::fy,paras::cy,0,0,1;
                pixel_xy_homo = k_camara.inverse() * pixel_ori_xy_homo;
//                LOG(INFO)<<"project finished!";
                // using LSM
                sxx += pixel_xy_homo * pixel_xy_homo.transpose();
                sxy += depth_inv * pixel_xy_homo;
//                LOG(INFO)<<"LSM finished!";
            }
        }
        // LOG(INFO)<<"sxx is "<<sxx(0,0)<<" "<<sxx(0,1)<<" "<<sxx(0,2)<<" "<<sxx(1,0)<<" "<<sxx(1,1)<<" "<<sxx(1,2)<<" "<<sxx(2,0)<<" "<<sxx(2,1)<<" "<<sxx(2,2);
        // LOG(INFO)<<"sxy is "<<sxy(0)<<" "<<sxy(1)<<" "<<sxy(2);
        Eigen::Vector3f res = sxx.inverse() * sxy;
        this->surfels[i].a = res(0);
        this->surfels[i].b = res(1);
        this->surfels[i].c = res(2);
        // MLE estimating std
        int valid_num = 0;
        for(auto it = this->pixel_in_surfel[i].begin(); it != this->pixel_in_surfel[i].end(); it++){
            float temp_depth = (float)this->cur_depth.at<uchar>(it->x, it->y);
            if(temp_depth != 0){
                Eigen::Vector3f pixel_ori_xy_homo; // pixel coor
                float depth_inv = 1.0 / temp_depth;
                // homo_coords << y, x, 1
                pixel_ori_xy_homo << float(it->y), float(it->x), 1.0;
                // project using camara intrinsic para
                Eigen::Vector3f pixel_xy_homo;  // world coor
                Eigen::Matrix3f k_camara;
                k_camara << paras::fx,0,paras::cx,0,paras::fy,paras::cy,0,0,1;
                pixel_xy_homo = k_camara.inverse() * pixel_ori_xy_homo;
                // LOG(INFO)<<"valid world coor xyz is "<< pixel_xy_homo(0) << ", "<<pixel_xy_homo(1)<< ", "<< pixel_xy_homo(2);
                this->depth_init.at<float>(it->x, it->y) = 1.0f / (pixel_xy_homo.transpose() * res)(0);
                float residual = depth_inv - (pixel_xy_homo.transpose() * res)(0);
                this->surfels[i].sigma += residual * residual;
                valid_num++;
            }
        }
        this->surfels[i].sigma /= (valid_num - 1);
        // LOG(INFO)<<"valid num is "<<valid_num<<" surfel sigma is "<<this->surfels[i].sigma<<" a, b, c is " << this->surfels[i].a<<this->surfels[i].b<<this->surfels[i].c;
        // calc self & cross corelation matrix
        this->surfels[i].std_mat = this->surfels[i].sigma * sxx.inverse();
        this->surfels[i].mean_mat =  res;
        if(valid_num == 0){
            invalid_sur_num++;
        }
    }
    LOG(INFO)<<"invalid surfel num is "<<invalid_sur_num;
    // LOG(INFO)<<"initialize surfel para finished!";
    // surfel filter
    for(int i = 0; i < this->surfel_num; i++){
        this->init_valid(i);
    }
    // EM initialization (only for Vmf Surfels)
}

void SurfelFilter::state_init_no_sem(cv::Mat& img, cv::Mat& depth){
    // set image and depth
    this->set_img(img);
    this->set_depth(depth);
    // superpixel segmentation
    cv::Ptr<cv::ximgproc::SuperpixelSLIC> slicer =
            cv::ximgproc::createSuperpixelSLIC(img,cv::ximgproc::SLIC::SLICO,10);
    slicer->iterate(4);
    slicer->enforceLabelConnectivity(25);
    slicer->getLabels(this->cur_spx_seg);
    this->surfel_num = slicer->getNumberOfSuperpixels();
    LOG(INFO)<<"Super pixel finished, num is "<<this->surfel_num;
    // assign superpixel centers
    for(int y=0;y<paras::IMG_WIDTH;y++){
        for(int x=0;x<paras::IMG_HEIGHT;x++){
            int seg_id = this->cur_spx_seg.at<int>(x,y);
            surfel & cur_surfel = this->surfels[seg_id];
            cur_surfel.px+=x;
            cur_surfel.py+=y;
            cur_surfel.pixel_num++;
//            LOG(INFO)<<"cur surfel pixel num is "<<cur_surfel.pixel_num<<" id is "<<seg_id;
            plain_point temp_point;
            temp_point.x = x;
            temp_point.y = y;
            this->pixel_in_surfel[seg_id].push_back(temp_point);
//            LOG(INFO)<<"cur y,x is "<<y<<", "<<x;
        }
    }
    // LOG(INFO)<<"finish assign superpixel centers ";
    auto end = this->surfels.begin()+this->surfel_num;
    for(auto it = this->surfels.begin();it!=end;it++){
        if(it->pixel_num>0){
            it->px = it->px/it->pixel_num;
            it->py = it->py/it->pixel_num;
        }
    }
    // LOG(INFO)<<"assign superpixel centers finished!";
    // initialize probability map
    for(int i = 0; i < this->surfel_num; i++){
        int sur_x = round(this->surfels[i].px);
        int sur_y = round(this->surfels[i].py);
        int num = 0;
        float average = 0.0f;
        for(int px = sur_x - ceil(paras::SEARCH_RANGE / 2.0f) + 1; px <= sur_x + floor(paras::SEARCH_RANGE / 2.0f); px++){
            for(int py = sur_y - ceil(paras::SEARCH_RANGE / 2.0f) + 1; py <= sur_y + floor(paras::SEARCH_RANGE / 2.0f); py++){
                if(py < 0 || py >= paras::IMG_WIDTH || px < 0 || px >= paras::IMG_HEIGHT){
                    // invalid search, designate it to -1
                    this->prob_map[i * this->surfel_pixels + num] = -1.0f;
                    num++;
                    continue;
                }
                float prob_color = this->calc_prob_color(px, py, sur_x, sur_y);
                float prob_distance = this->calc_prob_distance(px, py, sur_x, sur_y);
                this->prob_map[i * this->surfel_pixels + num] = prob_color * prob_distance;
                average += this->prob_map[i * this->surfel_pixels + num];
                num++;
            }
        }
        // LOG(INFO)<<"avarage probability of surfel is "<<average / num;
    }
    // LOG(INFO)<<"initialize probability map finished";
    // initialize surfel para
    int invalid_sur_num = 0;
    for(int i = 0; i < this->surfel_num; i++){
        Eigen::Matrix3f sxx = Eigen::Matrix3f::Zero();
        Eigen::Vector3f sxy = Eigen::Vector3f::Zero();
        int sur_x = round(this->surfels[i].px);
        int sur_y = round(this->surfels[i].py);
        int num = 0;
        for(int px = sur_x - ceil(paras::SEARCH_RANGE / 2.0f) + 1; px <= sur_x + floor(paras::SEARCH_RANGE / 2.0f); px++){
            for(int py = sur_y - ceil(paras::SEARCH_RANGE / 2.0f) + 1; py <= sur_y + floor(paras::SEARCH_RANGE / 2.0f); py++){
                if(py < 0 || py >= paras::IMG_WIDTH || px < 0 || px >= paras::IMG_HEIGHT){
                    num++;
                    continue;
                }
                if(this->prob_map[i * this->surfel_pixels + num] < paras::prob_threshold){
                    num++;
                    continue;
                }
                float temp_depth = (float)this->cur_depth.at<uchar>(px, py);
//            LOG(INFO)<<"cur depth is "<<temp_depth;
                if(temp_depth >paras::MIN_DEPTH){
                    Eigen::Vector3f pixel_ori_xy_homo; // pixel coor
                    float depth_inv = 1.0 / temp_depth;
                    // homo_coords << y, x, 1
                    pixel_ori_xy_homo << float(py), float(px), 1.0;
                    // project using camara intrinsic para
                    Eigen::Vector3f pixel_xy_homo;  // world coor
                    Eigen::Matrix3f k_camara;
                    k_camara << paras::fx,0,paras::cx,0,paras::fy,paras::cy,0,0,1;
                    pixel_xy_homo = k_camara.inverse() * pixel_ori_xy_homo;
    //                LOG(INFO)<<"project finished!";
                    // using LSM
                    sxx += pixel_xy_homo * pixel_xy_homo.transpose();
                    sxy += depth_inv * pixel_xy_homo;
    //                LOG(INFO)<<"LSM finished!";
                }
                num++;
            }
        }
        // LOG(INFO)<<"sxx is "<<sxx(0,0)<<" "<<sxx(0,1)<<" "<<sxx(0,2)<<" "<<sxx(1,0)<<" "<<sxx(1,1)<<" "<<sxx(1,2)<<" "<<sxx(2,0)<<" "<<sxx(2,1)<<" "<<sxx(2,2);
        // LOG(INFO)<<"sxy is "<<sxy(0)<<" "<<sxy(1)<<" "<<sxy(2);
        Eigen::Vector3f res = sxx.inverse() * sxy;
        this->surfels[i].a = res(0);
        this->surfels[i].b = res(1);
        this->surfels[i].c = res(2);
        // MLE estimating std
        int valid_num = 0;
        num = 0;
        for(int px = sur_x - ceil(paras::SEARCH_RANGE / 2.0f) + 1; px <= sur_x + floor(paras::SEARCH_RANGE / 2.0f); px++){
            for(int py = sur_y - ceil(paras::SEARCH_RANGE / 2.0f) + 1; py <= sur_y + floor(paras::SEARCH_RANGE / 2.0f); py++){
                if(py < 0 || py >= paras::IMG_WIDTH || px < 0 || px >= paras::IMG_HEIGHT){
                    num++;
                    continue;
                }
                if(this->prob_map[i * this->surfel_pixels + num] < paras::prob_threshold){
                    num++;
                    continue;
                }
                float temp_depth = (float)this->cur_depth.at<uchar>(px, py);
                if(temp_depth != 0){
                    Eigen::Vector3f pixel_ori_xy_homo; // pixel coor
                    float depth_inv = 1.0 / temp_depth;
                    // homo_coords << y, x, 1
                    pixel_ori_xy_homo << float(py), float(px), 1.0;
                    // project using camara intrinsic para
                    Eigen::Vector3f pixel_xy_homo;  // world coor
                    Eigen::Matrix3f k_camara;
                    k_camara << paras::fx,0,paras::cx,0,paras::fy,paras::cy,0,0,1;
                    pixel_xy_homo = k_camara.inverse() * pixel_ori_xy_homo;
                    // LOG(INFO)<<"valid world coor xyz is "<< pixel_xy_homo(0) << ", "<<pixel_xy_homo(1)<< ", "<< pixel_xy_homo(2);
                    // this->depth_init.at<float>(px, py) = 1.0f / (pixel_xy_homo.transpose() * res)(0);
                    float residual = depth_inv - (pixel_xy_homo.transpose() * res)(0);
                    this->surfels[i].sigma += residual * residual;
                    valid_num++;
                }
                num++;
            }
        }
        this->surfels[i].sigma /= (valid_num - 1);
        // LOG(INFO)<<"valid num is "<<valid_num<<" surfel sigma is "<<this->surfels[i].sigma<<" a, b, c is " << this->surfels[i].a<<this->surfels[i].b<<this->surfels[i].c;
        // calc self & cross corelation matrix
        this->surfels[i].std_mat = this->surfels[i].sigma * sxx.inverse();
        this->surfels[i].mean_mat =  res;
        if(valid_num == 0){
            invalid_sur_num++;
        }
    }
    LOG(INFO)<<"invalid surfel num is "<<invalid_sur_num;
    // LOG(INFO)<<"initialize surfel para finished!";
    // surfel filter
    for(int i = 0; i < this->surfel_num; i++){
        this->init_valid(i);
    }
    // store depth after initial
    for(int i = 0; i < this->surfel_num; i++){
        for(auto it = this->pixel_in_surfel[i].begin(); it != this->pixel_in_surfel[i].end(); it++){
            float temp_depth = (float)this->cur_depth.at<uchar>(it->x, it->y);
            if(temp_depth != 0){
                Eigen::Vector3f pixel_ori_xy_homo; // pixel coor
                float depth_inv = 1.0 / temp_depth;
                // homo_coords << y, x, 1
                pixel_ori_xy_homo << float(it->y), float(it->x), 1.0;
                // project using camara intrinsic para
                Eigen::Vector3f pixel_xy_homo;  // world coor
                Eigen::Matrix3f k_camara;
                k_camara << paras::fx,0,paras::cx,0,paras::fy,paras::cy,0,0,1;
                pixel_xy_homo = k_camara.inverse() * pixel_ori_xy_homo;
                // LOG(INFO)<<"valid world coor xyz is "<< pixel_xy_homo(0) << ", "<<pixel_xy_homo(1)<< ", "<< pixel_xy_homo(2);
                this->depth_init.at<float>(it->x, it->y) = 1.0f / (pixel_xy_homo.transpose() * this->surfels[i].mean_mat)(0);
            }
        }
    }
    // EM initialization (only for Vmf Surfels)
}

void SurfelFilter::state_filter(){
    // state fusion (Gauss distribution)(LDA)
    int sur_x,sur_y;
    int region_x[2],region_y[2];
    for(std::vector<surfel>::iterator it=surfels.begin();it!=surfels.end();it++){
        // if valid prior exists, calculate posterior.
        // search neighbors in range
        sur_x = int(it->py+0.5f);
        sur_y = int(it->px+0.5f);
        this->decide_region(sur_x,sur_y,region_x,region_y);
        
        int base_id = region_y[0]*paras::IMG_WIDTH;
        int prior_sur_id = -1;
        float prior_similarity = -1.0;
        for(int py=region_y[0];py<=region_y[1];py++){
            for(int px=region_x[0];px<=region_x[1];px++){
                if(this->search_map[base_id+px]>0){
                    // calculate similarity using color,distance and semantic
                    int tmp_sur_id = this->search_map[base_id+px]-1;
                    auto cur_surf = (*it);
                    std::cout<<tmp_sur_id<<" "<<base_id<<" "<<px<<" "<<py<<std::endl;
                    LOG(INFO)<<"state filter: "<< &cur_surf <<" "<< &(this->pre_surfels[tmp_sur_id]) <<" "<<this->pre_surfels[tmp_sur_id].px;
                    float cur_prob = this->calc_prob_surf(this->pre_surfels[tmp_sur_id], cur_surf, paras::semantic_on);
                    if(cur_prob>prior_similarity){// use the most similar surfel to fuse
                        prior_sur_id = tmp_sur_id;
                        prior_similarity = cur_prob;
                    }
                }
            }
            base_id+=paras::IMG_WIDTH;
        }
        // fuse two nearest neighbors
        if(prior_similarity>=paras::fuse_threshold)
            this->fuse(this->pre_surfels[prior_sur_id],*it);
    }
    // state fusion (Vmf distribution)
}

float SurfelFilter::calc_prob_surf(surfel& s1, surfel& s2, bool use_semantic){
    LOG(INFO)<<"calc_prob_surfel Start!";
    std::cout << &use_semantic <<" "<< &s1 <<" "<<&s2<<" "<<s1.px<< std::endl;
    float p_dist = sqrt((s1.px-s2.px)*(s1.px-s2.px)+(s1.py-s2.py)*(s1.py-s2.py));
    p_dist = paras::k1_distance*exp(-paras::k2_distance*p_dist);
    LOG(INFO)<<"calc_prob_surfel distance part END!";
    float p_color = sqrt(pow(s1.color_r-s2.color_r,2)+pow(s1.color_g-s2.color_g,2)+pow(s1.color_b-s2.color_g,2));
    if(p_color>=paras::color_thres)
        return -1.1;
    p_color = paras::k1_color*exp(-paras::k2_color*p_color);
    LOG(INFO)<<"calc_prob_surfel color part END!";
    if(use_semantic){
        float p_semantic = (s1.semantic_label==s2.semantic_label)? 1.0:paras::simple_semantic_penlty;
        return p_dist*p_color*p_semantic;
    }else
        return p_dist*p_color;
    LOG(INFO)<<"calc_prob_surfel END!";
}

void SurfelFilter::decide_region(int sur_x, int sur_y, int region_x[], int region_y[]){
    region_x[0]= sur_x - ceil(paras::SEARCH_RANGE / 2.0f) + 1;
    region_x[1]= sur_x + floor(paras::SEARCH_RANGE / 2.0f);
    region_x[0] = region_x[0]>=0? region_x[0]:0;
    region_x[1] = region_x[1]<paras::IMG_WIDTH? region_x[1]:paras::IMG_WIDTH-1;

    region_y[0]= sur_y - ceil(paras::SEARCH_RANGE / 2.0f) + 1;
    region_y[1]= sur_y + floor(paras::SEARCH_RANGE / 2.0f);
    region_y[0] = region_y[0]>=0? region_y[0]:0;
    region_y[1] = region_y[1]<paras::IMG_HEIGHT? region_y[1]:paras::IMG_HEIGHT-1;
}

void SurfelFilter::fuse(surfel& pre_surfel, surfel& post_surfel){
    // calculate posteriors
    // check if the surfel is an outlier
    if(!post_surfel.inlier){
        post_surfel.std_mat = pre_surfel.std_mat;
        post_surfel.mean_mat = pre_surfel.mean_mat;
        post_surfel.alpha = pre_surfel.alpha;
        post_surfel.beta = pre_surfel.beta;
        return;
    }
    // check the similarity of pre_surfel and cur_surefel
    if((post_surfel.mean_mat-pre_surfel.mean_mat).norm()>paras::diff_threshold)
        return;
    // fuse Gauss Mean
    auto pre_Sigma_inv = pre_surfel.std_mat.inverse();
    auto post_Sigma_inv = post_surfel.std_mat.inverse();
    Eigen::Matrix3f Sigma_inv = pre_Sigma_inv + post_Sigma_inv;
    Eigen::Vector3f m=Sigma_inv.inverse()*(pre_Sigma_inv*pre_surfel.mean_mat+post_Sigma_inv*post_surfel.mean_mat);
    float C1 = pre_surfel.alpha/(pre_surfel.beta+pre_surfel.alpha);
    float C2 = paras::uni_value*(1-C1);
    C1 = C1*Sigma_inv.determinant()/(pow(2*M_PI,1.5)*pre_Sigma_inv.determinant()*post_Sigma_inv.determinant());
    float C1_0 = m.dot(Sigma_inv*m);
    float C1_1 = pre_surfel.mean_mat.dot(pre_Sigma_inv*pre_surfel.mean_mat);
    float C1_2 = post_surfel.mean_mat.dot(post_Sigma_inv*post_surfel.mean_mat);
    C1 = C1*exp(-0.5*(C1_0-C1_1-C1_2));
    float L = C1+C2;
    C1 = C1/L;C2=C2/L;
    // fuse Gauss parameters
    auto new_mean = C1*m+C2*pre_surfel.mean_mat;
    auto new_std = C1*(m*m.transpose()+Sigma_inv.inverse())+
        C2*(post_surfel.mean_mat*post_surfel.mean_mat.transpose()+post_surfel.std_mat)-
        new_mean*new_mean.transpose();
    post_surfel.mean_mat = new_mean;
    post_surfel.std_mat = new_std;
    // fuse Beta alpha and beta parameter
    float a = pre_surfel.alpha, b = pre_surfel.beta;
    float k1 = C1*(a+1)/(a+b+1)+C2*a/(a+b+1);
    float k2 = (C1*(a+2)+C2*a)*(a+1)/((a+b+1)*(a+b+2)*k1);
    post_surfel.alpha = k1*(k2-1)/(k1-k2);
    post_surfel.beta = (1-k1)*post_surfel.alpha/k1;
    // end
}

void SurfelFilter::state_transform(trans_f & cam_pose){
    // state transform (Gauss distribution)
    memset(this->search_map,0,sizeof(int)*paras::IMG_WIDTH*paras::IMG_HEIGHT);
    Eigen::Vector3f center_coords;
    float x,y,disp,scale;
    auto first = pre_surfels.begin();
    for(std::vector<surfel>::iterator it=first;it!=pre_surfels.end();it++){
        // label tranformed position
        x = (it->py-paras::cx)/paras::fx;
        y = (it->px-paras::cy)/paras::fy;
        disp = 1/(it->a*x+it->b*y+it->c);
        // TODO: this check may be redundant. By Lzq
        if(disp<paras::MIN_DISP || disp>paras::MAX_DISP)
            continue;
        center_coords << x,y,1/disp;
        center_coords = cam_pose.first*center_coords+cam_pose.second;
        if(center_coords(2)<paras::MIN_DEPTH || center_coords(2)>paras::MAX_DEPTH)
            continue;
        disp = 1/center_coords(2);
        center_coords = center_coords*disp;
        it->py = center_coords(0)*paras::fx+paras::cx;
        it->px = center_coords(1)*paras::fy+paras::cy;
        if(it->px<0 || it->py<0  || it->px> paras::IMG_HEIGHT-1 || it->py> paras::IMG_WIDTH-1)
            continue;
        int transformed_id = int(it->px+0.5f)*paras::IMG_WIDTH+int(it->py+0.5f);
        // if(transformed_id==2)
        //     LOG(INFO)<<"pre_value: "<<this->search_map[transformed_id]<<"; new value:"<<std::distance(first,it)+1;
        this->search_map[transformed_id] = std::distance(first,it)+1;
        // if still in new view, then transform parameters
        // A*X+B*Y+C*Z+D=0 <=> A*(X/Z)+B*(Y/Z)+C = -D/Z <=> a*x+b*y+c=1/z
        // ==> normal vector: [a,b,c]/||[a,b,c]||_2
        // TODO: check if new normal vector is along with camera ray! By Lzq
        it->mean_mat = cam_pose.first*it->mean_mat;
        scale = disp/it->mean_mat.dot(center_coords); //.dot(center_coords);
        it->mean_mat = scale*it->mean_mat;
        it->std_mat = scale*scale*(cam_pose.first*it->std_mat*cam_pose.first.transpose());
        it->sigma = scale*scale*it->sigma;
    }
    // surfel filter
    for(int i = 0; i < this->surfel_num; i++){
        this->direction_valid(i);
    }
    // state transform (Vmf distribution)
}

void SurfelFilter::save_depth(int depth_case){

    for(int i = 0; i < this->surfel_num; i++){
        for(auto it = this->pixel_in_surfel[i].begin(); it != this->pixel_in_surfel[i].end(); it++){
            float temp_depth = (float)this->cur_depth.at<uchar>(it->x, it->y);
            if(temp_depth != 0){
                Eigen::Vector3f pixel_ori_xy_homo; // pixel coor
                float depth_inv = 1.0 / temp_depth;
                // homo_coords << y, x, 1
                pixel_ori_xy_homo << float(it->y), float(it->x), 1.0;
                // project using camara intrinsic para
                Eigen::Vector3f pixel_xy_homo;  // world coor
                Eigen::Matrix3f k_camara;
                k_camara << paras::fx,0,paras::cx,0,paras::fy,paras::cy,0,0,1;
                pixel_xy_homo = k_camara.inverse() * pixel_ori_xy_homo;
                // LOG(INFO)<<"valid world coor xyz is "<< pixel_xy_homo(0) << ", "<<pixel_xy_homo(1)<< ", "<< pixel_xy_homo(2);
                this->depth_surfel.at<float>(it->x, it->y) = 1.0f / (pixel_xy_homo.transpose() * this->surfels[i].mean_mat)(0);
            }
        }
    }

    if(depth_case == 0){
        // use surfel depth directly
        this->depth_res = this->depth_surfel;
    }
    else {
        // weight
        cv::Mat prob_sum;
        for(int i = 0; i < this->surfel_num; i++){
            int sur_x = round(this->surfels[i].px);
            int sur_y = round(this->surfels[i].py);
            int num = 0;
            for(int px = sur_x - ceil(paras::SEARCH_RANGE / 2.0f) + 1; px <= sur_x + floor(paras::SEARCH_RANGE / 2.0f); px++){
                for(int py = sur_y - ceil(paras::SEARCH_RANGE / 2.0f) + 1; py <= sur_y + floor(paras::SEARCH_RANGE / 2.0f); py++){
                    if(py < 0 || py >= paras::IMG_WIDTH || px < 0 || px >= paras::IMG_HEIGHT){
                        num++;
                        continue;
                    }
                    Eigen::Vector3f pixel_ori_xy_homo; // pixel coor
                    // homo_coords << y, x, 1
                    pixel_ori_xy_homo << float(py), float(px), 1.0;
                    // project using camara intrinsic para
                    Eigen::Vector3f pixel_xy_homo;  // world coor
                    Eigen::Matrix3f k_camara;
                    k_camara << paras::fx,0,paras::cx,0,paras::fy,paras::cy,0,0,1;
                    pixel_xy_homo = k_camara.inverse() * pixel_ori_xy_homo;
                    float depth_est = this->surfels[i].mean_mat.dot(pixel_xy_homo);
                    // TODO: ensure format of depth_res
                    this->depth_res.at<float>(px, py) = depth_est *  this->prob_map[i * this->surfel_pixels + num];
                    prob_sum.at<float>(px, py) += this->prob_map[i * this->surfel_pixels + num];
                    num++;
                }
            }
        }
        this->depth_res = this->depth_res / prob_sum;
    }
}

float SurfelFilter::calc_prob_color(const int& px, const int& py, const int& sur_x, const int& sur_y){
    // TODO: maybe exist some problems. By Lzq
    float pixel_int, center_int;
    pixel_int = this->cur_img.at<cv::Vec3b>(px, py)[0] + this->cur_img.at<cv::Vec3b>(px, py)[1] + this->cur_img.at<cv::Vec3b>(px, py)[2];
    pixel_int /= 3.0f;

    center_int = this->cur_img.at<cv::Vec3b>(sur_x, sur_y)[0] + this->cur_img.at<cv::Vec3b>(sur_x, sur_y)[1] + this->cur_img.at<cv::Vec3b>(sur_x, sur_y)[2];
    center_int /= 3.0f;

    float color_dist = pow(pixel_int - center_int, 2.0f);

    return paras::k1_color * exp(-paras::k2_color * color_dist);
}

float SurfelFilter::calc_prob_distance(const int& px, const int& py, const int& sur_x, const int& sur_y){

    float dist = pow(px - sur_x, 2.0f) + pow(py - sur_y, 2.0f);

    return paras::k1_distance * exp(-paras::k2_distance * dist);
}
// TODO: maybe donot use Euclier Distance
float SurfelFilter::calc_prob_semantic(const int& px, const int& py, const int& sur_x, const int& sur_y){
    
    float pixel_sem, center_sem;
    pixel_sem = (float)this->cur_semantic.at<uchar>(px, py);
    center_sem = (float)this->cur_semantic.at<uchar>(sur_x, sur_y);

    float sem_dist = pow(pixel_sem - center_sem, 2.0f);

    return paras::k1_semantic * exp(-paras::k2_semantic * sem_dist);
}

bool SurfelFilter::init_valid(int surfel_id){

    if(this->surfels[surfel_id].pixel_num < paras::valid_point_num){
        this->surfels[surfel_id].inlier = false;
        return false;
    }
    if(this->surfels[surfel_id].sigma > paras::std_threshold){
        this->surfels[surfel_id].inlier = false;
        return false;
    }
    if(!this->direction_valid(surfel_id)){
        this->surfels[surfel_id].inlier = false;
        return false;
    }
    return true;
}

bool SurfelFilter::direction_valid(int surfel_id){

    float angle = this->surfels[surfel_id].a * this->surfels[surfel_id].mean_mat(0) + this->surfels[surfel_id].b * this->surfels[surfel_id].mean_mat(1) - this->surfels[surfel_id].mean_mat(2);
    angle /= sqrt(pow(this->surfels[surfel_id].a, 2.0f) + pow(this->surfels[surfel_id].b, 2.0f) + 1);
    angle /= sqrt(pow(this->surfels[surfel_id].mean_mat(0), 2.0f) + pow(this->surfels[surfel_id].mean_mat(1), 2.0f) + pow(this->surfels[surfel_id].mean_mat(2), 2.0f));
    if(abs(angle) > paras::vert_direc_cos){
        this->surfels[surfel_id].inlier = false;
        return false;
    }

    return true;
}
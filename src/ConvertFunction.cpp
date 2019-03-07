#include "ConvertFunction.hpp"

using namespace std;
using namespace pcl;
using namespace cv;

///
///　入力データを変換した結果を出力する機能群
///

/// COPY PointCloud

//deepcopyPointCloud : 点群について深いコピーを施すメソッド(streamingな点群の扱い等に必要)
//input : pc_input (コピー元の点群オブジェクト)
//output : pc_output (コピー後の点群オブジェクト)
pcl::PointCloud<pcl::PointXYZ>::Ptr deepcopyPointCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr pc_input){
    pcl::PointCloud<pcl::PointXYZ>::Ptr pc_output (new pcl::PointCloud<pcl::PointXYZ>);
    int n_pc  = pc_input->points.size();
    for(int i=0;i<n_pc;i++){
        PointXYZ point;
        point.x = pc_input->points[i].x;
        point.y = pc_input->points[i].y;
        point.z = pc_input->points[i].z;
        pc_output->points.push_back(point);
    }
    return pc_output;
}

/// PCL -> Armadillo

//PointCloud2Mat : pcl点群形式のデータをarmadillo行列形式に変換するメソッド
//input : cloud(点群データ)
//output: pc_mat(arma::mat形式の変換データ, pc_mat(i,0):i番目の点群のx座標(横軸)データ, pc_mat(i,2):i番目の点群のz座標(深さ軸)データ)
arma::mat PointCloud2Mat(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud){
    int n_points = cloud->points.size();
    arma::mat pc_mat = arma::zeros<arma::mat>(n_points, 3);
    for(int i=0; i< n_points; i++){
        pc_mat(i,0) = cloud->points[i].x;
        pc_mat(i,1) = cloud->points[i].y;
        pc_mat(i,2) = cloud->points[i].z;
    }
    return pc_mat;
}


/// PCList Copy
//PClistPtr2PCPtr : PCListに格納された点群データをdeepコピーするメソッド
//input : cloud(点群オブジェクト, 呼び出し元ではPointCloudList[i]のような形でアドレス渡しをしている事を想定)
//output: pc(deepコピーした点群データ)
pcl::PointCloud<pcl::PointXYZ>::Ptr PClistPtr2PCPtr(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud){
    pcl::PointCloud<pcl::PointXYZ>::Ptr pc(new pcl::PointCloud<pcl::PointXYZ>);
    int n_points = cloud->points.size();
    for(int i=0; i< n_points; i++){
        pcl::PointXYZ point_tmp;
        point_tmp.x = cloud->points[i].x;
        point_tmp.y = cloud->points[i].y;
        point_tmp.z = cloud->points[i].z;
        pc->points.push_back(point_tmp);
    }
    return pc;
}

// カメラ取り付け角度を考慮した点群位置の変換
void transform_pointcloud(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_src, double theta_pitch, double theta_roll, bool flip_flag){

    Eigen::Affine3f transform = Eigen::Affine3f::Identity();
    transform.rotate(Eigen::AngleAxisf(theta_pitch / 180 * M_PI, Eigen::Vector3f::UnitX()));
    pcl::transformPointCloud(*cloud_src, *cloud_src, transform);

    transform = Eigen::Affine3f::Identity();
    transform.rotate(Eigen::AngleAxisf(theta_roll / 180 * M_PI, Eigen::Vector3f::UnitZ()));
    pcl::transformPointCloud(*cloud_src, *cloud_src, transform);
    if (flip_flag) {
        Eigen::Affine3f transform_flip = Eigen::Affine3f::Identity();
        transform_flip.rotate(Eigen::AngleAxisf(180 / 180 * M_PI, Eigen::Vector3f::UnitZ()));
        pcl::transformPointCloud(*cloud_src, *cloud_src, transform_flip);
    }
}

// 点群を距離画像上のマスク画像に変換する処理
cv::Mat Point2Mask(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, CameraParameter cam_p) {
    cv::Mat Mask = cv::Mat::zeros(Size(1280, 720), CV_8UC1);
    int n_points = cloud->points.size();

#ifdef _OPENMP
#pragma omp parallel for
#endif
    for(int i=0; i< n_points; i++){
        pcl::PointXYZ point_tmp;
        point_tmp.x = cloud->points[i].x;
        point_tmp.y = cloud->points[i].y;
        point_tmp.z = cloud->points[i].z;

        // 四捨五入
        int w_tmp = std::round(cam_p.cx + (point_tmp.x  * cam_p.fx ) / point_tmp.z);
        int h_tmp = std::round(cam_p.cy + (point_tmp.y  * cam_p.fy ) / point_tmp.z);
        Mask.at<uchar>(h_tmp, w_tmp) = 255;
    }

    return Mask;
}

// 距離画像をマスク画像に変換する処理
cv::Mat Depth2Mask(cv::Mat src) {

    cv::Mat Mask = cv::Mat::zeros(Size(src.cols, src.rows), CV_8UC1);

#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int h = 0; h < src.rows; h++) {
        for (int w = 0; w < src.cols; w++) {

            unsigned short z_value = src.at<cv::uint16_t>(h, w);

            if (z_value > 0) {
                Mask.at<uchar>(h, w) = 255;
            }
        }
    }

    return Mask;
}

// 距離画像を点群データに変換する処理
pcl::PointCloud<pcl::PointXYZ>::Ptr Depth2Point(cv::Mat src, CameraParameter cam_p) {

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);

    for (int h = 0; h < src.rows; h++) {
        for (int w = 0; w < src.cols; w++) {

            unsigned short z_value = src.at<cv::uint16_t>(h, w);

            if (z_value > 0) {
                Eigen::Vector3f v;
                v = Eigen::Vector3f::Zero();

                v.z() = z_value / 1000.0f;
                v.x() = v.z() * (w - cam_p.cx) * (1.0 / cam_p.fx);
                v.y() = v.z() * (h - cam_p.cy) * (1.0 / cam_p.fy);

                pcl::PointXYZ point_tmp;
                point_tmp.x = v.x();
                point_tmp.y = v.y();
                point_tmp.z = v.z();
                cloud->points.push_back(point_tmp);
            }
        }
    }

    return cloud;
}

/// memo :どのくらいの領域まで削除するか調整必要
// 距離画像を点群データに変換する処理の内、一定画素範囲のみ実行する(地面データを抜き出すため)
pcl::PointCloud<pcl::PointXYZ>::Ptr Depth2PointPlane(cv::Mat src, CameraParameter cam_p) {

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);

    // height / 2
    for (int h = src.rows / 2; h < src.rows; h++) {
        for (int w = 0; w < src.cols; w++) {

            unsigned short z_value = src.at<cv::uint16_t>(h, w);

            if (z_value > 0) {
                Eigen::Vector3f v;
                v = Eigen::Vector3f::Zero();

                v.z() = z_value / 1000.0f;
                v.x() = v.z() * (w - cam_p.cx) * (1.0 / cam_p.fx);
                v.y() = v.z() * (h - cam_p.cy) * (1.0 / cam_p.fy);

                pcl::PointXYZ point_tmp;
                point_tmp.x = v.x();
                point_tmp.y = v.y();
                point_tmp.z = v.z();
                cloud->points.push_back(point_tmp);
            }
        }
    }

    return cloud;
}
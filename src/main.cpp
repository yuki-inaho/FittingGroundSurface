#include "header.h"
#include "ConvertFunction.hpp"
#include "FilterFunction.hpp"
#include "GroundFittingManager.h"
#include <pcl/registration/icp.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/surface/convex_hull.h>
#include <map>

#include <Eigen/Dense>
#include <Eigen/SparseCore>
#include <Eigen/SparseCholesky>
#include <Eigen/IterativeLinearSolvers>
#include <Eigen/Sparse>

#include <Eigen/Core>
#include <Eigen/SparseQR>



#include <chrono>
#include <unistd.h>

using namespace std;
using namespace cv;
using namespace pcl;

bool view_flag = true;
bool draw_flag = true;

void 
updateViewer(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer, std::mutex &flagMutex)
{
    std::lock_guard<std::mutex> lock (flagMutex);

    auto f_update = viewer->updatePointCloud(cloud, "sample cloud");
    if (!f_update){
        viewer->addPointCloud<pcl::PointXYZ> (cloud, "sample cloud");
    }
    viewer->spinOnce(10);
}
boost::shared_ptr<pcl::visualization::PCLVisualizer> simpleVis (pcl::PointCloud<pcl::PointXYZ>::ConstPtr cloud)
{
  boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
  viewer->setBackgroundColor (0, 0, 0);
  viewer->addPointCloud<pcl::PointXYZ> (cloud, "sample cloud");
  viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "sample cloud");
  viewer->addCoordinateSystem (0.05);
  viewer->initCameraParameters ();
  return (viewer);
}

std::vector<cv::Point>
PointCloud2CVPoints(pcl::PointCloud<pcl::PointXYZ>::Ptr _cloud, CameraParameter cam_p, int img_height, int img_width){
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    cloud = deepcopyPointCloud(_cloud);
    transform_pointcloud(cloud, -30, 0, 0);
    transform_pointcloud(cloud, 0, 0, 1);
    std::vector<cv::Point> point_vec;

    for(int i=0;i<cloud->points.size();i++){
        cv::Point point;
        if(std::isfinite(cloud->points[i].y)){
            point.x = std::round(cam_p.cx + (cloud->points[i].x * cam_p.fx)  / (cloud->points[i].z));
            point.y = std::round(cam_p.cy + (cloud->points[i].y  * cam_p.fy)  / (cloud->points[i].z));

            if(point.x < 0 || point.y < 0 || point.x >= img_width || point.y >= img_height){ //画像範囲内に点群が収まらなければ
                point.x = 0;
                point.y = 0;
            }
            point_vec.push_back(point);
        }else{
            point.x = 0;
            point.y = 0;
            point_vec.push_back(point);
        }
    }
    return(point_vec);
}

void
drawingPoints(cv::Mat &img, const std::vector<cv::Point> &points){
    for(int i=0;i<points.size();i++){
        circle(img, points[i], 1 ,cv::Scalar(0,255,0));
    }
}

void
drawingGrid(cv::Mat &img, const std::vector<cv::Point> &points, std::multimap<int64_t, int> mp, std::map<int64_t, bool> mp_draw, int grid_size){
    for(int _x = 0; _x< grid_size; _x++){
        for(int _z = 0; _z< grid_size; _z++){

            if(_x < grid_size-1){
                if(_z < grid_size-1){

                    bool draw_flag = true;

                    int p_ind = grid_size*_x + _z;
                    int p_ind_zp1 = grid_size*_x + _z + 1;
                    int p_ind_xp1 = grid_size*(_x+1) + _z;
                    int p_ind_xzp1 = grid_size*(_x+1) + _z + 1;

                    draw_flag &= (points[p_ind].y != 0 );
                    draw_flag &= (points[p_ind_zp1].y != 0);
                    draw_flag &= (points[p_ind_xp1].y != 0);
                    draw_flag &= (points[p_ind_xzp1].y != 0);

                    if(draw_flag ){
//                        circle(img, points[p_ind], 5 ,cv::Scalar(0,255,0), -1);                                                        
//                        circle(img, points[p_ind_zp1], 5 ,cv::Scalar(255,0,0), -1);                                    
                        line(img, points[p_ind], points[p_ind_zp1], Scalar(0,255,0), 1, CV_AA); 
                        line(img, points[p_ind], points[p_ind_xp1], Scalar(0,255,0), 1, CV_AA); 
                        line(img, points[p_ind_xp1], points[p_ind_xzp1], Scalar(0,255,0), 1, CV_AA);                                                 
                        line(img, points[p_ind_zp1], points[p_ind_xzp1], Scalar(0,255,0), 1, CV_AA); 
                    }else{
                    //    cout << "test" << endl;
                    }
                }
            }
        }
    }
}

std::multimap<int64_t, int>
generateHash2PointIdxMap(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, double x_min, double x_max, double z_min, double z_max, int grid_size)
{
    std::multimap<int64_t, int> map_index2height;
    for(int i=0;i<cloud->points.size();i++){
        double px = cloud->points[i].x;
        double pz = cloud->points[i].z;
        int x_int = std::floor((px - x_min) /(x_max - x_min) * grid_size);
        int z_int = std::floor((pz - z_min) /(z_max - z_min) * grid_size);
        int64_t point_hash = x_int*grid_size + z_int;
        map_index2height.insert(std::pair<int64_t, int>(point_hash, i));
    }
    return map_index2height;
}

int main (int argc, char *argv[])
{
    std::mutex flagMutex;    
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);
    if (pcl::io::loadPCDFile<pcl::PointXYZ> ("../data/point_0002.pcd", *cloud) == -1) //* load the file
    {
        PCL_ERROR ("Couldn't read file test_pcd.pcd \n");
        return (-1);
    }

    CameraParameter cam_p;
    cam_p.fx = 900.191;
    cam_p.fy = 900.191;
    cam_p.cx = 640.477;
    cam_p.cy = 349.765;

    transform_pointcloud(cloud, 0, 0, 1);
    transform_pointcloud(cloud, 30, 0, 0);
    cloud = Z_Filter(cloud, 1.3);

    namedWindow ("Color", WINDOW_AUTOSIZE);
    moveWindow("Color", 1000,10);
    cv::Mat cimg;

    int currentKey = 0;
    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer;

    std::chrono::system_clock::time_point  start, end; // 型は auto で可
    start = std::chrono::system_clock::now(); // 計測開始時間

    //大域的地面フィッティング
    GroundFittingManager gm;
    std::vector<pcl::PointXYZ> aspara_ground_intersection_vec;
    std::vector<pcl::ModelCoefficients> ground_plane_coeffs;
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_ground(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
    gm.setPointCloudForFit(cloud);
    gm.fit();
    gm.getInlier(cloud, inliers);
    pcl::ModelCoefficients coeff = gm.getPlaneCoeff();
    pcl::ExtractIndices<pcl::PointXYZ> eifilter;

    double nx = coeff.values[0];
    double ny = coeff.values[1];
    double nz = coeff.values[2];
    double coeff_plane = coeff.values[3];
    double theta_ground = std::atan2(std::abs(nz),std::abs(ny))/M_PI*180.0;
    std::cout << "theta:" << theta_ground << std::endl;

    //地面点群の抽出
    std::vector<int> inliers_plane;
    Eigen::VectorXf model_coefficients(4);
    for(int i = 0;i<4;i++){
        model_coefficients(i) = coeff.values[i];
    }

    pcl::PointIndices::Ptr _inliers(new pcl::PointIndices);
    pcl::SampleConsensusModelPlane<pcl::PointXYZ>::Ptr model_p (new pcl::SampleConsensusModelPlane<pcl::PointXYZ> (cloud));
    model_p->selectWithinDistance 	(model_coefficients, 0.03, inliers_plane) ; //  3cm-range extraction from ground plane

    _inliers->indices.clear();
    for(int i = 0;i<inliers_plane.size();i++){
        _inliers->indices.push_back(inliers_plane[i]);
    }

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_extract (new pcl::PointCloud<pcl::PointXYZ>);
    eifilter.setInputCloud (cloud);
    eifilter.setIndices (_inliers);
    eifilter.setNegative (false);
    eifilter.filter (*cloud_extract);

    //地面点群のフィットした平面に対する射影(3次元点群を2次元平面に写す)
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_projected (new pcl::PointCloud<pcl::PointXYZ>);
    model_p->projectPoints(inliers_plane, model_coefficients, *cloud_projected, false);

    //地面の凸包の計算と、それを定義域とするグリッド空間の作成
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_hull (new pcl::PointCloud<pcl::PointXYZ>);
    pcl::ConvexHull<pcl::PointXYZ> chull;
    chull.setInputCloud (cloud_projected);
    chull.reconstruct (*cloud_hull);
    double x_min, x_max, z_min, z_max;
    x_min = 10000; z_min = 10000;
    x_max = -10000; z_max = 0;

    for(int i = 0; i<cloud_hull->points.size();i++){
        if(cloud_hull->points[i].x > x_max) x_max = cloud_hull->points[i].x;
        if(cloud_hull->points[i].x < x_min) x_min = cloud_hull->points[i].x;
        if(cloud_hull->points[i].z > z_max) z_max = cloud_hull->points[i].z;
        if(cloud_hull->points[i].z < z_min) z_min = cloud_hull->points[i].z;
    }

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_grid (new pcl::PointCloud<pcl::PointXYZ>);
    int grid_size = 25;

    //グリッドセルのインデックス->地面点群のインデックス（複数）と対応付けるハッシュマップの作成
    std::multimap<int64_t, int> map_hash2pointIdx = generateHash2PointIdxMap(cloud_projected, x_min, x_max, z_min, z_max, grid_size);    
    std::map<int64_t, bool> map_hash2draw; 

    //各グリッドセル上のメジアンの計算
    for(int _x = 0; _x< grid_size; _x++){
        for(int _z = 0; _z< grid_size; _z++){
            pcl::PointXYZ _point;
            double _x_f = double(_x)/double(grid_size)*(x_max - x_min);
            double _z_f = double(_z)/double(grid_size)*(z_max - z_min);
            _point.x = _x_f + x_min;
            _point.z = _z_f + z_min;

            int64_t grid_hash = grid_size* _x + _z;

            //同一グリッドセル範囲内に収まる点群の取得
            std::pair<multimap<int64_t, int>::iterator, multimap<int64_t, int>::iterator> range = map_hash2pointIdx.equal_range(grid_hash);
            
            if(range.first != range.second){ //グリッドセルに点群が存在すれば実行
                std::vector<double> height_vec;
                for(auto it = range.first; it != range.second; ++it){
                    height_vec.push_back(float(cloud_extract->points[it->second].y)) ;
                }

                //メジアンの計算
                int n = height_vec.size() / 2;
                double height_med;
                if(height_vec.size()%2 == 1)
                {
                    std::nth_element(height_vec.begin(), height_vec.begin()+n, height_vec.end());
                    height_med = height_vec[n];
                }else{
                    std::nth_element(height_vec.begin(), height_vec.begin()+n-1, height_vec.end());
                    float height_med_m = height_vec[n];                    
                    float height_med_p = height_vec[n-1];                    
                    height_med = 0.5*(height_med_m+height_med_p);
                }

                map_hash2draw.insert(std::pair<int64_t, bool>(grid_hash, true));                
                _point.y = height_med;
            }else{
                map_hash2draw.insert(std::pair<int64_t, bool>(grid_hash, false));                
                _point.y = std::numeric_limits<float>::quiet_NaN();
            }

            cloud_grid->points.push_back(_point);
        }
    }
    end = std::chrono::system_clock::now();  // 計測終了時間
    double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count(); //処理に要した時間をミリ秒に変換
    cout << elapsed << endl;


    cimg = cv::imread("../data/color_0002.jpg");

    std::vector<cv::Point> points_2d =  PointCloud2CVPoints(cloud_grid, cam_p, cimg.rows,cimg.cols);
    drawingGrid(cimg, points_2d, map_hash2pointIdx, map_hash2draw, grid_size);

    while (currentKey != 27) // ESCで終了 
    {
        if(cimg.cols != 0){
            imshow("Color",cimg);
        }
//        updateViewer(cloud_projected,viewer,flagMutex);
        currentKey = waitKey (10);            
        if(currentKey == 'q'){
            break;
        }
    }

    return 0;
}

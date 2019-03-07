#pragma once
#include "header.h"

pcl::PointCloud<pcl::PointXYZ>::Ptr deepcopyPointCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr pc_input);
arma::mat PointCloud2Mat(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud);
void transform_pointcloud(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_src, double theta_pitch, double theta_roll, bool flip_flag);

cv::Mat Point2Mask(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, CameraParameter cam_p);
cv::Mat Depth2Mask(cv::Mat src);
pcl::PointCloud<pcl::PointXYZ>::Ptr Depth2Point(cv::Mat src, CameraParameter cam_p);
pcl::PointCloud<pcl::PointXYZ>::Ptr Depth2PointPlane(cv::Mat src, CameraParameter cam_p);


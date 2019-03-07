#pragma once
#include "header.h"
#include "GroundFittingManager.h"
#include <pcl/kdtree/kdtree_flann.h>

class GroundFittingManager{
    public:
    GroundFittingManager();   
    void setPointCloudForFit(const pcl::PointCloud<pcl::PointXYZ>::Ptr cloud);
    void fit();
    void getInlier(const pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, pcl::PointIndices::Ptr inliers);
    void setLineCoeffs(const double &_x_0, const double &_y_0, const double &_z_0, const double &_l_x, const double &_l_y, const double &_l_z);
    void setGroundPointCloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_ground_input);
    void setKDTree();    
    void deriveIntersection();
    void getIntersectionPoint(pcl::PointXYZ &_intersection_point);
    void fitLocalPlane();
    void deriveIntersectionLocalView();
    void getIntersectionPointLocalView(pcl::PointXYZ &_intersection_point);
    pcl::ModelCoefficients getLocalPlaneCoeff();
    pcl::ModelCoefficients getPlaneCoeff();

    private:
    pcl::PointCloud<pcl::PointXYZ> cloud_ground;
    pcl::PointCloud<pcl::PointXYZ> cloud_global;
    pcl::ModelCoefficients coefficients;
    pcl::ModelCoefficients local_plane_coefficients;
    double line_coeff_x,line_coeff_y,line_coeff_z, x_0, y_0, z_0;
    pcl::PointXYZ intersection_point;
    pcl::PointXYZ intersection_point_local;
    pcl::KdTreeFLANN<pcl::PointXYZ> kdtree_ground;
};
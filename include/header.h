#pragma once
#include <iostream>
#include <fstream>
#include <chrono>
#include <string>
#include <omp.h>
#include <chrono>

/// Armadillo
#include <armadillo>

/// Eigen
#include <Eigen/Dense>

/// OpenCV
#include <opencv2/opencv.hpp>

/// PCL
#include <pcl/io/io.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/search/impl/search.hpp>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/conditional_removal.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/common/transforms.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/segmentation/min_cut_segmentation.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/sample_consensus/sac_model_line.h>
#include <pcl/sample_consensus/sac_model_plane.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/region_growing.h>
#include <pcl/surface/mls.h>
#include <pcl/ModelCoefficients.h>

/// RealSense
#include <librealsense2/rs.hpp>

/// カメラパラメータ
struct CameraParameter{
    float fx;
    float fy;
    float cx;
    float cy;
};
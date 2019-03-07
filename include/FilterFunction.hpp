#pragma once
#include "header.h"

pcl::PointCloud<pcl::PointXYZ>::Ptr removeNan(pcl::PointCloud<pcl::PointXYZ>::Ptr target);
pcl::PointCloud<pcl::PointXYZ>::Ptr RoughNessFilter(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud);

pcl::PointCloud<pcl::PointXYZ>::Ptr Z_Filter(pcl::PointCloud<pcl::PointXYZ>::Ptr src, float z_limit);
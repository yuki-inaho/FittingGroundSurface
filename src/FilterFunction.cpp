#include "FilterFunction.hpp"

using namespace pcl;
using namespace std;

///
///　入力データをフィルタリング処理した結果を出力する機能群
///

//removeNan: NaN要素を点群データから除去するメソッド
//input : target(NaN要素を除去する対象の点群)
//output: cloud(除去を行った点群)
PointCloud<PointXYZ>::Ptr removeNan(PointCloud<PointXYZ>::Ptr target){
    PointCloud<PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    int n_point = target->points.size();
    for(int i=0;i<n_point; i++){
        PointXYZ tmp_point;
        if(std::isfinite(target->points[i].x) || std::isfinite(target->points[i].y) || std::isfinite(target->points[i].z)){
            tmp_point.x = target->points[i].x;
            tmp_point.y = target->points[i].y;
            tmp_point.z = target->points[i].z;
            cloud->points.push_back(tmp_point);
        }
    }
//  cout << "varid points:" << cloud->points.size() << endl;
    return cloud;
}


/// RoughNess Filter
double _calc_roughness(int point_ind, pcl::PointCloud<PointXYZ>::Ptr cloud, std::vector<int> pointIdxRadiusSearch) {
    double rgns_var = 10000;

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_local(new pcl::PointCloud<pcl::PointXYZ>);
    int n_local = pointIdxRadiusSearch.size();
    cloud_local->points.push_back(cloud->points[point_ind]);
    for (int i = 0; i < n_local; i++) {
        cloud_local->points.push_back(cloud->points[pointIdxRadiusSearch[i]]);
    }

    pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices);

    pcl::SACSegmentation<pcl::PointXYZ> seg;
    // Optional
    // Mandatory
    seg.setModelType(pcl::SACMODEL_PLANE);
    seg.setMethodType(pcl::SAC_PROSAC);
    //seg.setMethodType (pcl::SAC_LMEDS);
    seg.setDistanceThreshold(0.10); //10cm
    seg.setMaxIterations(10);
    seg.setInputCloud(cloud_local);
    seg.segment(*inliers, *coefficients);
    if (inliers->indices.size() == 0) {
        return 10000;
    }

    vector<int> inliers_plane;
    Eigen::VectorXf model_coefficients(4);

    for (int i = 0; i < 4; i++) {
        model_coefficients(i) = coefficients->values[i];
    }

/*
    vector<int> ind;
    for(int i=0;i<cloud_local->points.size();i++){
        ind.push_back(i);
    }
*/
    vector<double> distances;
    pcl::SampleConsensusModelPlane<pcl::PointXYZ>::Ptr model_p(
            new pcl::SampleConsensusModelPlane<pcl::PointXYZ>(cloud_local));
    //model_p->computeModelCoefficients 	(ind, model_coefficients);
    model_p->getDistancesToModel(model_coefficients, distances);
    rgns_var = distances[0];
    return rgns_var;
}

vector<double> calc_roughness(pcl::PointCloud<PointXYZ>::Ptr cloud) {
    vector<double> roughness;

    pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
    kdtree.setInputCloud(cloud);
    std::vector<int> pointIdxRadiusSearch;
    std::vector<float> pointRadiusSquaredDistance;
    float radius = 0.03;
    double rgns_var;

    for (int i = 0; i < cloud->points.size(); i++) {
        pcl::PointXYZ searchPoint;
        searchPoint.x = cloud->points[i].x;
        searchPoint.y = cloud->points[i].y;
        searchPoint.z = cloud->points[i].z;
        int num_nb = kdtree.radiusSearch(searchPoint, radius, pointIdxRadiusSearch, pointRadiusSquaredDistance);
        if (num_nb > 30) {
            rgns_var = _calc_roughness(i, cloud, pointIdxRadiusSearch);
        } else {
            rgns_var = 10000;
        }
        roughness.push_back(rgns_var);
    }

    return roughness;
}

PointCloud<PointXYZ>::Ptr RoughNessFilter(PointCloud<PointXYZ>::Ptr cloud) {
    vector<double> roughness = calc_roughness(cloud);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_extract(new pcl::PointCloud<pcl::PointXYZ>);
    for (int i = 0; i < cloud->points.size(); i++) {
//      if(roughness[i] < 0.0035 & density[i]> 7000000){
        if (roughness[i] < 0.0035) {
            PointXYZ newPoint;
            newPoint.x = cloud->points[i].x;
            newPoint.y = cloud->points[i].y;
            newPoint.z = cloud->points[i].z;
            cloud_extract->points.push_back(newPoint);
        }
    }
    return cloud_extract;
}

/// 点群座標軸におけるZ方向(奥行方向)を一定以上の値を削除する
pcl::PointCloud<pcl::PointXYZ>::Ptr Z_Filter(pcl::PointCloud<pcl::PointXYZ>::Ptr src, float z_limit){

    pcl::PointCloud<pcl::PointXYZ>::Ptr dst(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::ConditionAnd<pcl::PointXYZ>::Ptr range_cond(new pcl::ConditionAnd<pcl::PointXYZ>());
    pcl::ConditionalRemoval<pcl::PointXYZ> condrem;

    // Z < 1.0m
    range_cond->addComparison(pcl::FieldComparison<pcl::PointXYZ>::ConstPtr(new pcl::FieldComparison<pcl::PointXYZ>("z", pcl::ComparisonOps::LT, z_limit)));
    condrem.setCondition(range_cond);
    condrem.setInputCloud(src);
    condrem.setKeepOrganized(true);
    condrem.filter(*dst);

    return dst;
}
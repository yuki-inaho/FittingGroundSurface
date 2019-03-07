#include "GroundFittingManager.h"

#include <pcl/surface/on_nurbs/fitting_surface_tdm.h>
#include <pcl/surface/on_nurbs/fitting_curve_2d_asdm.h>
#include <pcl/surface/on_nurbs/triangulation.h>

void
PointCloud2Vector3d (const pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, pcl::on_nurbs::vector_vec3d &data)
{
  for (unsigned i = 0; i < cloud->size (); i++)
  {
    pcl::PointXYZ &p = cloud->at (i);
    if (!std::isnan (p.x) && !std::isnan (p.y) && !std::isnan (p.z))
      data.push_back (Eigen::Vector3d (p.x, p.y, p.z));
  }
}

void
Vector3d2PointCloud (pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, const pcl::on_nurbs::vector_vec3d &data)
{
    for(int i=0;i<data.size();i++){
        Eigen::Vector3d _data;
        _data = data[i];
        pcl::PointXYZ _point;
        _point.x = _data(0);
        _point.y = _data(1);
        _point.z = _data(2);
        cloud->points.push_back(_point);
    }
}



GroundFittingManager::GroundFittingManager(){

}

void
GroundFittingManager::setPointCloudForFit(const pcl::PointCloud<pcl::PointXYZ>::Ptr cloud){
    pcl::copyPointCloud(*cloud, cloud_global);
}

void
GroundFittingManager::fit(){
    pcl::PointIndices::Ptr inliers (new pcl::PointIndices);
    pcl::PointCloud<pcl::PointXYZ>::Ptr _cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::copyPointCloud(cloud_global, *_cloud);

    pcl::SACSegmentation<pcl::PointXYZ> seg;
    seg.setOptimizeCoefficients (true);
    seg.setModelType (pcl::SACMODEL_PLANE);
    seg.setMethodType (pcl::SAC_PROSAC);
    seg.setDistanceThreshold (0.02); //2cm 
    seg.setProbability(0.99999);
    seg.setMaxIterations(20000);
    seg.setInputCloud (_cloud);
    seg.segment (*inliers, coefficients);
}

void
GroundFittingManager::getInlier(const pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, pcl::PointIndices::Ptr inliers){

    //ground point extraction using parameters of fitted plane
    std::vector<int> inliers_plane;
    Eigen::VectorXf model_coefficients(4);
    for(int i = 0;i<4;i++){
        model_coefficients(i) = coefficients.values[i];
    }

    pcl::SampleConsensusModelPlane<pcl::PointXYZ>::Ptr model_p (new pcl::SampleConsensusModelPlane<pcl::PointXYZ> (cloud));
    model_p->selectWithinDistance 	(model_coefficients, 0.05, inliers_plane) ; //  5cm-range extraction from ground plane
    inliers->indices.clear();
    for(int i = 0;i<inliers_plane.size();i++){
        inliers->indices.push_back(inliers_plane[i]);
    }

/*
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_nurbs_input(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::ExtractIndices<pcl::PointXYZ> eifilter; 
    eifilter.setInputCloud (cloud);
    eifilter.setIndices (inliers);
    eifilter.setNegative (false);
    eifilter.filter (*cloud_nurbs_input);

    std::chrono::system_clock::time_point  start, end; // 型は auto で可
    start = std::chrono::system_clock::now(); // 計測開始時間

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_nurbs(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::VoxelGrid<pcl::PointXYZ> sorv_nurbs;
//    sorv_nurbs.setInputCloud (boost::make_shared<pcl::PointCloud<pcl::PointXYZ>>(cloud_ground));
    sorv_nurbs.setInputCloud (cloud_nurbs_input);
    sorv_nurbs.setLeafSize (0.01f, 0.01f, 0.01f);
    sorv_nurbs.filter (*cloud_nurbs);
    
    pcl::search::Search<pcl::PointXYZ>::Ptr tree = boost::shared_ptr<pcl::search::Search<pcl::PointXYZ> >(new pcl::search::KdTree<pcl::PointXYZ>);
    pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
    pcl::NormalEstimationOMP<pcl::PointXYZ, pcl::Normal> normal_estimator(8);
    normal_estimator.setSearchMethod(tree);
    normal_estimator.setInputCloud(cloud_nurbs);
    //normal_estimator.setInputCloud(cloud_extract);
    normal_estimator.setKSearch(30);
    normal_estimator.compute(*normals);

    /// 領域分割
    std::vector<pcl::PointIndices> cluster_indices;
    pcl::RegionGrowing<pcl::PointXYZ, pcl::Normal> reg;
    reg.setMinClusterSize(1000);// 最小クラスタサイズ -> 大きさの妥当性
    reg.setMaxClusterSize(1000000); // 最大クラスタサイズ　-> 大きさの妥当性
    reg.setSearchMethod(tree);
    reg.setNumberOfNeighbours(100);
    reg.setInputCloud(cloud_nurbs);
    reg.setInputNormals(normals);
    reg.setSmoothnessThreshold(50.0 / 180.0 * M_PI);
    reg.setCurvatureThreshold(0.50); // 曲率しきい値
    std::vector<pcl::PointIndices> clusters;
    reg.extract(clusters);
    
    int max_num_point = 0;
    int arg_max_num_point = 0;
    for(int i=0;i<clusters.size();i++){
        if(max_num_point < clusters[i].indices.size()){
            max_num_point = clusters[i].indices.size();
            arg_max_num_point  = i;
        }
    }
    eifilter.setInputCloud (cloud_nurbs);
    eifilter.setIndices (boost::make_shared<pcl::PointIndices>(clusters[arg_max_num_point]));
    eifilter.setNegative (false);
    eifilter.filter (*cloud_nurbs_input);

//    inliers = clusters[arg_max_num_point];
    int K = 3;
    std::vector<int> pointIdxNKNSearch(K);
    std::vector<float> pointNKNSquaredDistance(K);
    pcl::KdTreeFLANN<pcl::PointXYZ> kdtree_nurbs;
    kdtree_nurbs.setInputCloud (cloud_nurbs_input);

    inliers->indices.clear();
    for(int i=0;i<cloud->points.size();i++)
    {
        if ( kdtree_nurbs.nearestKSearch (cloud->points[i], K, pointIdxNKNSearch, pointNKNSquaredDistance) > 0 )
        {
            pcl::PointXYZ _point;
            if(pointIdxNKNSearch.size() == 3){
                _point.x = (cloud_nurbs_input->points[pointIdxNKNSearch[0]].x + cloud_nurbs_input->points[pointIdxNKNSearch[1]].x + cloud_nurbs_input->points[pointIdxNKNSearch[2]].x)/3;
                _point.y = (cloud_nurbs_input->points[pointIdxNKNSearch[0]].y + cloud_nurbs_input->points[pointIdxNKNSearch[1]].y + cloud_nurbs_input->points[pointIdxNKNSearch[2]].y)/3;
                _point.z = (cloud_nurbs_input->points[pointIdxNKNSearch[0]].z + cloud_nurbs_input->points[pointIdxNKNSearch[1]].z + cloud_nurbs_input->points[pointIdxNKNSearch[2]].z)/3;
            }   
            double diff_x = cloud->points[i].x - _point.x;
            double diff_y = cloud->points[i].y - _point.y;
            double diff_z = cloud->points[i].z - _point.z;
            double diff = std::sqrt(diff_x*diff_x  + diff_y*diff_y + diff_z*diff_z);
            if(diff < 0.03){
                inliers->indices.push_back(i);
            }
        }
    }
*/

/*
    kdtree_nurbs.setInputCloud (cloud);
    inliers->indices.clear();
    for(int i=0;i<cloud_nurbs_input->points.size();i++)
    {
        if ( kdtree_nurbs.nearestKSearch (cloud_nurbs_input->points[i], K, pointIdxNKNSearch, pointNKNSquaredDistance) > 0 )
        {
            std::cout << pointIdxNKNSearch[0] << std::endl;
            if(pointNKNSquaredDistance[0] < 0.05){
                inliers->indices.push_back(i);
            }
        }
    }
    */
//    std::cout << "poitn" << inliers->indices.size() << std::endl;

/*
    unsigned order (3);
    unsigned refinement (3);
    unsigned iterations (3);
    unsigned mesh_resolution (1024);
    pcl::on_nurbs::FittingSurface::Parameter params;
    params.interior_smoothness = 0.2;
    params.interior_weight = 1.0;
    params.boundary_smoothness = 0.2;
    params.boundary_weight = 0.0;
    pcl::on_nurbs::NurbsDataSurface data;
    PointCloud2Vector3d (cloud_nurbs, data.interior);
    ON_NurbsSurface nurbs = pcl::on_nurbs::FittingSurface::initNurbsPCABoundingBox (order, &data);
    pcl::on_nurbs::FittingSurface fit (&data, nurbs);
    fit.refine (0);
    fit.refine (1);
    fit.assemble (params);
    fit.solve ();

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_nurbs_trans(new pcl::PointCloud<pcl::PointXYZ>);
    Vector3d2PointCloud (cloud_nurbs_trans, data.interior);

    int K = 3;
    std::vector<int> pointIdxNKNSearch(K);
    std::vector<float> pointNKNSquaredDistance(K);
    pcl::KdTreeFLANN<pcl::PointXYZ> kdtree_nurbs;
    kdtree_nurbs.setInputCloud (cloud_nurbs_trans);

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_nurbs_out(new pcl::PointCloud<pcl::PointXYZ>);
    inliers->indices.clear();
    for(int i=0;i<cloud->points.size();i++)
    {
        if ( kdtree_nurbs.nearestKSearch (cloud->points[i], K, pointIdxNKNSearch, pointNKNSquaredDistance) > 0 )
        {
            pcl::PointXYZ _point;
            if(pointIdxNKNSearch.size() == 3){
                _point.x = (cloud_nurbs_trans->points[pointIdxNKNSearch[0]].x + cloud_nurbs_trans->points[pointIdxNKNSearch[1]].x + cloud_nurbs_trans->points[pointIdxNKNSearch[2]].x)/3;
                _point.y = (cloud_nurbs_trans->points[pointIdxNKNSearch[0]].y + cloud_nurbs_trans->points[pointIdxNKNSearch[1]].y + cloud_nurbs_trans->points[pointIdxNKNSearch[2]].y)/3;
                _point.z = (cloud_nurbs_trans->points[pointIdxNKNSearch[0]].z + cloud_nurbs_trans->points[pointIdxNKNSearch[1]].z + cloud_nurbs_trans->points[pointIdxNKNSearch[2]].z)/3;
            }
            
            double diff_x = cloud->points[i].x - _point.x;
            double diff_y = cloud->points[i].y - _point.y;
            double diff_z = cloud->points[i].z - _point.z;
            double diff = std::sqrt(diff_x*diff_x  + diff_y*diff_y + diff_z*diff_z);
            if(diff < 0.05){
                inliers->indices.push_back(i);
            }
            //inliers->indices.push_back(inliers_plane[i]);
        }
    }

    //pcl::SampleConsensusModelPlane<pcl::PointXYZ>::Ptr model_p (new pcl::SampleConsensusModelPlane<pcl::PointXYZ> (cloud));
    //model_p->selectWithinDistance 	(model_coefficients, 0.05, inliers_plane) ; //  5cm-range extraction from ground plane
    //inliers->indices.clear();
    //for(int i = 0;i<inliers_plane.size();i++){
    //    inliers->indices.push_back(inliers_plane[i]);
    //}

    end = std::chrono::system_clock::now();  // 計測終了時間
    double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count(); //処理に要した時間をミリ秒に変換
    std::cout << "elapsed;" << elapsed << std::endl;
    std::cout << "cloud->points" << cloud->points.size() << std::endl;
*/
/*
    for(int i=0;i<cloud_vg->points.size();i++){
    Eigen::Vector3d p0;
    p0(0) = cloud->points[i].x;
    p0(1) = cloud->points[i].y;
    p0(2) = cloud->points[i].z;
    Eigen::Vector3d p1, tu1, tv1, p2, tu2, tv2, t1, t2;
    Eigen::Vector2d params1, params2;
    double error1, error2;
    params1 = pcl::on_nurbs::FittingSurface::findClosestElementMidPoint (nurbs, p0);
    params1 = pcl::on_nurbs::FittingSurface::inverseMapping (nurbs, p0, params1, error1, p1, tu1, tv1);
    }
    // 処理
*/

}

void
GroundFittingManager::setGroundPointCloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_ground_input){
    pcl::PointCloud<pcl::PointXYZ>::Ptr _cloud_ground(new pcl::PointCloud<pcl::PointXYZ>);

    pcl::copyPointCloud(*cloud_ground_input, *_cloud_ground);

    pcl::PointIndices::Ptr inliers (new pcl::PointIndices);
    //ground point extraction using parameters of fitted plane
    std::vector<int> inliers_plane;
    Eigen::VectorXf model_coefficients(4);
    for(int i = 0;i<4;i++){
        model_coefficients(i) = coefficients.values[i];
    }

    pcl::SampleConsensusModelPlane<pcl::PointXYZ>::Ptr model_p (new pcl::SampleConsensusModelPlane<pcl::PointXYZ> (_cloud_ground));
    model_p->selectWithinDistance 	(model_coefficients, 0.05, inliers_plane) ; //  3cm-range extraction from ground plane
    inliers->indices.clear();
    for(int i = 0;i<inliers_plane.size();i++){
        inliers->indices.push_back(inliers_plane[i]);
    }
    pcl::ExtractIndices<pcl::PointXYZ> eifilter; 
    eifilter.setInputCloud (_cloud_ground);
    eifilter.setIndices (inliers);
    eifilter.setNegative (false);
    eifilter.filter (cloud_ground);
}

void
GroundFittingManager::setLineCoeffs(const double &_x_0, const double &_y_0, const double &_z_0, const double &_l_x, const double &_l_y, const double &_l_z){
    x_0 = _x_0;
    y_0 = _y_0;
    z_0 = _z_0;
    line_coeff_x = _l_x;
    line_coeff_y = _l_y;
    line_coeff_z = _l_z;
}

void
GroundFittingManager::deriveIntersection(){
    double nx = coefficients.values[0];
    double ny = coefficients.values[1];
    double nz = coefficients.values[2];

    double coeff_plane = coefficients.values[3];


    double t_denominator = nx*line_coeff_x + ny*line_coeff_y + nz*line_coeff_z + 0.000000001; //avoid zero devision
    double t_enumerator = nx*x_0 + ny*y_0 + nz*z_0 +coeff_plane;
    double t = t_enumerator/t_denominator;
    intersection_point.x =  x_0 - t*line_coeff_x;
    intersection_point.y =  y_0 - t*line_coeff_y;
    intersection_point.z =  z_0 - t*line_coeff_z;
}

void
GroundFittingManager::getIntersectionPoint(pcl::PointXYZ &_intersection_point){
    _intersection_point.x = intersection_point.x;
    _intersection_point.y = intersection_point.y;
    _intersection_point.z = intersection_point.z;
}

void
GroundFittingManager::fitLocalPlane()
{
    int n_pc = cloud_ground.points.size();

    std::vector<int> pointIdxRadiusSearch;
    std::vector<float> pointRadiusSquaredDistance;

    pcl::PointCloud<pcl::PointXYZ>::Ptr ground_cloud_local(new pcl::PointCloud<pcl::PointXYZ>);

    pcl::PointIndices::Ptr inliers (new pcl::PointIndices);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_global_ptr (new pcl::PointCloud<pcl::PointXYZ>);
    pcl::copyPointCloud(cloud_global, *cloud_global_ptr);
    pcl::SACSegmentation<pcl::PointXYZ> seg; 

    double initial_radius = 0.05;
    double search_radius = initial_radius;
    int local_ground_point_num = 100;
    bool found_local_ground = false;
    bool flag_fitting_correctly_finished = true;

    while(!found_local_ground){
        if(kdtree_ground.radiusSearch (intersection_point, 0.05, pointIdxRadiusSearch, pointRadiusSquaredDistance) > 0){
            if(pointIdxRadiusSearch.size() >= local_ground_point_num) {
                found_local_ground = true;
            }else{
                found_local_ground = false;
                search_radius *= 1.5;
            }
        }else{
            found_local_ground = false;
            search_radius *= 1.5;
        }
        if(search_radius > 1.0){ //in the case of local point were not found
            flag_fitting_correctly_finished = false;
            break; 
        }
    }

    if(flag_fitting_correctly_finished){
        for(auto it=pointIdxRadiusSearch.begin(), it_end=pointIdxRadiusSearch.end();it!=it_end;++it){
            ground_cloud_local->points.push_back(cloud_ground.points[*it]);
        }

        seg.setOptimizeCoefficients (true);
        seg.setModelType (pcl::SACMODEL_PLANE);
        seg.setMethodType (pcl::SAC_PROSAC);
        seg.setDistanceThreshold (0.02); 
        seg.setMaxIterations(20000);
        seg.setInputCloud (ground_cloud_local);
        seg.segment (*inliers, local_plane_coefficients);

        if(inliers->indices.size() == 0){
            flag_fitting_correctly_finished = false;
        }
    }

    if(!flag_fitting_correctly_finished){
        seg.setOptimizeCoefficients (true);
        seg.setModelType (pcl::SACMODEL_PLANE);
        seg.setMethodType (pcl::SAC_PROSAC);
        seg.setDistanceThreshold (0.02); //10cm 
        seg.setMaxIterations(20000);
        seg.setInputCloud (cloud_global_ptr);
        seg.segment (*inliers, local_plane_coefficients);
    }
}

void
GroundFittingManager::setKDTree()
{
    kdtree_ground.setInputCloud (boost::make_shared<pcl::PointCloud<pcl::PointXYZ>>(cloud_ground));
}

pcl::ModelCoefficients
GroundFittingManager::getLocalPlaneCoeff(){
    return local_plane_coefficients;
}

pcl::ModelCoefficients
GroundFittingManager::getPlaneCoeff(){
    return coefficients;
}


void
GroundFittingManager::deriveIntersectionLocalView(){
    double nx = local_plane_coefficients.values[0];
    double ny = local_plane_coefficients.values[1];
    double nz = local_plane_coefficients.values[2];
    double coeff_plane = local_plane_coefficients.values[3];

    double t_denominator = nx*line_coeff_x + ny*line_coeff_y + nz*line_coeff_z + 0.000000001;
    double t_enumerator = nx*x_0 + ny*y_0 + nz*z_0 +coeff_plane;
    double t = t_enumerator/t_denominator;
    intersection_point_local.x =  x_0 - t*line_coeff_x;
    intersection_point_local.y =  y_0 - t*line_coeff_y;
    intersection_point_local.z =  z_0 - t*line_coeff_z;
}

void
GroundFittingManager::getIntersectionPointLocalView(pcl::PointXYZ &_intersection_point){
    _intersection_point.x  = intersection_point_local.x;
    _intersection_point.y  = intersection_point_local.y;
    _intersection_point.z  = intersection_point_local.z;
}


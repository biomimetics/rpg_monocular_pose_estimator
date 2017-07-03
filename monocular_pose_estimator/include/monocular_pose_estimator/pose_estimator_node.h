#ifndef POSE_ESTIMATOR_NODE_H
#define POSE_ESTIMATOR_NODE_H

#include "ros/ros.h"

#include <sensor_msgs/CameraInfo.h>

#include <geometry_msgs/PoseWithCovarianceStamped.h>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <opencv2/opencv.hpp>
#include <boost/thread/mutex.hpp>

#include <dynamic_reconfigure/server.h>
#include <monocular_pose_estimator/MonocularPoseEstimatorConfig.h>

#include <monocular_pose_estimator/BlobList.h>
#include "monocular_pose_estimator_lib/pose_estimator.h"

#include <exploration/MarkerAggregate.h>
#include <exploration/MarkerStateStamped.h>
#include <exploration/MarkerPositionState.h>
#include <geometry_msgs/Point.h>
#include <std_msgs/Header.h>
#include <geometry_msgs/Vector3.h>

namespace monocular_pose_estimator {

class PoseEstimatorNode {
  private:
    ros::NodeHandle nh_;
    ros::NodeHandle nh_private_;

    ros::Subscriber blob_list_sub_;
    ros::Subscriber active_markers_sub_;
    ros::Subscriber camera_info_sub_;

    ros::Publisher pose_pub_;

    dynamic_reconfigure::Server<monocular_pose_estimator::MonocularPoseEstimatorConfig> dr_server_;

    PoseEstimator trackable_object_;
    
    std::vector<exploration::MarkerStateStamped> active_markers_;

    boost::mutex mutex_;
    
    bool have_camera_info_;

  public:
    PoseEstimatorNode(const ros::NodeHandle& nh, const ros::NodeHandle& nh_private);
    PoseEstimatorNode() : PoseEstimatorNode( ros::NodeHandle(), ros::NodeHandle("~") ) {}
    ~PoseEstimatorNode();
    
    void dynamicParametersCallback(monocular_pose_estimator::MonocularPoseEstimatorConfig &config, uint32_t level);

    int state_to_hue(std::vector<uint8_t> state);
    
    void cameraInfoCallback(const sensor_msgs::CameraInfo::ConstPtr& msg);

    void blobListCallback(const monocular_pose_estimator::BlobList::ConstPtr& blob_list_msg);

    void activeMarkersCallback(const exploration::MarkerAggregate::ConstPtr& marker_aggregate_msg);

};

}

#endif

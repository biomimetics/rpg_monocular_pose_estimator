#ifndef VISUALIZER_H
#define VISUALIZER_H

#include "ros/ros.h"

#include <sensor_msgs/Image.h>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/image_encodings.h>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>

#include "monocular_pose_estimator_lib/visualization.h"
#include "monocular_pose_estimator/BlobList.h"

#include <boost/thread/mutex.hpp>

namespace monocular_pose_estimator {

class VisualizationNode {
  private:
    ros::NodeHandle nh_;
    ros::NodeHandle nh_private_;

    ros::Subscriber image_sub_;
    ros::Subscriber camera_info_sub_;
    ros::Subscriber distorted_blob_list_sub_;

    image_transport::Publisher image_pub_;

    bool have_camera_info_;
    sensor_msgs::CameraInfo cam_info_;

    cv::Mat camera_matrix_K_; //!< Variable to store the camera matrix as an OpenCV matrix
    std::vector<double> camera_distortion_coeffs_;
    
    cv::Mat last_image_;
    std::vector<geometry_msgs::Vector3> last_blobs_;
    boost::mutex mutex_;

  public:
    VisualizationNode(const ros::NodeHandle& nh, const ros::NodeHandle& nh_private);
    VisualizationNode() : VisualizationNode(ros::NodeHandle(), ros::NodeHandle("~")){}
    ~VisualizationNode();

    void cameraInfoCallback(const sensor_msgs::CameraInfo::ConstPtr& msg);

    void imageCallback(const sensor_msgs::Image::ConstPtr& image_msg);

    void distortedBlobListCallback(const monocular_pose_estimator::BlobList::ConstPtr& blob_list_msg);
};

}

#endif

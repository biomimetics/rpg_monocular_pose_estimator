#ifndef BLOB_FILTER_H
#define BLOB_FILTER_H

#include "ros/ros.h"

#include <sensor_msgs/Image.h>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/image_encodings.h>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>

#include <dynamic_reconfigure/server.h>
#include <monocular_pose_estimator/MonocularPoseEstimatorConfig.h>

#include "monocular_pose_estimator_lib/led_detector.h"
#include "monocular_pose_estimator/BlobList.h"

namespace monocular_pose_estimator {

class BlobFilterNode {
  private:
    ros::NodeHandle nh_;
    ros::NodeHandle nh_private_;

    ros::Subscriber image_sub_;
    ros::Subscriber camera_info_sub_;
    
    image_transport::Publisher image_pub_;

    ros::Publisher blob_list_pub_, distorted_blob_list_pub_;

    dynamic_reconfigure::Server<monocular_pose_estimator::MonocularPoseEstimatorConfig> dr_server_;
    bool have_camera_info_;
    sensor_msgs::CameraInfo cam_info_;
    
    cv::Mat camera_matrix_K_; //!< Variable to store the camera matrix as an OpenCV matrix
  std::vector<double> camera_distortion_coeffs_; //!< Variable to store the camera distortion parameters

  int detection_threshold_value_; //!< The current threshold value for the image for LED detection
  double gaussian_sigma_; //!< The current standard deviation of the Gaussian that will be applied to the thresholded image for LED detection
  double min_blob_area_; //!< The the minimum blob area (in pixels squared) that will be detected as a blob/LED. Areas having an area smaller than this will not be detected as LEDs.
  double max_blob_area_; //!< The the maximum blob area (in pixels squared) that will be detected as a blob/LED. Areas having an area larger than this will not be detected as LEDs.
  double max_width_height_distortion_; //!< This is a parameter related to the circular distortion of the detected blobs. It is the maximum allowable distortion of a bounding box around the detected blob calculated as the ratio of the width to the height of the bounding rectangle. Ideally the ratio of the width to the height of the bounding rectangle should be 1.
  double max_circular_distortion_; //!< This is a parameter related to the circular distortion of the detected blobs. It is the maximum allowable distortion of a bounding box around the detected blob, calculated as the area of the blob divided by pi times half the height or half the width of the bounding rectangle.
  unsigned roi_border_thickness_; //!< This is the thickness of the boarder (in pixels) around the predicted area of the LEDs in the image that defines the region of interest for image processing and detection of the LEDs.
  public:
    BlobFilterNode(const ros::NodeHandle& nh, const ros::NodeHandle& nh_private);
    BlobFilterNode() : BlobFilterNode(ros::NodeHandle(), ros::NodeHandle("~")){}
    ~BlobFilterNode();

    void cameraInfoCallback(const sensor_msgs::CameraInfo::ConstPtr& msg);

    void imageCallback(const sensor_msgs::Image::ConstPtr& image_msg);

    void dynamicParametersCallback(monocular_pose_estimator::MonocularPoseEstimatorConfig &config, uint32_t level);
};

}

#endif

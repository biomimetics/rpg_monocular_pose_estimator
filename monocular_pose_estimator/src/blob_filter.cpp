#include "monocular_pose_estimator/blob_filter.h"

int main(int argc, char* argv[]) {
  ros::init(argc, argv, "blob_filter");

  monocular_pose_estimator::BlobFilterNode blob_filter_node;

  ros::spin();

  return 0;
}

namespace monocular_pose_estimator
{

BlobFilterNode::BlobFilterNode(const ros::NodeHandle& nh, const ros::NodeHandle& nh_private)
  : nh_(nh), nh_private_(nh_private), have_camera_info_(false) {
  // Set up a dynamic reconfigure server.
  // This should be done before reading parameter server values.
  dynamic_reconfigure::Server<monocular_pose_estimator::MonocularPoseEstimatorConfig>::CallbackType cb_;
  cb_ = boost::bind(&BlobFilterNode::dynamicParametersCallback, this, _1, _2);
  dr_server_.setCallback(cb_);

  image_sub_ = nh_.subscribe("image_raw", 1, &BlobFilterNode::imageCallback, this);
  camera_info_sub_ = nh_.subscribe("camera_info", 1, &BlobFilterNode::cameraInfoCallback, this);
  
  // Initialize image publisher for visualization
  image_transport::ImageTransport image_transport(nh_);
  image_pub_ = image_transport.advertise("threshold_image", 1);

  blob_list_pub_ = nh_.advertise<monocular_pose_estimator::BlobList>("blob_list", 1);
  distorted_blob_list_pub_ = nh_.advertise<monocular_pose_estimator::BlobList>("distorted_blob_list", 1);
}

BlobFilterNode::~BlobFilterNode() {}

void BlobFilterNode::cameraInfoCallback(const sensor_msgs::CameraInfo::ConstPtr& msg) {
  if (!have_camera_info_) {
    cam_info_ = *msg;

    // Calibrated camera
    camera_matrix_K_ = cv::Mat(3, 3, CV_64F);
    camera_matrix_K_.at<double>(0, 0) = cam_info_.K[0];
    camera_matrix_K_.at<double>(0, 1) = cam_info_.K[1];
    camera_matrix_K_.at<double>(0, 2) = cam_info_.K[2];
    camera_matrix_K_.at<double>(1, 0) = cam_info_.K[3];
    camera_matrix_K_.at<double>(1, 1) = cam_info_.K[4];
    camera_matrix_K_.at<double>(1, 2) = cam_info_.K[5];
    camera_matrix_K_.at<double>(2, 0) = cam_info_.K[6];
    camera_matrix_K_.at<double>(2, 1) = cam_info_.K[7];
    camera_matrix_K_.at<double>(2, 2) = cam_info_.K[8];
    camera_distortion_coeffs_ = cam_info_.D;

    have_camera_info_ = true;
    ROS_INFO("Blob Filter camera calibration information obtained.");
  }
}

void BlobFilterNode::imageCallback(const sensor_msgs::Image::ConstPtr& image_msg) {
  // Check whether already received the camera calibration data
  if (!have_camera_info_) {
    ROS_WARN("Blob Filter no camera info yet...");
    return;
  }

  // Import the image from ROS message to OpenCV mat
  cv_bridge::CvImagePtr cv_ptr;
  try {
    cv_ptr = cv_bridge::toCvCopy(image_msg, sensor_msgs::image_encodings::BGR8);
  }
  catch (cv_bridge::Exception& e) {
    ROS_ERROR("Blob Filter cv_bridge exception: %s", e.what());
    return;
  }
  cv::Mat image = cv_ptr->image;

  cv::Rect region_of_interest;
  region_of_interest = cv::Rect(0, 0, image.cols, image.rows);
  List2DPoints detected_led_positions;
  std::vector<cv::Point2f> distorted_detection_centers;
  cv::Mat output_image;
  std::vector<int> blob_hues;

  LEDDetector::findColorLeds(
    image, region_of_interest, detection_threshold_value_, 
    gaussian_sigma_, min_blob_area_, max_blob_area_, 
    max_width_height_distortion_, max_circular_distortion_,
    detected_led_positions, distorted_detection_centers, camera_matrix_K_,
    camera_distortion_coeffs_, output_image, blob_hues);

  monocular_pose_estimator::BlobList blob_list, distorted_blob_list;

  blob_list.header.stamp = image_msg->header.stamp;
  blob_list.header.frame_id = image_msg->header.frame_id;
  distorted_blob_list.header.stamp = image_msg->header.stamp;
  distorted_blob_list.header.frame_id = image_msg->header.frame_id;

  for (unsigned i = 0; i < detected_led_positions.size(); i++) {
    geometry_msgs::Vector3 vec;
    vec.x = detected_led_positions[i](0);
    vec.y = detected_led_positions[i](1);
    vec.z = blob_hues[i];
    blob_list.blobs.push_back(vec);

    geometry_msgs::Vector3 dist_vec;
    dist_vec.x = distorted_detection_centers[i].x;
    dist_vec.y = distorted_detection_centers[i].y;
    dist_vec.z = blob_hues[i];
    distorted_blob_list.blobs.push_back(dist_vec);
  }
  
  blob_list_pub_.publish(blob_list);
  distorted_blob_list_pub_.publish(distorted_blob_list);

  if (image_pub_.getNumSubscribers() > 0) {
    cv_bridge::CvImage output_image_msg;
    output_image_msg.header = image_msg->header;
    output_image_msg.encoding = sensor_msgs::image_encodings::MONO8;
    output_image_msg.image = output_image;
    image_pub_.publish(output_image_msg.toImageMsg());
  }
}

void BlobFilterNode::dynamicParametersCallback(monocular_pose_estimator::MonocularPoseEstimatorConfig &config, uint32_t level) {
  detection_threshold_value_ = config.threshold_value;
  gaussian_sigma_ = config.gaussian_sigma;
  min_blob_area_ = config.min_blob_area;
  max_blob_area_ = config.max_blob_area;
  max_width_height_distortion_ = config.max_width_height_distortion;
  max_circular_distortion_ = config.max_circular_distortion;
  roi_border_thickness_ = config.roi_border_thickness;

  ROS_INFO("Blob Filter parameters changed");
}

}


#include "monocular_pose_estimator/visualization_node.h"

int main(int argc, char* argv[]) {
  ros::init(argc, argv, "visualization_node");

  monocular_pose_estimator::VisualizationNode visualization_node;

  ros::spin();

  return 0;
}

namespace monocular_pose_estimator {

VisualizationNode::VisualizationNode(const ros::NodeHandle& nh, const ros::NodeHandle& nh_private)
  : nh_(nh), nh_private_(nh_private), have_camera_info_(false) {

  image_sub_ = nh_.subscribe("image_raw", 1, &VisualizationNode::imageCallback, this);
  camera_info_sub_ = nh_.subscribe("camera_info", 1, &VisualizationNode::cameraInfoCallback, this);
  distorted_blob_list_sub_ = nh_.subscribe("distorted_blob_list", 1, &VisualizationNode::distortedBlobListCallback, this);
  estimated_pose_sub_ = nh_.subscribe("estimated_pose", 1, &VisualizationNode::estimatedPoseCallback, this);
  correspondence_list_sub_ = nh_.subscribe("correspondence_list", 1, &VisualizationNode::correspondenceListCallback, this);

  // Initialize image publisher for visualization
  image_transport::ImageTransport image_transport(nh_);
  image_pub_ = image_transport.advertise("augmented_image", 1);
}

VisualizationNode::~VisualizationNode() {}

void VisualizationNode::cameraInfoCallback(const sensor_msgs::CameraInfo::ConstPtr& msg) {
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
    ROS_INFO("Visualization Node camera calibration information obtained.");
  }
}

void VisualizationNode::imageCallback(const sensor_msgs::Image::ConstPtr& image_msg) {
  // Check whether already received the camera calibration data
  if (!have_camera_info_) {
    ROS_WARN("Visualization Node no camera info yet...");
    return;
  }

  // Import the image from ROS message to OpenCV mat
  cv_bridge::CvImagePtr cv_ptr;
  try {
    cv_ptr = cv_bridge::toCvCopy(image_msg, sensor_msgs::image_encodings::BGR8);
  }
  catch (cv_bridge::Exception& e) {
    ROS_ERROR("Visualization Node cv_bridge exception: %s", e.what());
    return;
  }

  if (!last_image_.empty()) {
    cv::Rect region_of_interest;
    region_of_interest = cv::Rect(0, 0, last_image_.cols, last_image_.rows);
    
    std::vector<cv::Point2f> distorted_detection_centers;
    std::vector<int> blob_hues;
    
    std::vector<std::string> pose_frames;
    std::vector<Eigen::Matrix4d> pose_transforms;

    mutex_.lock();

    for (unsigned i = 0; i < last_blobs_.size(); i++) {
      cv::Point2f pt;
      pt.x = last_blobs_[i].x;
      pt.y = last_blobs_[i].y;
      distorted_detection_centers.push_back(pt);
      blob_hues.push_back(last_blobs_[i].z);
    }

    //correspondence_lists_.clear();
    for (unsigned i = 0; i < estimated_poses_.size(); i++) {
      //ROS_INFO("Visualization got Pose %s", estimated_poses_[i].header.frame_id.c_str());
      
      std::string frame = estimated_poses_[i].header.frame_id;
      pose_frames.push_back(frame);

      double x = estimated_poses_[i].pose.pose.position.x;
      double y = estimated_poses_[i].pose.pose.position.y;
      double z = estimated_poses_[i].pose.pose.position.z;
      double qx = estimated_poses_[i].pose.pose.orientation.x;
      double qy = estimated_poses_[i].pose.pose.orientation.y;
      double qz = estimated_poses_[i].pose.pose.orientation.z;
      double qw = estimated_poses_[i].pose.pose.orientation.w;
      Eigen::Matrix4d transform;
      transform(0,0) = 1 - 2*qy*qy - 2*qz*qz;
      transform(0,1) = 2*qx*qy - 2*qz*qw;
      transform(0,2) = 2*qx*qz + 2*qy*qw;
      transform(0,3) = x;
      transform(1,0) = 2*qx*qy + 2*qz*qw;
      transform(1,1) = 1 - 2*qx*qx - 2*qz*qz;
      transform(1,2) = 2*qy*qz - 2*qx*qw;
      transform(1,3) = y;
      transform(2,0) = 2*qx*qz - 2*qy*qw;
      transform(2,1) = 2*qy*qz + 2*qx*qw;
      transform(2,2) = 1 - 2*qx*qx - 2*qy*qy;
      transform(2,3) = z;
      transform(3,3) = 1;
      pose_transforms.push_back(transform);
    }

    estimated_poses_.clear();

    mutex_.unlock();

    bool found_body_pose = false;
    Eigen::Matrix4d transform;

    Visualization::createVisualizationImage(last_image_, found_body_pose, transform, camera_matrix_K_, camera_distortion_coeffs_, region_of_interest, distorted_detection_centers, blob_hues);

    for (unsigned i = 0; i < pose_frames.size(); i++) {
      Visualization::addPoseToImage(last_image_, pose_transforms[i], camera_matrix_K_, camera_distortion_coeffs_);
    }

    if (image_pub_.getNumSubscribers() > 0) {
      cv_bridge::CvImage output_image_msg;
      output_image_msg.header = image_msg->header;
      output_image_msg.encoding = sensor_msgs::image_encodings::BGR8;
      output_image_msg.image = last_image_;
      image_pub_.publish(output_image_msg.toImageMsg());
    }
  }
  
  last_image_ = cv_ptr->image;
}

void VisualizationNode::distortedBlobListCallback(const monocular_pose_estimator::BlobList::ConstPtr& blob_list_msg) {
  mutex_.lock();
  last_blobs_ = blob_list_msg->blobs;
  mutex_.unlock();
}

void VisualizationNode::estimatedPoseCallback(const geometry_msgs::PoseWithCovarianceStamped::ConstPtr& pose_msg) {
  mutex_.lock();
  //geometry_msgs::PoseWithCovarianceStamped copy_msg = *pose_msg;
  estimated_poses_.push_back(*pose_msg);
  mutex_.unlock();
}

void VisualizationNode::correspondenceListCallback(const monocular_pose_estimator::BlobList::ConstPtr& correspondence_list_msg) {
  mutex_.lock();
  //correspondence_lists_.push_back(correspondence_list_msg);
  mutex_.unlock();
}

}

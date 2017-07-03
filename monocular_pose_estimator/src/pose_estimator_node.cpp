#include "monocular_pose_estimator/pose_estimator_node.h"

int main(int argc, char* argv[]) {
  ros::init(argc, argv, "pose_estimator_node");

  monocular_pose_estimator::PoseEstimatorNode pose_estimator_node;

  ros::spin();

  return 0;
}

namespace monocular_pose_estimator {

PoseEstimatorNode::PoseEstimatorNode(const ros::NodeHandle& nh, const ros::NodeHandle& nh_private)
  : nh_(nh), nh_private_(nh_private), have_camera_info_(false) {
  // Set up a dynamic reconfigure server.
  // This should be done before reading parameter server values.
  dynamic_reconfigure::Server<monocular_pose_estimator::MonocularPoseEstimatorConfig>::CallbackType cb_;
  cb_ = boost::bind(&PoseEstimatorNode::dynamicParametersCallback, this, _1, _2);
  dr_server_.setCallback(cb_);
  
  camera_info_sub_ = nh_.subscribe("camera_info", 1, &PoseEstimatorNode::cameraInfoCallback, this);

  blob_list_sub_ = nh_.subscribe("blob_list", 1, &PoseEstimatorNode::blobListCallback, this);
  active_markers_sub_ = nh_.subscribe("/active_markers", 1, &PoseEstimatorNode::activeMarkersCallback, this);

  pose_pub_ = nh_.advertise<geometry_msgs::PoseWithCovarianceStamped>("estimated_pose", 1);
}

PoseEstimatorNode::~PoseEstimatorNode() {}

void PoseEstimatorNode::dynamicParametersCallback(monocular_pose_estimator::MonocularPoseEstimatorConfig &config, uint32_t level) {
  trackable_object_.setBackProjectionPixelTolerance(config.back_projection_pixel_tolerance);
  trackable_object_.setNearestNeighbourPixelTolerance(config.nearest_neighbour_pixel_tolerance);
  trackable_object_.setCertaintyThreshold(config.certainty_threshold);
  trackable_object_.setValidCorrespondenceThreshold(config.valid_correspondence_threshold);

  ROS_INFO("Pose Estimator parameters changed");
}

void PoseEstimatorNode::cameraInfoCallback(const sensor_msgs::CameraInfo::ConstPtr& msg) {
  if (!have_camera_info_) {
    // Calibrated camera
    trackable_object_.camera_matrix_K_ = cv::Mat(3, 3, CV_64F);
    trackable_object_.camera_matrix_K_.at<double>(0, 0) = msg->K[0];
    trackable_object_.camera_matrix_K_.at<double>(0, 1) = msg->K[1];
    trackable_object_.camera_matrix_K_.at<double>(0, 2) = msg->K[2];
    trackable_object_.camera_matrix_K_.at<double>(1, 0) = msg->K[3];
    trackable_object_.camera_matrix_K_.at<double>(1, 1) = msg->K[4];
    trackable_object_.camera_matrix_K_.at<double>(1, 2) = msg->K[5];
    trackable_object_.camera_matrix_K_.at<double>(2, 0) = msg->K[6];
    trackable_object_.camera_matrix_K_.at<double>(2, 1) = msg->K[7];
    trackable_object_.camera_matrix_K_.at<double>(2, 2) = msg->K[8];
    trackable_object_.camera_distortion_coeffs_ = msg->D;
    have_camera_info_ = true;
    ROS_INFO("Pose Estimator Camera calibration information obtained.");
  }
}

// convert RGB marker state to 0-180 hue integer
int PoseEstimatorNode::state_to_hue(std::vector<uint8_t> state) {
  cv::Mat bgr_state(1, 1, CV_8UC3, cv::Scalar(state[2],state[1],state[0]));
  cv::Mat hsv_state;
  cv::cvtColor(bgr_state, hsv_state, cv::COLOR_BGR2HSV);
  std::vector<cv::Mat> hsv_channels(3);
  cv::split(hsv_state, hsv_channels);
  return hsv_channels[0].at<uint8_t>(0,0);
}

void PoseEstimatorNode::blobListCallback(const monocular_pose_estimator::BlobList::ConstPtr& blob_list_msg) {
  if (!have_camera_info_) {
    ROS_WARN("Pose Estimator no camera info");
    return;
  }// else {
  //  ROS_INFO("Camera Matrix");
  //  std::cout << trackable_object_.camera_matrix_K_ << "\n";
  //}

  std::vector<geometry_msgs::Vector3> blobs;
  blobs = blob_list_msg->blobs;
  
  //ROS_INFO("Pose estimator tracker params");
  //ROS_INFO("%f",trackable_object_.getBackProjectionPixelTolerance());
  //ROS_INFO("%f",trackable_object_.getNearestNeighbourPixelTolerance());
  //ROS_INFO("%f",trackable_object_.getCertaintyThreshold());
  //ROS_INFO("%f",trackable_object_.getValidCorrespondenceThreshold());

  //ROS_INFO("Blobs %d:", (int)blobs.size());
  // Only run pose estimation if at least 4 blobs were detected
  if ( blobs.size() >= 4 ) {  
    // Set detected_led_positions and blob_hues array from blob_list_msg
    List2DPoints detected_led_positions;
    detected_led_positions.resize(blobs.size());
    std::vector<int> blob_hues;

    for (unsigned b = 0; b < blobs.size(); b++) {
      Eigen::Vector2d pt;
      pt(0) = blobs[b].x;
      pt(1) = blobs[b].y;
      detected_led_positions(b) = pt;
      //std::cout << pt << ", ";
      blob_hues.push_back((int)blobs[b].z);
      //std::cout << blob_hues[b] << ",";
    }
    
    //std::cout << "\n";

    trackable_object_.setBlobHues(blob_hues);
    trackable_object_.setImagePoints(detected_led_positions);

    // Copy active_markers to sets of frames, positions, and hues of viewed robot markers
    std::vector<std::string> marker_frames;
    std::vector<std::vector<int>> marker_hues;
    std::vector<List4DPoints> marker_positions;
    mutex_.lock();
    for (unsigned m = 0; m < active_markers_.size(); m++) {
      marker_frames.push_back(active_markers_[m].header.frame_id);
      List4DPoints positions;
      positions.resize(active_markers_[m].marker_states.size());
      std::vector<int> hues;
      for (unsigned p = 0; p < active_markers_[m].marker_states.size(); p++) {
        Eigen::Matrix<double, 4, 1> point;
        point(0) = active_markers_[m].marker_states[p].position.x;
        point(1) = active_markers_[m].marker_states[p].position.y;
        point(2) = active_markers_[m].marker_states[p].position.z;
        point(3) = 1;
        positions(p) = point;
        hues.push_back(state_to_hue(active_markers_[m].marker_states[p].state));
      }
      marker_positions.push_back(positions);
      marker_hues.push_back(hues);
    }
    mutex_.unlock();

    // Calculate pose for each set of active markers
    for (unsigned m = 0; m < marker_hues.size(); m++) {
      trackable_object_.setMarkerHues(marker_hues[m]);
      trackable_object_.setMarkerPositions(marker_positions[m]);

      // A return value of 1 from initailiseWithHues indicates there was
      // a valid correspondence found, and the pose can be estimated
      //trackable_object_.findCorrespondences();
      //trackable_object_.printCorrespondences();
      int init_val = trackable_object_.initialiseWithHues();
      //ROS_INFO("Initialise result: %d", init_val);

      //trackable_object_.printCorrespondences();

      if ( init_val == 1) {
        // The time to predict doesn't actually do anything since we are brute force
        // initializing with hues
        double time_to_predict = blob_list_msg->header.stamp.toSec();
        trackable_object_.optimiseAndUpdatePose(time_to_predict);
        
        geometry_msgs::PoseWithCovarianceStamped predicted_pose;

        // Set timestamp and frame ID as camera_frame:marker_frame
        predicted_pose.header.stamp = blob_list_msg->header.stamp;
        predicted_pose.header.frame_id = blob_list_msg->header.frame_id + ":" + marker_frames[m];

        // Set position and orientation 
        Eigen::Matrix4d transform = trackable_object_.getPredictedPose();
        predicted_pose.pose.pose.position.x = transform(0, 3);
        predicted_pose.pose.pose.position.y = transform(1, 3);
        predicted_pose.pose.pose.position.z = transform(2, 3);
        Eigen::Quaterniond orientation = Eigen::Quaterniond(transform.block<3, 3>(0, 0));
        predicted_pose.pose.pose.orientation.x = orientation.x();
        predicted_pose.pose.pose.orientation.y = orientation.y();
        predicted_pose.pose.pose.orientation.z = orientation.z();
        predicted_pose.pose.pose.orientation.w = orientation.w();
        
        // Set covariance
        Matrix6d cov = trackable_object_.getPoseCovariance();
        for (unsigned i = 0; i < 6; ++i) {
          for (unsigned j = 0; j < 6; ++j) {
            predicted_pose.pose.covariance.elems[j + 6 * i] = cov(i, j);
          }
        }

        pose_pub_.publish(predicted_pose);
      }
    }
  }
}

void PoseEstimatorNode::activeMarkersCallback(const exploration::MarkerAggregate::ConstPtr& marker_aggregate_msg) {
  mutex_.lock();
  active_markers_ = marker_aggregate_msg->active_states;
  mutex_.unlock();
}

}


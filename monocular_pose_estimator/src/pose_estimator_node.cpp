#include "monocular_pose_estimator/pose_estimator_node.h"

namespace monocular_pose_estimator {

PoseEstimatorNode::PoseEstimatorNode(const ros::NodeHandle& nh, const ros::NodeHandle& nh_private)
  : nh_(nh), nh_private_(nh_private), have_camera_info_(false) {
  // Set up a dynamic reconfigure server.
  // This should be done before reading parameter server values.
  dynamic_reconfigure::Server<monocular_pose_estimator::MonocularPoseEstimatorConfig>::CallbackType cb_;
  cb_ = boost::bind(&PoseEstimatorNode::dynamicParametersCallback, this, _1, _2);
  dr_server_.setCallback(cb_);
  
  blob_list_sub_ = nh_.subscribe("blob_list", 1, &PoseEstimatorNode::blobListCallback, this);
  active_markers_sub_ = nh_.subscribe("/active_markers", 1, &PoseEstimatorNode::activeMarkersCallback, this);

  pose_pub_ = nh_.advertise<geometry_msgs::PoseWithCovarianceStamped>("/estimated_pose", 1);
}

PoseEstimatorNode::~PoseEstimatorNode() {}

void PoseEstimatorNode::dynamicParametersCallback(monocular_pose_estimator::MonocularPoseEstimatorConfig &config, uint32_t level) {
  trackable_object_.setBackProjectionPixelTolerance(config.back_projection_pixel_tolerance);
  trackable_object_.setNearestNeighbourPixelTolerance(config.nearest_neighbour_pixel_tolerance);
  trackable_object_.setCertaintyThreshold(config.certainty_threshold);
  trackable_object_.setValidCorrespondenceThreshold(config.valid_correspondence_threshold);

  ROS_INFO("Pose Estimator parameters changed");
}

int PoseEstimatorNode::state_to_hue(std::vector<uint8_t> state) {
  // TODO: convert RGB marker state to 0-180 hue integer
  return 90;
}

void PoseEstimatorNode::blobListCallback(monocular_pose_estimator::BlobList::ConstPtr& blob_list_msg) {
  std::vector<geometry_msgs::Vector3> blobs;
  blobs = blob_list_msg.blobs;
  if ( blobs.size() > min_num_leds_detected_ ) {
    std_msgs::Time blob_stamp;
    blob_stamp = blob_list_msg->header.stamp;
    
    // create detected_led_positions and blob_hues array from blob_list_msg
    List2DPoints detected_led_positions;
    detected_led_positions.resize(blobs.size());
    std::vector<int> blob_hues;

    for (unsigned b = 0; b < blobs.size(); b++) {
      Eigen::Vector2d pt;
      pt(0) = blobs[b].x;
      pt(1) = blobs[b].y;
      detected_led_positions(b) = pt;
      blob_hues.push_back((int)blobs[b].z);
    }

    trackable_object_.setBlobHues(blob_hues);
    trackable_object_.setImagePoints(detected_led_positions);
      
    // copy active_markers to positions of markers
    std::vector<std::string> marker_frames;
    std::vector<std::vector<int>> marker_hues;
    std::vector<List4DPoints> marker_positions;
    mutex_.lock();
    for (unsigned m = 0; m < active_markers_.size(); m++) {
      marker_frames.push_back(active_markers_[m].header.frame_id);
      List4DPoints positions;
      positions.resize(active_markers_[m].markers_states.size());
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

    // update positions_of_markers and marker_hues based on positions[m]
    for (unsigned m = 0; m < marker_hues.size(); m++) {
      trackable_object_.setMarkerHues(marker_hues[m]);
      trackable_object_.setMarkerPositions(marker_positions[m]);

      // A return value of 1 from initailiseWithHues indicates there was
      // a valid correspondence found, and the pose can be estimated
      if ( trackable_object_.initialiseWithHues() == 1) {
        trackable_object_.optimiseAndUpdatePose(blob_stamp.toSec());
        
        geometry_msgs::PoseWithCovarianceStamped predicted_pose;

        // Set timestamp and frame ID as camera_frame:marker_frame
        predicted_pose.header.stamp = blob_stamp;
        predicted_pose.header.frame_id = blob_list_msg.header.frame_id + ":" + marker_frames[m];

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

void PoseEstimatorNode::activeMarkersCallback(exploration::MarkerAggregate::ConstPtr& marker_aggregate_msg) {
  mutex_.lock();
  active_markers_ = marker_aggregate_msg.active_states;
  mutex_.unlock();
}

}


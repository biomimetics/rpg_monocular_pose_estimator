#!/usr/bin/python

import rospy
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Bool

class RecordExperiment():
  def __init__(self):
    rospy.init_node('record_experiment')
    
    self.recording = False

    self.setpoint_pub = rospy.Publisher('/record/desired_pose',PoseStamped,queue_size=1)
    self.info_pub = rospy.Publisher('/record/usb_cam/camera_info',CameraInfo,queue_size=1)
    self.image_pub = rospy.Publisher('/record/usb_cam/image_raw',Image,queue_size=1)
    self.pose_pub = rospy.Publisher('/record/estimated_pose',PoseWithCovarianceStamped,queue_size=1)
    
    rospy.Subscriber('/recording',Bool,self.recording_callback,queue_size=1)
    
    rospy.Subscriber('/desired_pose',PoseStamped,self.setpoint_callback,queue_size=1)
    rospy.Subscriber('/usb_cam/camera_info',CameraInfo,self.info_callback,queue_size=1)
    rospy.Subscriber('/usb_cam/image_raw',Image,self.image_callback,queue_size=1)
    rospy.Subscriber('/estimated_pose',PoseWithCovarianceStamped,self.pose_callback,queue_size=1)

  def run(self):
    while not rospy.is_shutdown():
      rospy.spin()

  def recording_callback(self,msg):
    self.recording = msg.data

  def info_callback(self,msg):
    if self.recording:
      self.info_pub.publish(msg)

  def image_callback(self,msg):
    if self.recording:
      self.image_pub.publish(msg)

  def pose_callback(self,msg):
    if self.recording:
      self.pose_pub.publish(msg)

  def setpoint_callback(self,msg):
    if self.recording:
      self.setpoint_pub.publish(msg)
  
if __name__ == '__main__':
  re = RecordExperiment()
  re.run()

#!/usr/bin/python

import rospy
import numpy
import sys

from geometry_msgs.msg import PoseStamped,PoseWithCovarianceStamped,Pose,Point,Quaternion
from std_msgs.msg import Bool
from tf.transformations import *

axes = ['x','y','z','r','p','q']

class CommandListener:
  def __init__(self):
    rospy.init_node('command_listener')
    self.rate = rospy.Rate(30)
    self.setpoint = [0.0]*6
    self.center = [0.0]*6
    self.sequence = 0
    self.record_pub = rospy.Publisher('/recording',Bool,queue_size=1)
    self.setpoint_pub = rospy.Publisher('/desired_pose',PoseStamped,queue_size=1)
    self.pose = [0.0]*6
    rospy.Subscriber('/monocular_pose_estimator/estimated_pose',PoseWithCovarianceStamped,self.pose_callback,queue_size=1)

  def run(self):
    while not rospy.is_shutdown():
      try:
        command = raw_input('Command:')

        # Take measurement
        if command == 't':
          sequence_goal = self.sequence + 30
          self.record_pub.publish(True)
          
          while(self.sequence < sequence_goal):
            self.publish_setpoint()
            self.rate.sleep()
          
          self.record_pub.publish(False)
        
        # Set command
        elif command[0] == 's':
          if len(command) > 1 and command[1] == 'c':
            self.center = self.pose
            print 'Center: ' + str(self.center)
          else:
            self.setpoint = [float(v) for v in command.split()[1:]]
            print 'Setpoint: ' + str(self.setpoint)
        
        # Step command
        elif command[0] in axes:
          self.setpoint[axes.index(command[0])] += float(command[1:])
          print 'Setpoint: ' + str(self.setpoint)

        # Center command
        elif command[0] == 'c':
          self.setpoint = self.center
          print 'Setpoint: ' + str(self.setpoint)

        self.rate.sleep()

      except EOFError:
        sys.exit(0) 

  def publish_setpoint(self):
    sp_msg = PoseStamped()
    sp_msg.pose = Pose(
      Point(*self.setpoint[0:3]),
      Quaternion(*quaternion_from_euler(*[numpy.pi*a/180.0 for a in self.setpoint[3:6]]))
    )
    sp_msg.header.seq = self.sequence
    sp_msg.header.stamp = rospy.Time.now()
    self.setpoint_pub.publish(sp_msg)
    self.sequence += 1
    
  def pose_callback(self, msg):
    pos = msg.pose.pose.position
    ori = msg.pose.pose.orientation
    angles = [180.0*a/numpy.pi for a in euler_from_quaternion([ori.x,ori.y,ori.z,ori.w])]
    self.pose = [pos.x, pos.y, pos.z] + angles

if __name__ == '__main__':
  cl = CommandListener()
  cl.run()
  

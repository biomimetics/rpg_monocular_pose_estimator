#!/usr/bin/python

# This node will take a color RGB image and produce a monochrome image with 
# only the Saturation channel of an HSV encoded version of of the input image

import rospy
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

rospy.init_node('color_to_saturation')

bridge = CvBridge()
image_pub = rospy.Publisher('image_out', Image, queue_size=1)

def image_callback(img_msg):
  try:
    cv_img = bridge.imgmsg_to_cv2(img_msg, 'bgr8')
    hsv = cv2.cvtColor(cv_img, cv2.COLOR_BGR2HSV)
    out_img = bridge.cv2_to_imgmsg(hsv[:,:,1], 'mono8')
    image_pub.publish(out_img)
  except CvBridgeError as e:
    print e

image_sub = rospy.Subscriber('image_raw', Image, image_callback, queue_size=1)

if __name__ == '__main__':
  while not rospy.is_shutdown():
    rospy.sleep(0.1)

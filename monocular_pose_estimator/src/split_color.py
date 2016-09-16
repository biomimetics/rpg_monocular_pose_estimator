#!/usr/bin/python

# This node will take a color RGB image and produce a monochrome image with 
# only the Saturation channel of an HSV encoded version of of the input image

import rospy
import cv2
import numpy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

rospy.init_node('color_to_saturation')

bridge = CvBridge()
image_pub = rospy.Publisher('image_out', Image, queue_size=1)

def image_callback(img_msg):
  try:
    cv_img = bridge.imgmsg_to_cv2(img_msg, 'bgr8')
    hsv = cv2.cvtColor(cv_img, cv2.COLOR_BGR2HSV)
    h,w,c = cv_img.shape
    out_img = numpy.zeros((h*2,w*2),dtype=numpy.uint8)
    out_img[0:h,0:w] = hsv[:,:,0]
    s_img = hsv[:,:,1]
    v_img = hsv[:,:,2]
    out_img[h:(2*h),0:w] = s_img
    out_img[0:h,w:(2*w)] = v_img
    svt_img = numpy.zeros((h,w),dtype=numpy.uint8)
    #svt_img[:] = (s_img ** 0.5) * (v_img ** 0.5)
    svt_img[:] = v_img
    _,svt_img = cv2.threshold(svt_img,20,255,cv2.THRESH_TOZERO)
    out_img[h:(2*h),w:(2*w)] = svt_img
    out_msg = bridge.cv2_to_imgmsg(out_img, 'mono8')
    image_pub.publish(out_msg)
  except CvBridgeError as e:
    print 'error'
    print e

image_sub = rospy.Subscriber('image_raw', Image, image_callback, queue_size=1)

if __name__ == '__main__':
  while not rospy.is_shutdown():
    rospy.sleep(0.1)

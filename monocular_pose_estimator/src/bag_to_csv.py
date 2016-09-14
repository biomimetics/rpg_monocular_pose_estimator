#!/usr/bin/python

import sys
from rosbag.bag import Bag
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped

# time (s), pos (m), orientation (qx, qy, qz, qw)
pose_columns = ['time', 'x', 'y', 'z', 'qx', 'qy', 'qz', 'qw']

def pose_to_csv(pose_entry):
  _, msg, _, start_time = pose_entry

  timestamp = msg.header.stamp
  pos = msg.pose.position
  ori = msg.pose.orientation

  return [timestamp.to_sec()-start_time, pos.x, pos.y, pos.z, ori.x, ori.y, ori.z, ori.w]

# time (s), pos (m), orientation (qx, qy, qz, qw), row-major 6x6 covariance matrix entries
pose_cov_columns = ['time', 'x', 'y', 'z', 'qx', 'qy', 'qz', 'qw'] + ['c%d%d'%(i,j) for i in range(6) for j in range(6)]

def pose_cov_to_csv(pose_entry):
  _, msg, _, start_time = pose_entry

  timestamp = msg.header.stamp
  pos = msg.pose.pose.position
  ori = msg.pose.pose.orientation
  cov = msg.pose.covariance

  return [timestamp.to_sec()-start_time, pos.x, pos.y, pos.z, ori.x, ori.y, ori.z, ori.w] + list(cov)

topic_formats = {
  'desired_pose': (pose_columns, pose_to_csv),
  'estimated_pose': (pose_cov_columns, pose_cov_to_csv)
}

base_topics = [
  'record/desired_pose',
  'record/estimated_pose',
  'monocular_pose_estimator/estimated_pose',
  'test/estimated_pose'
]

def init_topics(in_filename):
  topics = {}
  topic_names = []

  for base_topic in base_topics:
    topic_names.append('/' + base_topic)

  for topic_name in topic_names:
    topic_columns, topic_formatter = topic_formats[topic_name.split('/')[-1]]

    file_name = in_filename.split('.')[0] + '_' + '_'.join(topic_name.split('/')[1:]) + '.csv'

    csv_file = open(file_name, 'w')
    csv_file.write(', '.join(topic_columns) + '\n')
    topics[topic_name] = {
      'file':csv_file,
      'formatter':topic_formatter
    }

  return topics

def bag_to_csv(filename):
  bagfile = Bag(filename)
  topics = init_topics(filename)
  
  start_time = bagfile.get_start_time()
  for topic, msg, timestamp in bagfile.read_messages(topics=topics.keys()):
    formatter = topics[topic]['formatter']
    data_line = formatter((topic,msg,timestamp,start_time))
    topics[topic]['file'].write(', '.join(map(str,data_line)) + '\n')

  for topic in topics.keys():
    topics[topic]['file'].close()

  bagfile.close()

if __name__ == '__main__':
  bag_to_csv(sys.argv[1])

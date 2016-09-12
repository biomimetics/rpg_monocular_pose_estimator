#!/usr/bin/python

import numpy.lib.recfunctions as rfn
from load_csv_data import *
import sys
from tf.transformations import *

import matplotlib.pyplot as plt

data_series = ['x_axis','y_axis','z_axis','r_angle','p_angle','q_angle']
series_titles = {'x':'X','y':'Y','z':'Z','r':'Roll','p':'Pitch','q':'Yaw'}

def time_cluster(data, max_delta = 1.0):
  diff_idx = numpy.hstack([ [0], numpy.diff(data['time']) > max_delta ])
  return numpy.cumsum(diff_idx)

def quat_to_rpq(data):
  rpq = numpy.zeros(len(data),dtype=[('r', '<f8'), ('p', '<f8'), ('q', '<f8')])
  q_data = data[['qx','qy','qz','qw']]
  for i in range(len(data)):
    rpq[i] = euler_from_quaternion(list(q_data[i]))
  return rpq

def get_series(data, series_name):
  desired = data['calibration_%s_record_desired_pose' % series_name]
  d_i = time_cluster(desired)
  desired = desired[numpy.hstack([numpy.diff(d_i)>0,[True]])]
  desired = rfn.merge_arrays([desired,quat_to_rpq(desired)],flatten=True)
  estimated = data['calibration_%s_record_estimated_pose' % series_name]
  e_i = time_cluster(estimated)
  return desired, estimated, e_i
  
def series_error(desired, estimated, e_i, flip_roll=False, euler=True):
  res = numpy.zeros(len(estimated),dtype=[('x', '<f8'), ('y', '<f8'), ('z', '<f8'), ('r', '<f8'), ('p', '<f8'), ('q', '<f8')])
  for s in ['x','y','z']:
    res[s] = estimated[s] - desired[s][e_i]

  if euler:
    rpq_d = quat_to_rpq(desired)
    if flip_roll:
      # Correct for reversing desired roll angle measurement
      rpq_d['r'] *= -1
    rpq_e = quat_to_rpq(estimated)
    for s in ['r','p','q']:
      # Keep angles in [-pi,pi]
      res[s] = numpy.mod(numpy.pi + rpq_e[s] - rpq_d[s][e_i],2*numpy.pi)-numpy.pi
  else:
    d_q = desired[['qx','qy','qz','qw']]
    e_q = estimated[['qx','qy','qz','qw']]
    for i in range(len(e_q)):
      rpq_err = euler_from_quaternion(
        quaternion_multiply(
          list(e_q[i]),
          quaternion_inverse(list(d_q[e_i[i]]))
        )
      )
      for s,j in zip(['r','p','q'],range(3)):
        res[i][s] = rpq_err[j]
  
  return res

def pos_angle_error(axis_error):
  pos = numpy.zeros(len(axis_error))
  for axis in ['x','y','z']:
    pos += axis_error[axis] ** 2
  pos **= 0.5

  angle = numpy.zeros(len(axis_error))
  for i in range(len(axis_error)):
    angle[i] = 180.0*abs(rotation_from_matrix(euler_matrix(*axis_error[['r','p','q']][i]))[0])/numpy.pi
  
  return pos,angle

def process_data(data):
  series = []
  for ds in data_series:
    series.append(series_error(*get_series(data,ds)))
  return series

def plot_series(data):  
  for s in data.dtype.names:
    plt.plot(data[s],label=s)
  plt.legend()
  plt.grid(True)
  plt.show()

def plot_series_box(data, series_name, title):
  desired, estimated, e_i = get_series(data,series_name)
  series_label = series_name[0]

  pos, angle = pos_angle_error(series_error(
    desired, estimated, e_i, flip_roll = series_label == 'r'
  ))
  pos_samps = [pos[e_i==i] for i in range(max(e_i))]
  angle_samps = [angle[e_i==i] for i in range(max(e_i))]
  
  plt.subplot(211)
  plt.boxplot(pos_samps)
  ax = plt.gca()
  ax.set_xticklabels([])
  plt.ylabel('Position Error (m)')
  plt.title(title)

  plt.subplot(212)
  plt.boxplot(angle_samps)
  ax = plt.gca()
  
  if series_label in ['x','y','z']:
    x_labels = ['%0.2f' % v for v in desired[series_label]]
    plt.xlabel('%s Position (m)' % series_titles[series_label])
  else:
    x_labels = ['%d' % int(180.0*v/numpy.pi) for v in desired[series_label]]
    plt.xlabel('%s Angle (deg.)' % series_titles[series_label])
    
  ax.set_xticklabels(x_labels)
  plt.xticks(rotation='vertical')
  plt.gcf().subplots_adjust(bottom=0.15)
  plt.ylabel('Angle Error (deg.)')
  
  plt.show()

def make_all_plots(data):
  plot_series_box(data,'x_axis','X Axis Errors')
  plot_series_box(data,'y_axis','Y Axis Errors')
  plot_series_box(data,'z_axis','Z Axis Errors')
  plot_series_box(data,'r_angle','Roll Angle Errors')
  plot_series_box(data,'p_angle','Pitch Angle Errors')
  plot_series_box(data,'q_angle','Yaw Angle Errors')
  
if __name__ == '__main__':
  data = load_csv_data(sys.argv[1])
  make_all_plots(data)

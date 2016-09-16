#!/usr/bin/python

import numpy.lib.recfunctions as rfn
import numpy.linalg

from load_csv_data import *
import sys
from tf.transformations import *

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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
  desired,_ = transform_from_start(desired)
  d_i = time_cluster(desired)
  desired = desired[numpy.hstack([numpy.diff(d_i)>0,[True]])]
  desired = rfn.merge_arrays([desired,quat_to_rpq(desired)],flatten=True)
  estimated = data['calibration_%s_record_estimated_pose' % series_name]
  estimated,_ = transform_from_start(estimated)
  e_i = time_cluster(estimated)
  return desired, estimated, e_i
  
def series_error(desired, estimated, e_i, flip_roll=False, euler=True, flip_quat=False):
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

def transform_from_start(series,idx=0,H0=None):
  
  if H0 is None:
    R0 = quaternion_matrix(list(series[['qx','qy','qz','qw']][idx]))
    T0 = translation_matrix(list(series[['x','y','z']][idx]))
    H0 = T0.dot(R0)

  H0_inv = numpy.linalg.inv(H0)
  series_tf = series.copy()
  
  for i in range(len(series)):
    Rt = quaternion_matrix(list(series[['qx','qy','qz','qw']][i]))
    Tt = translation_matrix(list(series[['x','y','z']][i]))
    Ht = H0_inv.dot(Tt.dot(Rt))
    pose = list(Ht[:3,3]) + list(quaternion_from_matrix(Ht))
    for s,v in zip(['x','y','z','qx','qy','qz','qw'],pose):
      series_tf[i][s] = v
  
  return series_tf, H0

def plot_series(data):  
  for s in data.dtype.names:
    plt.plot(data[s],label=s)
  plt.legend()
  plt.grid(True)
  plt.show()

def reject_outliers(data, m=4.0):
  data = numpy.array(data)
  return data[abs(data - data.mean()) < m * data.std()]

def plot_series_box(data, series_name, title, euler=True):
  fig = plt.figure(title.replace(' ','_').lower(),(8,6))
  desired, estimated, e_i = get_series(data,series_name)
  series_label = series_name[0]

  pos, angle = pos_angle_error(series_error(
    desired, estimated, e_i, flip_roll = series_label == 'r', euler=euler
  ))
  
  pos_samps = [reject_outliers(pos[e_i==i]) for i in range(max(e_i))]

  angle_samps = [reject_outliers(angle[e_i==i]) for i in range(max(e_i))]
  
  plt.subplot(211)
  plt.boxplot(pos_samps,sym='')
  ax = plt.gca()
  ax.set_xticklabels([])
  plt.ylabel('Position Error (m)')
  plt.title(title)
  #plt.ylim([0.0,0.51])

  plt.subplot(212)
  plt.boxplot(angle_samps,sym='')
  #plt.ylim([0.0,61.0])
  ax = plt.gca()
  
  if series_label in ['x','y','z']:
    #x_labels = ['%0.2f' % abs(v) for v in desired[series_label]]
    x_labels = ['%0.2f' % (i * 0.05) for i in range(len(desired[series_label]))]
    plt.xlabel('%s Difference (m)' % title.split()[0])
  else:
    #x_labels = ['%d' % int(180.0*v/numpy.pi) for v in desired[series_label]]
    x_labels = ['%d' % (i * 20) for i in range(len(desired[series_label]))]
    plt.xlabel('%s Difference (deg)' % title.split()[0])
    
  ax.set_xticklabels(x_labels)
  plt.xticks(rotation='vertical')
  plt.gcf().subplots_adjust(bottom=0.15)
  plt.ylabel('Angle Error (deg)')

# Homogeneous points representing coordinate axes
axes_pts = 1.0*numpy.array([
  [0,1,0,0,0,0],
  [0,0,0,1,0,0],
  [0,0,0,0,0,1],
  [1,1,1,1,1,1]
])

def plot_trajectory(data, series_name, title='Estimated Trajectory', start=0, end=-1, axes_scale=0.03,n_axes=3):
  series = data[series_name][start:end]
  series,_ = transform_from_start(series)
  
  fig = plt.figure(title.replace(' ','_').lower(),(8,8))
  
  ax = fig.add_subplot(111, projection='3d')
  
  ax.plot([series['x'][0]], [series['y'][0]], [series['z'][0]], 'yo',label='Start',markersize=10)
  ax.plot([series['x'][-1]], [series['y'][-1]], [series['z'][-1]], 'y*',label='End',markersize=16)
  
  
  n_points = len(series)
  t_step = int(n_points/n_axes)
  for t_i in range(0,n_points,t_step) + [-1]:
    R = quaternion_matrix(list(series[['qx','qy','qz','qw']][t_i]))
    T = translation_matrix(list(series[['x','y','z']][t_i]))
    S = scale_matrix(axes_scale)
    pts = T.dot(R).dot(S).dot(axes_pts)
    for i,c in zip(range(3),['r','g','b']):
      l_pts = pts[0:3,(2*i):(2*i+2)]
      if i == 0:
        ax.plot([l_pts[0,1]],[l_pts[1,1]],[l_pts[2,1]],color=c,linewidth=2,marker='D',markeredgecolor='r')
      ax.plot(l_pts[0],l_pts[1],l_pts[2],color=c,linewidth=2)

  ax.plot(series['x'], series['y'], series['z'], 'k.',markersize=3)
  
  ax.set_xlabel('\nX (m)      ')
  ax.set_ylabel('\n\n     Y (m)')
  ax.set_zlabel('Z (m)')
  ax.view_init(11,140)
  plt.xticks(rotation=30)
  plt.yticks(rotation=-30)
  #plt.axis('scaled')
  plt.axis('equal')
  plt.axis('tight')
  plt.grid(True)
  plt.legend(numpoints=1)
  plt.title(title)

def plot_time_trajectory(data, series_name, title='Robot Temporal Trajectory', start=0, end=-1, H0=None):
  series = data[series_name][start:end]
  series['time'] -= series['time'][0]
  series,_ = transform_from_start(series,H0=H0)
  rpq = quat_to_rpq(series)
  tmin, tmax = series['time'][0], series['time'][-1]
  
  fig = plt.figure(title.replace(' ','_').lower(),(8,6))
  ax = plt.subplot(211)
  plt.title(title) 
  for field,marker,label,size in zip(['x','y','z'],['r+','gx','b.'],['X','Y','Z'],[3,3,6]):
    plt.plot(series['time'],series[field],marker,label=label,markersize=size)
  ax.set_xticklabels([])
  plt.ylabel('Posisition (m)')
  plt.legend(numpoints=1)
  plt.xlim(tmin,tmax)

  plt.subplot(212)
  for field,marker,label,size in zip(['r','p','q'],['r+','gx','b.'],['Roll','Pitch','Yaw'],[3,3,6]):
    plt.plot(series['time'],180.0*rpq[field]/numpy.pi,marker,label=label,markersize=size)
  plt.ylabel('Angle (deg)')
  plt.ylim(-180.0,180.0)
  plt.legend(numpoints=1)
  plt.xlabel('Time (s)')
  plt.xlim(tmin,tmax)

def make_box_plots(data):
  # Swap camera frames with robot frames (so that 'Yaw' plot is robot yaw)
  plot_series_box(data,'x_axis','X Sweep Errors')
  plot_series_box(data,'z_axis','Y Sweep Errors')
  plot_series_box(data,'y_axis','Z Sweep Errors')
  plot_series_box(data,'r_angle','Roll Sweep Errors')
  plot_series_box(data,'q_angle','Pitch Sweep Errors')
  plot_series_box(data,'p_angle','Yaw Sweep Errors',False)

def make_trajectory_plots(data):
  plot_trajectory(data, 'downward_upward_spiral_0_monocular_pose_estimator_estimated_pose', start=1530, end=2330,axes_scale=0.04,n_axes=7)
  H0 = transform_from_start(data['test_test_estimated_pose'])[1]
  plot_time_trajectory(data,'test_test_estimated_pose','Trajectory Estimate with Hue Information')
  plot_time_trajectory(data,'test2_test_estimated_pose','Trajectory Estimate without Hue Information',H0=H0)

def make_all_plots(data):
  make_box_plots(data)
  make_trajectory_plots(data)

def show_all_plots(data):
  make_all_plots(data)
  plt.show()

def save_all_plots(data, outdir='.'):
  make_all_plots(data)
  for fignum in plt.get_fignums():
    fig = plt.figure(fignum)
    fig.savefig(outdir + '/' + fig.get_label() + '.pdf')
    
if __name__ == '__main__':
  data = load_csv_data(sys.argv[1])
  save_all_plots(data,sys.argv[2])

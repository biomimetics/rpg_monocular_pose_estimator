#!/usr/bin/python

import yaml
from scipy.spatial.distance import pdist
from scipy.optimize import minimize
import numpy
import sys
from tf.transformations import *

marker_hues = [
  0.0,    # M0 (front) - Red
  60.0,   # M1 (rotor1) - Green
  90.0,   # M2 (rotor2) - Cyan
  30.0,   # M3 (rotor3) - Yellow
  120.0,  # M4 (rotor4) - Blue
  150.0   # M5 (back) - Magenta
]

def load_distances(filename):
  f = open(filename)
  data = yaml.load(f)
  f.close()
  return numpy.array(data['distances'])

def distance_error(positions, distances):
  return pdist(positions)-distances

def objective(positions, distances):
  return numpy.linalg.norm(distance_error(positions,distances))

# M5 is fixed at origin, M0 is on x-axis, and M1 is in XY plane
def feature_to_positions(feature):
  positions = numpy.zeros(18)
  positions[0] = feature[0]
  positions[3:5] = feature[1:3]
  positions[6:15] = feature[3:]
  return positions.reshape(6,3)

# Start markers at midpoint of long axis, and average offset from center axis
def x0_from_distances(distances):
  midpoint = distances[4] / 2.0 # |M0-M5|/2
  offset = (distances[6] + distances[10]) / 4.0 # avg(|M1-M3|,|M2-M4|)/2
  return numpy.array([
    distances[4],         # M0_x
    midpoint, offset,     # M1_xy
    midpoint, 0, offset,  # M2_xyz
    midpoint, -offset, 0, # M3_xyz
    midpoint, 0, -offset  # M4_xyz
  ])

def positions_from_distances(distances):
  x0 = x0_from_distances(distances)
  fun = lambda x : objective(feature_to_positions(x), distances)
  #jac = lambda x : distance_error(feature_to_positions(x), distances)
  res = minimize(fun,x0,options={'disp':True})
  return feature_to_positions(res.x)
  
def correct_positions(positions):
  hom_positions = numpy.vstack([positions.T,numpy.ones((1,6))])
  midpoint = positions[0,0]/2
  pose_correction = euler_matrix(numpy.pi/4,0,0).dot(translation_matrix([-midpoint,0,0,]))
  corrected_positions = pose_correction.dot(hom_positions)
  return corrected_positions.T[:,:3]

def save_positions(positions, filename, hues=marker_hues):
  marker_positions = []
  for position, hue in zip(positions,hues):
    marker_positions.append({
      'x':float(position[0]),
      'y':float(position[1]),
      'z':float(position[2]),
      'h':hue
    })
  yaml_data = {'marker_positions':marker_positions}
  f = open(filename,'w')
  yaml.dump(yaml_data,f)
  f.close()

if __name__ == '__main__':
  distances_filename = sys.argv[1]
  positions_filename = sys.argv[2]
  distances = load_distances(distances_filename)
  positions = positions_from_distances(distances)
  corrected_positions = correct_positions(positions)
  save_positions(corrected_positions, positions_filename)


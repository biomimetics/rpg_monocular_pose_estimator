#!/usr/bin/python

import glob
import numpy

def csv_to_numpy(filename):
  return numpy.genfromtxt(filename, dtype=float, delimiter=',', names=True)

def numpy_to_csv(filename,data):
  numpy.savetxt(filename,data,delimiter=', ',header=', '.join(data.dtype.names),comments='')

def load_csv_data(path='.',filename=None):
  if filename is not None:
    files = [filename]
  else:
    files = glob.glob(path + '/*.csv')

  print "Loading data from " + str(files)

  data = {}
  for f in files:
    series = f.split('.')[-2].split('/')[-1]
    data[series] = csv_to_numpy(f)
 
  if filename is not None:
    return data.values()[0]
  else:
    return data

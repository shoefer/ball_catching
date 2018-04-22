# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 11:24:41 2016

@author: shoefer
"""

import numpy as np

from warnings import warn

def windowed_averaged_dot(X, DT, window_size=10, gap=30):
  i = len(X)-1
  wnd1_left = np.max([i-gap-window_size, 0])
  wnd1_right = np.max([i-gap, 0])+1
  wnd2_left = np.max([i-window_size, 0])
  wnd2_right = i+1

  real_gap = wnd2_right-wnd1_right
#  print wnd1_left, wnd1_right 
#  print wnd2_left, wnd2_right 
#  print real_gap
  if real_gap == 0:
    return 0.
  res = (np.mean(X[wnd2_left:wnd2_right]) - np.mean(X[wnd1_left:wnd1_right]))/(real_gap*DT)
  return res

def windowed_averaged_diff(X1, X2, window_size=10, gap=30):
  assert (len(X1) == len(X2))
  i = len(X1)-1
  wnd1_left = np.max([i-gap-window_size, 0])
  wnd1_right = np.max([i-gap, 0])+1
  wnd2_left = np.max([i-window_size, 0])
  wnd2_right = i+1

  real_gap = wnd2_right-wnd1_right
#  print wnd1_left, wnd1_right 
#  print wnd2_left, wnd2_right 
#  print real_gap
  if real_gap == 0:
    return 0., 0.
  res = (np.mean(X1[wnd2_left:wnd2_right]) - np.mean(X2[wnd1_left:wnd1_right]))
  return res, real_gap


def windowed_averaged_dot_list(X, DT, window_size=10, gap=30):
  dot = []
  for i in range(len(X)):
    if i == 0:
      continue
    dot.append(windowed_averaged_dot(X, DT, window_size, gap))
  return dot    


def queue_push(lst, v):
  v_ = lst.pop()
  return [v] + lst

#-----------------------------------------------------------------------------------------

def compute_rv_index(size, rv_delay, rv_min=3):
  """
  Used by COV-OAC strategy
  """
  if size == 0:
    warn("compute_rv_index undefined for size=0")
    return None

  return min(size-1, 
             size - 
                 min(size-rv_min, rv_delay if rv_delay > 3  else 2)
             )
    

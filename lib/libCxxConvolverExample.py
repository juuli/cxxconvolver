# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 18:28:40 2018

@author: jks
"""

import numpy as np
import pyaudio as paudio
import matplotlib.pyplot as plt
import scipy.io.wavfile as swav
import scipy.signal as sig
import ctypes as ct
import atexit
from numpy.ctypeslib import ndpointer


def readAndResampleTo(fp, fs, normalize=False):
  fs_a, data = swav.read(fp)
  y = np.floor(np.log2(len(data)))
  nextpow2 = np.power(2, y)
  n_data = sig.resample(data[:int(nextpow2)], int(float(fs_a)/float(fs)*float(len(data[:int(nextpow2)]))))
  if(normalize):
    n_data = n_data/np.max(np.abs(n_data))
  n_data = n_data.astype(np.float32)
  return n_data


class Convolver:
  def __init__(self):
    import os
    folder =  os.path.dirname(os.path.abspath(__file__))
    self.init_ = False
    self.cc_ptr_ = None
    atexit.register(self.cleanup)

    # Load extern "C" functions from the library
    if(os.name == 'nt'):
      self.cc = ct.cdll.LoadLibrary(folder + "/CxxConvolver.dll")
    else:
      self.cc = ct.cdll.LoadLibrary(folder + "/libCxxConvolver.dylib")

    self.cc.initialize.restype = ct.c_void_p

    self.cc.initialize.argtypes = [ct.c_uint64,
                              ct.c_uint64,
                              ct.c_uint64,
                              ct.c_uint64]

    self.cc.destroy.argtypes = [ct.c_void_p]

    self.cc.addInputFrame.argtypes = [ct.c_void_p,
                                      ndpointer(ct.c_float,
                                                flags="F_CONTIGUOUS")]

    self.cc.processOutput.argtypes = [ct.c_void_p,
                                      ct.c_uint64]

    self.cc.getOutputFrame.argtypes = [ct.c_void_p,
                                       ndpointer(ct.c_float,
                                                 flags="F_CONTIGUOUS")]

    self.cc.setAllSwap.argtypes = [ct.c_void_p]

    self.cc.getNumSwapping.argtypes = [ct.c_void_p]
    self.cc.getNumSwapping.restype = ct.c_int32


    self.cc.isSwapping.argtypes = [ct.c_void_p,
                              ct.c_uint64,
                              ct.c_uint64]

    self.cc.isSwapping.restype = ct.c_int32

    self.cc.updateFilter.argtypes = [ct.c_void_p,
                                     ct.c_uint64,
                                     ct.c_uint64,
                                     ndpointer(ct.c_float,
                                               flags="F_CONTIGUOUS"),
                                    ct.c_uint64]

    self.cc.execPartSwap.argtypes = [ct.c_void_p,
                                    ct.c_uint64,
                                    ct.c_uint64,
                                    ndpointer(ct.c_float,
                                              flags="F_CONTIGUOUS"),
                                    ct.c_uint64]

  def initialize(self, num_input, num_output, fs, frame_len):
    self.init_ = True
    self.cc_ptr_ = self.cc.initialize(num_input, num_output,
                                      fs, frame_len)

  def updateFilter(self, input_, output_, filter_, filter_len):
    self.cc.updateFilter(self.cc_ptr_, input_, output_, filter_, filter_len)

  def processOutput(self, output):
    self.cc.processOutput(self.cc_ptr_, output)

  def getOutputFrame(self, output_frame):
    if(output_frame.dtype != np.float32):
      print "Warning, output frame not float32"

    self.cc.getOutputFrame(self.cc_ptr_, output_frame)

  def addInputFrame(self, input_frame):
    if(input_frame.dtype != np.float32):
      print "Warning, output frame not float32"

    self.cc.addInputFrame(self.cc_ptr_, input_frame)

  def cleanup(self):
    if(self.init_):
      print "Destroying Convolver"
      self.cc.destroy(self.cc_ptr_)
      print "Done"
      self.init_ = False
    else:
      print "Convoler not initialized, no cleanup"


num_inputs = 2
num_outputs = 2
num_filters = num_inputs*num_outputs
filter_len = 48000
fs = 48000
frame_len = 512

cc = Convolver()
cc.initialize(num_inputs, num_outputs, filter_len, frame_len)

# Load an array response
root = "/Users/jukkasaarelma/Documents/koodit/cxxsdm/bin/audio/"
fp = root + "VM_R2_large_3.json"
import json
f = open(fp, "r")
data = json.load(f)
f.close()

resp_len = np.shape(data['ir0'])[0]
irs = np.zeros((resp_len, 6), dtype=np.float32, order='F')

for i in range(6):
  irs[:,i] = data['ir%u'%i]

filters = np.zeros((filter_len, num_filters), dtype=np.float32, order='F')
filters[:,0] = irs[:filter_len, 2]
filters[:,3] = irs[:filter_len, 3]
filters = filters/np.max(np.abs(filters))*0.1

for i in range(num_filters):
  ii = i/num_outputs
  oi = i%num_outputs
  cc.updateFilter(ii, oi, filters[:,i], filter_len)
pa = paudio.PyAudio()

#pa.is_format_supported(48000, input_channels=2, output_channels=4,
#                       input_device=6, output_device=6,
#                       input_format=paudio.paFloat32,
#                       output_format=paudio.paFloat32)
#print pa.get_host_api_count()
#print pa.get_default_host_api_info()
#print pa.get_device_info_by_index(2)

output_frame = np.zeros(num_outputs*frame_len, dtype=np.float32, order='F')


import glob
files = glob.glob("audio/*.wav")

samples = []
for f in files:
  cur = readAndResampleTo(f, fs, normalize=True)
  samples.append(cur)

input_switch = 1
frame_idx = 0
played = True
in_frame = np.zeros((frame_len*2, 1), dtype=np.float32, order='F')

def callback(in_data, frame_count, time_info, status):
  global cc
  global played_frames
  global in_frame
  global output_frame
  global input_switch
  global frame_idx

  in_frame[:,:] = 0.0
  output_frame[:,:] = 0.0

  if(input_switch == 0):
    in_frame = np.fromstring(in_data, dtype=np.float32)

  if(input_switch >= 1 and input_switch < len(samples)):
    total_frames = np.shape(samples[input_switch])[0]/frame_count
    if frame_idx < total_frames-1:
      si = frame_idx*(2*frame_len)
      ei = (frame_idx+1)*(2*frame_len)
      in_frame[:, 0] = (samples[input_switch].flatten())[si:ei]

    frame_idx += 1

  else:
    pass

  cc.addInputFrame(in_frame)
  for i in range(num_outputs):
    cc.processOutput(i)

  cc.getOutputFrame(output_frame)
  return (output_frame.data, paudio.paContinue)

stream = pa.open(rate=int(fs),
                 input_channels=int(0),
                 output_channels=int(2),
                 format=paudio.paFloat32,
                 input=int(0),
                 output=int(1),
                 input_device_index=int(0),
                 output_device_index=int(2),
                 frames_per_buffer=int(512),
                 stream_callback=callback)
#

def closeStream(stream):
  stream.stop_stream()
  stream.close()

print "Start stream"
stream.start_stream()

#import time
#time.sleep(1.1)

closeStream(stream)
cc.cleanup()
del cc
#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Slicemap script
import pymses,os
from pymses.analysis.visualization import *
from pymses.analysis.visualization.image_plot_utils import *
from optparse import OptionParser
parser = OptionParser()
parser.usage = "%prog ramses_directory ramses_output_number map_max_resolution=512"
(opts, args) = parser.parse_args()
try:
	fileDir = args[0]
	outNumber = int(args[1])
except:
	fileDir = None
	outNumber = None
try:
	mms = int(args[2])
except:
	mms = 512
ro = pymses.RamsesOutput(fileDir, outNumber)
outNumber = ro.iout
source = ro.amr_source(["rho"])
cam  = Camera(center=[0.5, 0.5, 0.5], line_of_sight_axis='z',
		region_size=[1., 1.], up_vector='y',
		distance=0.0, far_cut_depth=0.0, map_max_size=mms)
op = ScalarOperator(lambda dset: dset["rho"])
from time import time
t0 = time()
# Optional CameraOctreeDatasource creation (may be faster...)
#from pymses.sources.ramses.octree import CameraOctreeDatasource
#esize = 0.5**(ro.info["levelmin"]+1)
#cod = CameraOctreeDatasource(cam, esize, source)
#source = cod.dset
#op = MaxLevelOperator()
# Compute the SliceMap
map = SliceMap(source, cam, op, verbose=False)
t1=time()
print("SliceMap time = %.1f s"%(t1-t0), " Octree levelmax read :", cam.get_required_resolution())
#mapPylab = apply_log_scale(map)
#import pylab as P
#P.imshow(mapPylab)
#P.show()
save_map_HDF5(map, cam, map_name="SliceMap_%s"%(outNumber))
save_HDF5_to_img("./SliceMap_%s.h5"%(outNumber), cmap="jet",img_path="./")
os.remove("./SliceMap_%s.h5"%(outNumber))


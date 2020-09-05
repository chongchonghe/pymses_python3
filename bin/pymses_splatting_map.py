#!/usr/bin/env python
# -*- coding: utf-8 -*-
# splatting map script
import pymses,os
from pymses.analysis.visualization import *
from pymses.analysis.visualization.fft_projection import *
from pymses.analysis.visualization.image_plot_utils import *
from time import time
from optparse import OptionParser
parser = OptionParser()
parser.usage = "%prog ramses_directory ramses_output_number map_max_resolution=512"
(opts, args) = parser.parse_args()
try:
	fileDir = args[0]
	outNumber = int(args[1])
except:
	# "None" leads to an automatic look for 
	# RAMSES output in the current directory
	fileDir = None 
	outNumber = None
try:
	mms = int(args[2])
except:
	mms = 512
ro = pymses.RamsesOutput(fileDir, outNumber)
outNumber = ro.iout
cam = Camera(center=[ 0.5, 0.5, 0.5 ], line_of_sight_axis="z", up_vector="y", region_size=[5.0E-1, 5.0E-1], \
			distance=2.5E-1, far_cut_depth=2.5E-1, map_max_size=mms)
#import tables
#file= tables.openFile("camera/cameraAMRviewer.h5", "r")
#cam = Camera.from_HDF5(file)
#cam.map_max_size=mms
#file.close()
#ro.verbose=False
source = ro.amr_source(["rho"])
scal_func = FractionOperator(lambda dset: (dset["rho"]**2 * dset["size"]**3), lambda dset: (dset["rho"] * dset["size"]**3))
# particles :
#source = ro.particle_source(["mass", "level"])
#scal_func = ScalarOperator(lambda dset: dset["mass"])
t0 = time()
mp = MapFFTProcessor(source,ro.info, pre_flatten=True)
splatting = mp.process(scal_func, cam)
t1=time()
print("Total time = %.1f s"%(t1-t0))
#mapPylab = apply_log_scale(splatting)
#import pylab as P
#P.imshow(mapPylab)
#P.show()

import pylab as m
cdict = {
'red'  :  ((0., 0., 0.), (0.2, 0., 0.), (0.5, 0.9, .9), (1., 1., 1.)),
'green':  ((0., 0., 0.), (0.2, 0.0, 0.0), (0.5, .9, .9), (1., 0., 0.)),
'blue' :  ((0., 0., 0.), (0.2, .8, .8), (0.5, 0.4, 0.4), (1., 0., 0.))
}
#generate the colormap with 1e5 interpolated values
my_cmap = m.matplotlib.colors.LinearSegmentedColormap('my_colormap', cdict, 1e5)

save_map_HDF5(splatting, cam, map_name="splatting_%s"%(outNumber))
save_HDF5_to_img("./splatting_%s.h5"%(outNumber), cmap=my_cmap, img_path="./")
os.remove("./splatting_%s.h5"%(outNumber))

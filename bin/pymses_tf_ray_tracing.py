#!/usr/bin/env python
# -*- coding: utf-8 -*-
# transfer function ray tracing map script
import pymses, os
from pymses.analysis.visualization import *
from pymses.analysis.visualization.image_plot_utils import *
from time import time
from pymses.analysis.visualization.raytracing import OctreeRayTracer, RayTracer
from pymses.sources.ramses import CameraOctreeDatasource, CameraOctreeDataset
from optparse import OptionParser
from numpy import log10
t0 = time()
parser = OptionParser()
parser.usage = "%prog ramses_directory ramses_output_number map_max_resolution=512"
(opts, args) = parser.parse_args()
try:
	fileDir = args[0]
	outNumber = int(args[1])
	mms = int(args[2])
except:
	# "None" leads to an automatic look for 
	# RAMSES output in the current directory
	fileDir = None 
	outNumber = None
	mms = 512
ro = pymses.RamsesOutput(fileDir, outNumber)
outNumber = ro.iout
cam = Camera(center=[ 0.5, 0.5, 0.5 ], line_of_sight_axis="z", up_vector="y",\
		region_size=[5.0E-1, 5.0E-1], distance=2.5E-1,\
		far_cut_depth=2.5E-1, map_max_size=mms)
#import tables
#file= tables.openFile("camera.h5", "r")
#cam = Camera.from_HDF5(file)
#file.close()
source = ro.amr_source(["rho"])
esize = 0.5**(ro.info["levelmin"]+1)

#####################################################################################
######## 1) OctreeRayTracer with ColorLinesTransferFunction definition ##############
#####################################################################################
cltf = ColorLinesTransferFunction( (-5.0, 2.0) )
cltf.add_line(-2.0, 0.1)
cltf.add_line(.0, 0.1)
cltf.add_line(2., 0.1)
cam.set_color_transfer_function(cltf)
# We add 1e-8 to avoid NaN and -Inf log result problems with approximative null values
op = ScalarOperator(lambda dset: log10(dset["rho"]+1e-8))
###### Option A : Create OctreeRayTracer = implicit OctreeDataSource creation #######
#OctreeRT = OctreeRayTracer(ro, ["rho"])
###### Option B : Explicit OctreeDataSource creation (can be explicitly reused) #####
######### fullOctreeDataSource = Build a local octree for this aera #################
fullOctreeDataSource = CameraOctreeDatasource(cam, esize, source).dset
# Here is how to save and load local octree option for faster reuse if wanted :
#fullOctreeDataSource.write_hdf5("myDset")
#fullOctreeDataSource = CameraOctreeDataset.from_hdf5("myDset")
OctreeRT = OctreeRayTracer(fullOctreeDataSource)
#########            Start OctreeRayTracer process :                 ################
img=OctreeRT.process(op, cam)
#img.show()
img.save("rt_tf_%s.png"%outNumber)
print("rt_tf_%s.png saved"%outNumber)

#####################################################################################
######## 2) RayTracer with PyMSES Classic Operator definition #######################
#####################################################################################
#op = ScalarOperator(lambda dset: dset["rho"])
op = FractionOperator(lambda dset: (dset["rho"]**2), lambda dset: (dset["rho"]))
######### Option A : Create RayTracer = multiprocessing on data #####################
#rt = RayTracer(ro, ["rho"])
##map = rt.process(op, cam) # multiprocessing cpu on data files loaded
#map = rt.process(op, cam, source=fullOctreeDataSource) # reuse local octree
######### Option B : use OctreeRT = multiprocessing on image pixels/rays ############
map, levelmax_map = OctreeRT.process(op, cam, rgb=False)  # reuse OctreeRT from part 1)
# This should give the same result as RayTracer(ro, ["rho"]).process(op, cam)
#mapPylab = apply_log_scale(splatting)
#import pylab as P
#P.imshow(mapPylab)
#P.show()
save_map_HDF5(map, cam, map_name="rt_%s"%outNumber)
save_HDF5_to_img(("./rt_%s.h5"%outNumber), cmap="jet",img_path="./")
os.remove(("./rt_%s.h5"%outNumber))
save_map_HDF5(levelmax_map, cam, map_name="rt_lvlmax_%s"%outNumber)
save_HDF5_to_img(("./rt_lvlmax_%s.h5"%outNumber), cmap="jet",img_path="./", log_sensitive=False)
os.remove(("./rt_lvlmax_%s.h5"%outNumber))
print("rt total time = %.1f s"%(time()-t0), "mms = ", mms, "max AMR read = ",\
	cam.get_required_resolution())
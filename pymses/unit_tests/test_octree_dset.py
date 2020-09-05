# License:
#   Copyright (C) 2011 Thomas GUILLET, Damien CHAPON, Marc LABADENS. All Rights Reserved.
#
#   This file is part of PyMSES.
#
#   PyMSES is free software: you can redistribute it and/or modify
#   it under the terms of the GNU General Public License as published by
#   the Free Software Foundation, either version 3 of the License, or
#   (at your option) any later version.
#
#   PyMSES is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU General Public License for more details.
#
#   You should have received a copy of the GNU General Public License
#   along with PyMSES.  If not, see <http://www.gnu.org/licenses/>.
"""
test_octree_dset.py -- test module for Hilbert space-filling curve
"""

import pymses
import numpy
from pymses.analysis.visualization import *
from pymses.analysis.visualization.raytracing import RayTracer, OctreeRayTracer
from pymses.analysis.visualization.fft_projection import *
from pymses.analysis.visualization.image_plot_utils import *
import os
from pymses.utils import misc

PWD = os.path.dirname(__file__)

asserted_SliceMap_result = numpy.array([[-5.98442495, -5.98442495, -6.00830233],
			[-5.98442495, -5.98442495, -6.00830233],
			[-6.03202943, -6.03202943, -6.07758903]])

asserted_interpolated_SliceMap_result = numpy.array([[-5.99014809, -5.96351829, -5.96968206],
			[-6.01119872, -5.99727729, -5.99045202],
			[-6.02887729, -6.01696159, -6.01550138]])

asserted_rt_map = numpy.array([[-4.61547079, -4.50387064, -4.50387064],
			[-5.17418872, -4.88956574, -4.88956574],
			[-5.17418872, -4.88956574, -4.88956574]])

asserted_rotated_rt_map = numpy.array([[4.43912944e-06, 9.40160105e-06, 6.78405933e-06],
			[5.16985554e-06, 3.88652536e-05, 4.32777455e-05],
			[2.06883743e-06, 7.71202140e-06, 8.58699623e-06]])


asserted_lvlmax_map = numpy.array([[3, 3, 3],
			[3, 3, 3],
			[3, 3, 3]])

class EmptyOutput(object):
	def __init__(self):
		self.info = {}
		self.info["levelmin"] = 3
		self.info["dom_decomp"] = None
		self.output_repos = None
		self.iout = None

def test_loading_OctreeDataset():#{{{
	amr_source = pymses.sources.ramses.octree.OctreeDataset\
	      .from_hdf5(os.path.join(PWD, "test_amr_dset.h5"))
	assert(amr_source.read_lmax == 3)
	assert(amr_source._data_list == [0])
	assert(amr_source.fields["rho"].shape[0] == 73)
	assert(amr_source.fields["rho"][5][5] - 1.33670771473e-07 < 1e-6)
#}}}

def test_SliceMap():#{{{
	amr_source = pymses.sources.ramses.octree.OctreeDataset.\
	      from_hdf5(os.path.join(PWD, "test_amr_dset.h5"))
	rho_op = ScalarOperator(lambda dset: dset["rho"])
	cam  = Camera(center=[0.5, 0.5, 0.5], line_of_sight_axis='z', region_size=[.3, .3],\
		      distance=0., far_cut_depth=0.,\
		up_vector='y', map_max_size=3, log_sensitive=True)
	map = SliceMap(amr_source, cam, rho_op, z=0.3, use_C_code=False)
	map = apply_log_scale(map)
	print(map)
	print(asserted_SliceMap_result)
	assert(abs(map - asserted_SliceMap_result) < 1e-6).all()
#}}}

def test_SliceMap_C_code():#{{{
	amr_source = pymses.sources.ramses.octree.OctreeDataset.\
	      from_hdf5(os.path.join(PWD, "test_amr_dset.h5"))
	rho_op = ScalarOperator(lambda dset: dset["rho"])
	cam  = Camera(center=[0.5, 0.5, 0.5], line_of_sight_axis='z', region_size=[.3, .3],\
		      distance=0., far_cut_depth=0.,\
		up_vector='y', map_max_size=3, log_sensitive=True)
	map = SliceMap(amr_source, cam, rho_op, z=0.3, use_C_code=True)
	map = apply_log_scale(map)
	print(map)
	print(asserted_SliceMap_result)
	assert(abs(map - asserted_SliceMap_result) < 1e-6).all()
#}}}

def test_SliceMap_C_code_interpolation():#{{{
	amr_source = pymses.sources.ramses.octree.OctreeDataset.\
	      from_hdf5(os.path.join(PWD, "test_amr_dset.h5"))
	rho_op = ScalarOperator(lambda dset: dset["rho"])
	cam  = Camera(center=[0.5, 0.5, 0.5], line_of_sight_axis='z', region_size=[.3, .3],\
		      distance=0., far_cut_depth=0.,\
		up_vector='y', map_max_size=3, log_sensitive=True)
	map = SliceMap(amr_source, cam, rho_op, z=0.3, interpolation=True, use_C_code=True)
	map = apply_log_scale(map)
	print(map)
	print(asserted_interpolated_SliceMap_result)
	assert(abs(map - asserted_interpolated_SliceMap_result) < 1e-6).all()
#}}}

def test_SliceMap_OpenCL():#{{{
	misc.init_OpenCl()
	if misc.OpenCL_initialized:
		amr_source = pymses.sources.ramses.octree.OctreeDataset.\
		      from_hdf5(os.path.join(PWD, "test_amr_dset.h5"))
		rho_op = ScalarOperator(lambda dset: dset["rho"])
		cam  = Camera(center=[0.5, 0.5, 0.5], line_of_sight_axis='z', region_size=[.3, .3],\
			      distance=0., far_cut_depth=0.,\
			up_vector='y', map_max_size=3, log_sensitive=True)
		map = SliceMap(amr_source, cam, rho_op, z=0.3, use_C_code=True, use_openCL=True)
		map = apply_log_scale(map)
		print(map)
		print(asserted_SliceMap_result)
		assert(map - asserted_SliceMap_result < 1e-6).all()
		#assert(False)
#}}}

def test_loading_camera():#{{{
	cam  = Camera.from_HDF5(os.path.join(PWD, "camera_test.h5"))
	cam.printout()
	center = [ 0.49414021,  0.48413301,  0.49849942]
	line_of_sight = [ 0.59924788,  0.32748786,  0.73051603]
	assert(abs(cam.los_axis - line_of_sight) < 1e-5).all()
	assert(abs(cam.center - center) < 1e-5).all()
#}}}

def test_RT():#{{{
	octree_dset = pymses.sources.ramses.octree.OctreeDataset.\
	      from_hdf5(os.path.join(PWD, "test_amr_dset.h5"))
	rho_op = ScalarOperator(lambda dset: dset["rho"])
	cam  = Camera(center=[0.5, 0.5, 0.5], line_of_sight_axis='z', region_size=[.3, .3],\
		      distance=0.3, far_cut_depth=0.3,\
		up_vector='y', map_max_size=3, log_sensitive=True)
	
	ro = EmptyOutput()
	rt = RayTracer(ro, ["rho"])
	map=rt.process(rho_op, cam, source=octree_dset, use_C_code=False)
	map = apply_log_scale(map)
	print(map)
	print(asserted_rt_map)
	assert(abs(map - asserted_rt_map) < 1e-6).all()
#}}}

def test_RT_C_code():#{{{
	octree_dset = pymses.sources.ramses.octree.OctreeDataset.\
	      from_hdf5(os.path.join(PWD, "test_amr_dset.h5"))
	rho_op = ScalarOperator(lambda dset: dset["rho"])
	cam  = Camera(center=[0.5, 0.5, 0.5], line_of_sight_axis='z', region_size=[.3, .3],\
		      distance=0.3, far_cut_depth=0.3,\
		up_vector='y', map_max_size=3, log_sensitive=True)

	ro = EmptyOutput()
	rt = RayTracer(ro, ["rho"])
	map=rt.process(rho_op, cam, source=octree_dset, use_C_code=True)
	map = apply_log_scale(map)
	print(map)
	print(asserted_rt_map)
	assert(abs(map - asserted_rt_map) < 1e-6).all()
#}}}

def test_RayTracer_rotated():#{{{
	octree_dset = pymses.sources.ramses.octree.CameraOctreeDataset.\
	      from_hdf5(os.path.join(PWD, "test_amr_dset.h5"))
	rho_op = ScalarOperator(lambda dset: dset["rho"])
	# camera center small shift is still with OctreeRayTracer to avoid grid limit pb
	cam  = Camera(center=[0.501, 0.501, 0.501], line_of_sight_axis=[0.401, 0.601, -0.701], region_size=[.3, .3],\
		      distance=0.3, far_cut_depth=0.3,\
		up_vector='y', map_max_size=3, log_sensitive=False)
	
	ro = EmptyOutput()
	rt = RayTracer(ro, ["rho"])
	map=rt.process(rho_op, cam, source=octree_dset)
	print("map :")
	print(map)
	print("asserted_rt_map :")
	print(asserted_rotated_rt_map)
	assert(abs(map - asserted_rotated_rt_map) < 1e-3).all()
	ro = EmptyOutput()
	rt = RayTracer(ro, ["rho"])
	map=rt.process(rho_op, cam, source=octree_dset, use_bottom_up=True)
	print("map :")
	print(map)
	print("asserted_rt_map :")
	print(asserted_rotated_rt_map)
	assert(abs(map - asserted_rotated_rt_map) < 1e-3).all()
#}}}

def test_OctreeRayTracer():#{{{
	octree_dset = pymses.sources.ramses.octree.CameraOctreeDataset.\
	      from_hdf5(os.path.join(PWD, "test_amr_dset.h5"))
	rho_op = ScalarOperator(lambda dset: dset["rho"])
	# camera center small shift is still with OctreeRayTracer to avoid grid limit pb
	cam  = Camera(center=[0.501, 0.501, 0.501], line_of_sight_axis='z', region_size=[.3, .3],\
		      distance=0.3, far_cut_depth=0.3,\
		up_vector='y', map_max_size=3, log_sensitive=True)
	
	rt = OctreeRayTracer(octree_dset)
	
	map, lvlmax_map = rt.process(rho_op, cam, rgb=False, use_C_code=False)
	map = apply_log_scale(map)
	print(map)
	print(asserted_rt_map)
	assert(abs(map - asserted_rt_map) < 1e-6).all()
	assert(lvlmax_map == asserted_lvlmax_map).all()
#}}}

def test_OctreeRayTracer_C_code():#{{{
	octree_dset = pymses.sources.ramses.octree.CameraOctreeDataset.\
	      from_hdf5(os.path.join(PWD, "test_amr_dset.h5"))
	rho_op = ScalarOperator(lambda dset: dset["rho"])
	# camera center small shift is still with OctreeRayTracer to avoid grid limit pb
	cam  = Camera(center=[0.501, 0.501, 0.501], line_of_sight_axis='z', region_size=[.3, .3],\
		      distance=0.3, far_cut_depth=0.3,\
		up_vector='y', map_max_size=3, log_sensitive=True)
	
	rt = OctreeRayTracer(octree_dset)
	
	map, lvlmax_map = rt.process(rho_op, cam, rgb=False, use_C_code=True)
	map = apply_log_scale(map)
	print(map)
	print(asserted_rt_map)
	assert(abs(map - asserted_rt_map) < 1e-6).all()
	print(lvlmax_map)
	print(asserted_lvlmax_map)
	assert(lvlmax_map == asserted_lvlmax_map).all()
#}}}

def test_OctreeRayTracer_rotated():#{{{
	octree_dset = pymses.sources.ramses.octree.CameraOctreeDataset.\
	      from_hdf5(os.path.join(PWD, "test_amr_dset.h5"))
	rho_op = ScalarOperator(lambda dset: dset["rho"])
	# camera center small shift is still with OctreeRayTracer to avoid grid limit pb
	cam  = Camera(center=[0.501, 0.501, 0.501], line_of_sight_axis=[0.401, 0.601, -0.701], region_size=[.3, .3],\
		      distance=0.3, far_cut_depth=0.3,\
		up_vector='y', map_max_size=3, log_sensitive=False)
	
	rt = OctreeRayTracer(octree_dset)
	map, lvlmax_map = rt.process(rho_op, cam, rgb=False, use_C_code=False)
	print("map :")
	print(map)
	print("asserted_rt_map :")
	print(asserted_rotated_rt_map)
	assert(abs(map - asserted_rotated_rt_map) < 1e-3).all()
	print(lvlmax_map)
	print(asserted_lvlmax_map)
	assert(lvlmax_map == asserted_lvlmax_map).all()
	print("Coherence with C Code... :")
	mapC, lvlmax_mapC = rt.process(rho_op, cam, rgb=False, use_C_code=True)
	assert(abs(map - mapC) < 1e-3).all()
	assert(lvlmax_map == lvlmax_mapC).all()
	print("Coherence with RayTracer... :")
	class EmptyOutput(object):
		def __init__(self):
			self.info = {}
			self.info["levelmin"] = 3
			self.info["dom_decomp"] = None
			self.output_repos = None
			self.iout = None
	ro = EmptyOutput()
	top_down_rt = RayTracer(ro, ["rho"])
	mapRayT=top_down_rt.process(rho_op, cam, source=octree_dset)
	assert(abs(map - mapRayT) < 1e-3).all()
	misc.init_OpenCl()
	if misc.OpenCL_initialized:
		mapO, lvlmax_mapO = rt.process(rho_op, cam, rgb=False, use_openCL=True)
		print("OK\nCoherence with OpenCl Code... :")
		print(lvlmax_mapC, numpy.min(lvlmax_mapC))
		print(lvlmax_mapO, numpy.min(lvlmax_mapO))
		assert(abs(mapC - mapO) < 1e-3).all()
		assert(lvlmax_mapC == lvlmax_mapO).all()
	
	# OpenCL Single precision problem : in "Initialisation : first cell ray/face intersection computation"
	# Can be seen with map_max_size=2000 in this test or with (change in ray_trace_maps.py code):
	# ray_vectors,ray_origins,ray_lengths = numpy.array([[ 0.39833941,  0.59701244, -0.69634895]]),\
	# numpy.array([[ 0.4545077,   0.21786684,  0.66247976]]), numpy.array([ 0.6])
	# nx_map,ny_map=1,1
	# for (idim=0 ; idim < 2 ; idim++){
	# v = 0,500000124057283 = (0,624999984492840-0,6875)/0,125 vs v = 0,5 =(0,625-0,6875)/0,125
	# and  0,624999984492840 = 0,66247976-(0,053823267066261*0,69634895) vs 0,625 = 0,66247976-(0,05382327*0,69634897)
	
#}}}
	
def test_OctreeRayTracer_OpenCL():#{{{
	misc.init_OpenCl()
	if misc.OpenCL_initialized:
	#if True:
		octree_dset = pymses.sources.ramses.octree.CameraOctreeDataset.\
		      from_hdf5(os.path.join(PWD, "test_amr_dset.h5"))
		rho_op = ScalarOperator(lambda dset: dset["rho"])
		# camera center small shift is still with OctreeRayTracer to avoid grid limit pb
		cam  = Camera(center=[0.501, 0.501, 0.501], line_of_sight_axis='z', region_size=[.3, .3],\
			      distance=0.3, far_cut_depth=0.3,\
			up_vector='y', map_max_size=3, log_sensitive=True)
		
		rt = OctreeRayTracer(octree_dset)
		# print rt.process(rho_op, cam, rgb=False, use_C_code=True)
		map, lvlmax_map = rt.process(rho_op, cam, rgb=False, use_openCL=True)
		
		map = apply_log_scale(map)
		print("map :")
		print(map)
		print("asserted_rt_map :")
		print(asserted_rt_map)
		assert(abs(map - asserted_rt_map) < 1e-3).all()
		print(lvlmax_map)
		print(asserted_lvlmax_map)
		assert(lvlmax_map == asserted_lvlmax_map).all()
			
#}}}

def test_Splatting():#{{{
	octree_dset = pymses.sources.ramses.octree.OctreeDataset.\
	      from_hdf5(os.path.join(PWD, "test_amr_dset.h5"))
	rho_op = ScalarOperator(lambda dset: dset["rho"])
	cam  = Camera(center=[0.5, 0.5, 0.5], line_of_sight_axis='z', region_size=[.3, .3],\
		      distance=0.3, far_cut_depth=0.3,\
		up_vector='y', map_max_size=3, log_sensitive=True)
	
	class EmptyOutput(object):
		def __init__(self):
			self.info = {}
			self.info["levelmin"] = 3
			self.info["levelmax"] = 3
			self.info["dom_decomp"] = None
			self.output_repos = None
			self.iout = None
	ro = EmptyOutput()
	
	mp = MapFFTProcessor(octree_dset, ro.info, pre_flatten=False)
	map = mp.process(rho_op, cam)
	map = apply_log_scale(map)
	print(map)
	asserted_map = numpy.array([[-0.82526528, -0.62671673, -0.71275912],
				    [-0.86070975, -0.64625094, -0.71627191],
				    [-1.27243366, -1.02005721, -1.05291437]])
	print(asserted_map)
	assert(abs(map - asserted_map) < 1e-6).all()
#}}}

def test_OctreeRayTracer_RGB():#{{{
	octree_dset = pymses.sources.ramses.octree.CameraOctreeDataset.\
	      from_hdf5(os.path.join(PWD, "test_amr_dset.h5"))
	rho_op = ScalarOperator(lambda dset: dset["rho"])
	cam  = Camera(center=[0.5, 0.5, 0.5], line_of_sight_axis='z', region_size=[.3, .3],\
		      distance=0.3, far_cut_depth=0.3,\
		up_vector='y', map_max_size=3, log_sensitive=True)
	
	cltf = ColorLinesTransferFunction( (-5.0, 2.0) )
	cltf.add_line(-2.0, 0.1)
	cltf.add_line(.0, 0.1)
	cltf.add_line(2., 0.1)
	cam.set_color_transfer_function(cltf)
			
	rt = OctreeRayTracer(octree_dset)
	map=rt.process(rho_op, cam, return_image=False)
	
	print(map[0])
	asserted_map = numpy.array([[0.12494199, 0., 0.57994005],
				   [0.12494199, 0., 0.57994005],
				   [0.12446702, 0., 0.57773542]])
	print(asserted_map)
	assert(abs(map[0] - asserted_map) < 1e-6).all()
#}}}
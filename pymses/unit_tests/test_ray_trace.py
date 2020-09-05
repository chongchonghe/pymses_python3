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
test_ray_trace.py -- test module for pymses octree ray tracing
"""

from pymses.analysis.visualization.raytracing.ray_trace import ray_trace_amr
from pymses.analysis.visualization import Operator
import pymses.sources.ramses.hilbert as hilbert
from pymses.sources.ramses.octree import RamsesOctreeDataset

import numpy

def bluid_simple_dset(various_value = False):
	###############################################
	#
	# Hydro rho value :
	#     
	#   .-------------------------------.
	#   |       |       |       | 4 | 5 |
	#   |   3   |   3   |   4   |---|---|
	#   |       |       |       | 4 | 4 |
	#   |-------|-------|-------|-------|
	#   |       |       |       |       |
	#   |   3   |   3   |   4   |   4   |
	#   |       |       |       |       |
	#   |-------------------------------|
	#   |       |       |       |       |
	#   |   1   |   1   |   2   |   2   |
	#   |       |       |       |       |
	#   |-------|-------|-------|-------|
	#   |       |       |       |       |
	#   |   1   |   1   |   2   |   2   |
	#   |       |       |       |       |
	#   |-------------------------------|
	#
	# Block with rho value 1 and 3 belongs to cpu (or dset) 1
	# Block with rho value 2 and 4 and 5 belongs to cpu (or dset) 2
	################################################
	
	ndim = 2
	simulation_level_max = 3 # max / data structure
	read_level_max = 3 # max / reading level (user can set depthmax < simulation_level_max to go faster) = amr_struct.readlmax
	ncpu = 2
	
	# Define the HilbertDomainDecomp needed with the dset to do the ray tracing
	info = {}
	cell_keymin = numpy.array([0.0, 32.0])
	cell_keymax = numpy.array([32.0, 64.0])
	corner_keymin = cell_keymin * 2**ndim
	corner_keymax = cell_keymax * 2**ndim
	dd = hilbert.HilbertDomainDecomp(ndim, corner_keymin, corner_keymax, (1,simulation_level_max))
	info["dom_decomp"] = dd
	
	# Define a ramses amr octree test dset 1
	
	header = {}
	header["ndim"] = ndim
	header["levelmax"] = simulation_level_max
	header["noutputs"] = 1
	header["ncpu"] = ncpu
	header["ncoarse"] = 1
	header["boxlen"] = 1.0
	struct = {}
	struct["ndim"] = ndim
	struct["twotondim"] = 2**ndim
	struct["ngrids"] = 3
	struct["readlmax"] = read_level_max
	struct["levelmax"] = simulation_level_max
	struct["son_indices"] = numpy.array([[1,-1,2,-1],[-1,-1,-1,-1],[-1,-1,-1,-1]], 'i')
	struct["grid_centers"] = numpy.array([[0.5,0.5],[0.25,0.25],[0.25,0.75]])
	struct["ncpu"] = ncpu
	struct["ngridlevel"] = numpy.array([[1,2,0],[0,0,0]], 'i') # 1 lvl 1 block, 2 lvl 2 blocks and 0 lvl 3 block
	struct["ngridbound"] = numpy.zeros((0,simulation_level_max), 'i') # not used??
	struct["grid_indices"] = numpy.array(list(range(struct["ngrids"])), 'i') # utility?
	amr_dicts = (header,struct)
	dset_1 = RamsesOctreeDataset(amr_dicts)
	dset_1.icpu = 0
	
	# Add the hydro rho data to the dset 1
	if various_value:
		dset_1.add_scalars("rho",numpy.array([[0.1,0.2,0.4,0.8],[0.01,0.02,0.04,0.08],[0.0001,0.0002,0.0004,0.0008]]))
	else:
		dset_1.add_scalars("rho",numpy.array([[1,2,3,4.0625],[1,1,1,1],[3,3,3,3]]))
	
	# Define the ramses amr octree test dset 2
	header = {}
	header["ndim"] = ndim
	header["levelmax"] = simulation_level_max
	header["noutputs"] = 2
	header["ncpu"] = ncpu
	header["ncoarse"] = 1
	header["boxlen"] = 1.0
	struct = {}
	struct["ndim"] = ndim
	struct["twotondim"] = 2**ndim
	struct["ngrids"] = 4
	struct["readlmax"] = read_level_max
	struct["levelmax"] = simulation_level_max
	struct["son_indices"] = numpy.array([[-1,1,-1,2],[-1,-1,-1,-1],[-1,-1,-1,3],[-1,-1,-1,-1]], 'i')
	struct["grid_centers"] = numpy.array([[0.5,0.5],[0.75,0.25],[0.75,0.75],[0.875,0.875]])
	struct["ncpu"] = ncpu
	struct["ngridlevel"] = numpy.array([[0,0,0],[1,2,1]], 'i') # 1 lvl 1 block, 2 lvl 2 blocks and 1 lvl 3 block
	struct["ngridbound"] = numpy.zeros((0,simulation_level_max), 'i') # not used??
	struct["grid_indices"] = numpy.array(list(range(struct["ngrids"])), 'i') # utility?
	amr_dicts = (header,struct)
	dset_2 = RamsesOctreeDataset(amr_dicts)
	dset_2.icpu = 1
	
	# Add the hydro rho data to the dset 2
	# Note that amr cells order in a block is:
	#   .-------.
	#   | 3 | 4 |
	#   |---|---|
	#   | 1 | 2 |
	#   .-------.
	if various_value:
		dset_2.add_scalars("rho",numpy.array([[0.1,0.2,0.4,0.8],[0.001,0.002,0.004,0.008],[0.00001,0.00002,0.00004,0.00008],[0.000001,0.000002,0.000004,0.000008]]))
	else:
		dset_2.add_scalars("rho",numpy.array([[1,2,3,4.0625],[2,2,2,2],[4,4,4,4.25],[4,4,4,5]]))
	
	return (dset_1, dset_2, info)

def build_simple_operator():
	# Define the data operator which will be used in the final ray trace process
	class myOp(Operator):
		def __init__(self, d):
			Operator.__init__(self, d, is_max_alos=False)

		def operation(self, int_dict):
			map = list(int_dict.values())[0]
			return map

	denom_func = lambda dset: (dset["rho"])
	op = myOp({"rho": denom_func})
	
	return op

def test_rt2d_simple():#{{{
	# Test the ray trace routine with a small test case
	###############################################
	#
	# Hydro rho value :
	#     
	#   .-------------------------------.
	#   |       |       |       | 4 | 5 |
	#   |   3   |   3   |   4   |---|---|
	#   |       |       |       | 4 | 4 |
	#   |-------|-------|-------|-------|
	#   |       |       |       |       |
	#   |   3   |   3   |   4   |   4   |
	#   |       |       |       |       |
	#   |-------------------------------|
	#   |       |       |       |       |
	#   |   1   |   1   |   2   |   2   |
	#   |       |       |       |       |
	#   |-------|-------|-------|-------|
	#   |       |       |       |       |
	#   |   1   |   1   |   2   |   2   |
	#   |       |       |       |       |
	#   |-------------------------------|
	#      ^      ^        ^          ^ 
	#      |      |        |          |
	# ray  0      1        2          3
	#
	# Block with rho value 1 and 3 belongs to cpu (or dset) 1
	# Block with rho value 2 and 4 and 5 belongs to cpu (or dset) 2
	################################################
	
	# Define rays to be traced 
	ray_origins = numpy.array([[0.1,0.],[0.3,0.],[0.7,0.],[0.9,0.]])
	ray_vectors = numpy.array([[0.,1.],[0.,1.],[0.,1.],[0.,1.]])
	ray_lengths = numpy.array([1.,.6,1.,1.])
	
	# Get datasets and operator
	(dset_1, dset_2, info) = bluid_simple_dset()
	op = build_simple_operator()
	
	# Do the ray trace through the first dset:
	map, ray_length_mapsDset = ray_trace_amr(dset_1, ray_origins, ray_vectors, ray_lengths, op, info)
	
	map_asserted_result=numpy.array([[1*0.25+1*0.25+3*0.25+3*0.25],[1*0.25+1*0.25+3*.1],[0.],[0.]])
	print("\nmap_asserted_result =",map_asserted_result)
	print("map =",map)
	assert (abs(map_asserted_result - map) < 10e-15).all()
	print("ray_length_mapsDset =", ray_length_mapsDset)
	assert (ray_length_mapsDset == [[1.],[.6],[0.],[0.]]).all()
	
	# Do the ray trace through the second dset:
	map, ray_length_mapsDset= ray_trace_amr(dset_2, ray_origins, ray_vectors, ray_lengths, op, info)
	
	map_asserted_result=numpy.array([[0.],[0.],[2*0.25+2*0.25+4*0.25+4*0.25],[2*0.25+2*0.25+4*0.25+4*0.125+5*0.125]])
	print("\nmap_asserted_result =",map_asserted_result)
	print("map =",map)
	assert (abs(map_asserted_result - map) < 10e-15).all()
	
	# Do the ray trace through the second dset with level_max:
	map, ray_length_mapsDset= ray_trace_amr(dset_2, ray_origins, ray_vectors, ray_lengths, op, info, level_max = 2)

	map_asserted_result=numpy.array([[0.],[0.],[2*0.25+2*0.25+4*0.25+4*0.25],[2*0.25+2*0.25+4*0.25+4.25*0.25]])
	print("\nmap_asserted_result =",map_asserted_result)
	print("map =",map)
	assert (abs(map_asserted_result - map) < 10e-15).all()
#}}}

def test_rt2d_simple_diagonal():#{{{
	# Test the ray trace routine with a small test case
	###############################################
	#
	# Hydro rho value :
	#     
	#   .-------------------------------.
	#   |       |       |       | 4 | 5 |
	#   |   3   |   3   |   4   |---|---|
	#   |       |       |       | 4 | 4 |
	#   |-------|-------|-------|-------|
	#   |       |       |       |       |
	#   |   3   |   3   |   4   |   4   |
	#   |       |       |       |       |
	#   |-------------------------------|
	#   |       |       |       |       |
	#   |   1   |   1   |   2   |   2   |
	#   |       |       |       |       |
	#   |-------|-------|-------|-------|
	#   |       |       |       |       |
	#   |   1   |   1   |   2   |   2   |
	#   |       |       |       |       |
	#   |-------------------------------|
	#     /  
	#ray
	#
	# Block with rho value 1 and 3 belongs to cpu (or dset) 1
	# Block with rho value 2 and 4 and 5 belongs to cpu (or dset) 2
	################################################
	
	# Define rays to be traced 
	ray_origins = numpy.array([[.1,0.]])
	ray_vectors = numpy.array([[.1*1.01**-.5,1.*1.01**-.5]]) # so that ray ends at ([0.1,1])
	ray_lengths = numpy.array([2.])
	
	# Get datasets and operator
	(dset_1, dset_2, info) = bluid_simple_dset()
	op = build_simple_operator()
	
	# Do the ray trace through the first dset:
	map, ray_length_mapsDset = ray_trace_amr(dset_1, ray_origins, ray_vectors, ray_lengths, op, info)
	
	map_asserted_result=numpy.array([[2*1.01**.5]])
	print("\nmap_asserted_result =",map_asserted_result)
	print("map =",map)
	print("ray_length_mapsDset", ray_length_mapsDset)
	assert (abs(map_asserted_result - map) < 10e-15).all()
	assert (abs(ray_length_mapsDset - [[1.01**.5]]) < 1e-15).all()
#}}}

def test_rt2d_vicious_boundary_case():#{{{
	# Test the ray trace routine with a small test case
	###############################################
	#
	# Hydro rho value :
	#     
	#   .-------------------------------.
	#   |       |       |       | 4 | 5 |
	#   |   3   |   3   |   4   |---|---|
	#   |       |       |       | 4 | 4 |
	#   |-------|-------|-------|-------|
	#   |       |       |       |       |
	#   |   3   |   3   |   4   |   4   |
	#   |       |       |       |       |
	#   |-------------------------------|
	#   |       |       |       |       |
	#   |   1   |   1   |   2   |   2   |
	#   |       |       |       |       |
	#   |-------|-------|-------|-------|
	#   |       |       |       |       |
	#   |   1   |   1   |   2   |   2   |
	#   |       |       |       |       |
	#   |-------------------------------|
	#   ^ /     ^       ^           ^ 
	#   |       |       |           |
	#   0 1     2       3           4
	#
	# The second ray pass trough the middle of block with value 3
	#
	# Block with rho value 1 and 3 belongs to cpu (or dset) 1
	# Block with rho value 2 and 4 and 5 belongs to cpu (or dset) 2
	################################################
	
	# Define rays to be traced 
	ray_origins = numpy.array([[0.,0.],[.1,0.],[.25,0.],[.5,0.],[0.875,0.]])
	ray_vectors = numpy.array([[0.,1.],[.2*1.04**-.5,1.*1.04**-.5],[0.,1.],[0.,1.],[0.,1.]])
	ray_lengths = numpy.array([2.,2.,2.,2.,2.])
	# The second ray pass trough the middle of block with value 3
	
	# Get datasets and operator
	(dset_1, dset_2, info) = bluid_simple_dset()
	op = build_simple_operator()
	
	# Do the ray trace through the first dset:
	map, ray_length_mapsDset = ray_trace_amr(dset_1, ray_origins, ray_vectors, ray_lengths, op, info)
	
	map_asserted_result=numpy.array([[2.],[2*1.04**.5],[2.],[0.],[0.]])
	print("\nThe second ray is a simple diagonal")
	print("map_asserted_result =",map_asserted_result)
	print("map =",map)
	assert (abs(map_asserted_result - map) < 10e-15).all()
	
	# Do the ray trace through the second dset:
	map, ray_length_mapsDset = ray_trace_amr(dset_2, ray_origins, ray_vectors, ray_lengths, op, info)
	
	map_asserted_result=numpy.array([[0.],[0.],[0.],[3.],[3.+1./8.]])
	print("\nmap_asserted_result =",map_asserted_result)
	print("map =",map)
	assert (abs(map_asserted_result - map) < 10e-15).all()
#}}}

def test_rt2d_complex():#{{{
	# Test the ray trace routine with the same amr structure as test_rt2d_simple()
	# but with more specific values to reveal eventual mistakes
	
	# Define rays to be traced trough the first dataset
	ray_origins = numpy.array([[0.1,0.],[0.3,0.]])
	ray_vectors = numpy.array([[0.,1.],[0.,1.]])
	ray_lengths = numpy.array([2.,2.])
	
	# Get datasets and operator
	(dset_1, dset_2, info) = bluid_simple_dset(various_value = True)
	op = build_simple_operator()
	
	# Do the ray trace through the first dset:
	map, ray_length_mapsDset = ray_trace_amr(dset_1, ray_origins, ray_vectors, ray_lengths, op, info)
	
	map_asserted_result=numpy.array([[0.01*0.25+0.04*0.25+0.0001*0.25+0.0004*0.25],\
		[0.02*0.25+0.08*0.25+0.0002*0.25+0.0008*0.25]])
	print("\nmap_asserted_result =",map_asserted_result)
	print("map =",map)
	assert (abs(map_asserted_result - map) < 10e-15).all()
#}}}

def test_rt2d_complex_WITHOUT_C_CODE():#{{{
	# Test the ray trace routine with the same amr structure as test_rt2d_simple()
	# but with more specific values to reveal eventual mistakes
	
	# Define rays to be traced trough the first dataset
	ray_origins = numpy.array([[0.1,0.],[0.3,0.]])
	ray_vectors = numpy.array([[0.,1.],[0.,1.]])
	ray_lengths = numpy.array([1.,1.])
	
	# Get datasets and operator
	(dset_1, dset_2, info) = bluid_simple_dset(various_value = True)
	op = build_simple_operator()
	
	# Do the ray trace through the first dset:
	map, ray_length_mapsDset = ray_trace_amr(dset_1, ray_origins, ray_vectors, ray_lengths, op, info, use_C_code = False)
	
	map_asserted_result=numpy.array([[0.01*0.25+0.04*0.25+0.0001*0.25+0.0004*0.25],\
		[0.02*0.25+0.08*0.25+0.0002*0.25+0.0008*0.25]])
	print("\nmap_asserted_result =",map_asserted_result)
	print("map =",map)
	assert (abs(map_asserted_result - map) < 10e-15).all()
#}}}

def bluid_dset_with_overlap():
	###############################################
	#
	# Hydro rho value :
	#     
	#   .---------------------------------------------------------------.
	#   |               |       | 3 | 1 |       |       |               |
	#   |               |   1   |---|---|   4   |   4   |               |
	#   |               |       | 1 | 1 |       |       |               |
	#   |       1       |-------|-------|-------|-------|       6       |
	#   |               |       |       |       |       |               |
	#   |               |   1   |   2   |   4   |   4   |               |
	#   |               |       |       |       |       |               |
	#   |-------------------------------|---------------|---------------|
	#   |       |       |       |       | 4 | 5 |       |               |
	#   |   1   |   3   |   2   |   2   |---|---|   4   |               |
	#   |       |       |       |       | 4 | 4 |       |               |
	#   |-------|-------|-------|-------|---|---|-------|       6       |
	#   |       |       |       | 2 | 2 | 1 | 1 |       |               |
	#   |   1   |   1   |   2   |---|---|---|---|   4   |               |
	#   |       |       |       | 1 | 1 | 1 | 4 |       |               |
	#   |-------------------------------|-------------------------------|
	#   |       |       | 1 | 1 | 1 | 2 |       |       |               |
	#   |   2   |   2   |---|---|---|---|   4   |   4   |               |
	#   |       |       | 1 | 2 | 1 | 1 |       |       |               |
	#   |-------|-------|-------|-------|-------|-------|       6       |
	#   |       |       |       |       |       |       |               |
	#   |   2   |   2   |   2   |   1   |   4   |   4   |               |
	#   |       |       |       |       |       |       |               |
	#   |-------------------------------|---------------|---------------|
	#   |       |       |               |               |               |
	#   |   1   |   1   |               |               |               |
	#   |       |       |               |               |               |
	#   |-------|-------|       1       |       6       |       6       |
	#   |       |       |               |               |               |
	#   |   1   |   1   |               |               |               |
	#   |       |       |               |               |               |
	#   .---------------------------------------------------------------.
	#      ^              ^           ^       ^ 
	#      |              |           |       |
	# ray  0              1           2       3
	# Block with rho value 1 and 2 and 3 belongs to cpu (or dset) 1
	# Block with rho value 4 and 5 and 6 belongs to cpu (or dset) 2
	################################################
	
	
	
	
	
	ndim = 2
	simulation_level_max = 4 # max / data structure
	read_level_max = 4 # max / reading level (user can set depthmax < simulation_level_max to go faster)
	ncpu = 2
	
	# Define the HilbertDomainDecomp needed with the dset to do the ray tracing
	info = {}
	cell_keymin = numpy.array([0.0, 131.0]) # 131 = 128 + 3 additional blocks for cpu 1
	cell_keymax = numpy.array([131.0, 256.0])
	corner_keymin = cell_keymin * 2**ndim
	corner_keymax = cell_keymax * 2**ndim
	dd = hilbert.HilbertDomainDecomp(ndim, corner_keymin, corner_keymax, (1,simulation_level_max))
	info["dom_decomp"] = dd
	grids, orders, no = dd.minimal_domain(0, read_lmax=4)
	assert (grids == numpy.array([[ 0.25,    0.25   ],
								  [ 0.25,    0.75   ],
								  [ 0.53125, 0.53125],
								  [ 0.53125, 0.59375],
								  [ 0.59375, 0.59375]])).all()
	assert (orders == numpy.array([1,1,4,4,4])).all()
	assert no == 0
	
	# Define a ramses amr octree test dset 1
	
	header = {}
	header["ndim"] = ndim
	header["levelmax"] = simulation_level_max
	header["noutputs"] = 1
	header["ncpu"] = ncpu
	header["ncoarse"] = 1
	header["boxlen"] = 1.0
	struct = {}
	struct["ndim"] = ndim
	struct["twotondim"] = 2**ndim
	struct["ngrids"] = 18
	struct["readlmax"] = read_level_max
	struct["levelmax"] = simulation_level_max
	struct["son_indices"] = numpy.array([[1,-1,2,3],\
		[4,-1,5,6],[7,8,-1,9],[10,-1,11,-1],\
		[-1,-1,-1,-1],[-1,-1,-1,-1],[-1,-1,12,13],[-1,-1,-1,-1],[-1,14,-1,-1],[-1,-1,-1,15],[16,-1,17,-1],[-1,-1,-1,-1],\
		[-1,-1,-1,-1],[-1,-1,-1,-1],[-1,-1,-1,-1],[-1,-1,-1,-1],[-1,-1,-1,-1],[-1,-1,-1,-1]], 'i')
	struct["grid_centers"] = numpy.array([[0.5,0.5],\
		[0.25,0.25],[0.25,0.75],[0.75,0.75],\
		[0.125,0.125],[0.125,0.375],[0.375,0.375],[0.125,0.625],[0.375,0.625],[0.375,0.875],[0.625,0.625],[0.625,0.875],\
		[0.3125,0.4375],[0.4375,0.4375],[0.4375,0.5625],[0.4375,0.9375],[0.5625,0.5625],[0.5625,0.6875]])
	struct["ncpu"] = ncpu
	struct["ngridlevel"] = numpy.array([[1,2,7,5],[0,1,1,1]], 'i')
	struct["ngridbound"] = numpy.zeros((0,simulation_level_max), 'i')
	struct["grid_indices"] = numpy.array(list(range(struct["ngrids"])), 'i')
	amr_dicts = (header,struct)
	dset_1 = RamsesOctreeDataset(amr_dicts)
	dset_1.icpu = 0
	
	# Add the hydro rho data to the dset 1
	dset_1.add_scalars("rho",numpy.array([[1,5,2,5],\
		[1,1,2,1],[1,2,1,2],[4,6,4,6],\
		[1,1,1,1],[2,2,2,2],[2,1,1,1],[1,1,1,3],[2,2,2,2],[1,2,1,2],[2,4,4,4],[4,4,4,4],\
		[1,2,1,1],[1,1,1,2],[1,1,2,2],[1,1,3,1],[1,4,1,1],[4,4,4,5]]))
	
	# Define the ramses amr octree test dset 2
	header = {}
	header["ndim"] = ndim
	header["levelmax"] = simulation_level_max
	header["noutputs"] = 2
	header["ncpu"] = ncpu
	header["ncoarse"] = 1
	header["boxlen"] = 1.0
	struct = {}
	struct["ndim"] = ndim
	struct["twotondim"] = 2**ndim
	struct["ngrids"] = 4
	struct["readlmax"] = read_level_max
	struct["levelmax"] = simulation_level_max
	struct["son_indices"] = numpy.array([[-1,1,-1,2],[-1,-1,-1,-1],[-1,-1,-1,3],[-1,-1,-1,-1]], 'i')
	struct["grid_centers"] = numpy.array([[0.5,0.5],[0.75,0.25],[0.75,0.75],[0.875,0.875]])
	struct["ncpu"] = ncpu
	struct["ngridlevel"] = numpy.array([[0,0,0],[1,2,1]], 'i') # 1 lvl 1 block, 2 lvl 2 blocks and 1 lvl 3 block
	struct["ngridbound"] = numpy.zeros((0,simulation_level_max), 'i')
	struct["grid_indices"] = numpy.array(list(range(struct["ngrids"])), 'i')
	amr_dicts = (header,struct)
	dset_2 = RamsesOctreeDataset(amr_dicts)
	dset_2.icpu = 1
	
	# Add the hydro rho data to the dset 2
	dset_2.add_scalars("rho",numpy.array([[1,2,3,4.0625],[2,2,2,2],[4,4,4,4.25],[4,4,4,5]]))
	
	return (dset_1, dset_2, info)
	
def test_rt2d_with_overlap():#{{{
	# Test the ray trace routine with a small test case

	# Define rays to be traced 
	ray_origins = numpy.array([[.1,0.],[.3,0.],[.45,0.],[.6,0.]])
	ray_vectors = numpy.array([[0.,1.],[0.,1.],[0.,1.],[0.,1.]])
	ray_lengths = numpy.array([1.,1.,1.,1.])
	#ray_origins = numpy.array([[.6,0.]])
	#ray_vectors = numpy.array([[0.,1.]])
	# Get,[ datasets and operator
	(dset_1, dset_2, info) = bluid_dset_with_overlap()
	op = build_simple_operator()
	
	# Do the ray trace through the first dset:
	map, ray_length_mapsDset = ray_trace_amr(dset_1, ray_origins, ray_vectors, ray_lengths, op, info)
	
	map_asserted_result=numpy.array([[1.*.75+2.*.25],[1.*.5+2.*.375+1.*.125],[1.*.5+2.*.375+1.*.125],[0.0625]])
	print("\nmap_asserted_result =",map_asserted_result)
	print("map =",map)
	assert (abs(map_asserted_result - map) < 10e-15).all()
#}}}

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
import numpy
from time import time
from pymses.utils import misc
import pymses.analysis.visualization.raytracing
from pymses.analysis.visualization.transfer_functions import ColorLinesTransferFunction
from os.path import dirname

class openCL_RT_singleton(object):
	instance = None
	# WARNING : this singleton has class variables that are initialized
	# in pymses.utils.misc.init_OpenCl() like that :
	# prg = None
	# dset_loaded = False
	# rgb_buffers_loaded = False
	def __new__(classe, *args, **kargs): 
		if classe.instance is None:
			classe.instance = object.__new__(classe, *args, **kargs)
		return classe.instance
	
	def build_prg(self,):
		import pyopencl as cl
		# create kernel program:
		from string import Template
		file_dir = dirname(pymses.analysis.visualization.raytracing.__file__)
		opencl_code_linestring = open(file_dir+'/ray_trace_openCL.c', 'r').read()
		code_pattern= Template(opencl_code_linestring)
		if misc.OpenCL_selected_precision == 'double':
			precision_pragma = '#pragma OPENCL EXTENSION cl_khr_fp64: enable'
		else:
			precision_pragma = ''
		substitutions = {'precision':misc.OpenCL_selected_precision, 'precision_pragma':precision_pragma}
		code = code_pattern.substitute(substitutions)
		self.prg = cl.Program(misc.OpenCL_ctx, code).build()
	
	def load_dset(self, dset, op, np_precision):
		print("OpenCL load_dset")
		import pyopencl as cl
		max_read_level = dset.amr_struct["readlmax"]
		dx = numpy.empty(max_read_level+2,np_precision)
		sons = dset.amr_struct["son_indices"].astype(numpy.int32)
		cell_levels = dset.amr_struct["cell_levels"].astype(numpy.uint32)
		neighbors = dset.amr_struct["neighbors"].astype(numpy.uint32)
		grid_centers = dset.amr_struct["grid_centers"].astype(np_precision)
		cell_centers = dset.get_cell_centers().astype(np_precision)
		################### Precompute the cell sizes #########################
		dx[0]=0.5
		for ilevel in range(1,max_read_level+2):
			dx[ilevel] = dx[ilevel-1] / 2.0
		###########################################################################
		# create dset related buffers:
		self.grid_centers_buf = cl.Buffer(misc.OpenCL_ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=grid_centers)
		self.cell_centers_buf = cl.Buffer(misc.OpenCL_ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=cell_centers)
		self.sons_buf = cl.Buffer(misc.OpenCL_ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=sons)
		self.cell_levels_buf = cl.Buffer(misc.OpenCL_ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=cell_levels)
		self.neighbors_buf = cl.Buffer(misc.OpenCL_ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=neighbors)
		self.dx_buf = cl.Buffer(misc.OpenCL_ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=dx)
		self.load_scalar_data(dset, op, np_precision)
		
		self.dset_loaded = True
	
	def load_scalar_data(self, dset, op, np_precision):
		import pyopencl as cl
		ndim = dset.amr_header["ndim"]
		ngrids = dset.amr_struct["ngrids"]
		twotondim = 1 << ndim
		nfunc = 3 #op.nscal_func()
		scalar_data = numpy.empty((nfunc, ngrids, twotondim),np_precision)
		ifunc = 0
		for (key, scal_func) in op.iter_scalar_func():
			scalar_data[ifunc,:,:] = scal_func(dset)
			ifunc += 1
		self.scalar_data_buf = cl.Buffer(misc.OpenCL_ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=scalar_data)
	
	def load_rgb_buffers(self, tf, np_precision):
		import pyopencl as cl
		self.gamma = np_precision(tf.gamma)
		vlines, wlines, rlines, glines, blines, alines = tf.vwrgba
		self.nlines = numpy.uint32(vlines.size)
		# create rgb buffers
		self.vlines_buf = cl.Buffer(misc.OpenCL_ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=vlines.astype(np_precision))
		self.wlines_buf = cl.Buffer(misc.OpenCL_ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=wlines.astype(np_precision))
		self.rlines_buf = cl.Buffer(misc.OpenCL_ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=rlines.astype(np_precision))
		self.glines_buf = cl.Buffer(misc.OpenCL_ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=glines.astype(np_precision))
		self.blines_buf = cl.Buffer(misc.OpenCL_ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=blines.astype(np_precision))
		self.alines_buf = cl.Buffer(misc.OpenCL_ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=alines.astype(np_precision))
	
		self.tf = tf
		self.rgb_buffers_loaded = True

	def ray_trace_octree_openCL(self, dset, ray_origins, ray_vectors, ray_lengths,\
					op, color_tf, rgb, level_max=None, reload_scalar_field=False):
		# This code is an OpenCL adapted copy of ray_trace_octree_C (in ray_trace_C_code.c)
		import pyopencl as cl
		#rgb=False #todo
		max_op = numpy.uint32(op.is_max_alos()) #todo : test !
		#print rgb
		rgb = numpy.uint32(rgb)
		misc.init_OpenCl()
		precision_dict = {'float':numpy.float32, 'double':numpy.float64}
		np_precision = precision_dict[misc.OpenCL_selected_precision]
		nrays= ray_origins.shape[0]
		ndim = numpy.uint32(dset.amr_header["ndim"])
		ngrids = numpy.uint32(dset.amr_struct["ngrids"])
		max_read_level = numpy.uint32(dset.amr_struct["readlmax"])
		if level_max != None and max_read_level>level_max:
			max_read_level = numpy.uint32(level_max)
		ray_origins = ray_origins.astype(np_precision)
		ray_vects = ray_vectors.astype(np_precision)
		ray_lengths = ray_lengths.astype(np_precision)
		nfunc = numpy.uint32(3) #op.nscal_func()
		# output array:
		I = numpy.zeros((nrays, 3),np_precision)
		###########################################################################
		if misc.OpenCL_selected_precision == 'float':
			print("WARNING : simple float precision is very buggy if you rotate the view...")
		# create view related buffers:
		I_buf = cl.Buffer(misc.OpenCL_ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=I)
		ray_origins_buf = cl.Buffer(misc.OpenCL_ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=ray_origins)
		ray_vects_buf = cl.Buffer(misc.OpenCL_ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=ray_vects)
		ray_lengths_buf = cl.Buffer(misc.OpenCL_ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=ray_lengths)
		
		# build program
		#op_time = time()
		if self.prg==None: self.build_prg()
		
		# load needed data on device (i.e. create buffer)
		if not self.dset_loaded :
			self.load_dset(dset, op, np_precision)
		elif reload_scalar_field:
			self.load_scalar_data(dset, op, np_precision)
		if not isinstance(color_tf, ColorLinesTransferFunction):
			#  by default : init variable like in the amrviewer GUI
			color_tf = ColorLinesTransferFunction( (-5.0, 2.0) )
			color_tf.add_line(-2.0, 0.1)
			color_tf.add_line(.0, 0.1)
			color_tf.add_line(2., 0.1)
		if not self.rgb_buffers_loaded or not color_tf.similar(self.tf):
			self.load_rgb_buffers(color_tf, np_precision)
		
		# start executing the program:
		self.prg.rt(misc.OpenCL_queue, (nrays,), None, I_buf, ray_origins_buf,
				ray_vects_buf, self.grid_centers_buf,
				self.cell_centers_buf, self.sons_buf,
				self.cell_levels_buf, self.scalar_data_buf,
				self.neighbors_buf, ray_lengths_buf,	self.dx_buf,
				ndim, max_read_level, nfunc, max_op, ngrids, rgb, self.nlines,
				self.gamma, self.vlines_buf, self.wlines_buf,
				self.rlines_buf, self.glines_buf,
				self.blines_buf, self.alines_buf)
		# get the result:
		cl.enqueue_copy(misc.OpenCL_queue, I, I_buf).wait()
		#print "Opencl execution time : %.3fs"%(time() - op_time)
	
		return I

__all__ = ["openCL_RT_singleton"]

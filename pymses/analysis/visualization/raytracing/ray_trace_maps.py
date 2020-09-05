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
from pymses.filters import *
from .ray_trace import ray_trace_amr, ray_trace_octree
from pymses.analysis.visualization.raytracing.ray_trace_openCL import openCL_RT_singleton
from ..fft_projection.cmp_maps import MapProcessor
from ....sources.ramses.octree import CameraOctreeDatasource, CameraOctreeDataset
from ....sources.ramses import tree_utils
from time import time

class RayTracer(MapProcessor):#{{{
	r"""
	RayTracer class

	Parameters
	----------
	ramses_output   : :class:`~pymses.sources.ramses.output.RamsesOutput`
		ramses output from which data will be read to compute the map
	field_list      : ``list`` of ``string``
		list of all the required AMR fields to read (see :meth:`~pymses.sources.ramses.output.RamsesOutput.amr_source`)

	"""
	def __init__(self, ramses_output, field_list):#{{{
		MapProcessor.__init__(self, None)
		self.ro = ramses_output
		self.fields_to_read = field_list
	#}}}

	def process(self, op, camera, surf_qty=False, verbose=True, multiprocessing=True,\
		    source=None, use_hilbert_domain_decomp=True, use_C_code=True, use_bottom_up=False):#{{{
		r"""
		Map processing method : ray-tracing through data cube

		Parameters
		----------
		op              : :class:`~pymses.analysis.visualization.Operator`
			physical scalar quantity data operator
		camera          : :class:`~pymses.analysis.visualization.Camera`
			camera containing all the view params
		surf_qty        : ``boolean`` (default False)
			whether the processed map is a surface physical quantity. If True, the map
			is divided by the surface of a camera pixel.
		verbose		: ``boolean`` (default False)
			show more console printouts
		multiprocessing : ``boolean`` (default True)
			try to use multiprocessing (process cpu data file in parallel) to speed up
			the code (need more RAM memory, python 2.6 or higher needed)
		source : class:`~pymses.sources...` (default None)
			Optional : The source to process may be specified here if you want to reuse
			a CameraOctreeDatasource already loaded in memory for example (see
			pymses/bin/pymses_tf_ray_tracing.py)
		use_hilbert_domain_decomp : ``boolean`` (default True)
			If False, iterate on the whole octree for each cpu file(instead of iterating
			on the cpu minimal domain decomposition, which is faster)
		use_C_code : ``boolean`` (default True)
			Our pure C code is faster than the (not well optimized) Cython code,
			and should give the same result
		use_bottom_up : ``boolean`` (default False)
			Force the use of the bottom-up algorithm instead of the classic top-down on
			the octree. Use the "neighbors" array. DOESN'T WORK YET
		"""
		begin_time = time()
		# AMR DataSource preparation
		rlev = camera.get_required_resolution()
		self.ro.verbose = verbose
		if not verbose:
			# print minimal informations :
			print(self.ro.output_repos, "output", self.ro.iout)
		
		if source is None:
			source = self.ro.amr_source(self.fields_to_read)
			source.set_read_lmax(rlev)

		# Get rays info
		ray_vectors, ray_origins, ray_lengths = camera.get_rays()
		n_rays = ray_origins.shape[0]
		nx_map, ny_map = camera.get_map_size()

		# Data bounding box
		domain_bounding_box = camera.get_bounding_box()

		# Extended domain bounding box for big octree cells
		ext = 0.5**(self.ro.info["levelmin"])
		domain_bounding_box.min_coords = numpy.amax([domain_bounding_box.min_coords - ext,[0.,0.,0.]], axis=0)
		domain_bounding_box.max_coords = numpy.amin([domain_bounding_box.max_coords + ext,[1.,1.,1.]], axis=0)
	
		# Data spatial filtering
		rsource = RegionFilter(domain_bounding_box, source)
		print("dsets to process:",len(rsource._data_list))

		# Maps initialisation
		if use_bottom_up:
			maps = numpy.zeros((n_rays,3), dtype='d')
		else:
			maps = numpy.zeros((n_rays, op.nscal_func()), dtype='d')
		ray_length_maps = numpy.zeros((n_rays, op.nscal_func()), dtype='d')
		
		if isinstance(source, CameraOctreeDataset):
			# In this case there is only one dataset to process so:
			use_hilbert_domain_decomp = False
			multiprocessing = False
		
		# Try to use multiprocessing
		if multiprocessing:
			try:
				from multiprocessing import Process, Queue, cpu_count
				if cpu_count() == 1 :
					multiprocessing = False # don't use multiprocessing if there is only one cpu
			except Exception:
				print('WARNING: multiprocessing unavailable')
				multiprocessing = False
		if multiprocessing:
		#if False: # TODO : fix bug with bottum up
			#############################################################################
			# multiprocessing ray tracing, cpu files distribution (sort last rendering) #
			#############################################################################
			# CPU files to read
			cpu_full_list = rsource._data_list
			ncpufile = len(cpu_full_list)
			
			# Create one cpu_list for each node:
			t0 = time()
			ray_time = 0
			from pymses.utils import misc
			NUMBER_OF_PROCESSES = min(len(cpu_full_list), cpu_count(), misc.NUMBER_OF_PROCESSES_LIMIT)
			
			
			def process_dset(rsource, ray_origins, ray_vectors, ray_lengths, op, ro_info,\
					 cpu_task_queue, maps_queue, ray_length_maps_queue):
				# Utility method for the ray tracing multiprocessing method. 
				# It reads the source files and sum resulting dsets ray tracing
				maps = numpy.zeros((nx_map * ny_map, op.nscal_func()), dtype='d')
				mapsDset = numpy.zeros((nx_map * ny_map, op.nscal_func()), dtype='d')
				ray_length_maps = numpy.zeros((nx_map * ny_map, op.nscal_func()), dtype='d')
				for icpu in iter(cpu_task_queue.get, 'STOP'):
					dset = rsource.get_domain_dset(icpu)
					active_mask = dset.get_active_mask()
					g_levels =  dset.get_grid_levels()
					#dset.add_scalars("level", g_levels.reshape(g_levels.shape[0],1).repeat(2**dset.amr_header["ndim"],1))
					# We do the processing only if needed, i.e. only if the amr level min
					# of active cells in the dset is <= rlev
					if len(g_levels[active_mask]) > 0 and numpy.min(g_levels[active_mask]) <= rlev:
						mapsDset,ray_length_mapsDset = ray_trace_amr(dset, ray_origins, ray_vectors, \
							ray_lengths, op, self.ro.info, rlev, active_mask, g_levels, use_C_code = use_C_code,\
							use_hilbert_domain_decomp=use_hilbert_domain_decomp)
						ray_length_maps += ray_length_mapsDset
						if op.is_max_alos():
							numpy.maximum(maps, mapsDset, maps)
						else:
							maps += mapsDset
				maps_queue.put(maps)
				ray_length_maps_queue.put(ray_length_maps)
			# Create queues
			cpu_task_queue = Queue()
			maps_queue = Queue()
			ray_length_maps_queue = Queue()
			# Submit tasks
			for task in cpu_full_list:
				cpu_task_queue.put(task)
			# Start worker processes
			for i in range(NUMBER_OF_PROCESSES):
				Process(target=process_dset, args=(rsource, ray_origins, ray_vectors, \
					ray_lengths, op, self.ro.info, cpu_task_queue, maps_queue,\
					ray_length_maps_queue)).start()
			# Tell child processes to stop when they have finished
			for i in range(NUMBER_OF_PROCESSES):
				cpu_task_queue.put('STOP')
			# Get results
			for i in range(NUMBER_OF_PROCESSES):
				if op.is_max_alos():
					numpy.maximum(maps, maps_queue.get(), maps)
				else:
					maps += maps_queue.get()
				ray_length_maps += ray_length_maps_queue.get()
		else:
			###############################################
			# no multiprocessing : sequential ray tracing #
			###############################################
			for icpu in rsource._data_list:
				dset = rsource.get_domain_dset(icpu)
				active_mask = dset.get_active_mask()
				g_levels =  dset.get_grid_levels()
				#dset.add_scalars("level", g_levels.reshape(g_levels.shape[0],1).repeat(2**dset.amr_header["ndim"],1))
				# We do the processing only if needed, i.e. only if the amr level min of active cells in the dset is <= rlev
				if len(g_levels[active_mask]) > 0 and numpy.min(g_levels[active_mask]) <= rlev:
					if use_bottom_up:
						if "neighbors" not in dset.amr_struct:
							dset.amr_struct["neighbors"] = -numpy.ones((dset.amr_struct["ngrids"],
							   2*dset.amr_header["ndim"]), dtype='i')
							#dset.amr_struct["son_indices"][dset.amr_struct["son_indices"]>=dset.amr_struct["ngrids"]] = -1
							tree_utils.octree_compute_neighbors(dset, verbose=False)
						mapsDset = ray_trace_octree(dset, ray_origins, ray_vectors, ray_lengths,
							op, rgb=False, level_max=rlev, use_C_code=False, verbose=False)
						# WARNING : I forget the lvl_max map here, and I don't compute ray_length_maps
					else:
						mapsDset,ray_length_mapsDset = ray_trace_amr(dset, ray_origins, ray_vectors, \
							ray_lengths, op, self.ro.info, rlev, active_mask, g_levels, use_C_code=use_C_code,
							use_hilbert_domain_decomp=use_hilbert_domain_decomp)
						ray_length_maps += ray_length_mapsDset
					if op.is_max_alos():
						numpy.maximum(maps, mapsDset, maps)
					else:
						maps += mapsDset
					
		print("Ray trace process time = %.3fs"%(time() - begin_time))
		
		if not use_bottom_up:
			different_ray_length = numpy.unique(ray_length_maps)
			equal = True
			if len(different_ray_length) > 1:
				if (numpy.max(different_ray_length) - numpy.min(different_ray_length)) > ray_lengths[0] * 10e-3:
					equal = False
					print("Calculated ray lengths during the ray trace process are not always equal")
					if len(different_ray_length) < 5:
						print("ray_lengths[0] =", ray_lengths[0])
						for value in different_ray_length:
							print("There are", sum(ray_length_maps == value), "ray(s) with value", value)
			if equal:
				print("Calculated ray lengths during the ray trace process are all equal : visualized volume is complete ! :-)")
		
		ifunc=0
		map_dict = {}
		S = camera.get_pixel_surface()
		for key, func in op.iter_scalar_func():
			if (op.is_max_alos()+surf_qty):
				map_dict[key] = maps[:,ifunc].reshape(nx_map, ny_map)
			else:
				map_dict[key] = S * maps[:,ifunc].reshape(nx_map, ny_map)
			ifunc+=1
		map = op.operation(map_dict)
		
		return map
	#}}}

#}}}

class OctreeRayTracer(MapProcessor):#{{{
	r"""
	RayTracerDir class

	Parameters
	----------
	ramses_output   : :class:`~pymses.sources.ramses.output.RamsesOutput`
		ramses output from which data will be read to compute the map
	field_list      : ``list`` of ``string``
		list of all the required AMR fields to read (see :meth:`~pymses.sources.ramses.output.RamsesOutput.amr_source`)

	"""
	def __init__(self, * args):#{{{
		nargs = len(args)
		assert nargs in [1,2]
		if nargs == 1:
			dset = args[0]
			assert isinstance(dset, CameraOctreeDataset)
			MapProcessor.__init__(self, dset)
		else:
			MapProcessor.__init__(self, None)
			self.ro, self.fields_to_read = args
	#}}}

	def process(self, op, camera, surf_qty=False, return_image=True, rgb=True,
			use_C_code=True, use_openCL=False, dataset_already_loaded=False,
			reload_scalar_field=False):#{{{
		r"""
		Map processing method : directional ray-tracing through AMR tree

		Parameters
		op              : :class:`~pymses.analysis.visualization.Operator`
			physical scalar quantity data operator
		camera          : :class:`~pymses.analysis.visualization.Camera`
			camera containing all the view params
		surf_qty        : ``boolean`` (default False)
			whether the processed map is a surface physical quantity. If True, the map
			is divided by the surface of a camera pixel.
		return_image        : ``boolean`` (default True)
			if True, return a PIL image (when rgb option is also True), else it returns
			a numpy array map
		rgb        : ``boolean`` (default True)
			if True, this code use the camera.color_tf to compute a rgb image
			if False, this code doesn't use the camera.color_tf, and works like the
			standard RayTracer. Then it returns two maps : the requested map,
			and the AMR levelmax map
		use_C_code : ``boolean`` (default True)
			Our pure C code is faster than the (not well optimized) Cython code,
			and should give the same result
		use_openCL : ``boolean`` (default False)
			Experimental : use "pyopencl" http://pypi.python.org/pypi/pyopencl
		dataset_already_loaded : ``boolean`` (default False)
			Flag used with use_openCL=True to avoid reloading a dataset on the device
		reload_scalar_field : ``boolean`` (default False)
			Flag used with use_openCL=True and dataset_already_loaded=True to avoid
			reloading the dataset structure on the device while using a different scalar field
		"""
		if self.source is None:
			begin_time = time()
			# CameraOctreeDatasource creation
			source = self.ro.amr_source(self.fields_to_read)
			esize = 0.5**(self.ro.info["levelmin"]+1)
			cod = CameraOctreeDatasource(camera, esize, source, include_split_cells=False)
			self.source = cod.dset
			print("CameraOctreeDatasource loaded up to level", camera.get_required_resolution(),\
			"with ngrids =", self.source.amr_struct["ngrids"],\
			"(loading time = %.2fs"%(time() - begin_time), ")")
		
		# Get rays info
		t0=time()
		nx_map, ny_map = camera.get_map_size()
		ray_vectors, ray_origins, ray_lengths = camera.get_rays()
		n_rays = ray_origins.shape[0]
		rlev = camera.get_required_resolution()

		begin_time = time()
		if use_openCL:
			from pymses.utils import misc
			misc.init_OpenCl()
			if not misc.OpenCL_initialized:
				print("Error in init_OpenCl() : OpenCL won't be used !")
				use_openCL = False
		if use_openCL:
			if not dataset_already_loaded:
				# force the reloading of the (new?) dataset on the device for this new view
				openCL_RT_singleton().dset_loaded = False
			I = openCL_RT_singleton().ray_trace_octree_openCL(self.source, ray_origins, ray_vectors, ray_lengths,\
					op, camera.color_tf, rgb=rgb, level_max=rlev, reload_scalar_field=reload_scalar_field)
		else:
			try:
				# try to use multiprocessing on rays
				##########################################################################
				# multiprocessing ray tracing, pixel distribution (sort first rendering) #
				##########################################################################
				from multiprocessing import Process, cpu_count, Pipe
				from pymses.utils import misc
				NUMBER_OF_PROCESSES = min(n_rays/10000 + 1, cpu_count(), misc.NUMBER_OF_PROCESSES_LIMIT)
				if NUMBER_OF_PROCESSES == 1 :
					raise(Exception) # don't use multiprocessing
				s = n_rays/NUMBER_OF_PROCESSES+1
				
				def process_dset(child_conn, i_1, i_2):
					# Utility method for multiprocessing ray tracing
					maps = numpy.zeros((s+1, 3), dtype='d')
					maps = ray_trace_octree(self.source, ray_origins[i_1:i_2], ray_vectors[i_1:i_2],\
							ray_lengths[i_1:i_2], op, camera.color_tf, level_max=rlev, verbose=False, rgb=rgb,\
							use_C_code=use_C_code)
					child_conn.send(maps)
					child_conn.close()
				
				# Start worker processes
				parent_conn = []
				for i in range(NUMBER_OF_PROCESSES):
					p_c, child_c = Pipe()
					parent_conn.append(p_c)
					Process(target=process_dset, args=(child_c, i*s,(i+1)*s)).start()
				
				# Get results
				I = numpy.zeros((n_rays,3), dtype='d')
				for i in range(NUMBER_OF_PROCESSES):
					I[i*s:(i+1)*s] = parent_conn[i].recv()
			except Exception:
				print('No multiprocessing...')
				I = ray_trace_octree(self.source, ray_origins, ray_vectors, ray_lengths,\
						op, camera.color_tf, rgb=rgb, level_max=rlev, use_C_code=use_C_code)
		print("Octree ray trace processing time = %.3fs"%(time() - begin_time))
		if rgb:
			shape = I.shape
			I = I.reshape(shape[0]*shape[1])
			if sum(I!=I) !=0:
				print("Error : There are",sum(I!=I)," NaN value")
			if not return_image :
				return I.reshape((nx_map, ny_map, 3))
			else:
				import Image
				map = I.reshape((nx_map*ny_map, 3))
				#map = (map - min(map)) / (max(map) - min(map))
				map[:,0] = (map[:,0] - min(map[:,0])) / (max(map[:,0]) - min(map[:,0]))
				map[:,1] = (map[:,1] - min(map[:,1])) / (max(map[:,1]) - min(map[:,1]))
				map[:,2] = (map[:,2] - min(map[:,2])) / (max(map[:,2]) - min(map[:,2]))
				map = numpy.asarray(map*255, dtype='i')
				R_band = Image.new("L",(nx_map,ny_map))
				R_band.putdata(map[:,0])
				#import pylab as P
				#P.imshow(map[:,2].reshape((nx_map,ny_map)))
				#P.show()
				G_band = Image.new("L",(nx_map,ny_map))
				G_band.putdata(map[:,1])
				B_band = Image.new("L",(nx_map,ny_map))
				B_band.putdata(map[:,2])
				img = Image.merge("RGB", (R_band, G_band, B_band))
				return img.rotate(90)
		else:
			ifunc=0
			map_dict = {}
			S = camera.get_pixel_surface()
			for key, func in op.iter_scalar_func():
				if (op.is_max_alos()+surf_qty):
					map_dict[key] = I[:,ifunc].reshape(nx_map, ny_map)
				else:
					map_dict[key] = S * I[:,ifunc].reshape(nx_map, ny_map)
				ifunc+=1
			map = op.operation(map_dict)
	
			levelmax_map = I[:,2].reshape(nx_map, ny_map)
			return map, levelmax_map
	#}}}
#}}}

__all__ = ["RayTracer", "OctreeRayTracer"]

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
from .ray_trace import ray_trace_amr 
from ..fft_projection.cmp_maps import MapProcessor
from time import time
from pymses.utils.misc import balanced_cpu_list, chunks

class RayTracerMPI(MapProcessor):
	r"""
	RayTracer class

	Parameters
	----------
	ramses_output   : :class:`~pymses.sources.ramses.output.RamsesOutput`
		ramses output from which data will be read to compute the map
	field_list      : ``list`` of ``string``
		list of all the required AMR fields to read (see :meth:`~pymses.sources.ramses.output.RamsesOutput.amr_source`)
	remember_data   : ``boolean`` (default False)
		option to remember dataset loaded. Avoid reading the data again for each frame of a rotation movie.
		WARNING : The saved cache data don't update yet it's levelmax and cpu list, so use carefully this
				if zooming / moving too much inside the simulation box.

	"""
	def __init__(self, ramses_output, field_list, remember_data=False):
		MapProcessor.__init__(self, None)
		self.ro = ramses_output
		self.fields_to_read = field_list
		self.remember_data = remember_data
		self.cpu_node_list = []
		if remember_data:
			self.dsets={}
	def process(self, op, camera, surf_qty=False, use_balanced_cpu_list=False,
		    testing_ray_number_max=100, verbose=False, use_C_code=True):
		r"""
		Map processing method using MPI: ray-tracing through data cube

		Parameters
		----------
		op              : :class:`~pymses.analysis.visualization.Operator`
			physical scalar quantity data operator
		camera          : :class:`~pymses.analysis.visualization.Camera`
			camera containing all the view params
		surf_qty        : ``boolean`` (default False)
			whether the processed map is a surface physical quantity.
			If True, the map is divided by the surface of a camera pixel.
		use_balanced_cpu_list : ``boolean`` (default False)
			option to optimize the load balancing between MPI process,
			add an intial dsets testing before processing every rays
		testing_ray_number_max : ``boolean`` (default 100)
			number of testing ray for the balanced cpu list option
		verbose : ``boolean`` (default False)
			more printout (may flood the console out for big simulation with many cpu)
		use_C_code : ``boolean`` (default True)
			Our pure C code is faster than the (not well optimized) Cython code,
			and should give the same result
		"""
		from mpi4py import MPI
		# AMR DataSource preparation
		rlev = camera.get_required_resolution()
		#rlev = self.ro.info["levelmax"] # uncomment to get full amr lvl data used
		self.ro.verbose = verbose
		if verbose:
			print(self.ro.output_repos, "output", self.ro.iout)
	
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
	
		# Data spatial filtering <-> CPU files to read
		cpu_full_list = self.ro.dom_decomp.map_region(domain_bounding_box)
		ncpufile = len(cpu_full_list)
		
		# Maps initialisation
		mapsNodeI = numpy.zeros((n_rays, op.nscal_func()), dtype='d')
		maps = numpy.zeros((n_rays, op.nscal_func()), dtype='d')
		ray_length_mapsNodeI = numpy.zeros((n_rays, op.nscal_func()), dtype='d')
		ray_length_maps = numpy.zeros((n_rays, op.nscal_func()), dtype='d')
		
		#
		# parallelisation step on processing nodes with MPI
		#

		comm = MPI.COMM_WORLD
		processingNodeNumber = comm.Get_size()
		myrank = comm.Get_rank()
		name = MPI.Get_processor_name()
		processingNodeNumber = min(ncpufile, processingNodeNumber)
		if myrank == 0:print("ncpufile", ncpufile)
		balanced_cpu_list_overhead_timecost = 0
		if not self.remember_data or self.cpu_node_list == []:
			# build the appropriate cpu_node_list distribution if not already done
			if use_balanced_cpu_list:
				# MPI load balacing optimisation
				if myrank == 0:
					t0_balanced_cpu_list = time()
					if verbose: print("cpu_full_list=",cpu_full_list)
				testing_mask = numpy.zeros(nx_map*ny_map, dtype=bool)
				for i in range(testing_ray_number_max):
					testing_mask[int(numpy.random.rand()*nx_map*ny_map)] = 1
				cpu_cost_list = []
				self.cpu_node_list = chunks(cpu_full_list,processingNodeNumber)
				if myrank < len(self.cpu_node_list):
					# we create the source files binding only for the cpu file needed:
					source = self.ro.amr_source(self.fields_to_read, self.cpu_node_list[myrank])
					if not self.remember_data:
						source.set_read_lmax(rlev)
					for icpu in self.cpu_node_list[myrank]:
						#pos_blocks, order_list, noverlap = self.ro.info["dom_decomp"].minimal_domain(cpu_full_list[i]-1)
						#cost = pos_blocks.shape[0]
						dset = source.get_domain_dset(icpu)
						active_mask = dset.get_active_mask()
						g_levels =  dset.get_grid_levels()
						# We do the processing only if needed, i.e. only if the amr level min
						# of active cells in the dset is <= rlev
						if len(g_levels[active_mask]) > 0 and numpy.min(g_levels[active_mask]) <= rlev:
							t0 = time()
							ray_trace_amr(dset, ray_origins[testing_mask], ray_vectors[testing_mask],\
								      ray_lengths, op, self.ro.info, rlev, active_mask, g_levels,\
								      use_C_code = use_C_code)
							cost = time()-t0
							if verbose: print("relative estimated time cost for cpu file",icpu, " is ", cost, "s")
							cpu_cost_list.append([cost,icpu])
				if myrank == 0:
					for i in range(1,processingNodeNumber):
						cpu_cost_list_I = comm.recv(source=i, tag=3)
						cpu_cost_list.extend(cpu_cost_list_I)
					self.cpu_node_list = balanced_cpu_list(cpu_cost_list,processingNodeNumber)
					balanced_cpu_list_overhead_timecost = time() - t0_balanced_cpu_list
					if verbose: print("balanced_cpu_list_overhead_timecost",balanced_cpu_list_overhead_timecost)
				else:
					comm.send(cpu_cost_list, dest=0, tag=3)
				self.cpu_node_list = comm.bcast(self.cpu_node_list, root=0)
			else:
				# No MPI load balacing optimisation : dumb scattering
				self.cpu_node_list = chunks(cpu_full_list,processingNodeNumber)
		if myrank == 0 and verbose:
			print("self.cpu_node_list",self.cpu_node_list)
		
		t0 = time()
		ray_time = 0
		reading_time = 0
		if myrank < len(self.cpu_node_list):
			# we create the source files binding only for the cpu file needed:
			source = self.ro.amr_source(self.fields_to_read, self.cpu_node_list[myrank])
			if not self.remember_data:
				source.set_read_lmax(rlev)
			j=0
			for icpu in self.cpu_node_list[myrank]:
				j+=1
				# read data file
				reading_t0 = time()
				if self.remember_data:
					if icpu not in self.dsets:
						self.dsets[icpu] = source.get_domain_dset(icpu)
					dset = self.dsets[icpu]
				else:
					dset = source.get_domain_dset(icpu)
				reading_time += time() - reading_t0
				useless_dset =  False
				if use_balanced_cpu_list:
					# We do the processing
					mapsDset,ray_length_mapsDset = ray_trace_amr(dset, ray_origins, ray_vectors, ray_lengths,\
										     op, self.ro.info, rlev, use_C_code = use_C_code)
				else:
					active_mask = dset.get_active_mask()
					g_levels =  dset.get_grid_levels()
					# We do the processing only if needed, i.e. only if the amr level min of active cells in the dset is <= rlev
					if len(g_levels[active_mask]) > 0 and numpy.min(g_levels[active_mask]) <= rlev:
						# compute ray trace process
						mapsDset,ray_length_mapsDset = ray_trace_amr(dset, ray_origins, ray_vectors, ray_lengths,\
											     op, self.ro.info, rlev, active_mask, g_levels,\
											     use_C_code = use_C_code)
					else:
						useless_dset = True
				if not useless_dset:
					ray_length_mapsNodeI += ray_length_mapsDset
					if op.is_max_alos():
						numpy.maximum(mapsNodeI, mapsDset, mapsNodeI)
					else:
						mapsNodeI += mapsDset
				if verbose:
					print("MPI activated : node",myrank,"result", j, "out of", len(self.cpu_node_list[myrank]), "computed! ",\
					"Ray Trace Process time = %.1f s"%(time()-t0))
			ray_time = time()-t0
			if verbose: print("MPI activated with",processingNodeNumber,"nodes","on",name,".",\
				"Ray Trace Process time = %.1f s"%(ray_time), "reading time =",reading_time,"s", \
				"max(mapI)=",numpy.max(mapsNodeI), "self.cpu_node_list[myrank]",self.cpu_node_list[myrank])
		else:
			print("MPI activated : node",myrank," of ",processingNodeNumber," not used!")
		
		if myrank != 0:
			results_to_send = {'ray_time': ray_time, 'reading_time':reading_time} 
			comm.send(results_to_send, dest=0, tag=2)
			if verbose:
				print("MPI job for node",myrank," done !")
		else:
			#process 0 collects results
			if verbose:
				print("MPI node",myrank," start to collect results!")
			total_processing_time = ray_time
			total_reading_time = reading_time
			for i in range(1,processingNodeNumber):
				if verbose:
					print("MPI node",myrank," is waiting for result", i)
				results_recv = comm.recv(source=i, tag=2)
				total_processing_time += results_recv["ray_time"]
				total_reading_time += results_recv["reading_time"]
				if verbose:
					print("MPI node",myrank," has just collected result", i, "time=", time()-t0)
		if myrank == 0:
			if verbose:
				print("MPI root node",myrank," has finished collecting results!")
			this_process_overall_time = time()-t0
			print("total reading time = %.1f"%(total_reading_time/total_processing_time*100.), "% of the total processing time")
			print("MPI parallel load balancing result: processing time/node used at %.1f"%(\
				total_processing_time/(this_process_overall_time*processingNodeNumber)*100), "% on", processingNodeNumber, "mpi process")
			if use_balanced_cpu_list and balanced_cpu_list_overhead_timecost != 0:
				print("balanced_cpu_list_overhead_timecost",balanced_cpu_list_overhead_timecost)
		if op.is_max_alos():
			comm.Reduce(mapsNodeI, maps, op=MPI.MAX, root=0)
		else:
			comm.Reduce(mapsNodeI, maps, op=MPI.SUM, root=0)
		comm.Reduce(ray_length_mapsNodeI, ray_length_maps, op=MPI.SUM, root=0)
		if myrank != 0:
			return None
		
		different_ray_length = numpy.unique(ray_length_maps)
		if len(different_ray_length) > 1:
			if (numpy.max(different_ray_length) - numpy.min(different_ray_length)) > ray_lengths[0] * 10e-3:
				print("Calculated ray lengths during the ray trace process are not always equal")
				if len(different_ray_length) < 5:
					print("ray_lengths[0] =", ray_lengths[0])
					for value in different_ray_length:
						print("There are", sum(ray_length_maps == value), "ray(s) with value", value)
				if not op.is_max_alos():
					print("Correction: maps = maps / ray_length_maps")
					maps = maps / ray_length_maps
					# Nullify ray with length = 0
					maps[maps == numpy.inf] = 0
		else:
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

__all__ = ["RayTracerMPI"]

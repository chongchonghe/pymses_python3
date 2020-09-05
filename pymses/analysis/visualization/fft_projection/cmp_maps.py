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
import numpy as N
from pymses.filters import *
from pymses.core import Source, IsotropicExtPointDataset
from pymses.utils import misc
from .convolution_kernels import *
from ..camera import ExtendedCamera, CameraFilter
from .map_bin2d import histo2D
from ..operator import *
from ....sources.ramses.octree import CameraOctreeDataset
import types
from time import time
import pymses.utils.constants as C


class MapProcessor:#{{{
	r"""
	Map-processing generic class

	"""
	def __init__(self, source):
		self.source = source

	def _surf_qty_map(self, camera, map):
		S = camera.get_pixel_surface()
		return map / S

	def process(self, op, camera, surf_qty=False):#{{{
		raise NotImplementedError()

class MapFFTProcessor(MapProcessor):#{{{
	r"""
	MapFFTProcessor class
	Parameters
	----------
	source : ``Source``
		data source
	info   : ``dict``
		RamsesOutput info dict.
	ker_conv : :class:`~pymses.analysis.visualization.ConvolKernel'  (default None leads to use a GaussSplatterKernel)
		Convolution kernel used for the map processing
	pre_flatten : ``boolean`` (default False)
		Option to flatten the data source (using multiprocessing if possible) before computing the map
		The filtered data are then saved into the "self.filtered_source" source attribute.
	remember_data : ``boolean`` (default False)
		Option which uses a "self.cache_dset" dictionarry attribute as a cache
		to avoid reloading dset from disk.
		This uses a lot of memory as it currently remembers a active_mask by
		levelmax filtering for each (dataset, levelmax) couple
	cache_dset : : ``python dictionary``  (default {})
		Cache dsets dictionnary reference, used only if remember_data == True,
		to share the same cache between various MapFFTProcessor. It is a dictionary
		of PointDatasets created with the CellsToPoints filter, referenced by
		[icpu, lmax] where icpu is the cpu number and lmax is the max AMR level used.
	use_camera_lvlmax : ``boolean`` (default True)
		Limit the transformation of the AMR grid to particles
		to AMR cells under the camera octree levelmax (so that visible cells
		are only the ones that have bigger size than the camera pixel size).
		Set this to False when using directly particle data from ".part"
		particles files (dark matter and stars particles), so as to get
		the cache_dset working without the levelmax specification
	"""
	def __init__(self, source, info, ker_conv=None, pre_flatten=False,
		     remember_data=False, cache_dset={}, use_camera_lvlmax=True):
		self.init_source = source
		self.info = info
		if ker_conv is None:
			max_ker_size = 0.5**(info["levelmin"]+1)
			self.convol_kernel = GaussSplatterKernel(max_size=max_ker_size)
		else:
			self.convol_kernel = ker_conv
		self.pre_flatten = pre_flatten
		self.filtered_source = None
		self.remember_data = remember_data
		self.cache_dset = cache_dset
		self.use_camera_lvlmax = use_camera_lvlmax
		
	def free_filtered_source(self):
		self.filtered_source = None
		
	def free_cache_dset(self):
		self.cache_dset.clear()
		
	def prepare_data(self, camera, field_list=None):
		"""prepare data method : it computes the "self.filtered_source" source attribute
		for the process(...) method. Load data from disk or from cache if remember_data option is activated.
		The data are then filtered with the CameraFilter class
		This uses multiprocessing if possible.
		Parameters
		----------
		camera          : :class:`~pymses.analysis.visualization.Camera`
			camera containing all the view params, the filtering is done according to those param
		field_list	``list`` of ``strings``
			list of AMR data fields needed to be read

		"""
		ext_size = self.convol_kernel.get_max_size()
		# Make sure we have a points dset source
		if self.init_source.get_source_type() == Source.AMR_SOURCE:
			if isinstance(self.init_source, CameraOctreeDataset):
				rlev = min(camera.get_required_resolution(), self.info["levelmax"])
				self.points_dset_source = CellsToPoints(self.init_source, smallest_cell_level=rlev)
			else:
				self.points_dset_source = CellsToPoints(self.init_source,
								remember_data=self.remember_data,
								cache_dset=self.cache_dset)
		elif not (isinstance(self.init_source, ExtendedPointFilter) or \
			  isinstance(self.init_source, IsotropicExtPointDataset)):
			self.points_dset_source = ExtendedPointFilter(self.init_source,
								      remember_data=self.remember_data,
								      cache_dset=self.cache_dset)
		else:
			self.points_dset_source = self.init_source
		if self.use_camera_lvlmax:
			lmax = min(camera.get_required_resolution(), self.info["levelmax"])
		else:
			lmax = self.info["levelmax"]
		self.points_dset_source.set_read_lmax(lmax)
		
		#if self.remember_data: # Deprecated code : use this commented code 
		# only to benchmark the remember_data cache_dset performance
		#	# cpu list filtering with Extended camera
		#	self.cameraFilter = CameraFilter(self.init_source, camera, ext_size,
		#				use_camera_lvlmax=self.use_camera_lvlmax)
		#	cpu_list = self.cameraFilter._data_list
		#	# read and transform each amr dset into points dset
		#	tInit = time()
		#	dsets = []
		#	try:
		#		len_cpu_list = len(cpu_list)
		#		if len_cpu_list == 1:
		#			raise(Exception) # don't use multiprocessing if there is only one dset
		#		from multiprocessing import Process, Queue, cpu_count
		#		if cpu_count() == 1:
		#			raise(Exception) # don't use multiprocessing if there is only one cpu
		#		
		#		NUMBER_OF_PROCESSES = min(len_cpu_list, cpu_count(), misc.NUMBER_OF_PROCESSES_LIMIT)
		#
		#		# define long computing process method to do in parallel
		#		def read_dsets(cpu_task_queue, dsets_queue, field_list):
		#			for icpu in iter(cpu_task_queue.get, 'STOP'):
		#				dsets_queue.put((icpu, self.points_dset_source.get_domain_dset(icpu, fields_to_read=field_list)))
		#			
		#		# Create queues
		#		cpu_task_queue = Queue()
		#		cpu_filter_task_queue = Queue()
		#		task_sent_number = 0
		#		# Submit tasks when necessary
		#		for icpu in cpu_list:
		#			if self.cache_dset.has_key((icpu, lmax)) and set(field_list).issubset(self.cache_dset[icpu, lmax].scalars):
		#				# we have data for this (dset, field_list) in cache
		#				cpu_filter_task_queue.put(self.cache_dset[icpu, lmax])
		#			else :
		#				# we need to read data from disk
		#				cpu_task_queue.put(icpu)
		#				task_sent_number += 1
		#		reading_NUMBER_OF_PROCESSES = min(NUMBER_OF_PROCESSES, task_sent_number)
		#		# Tell child processes to stop when they have finished
		#		for i in range(reading_NUMBER_OF_PROCESSES):
		#			cpu_task_queue.put('STOP')
		#		if task_sent_number > 1:
		#			# Start worker processes only if there are more than 1 task
		#			dsets_queue = Queue()
		#			for i in range(reading_NUMBER_OF_PROCESSES):
		#				Process(target=read_dsets, args=(cpu_task_queue, dsets_queue, field_list)).start()
		#			
		#			# Get results
		#			for i in range(task_sent_number):
		#				icpu_got, dsets_queue_got = dsets_queue.get()
		#				self.cache_dset[icpu_got, lmax] = dsets_queue_got
		#				cpu_filter_task_queue.put(dsets_queue_got)
		#		elif task_sent_number > 0:
		#			for icpu in iter(cpu_task_queue.get, 'STOP'):
		#				dset = self.points_dset_source.get_domain_dset(icpu, fields_to_read=field_list)
		#				self.cache_dset[icpu, lmax] = dset
		#				cpu_filter_task_queue.put(dset)
		#		print "Cache dset load time = %.4fs"%(time() - tInit)
		#		# Filter the points dset source according to the camera
		#		tInit = time()
		#		filtered_dsets_queue = Queue()
		#		for i in range(NUMBER_OF_PROCESSES):
		#			cpu_filter_task_queue.put('STOP')
		#		# define long computing process method to do in parallel
		#		def filter_dsets(cpu_filter_task_queue, filtered_dsets_queue):
		#			for dset in iter(cpu_filter_task_queue.get, 'STOP'):
		#				filtered_dsets_queue.put(self.cameraFilter.filtered_dset(dset))
		#		for i in range(NUMBER_OF_PROCESSES):
		#			Process(target=filter_dsets, args=(cpu_filter_task_queue, filtered_dsets_queue)).start()
		#		# Get results
		#		for i in range(len_cpu_list):
		#			dsets.append(filtered_dsets_queue.get())
		#		print "Filtering time = %.3fs"%(time() - tInit)
		#		# concatenate dset with a trick to keep only our field list:
		#		field_list_intersection_set = set(dsets[0].scalars)
		#		for dset in dsets:
		#			field_list_intersection_set = field_list_intersection_set.intersection(set(dset.scalars))
		#		field_list_intersection = list(field_list_intersection_set)
		#		dsets_0_scalars_saved = dsets[0].scalars
		#		dsets[0].scalars = field_list_intersection
		#		self.filtered_source = IsotropicExtPointDataset.concatenate(dsets)
		#		dsets[0].scalars = dsets_0_scalars_saved
		#	except Exception:
		#		# No multiprocessing
		#		for icpu in cpu_list:
		#			if self.cache_dset.has_key((icpu, lmax)) and set(field_list).issubset(self.cache_dset[icpu, lmax].scalars):
		#				# we have data for this (dset, field_list) in cache
		#				dsets.append(self.cameraFilter.filtered_dset(self.cache_dset[icpu, lmax]))
		#			else :
		#				# we need to read data from disk
		#				dset = self.points_dset_source.get_domain_dset(icpu, fields_to_read=field_list)
		#				self.cache_dset[icpu, lmax] = dset
		#				# Filter the points dset source according to the camera
		#				dsets.append(self.cameraFilter.filtered_dset(dset))
		#		print "Cache dset load and filter time = %.4fs"%(time() - tInit)
		#		# concatenate dset with a trick to keep only our field list:
		#		field_list_intersection_set = set(dsets[0].scalars)
		#		for dset in dsets:
		#			field_list_intersection_set = field_list_intersection_set.intersection(set(dset.scalars))
		#		field_list_intersection = list(field_list_intersection_set)
		#		dsets_0_scalars_saved = dsets[0].scalars
		#		dsets[0].scalars = field_list_intersection
		#		self.filtered_source = IsotropicExtPointDataset.concatenate(dsets)
		#		dsets[0].scalars = dsets_0_scalars_saved
		#else:
		# Filter the points dset source according to the camera
		self.filtered_source = CameraFilter(self.points_dset_source,
				camera, ext_size, keep_cache_dset=self.remember_data,
				use_camera_lvlmax=self.use_camera_lvlmax)
		if self.pre_flatten:
			self.filtered_source = self.filtered_source.flatten()
		# Those following objects are used only in this method, but are class attributes
		# for the multiprocessing part, so we delete them here :
		self.points_dset_source = None
		self.cameraFilter = None
		
	def process(self, op, camera, surf_qty=False, multiprocessing=True, FFTkernelSizeFactor=1,
			data_already_prepared=False, random_shift=False, stars_age_instensity_dimming=False):
		"""Map processing method

		Parameters
		----------
		op              : :class:`~pymses.analysis.visualization.Operator`
			physical scalar quantity data operator
		camera          : :class:`~pymses.analysis.visualization.Camera`
			camera containing all the view params
		surf_qty        : ``boolean`` (default False)
			whether the processed map is a surface physical quantity. If True, the map is divided by the surface of a camera pixel.
		multiprocessing : ``boolean`` (default True)
			try to use multiprocessing to compute both of the FractionOperator's "top" and "down" FFT maps in parallel 
		FFTkernelSizeFactor  : ``int or float`` (default 1)
			allow to change the convolution kernel size by a multiply factor to adjust points size
		data_already_prepared : ``boolean`` (default False)
			set this option to true if you have already called the prepare_data() method : this method will
			then simply used it's "self.filtered_source" source attribute without computing it again
		random_shift : ``boolean`` (default False)
			add a random shift to point positions to avoid seeing the grid on resulting image
		stars_age_instensity_dimming : ``boolean`` (default False)
			Requires the "epoch" field. Make use of this formula :
				if star_age < 10 Million years (Myr) : intensity_weights = operator_weights (young stars are normally bright)
				else : intensity_weights = operator_weights * [star_age/10**6 Myr]**-0.7 (intensity dimming with years old)

		Returns
		-------
		map : ``array``
			FFT-convolved processed map

		"""
		t0 = time()
		self.convol_kernel.FFTkernelSizeFactor = FFTkernelSizeFactor

		if not data_already_prepared: # We don't prepare data again only if it has just been done
			self.prepare_data(camera)
		
		region_level = camera.get_region_size_level()
		cs = 1./2**region_level

		# Map initial value
		nx, ny = camera.get_map_size()
		map = N.zeros((nx, ny))
		
		#from pymses.analysis import find_galaxy_axis
		#print find_galaxy_axis(self.filtered_source, camera, nbSample=2000)
				
		# Map processing (FFT convolution)
		print("Processing map dict. (kernel size by kernel size 2D binning)")
		map_dict = {}
		maps = {}
		for key, func in op:
			map_dict[key] = {}
			maps[key] = N.zeros((nx, ny))
		cam_dict = {}
		big_cells = []
		big_cells_number_limit = 1000 # Needed to avoid a very slow rendering with the python loop
		big_cells_size = cs/2.**2 # *4., *2., /2., /4., /8 ?
		while (True):
			# this infinite loop is used to automatically adjust the big_cells size
			# so as to get an appropriate number (not too big).
			for dset in self.filtered_source:
				if random_shift:
					dset.add_random_shift()
				tInit2 = time()
				if dset.npoints == 0: # No point in dset => fetch next dataset
					print("No interesting point in this dataset")
					continue
				print("npoints = %i"%dset.npoints)
				
				# Sizes of the points
				sizes = self.convol_kernel.get_size(dset)
				# Big cells => straightforward summation
				mask = (sizes > big_cells_size)
				if mask.any():
					big_cells_dset = dset.filtered_by_mask(mask)
					big_cells.append(big_cells_dset)
					# Small cells => FFT processing
					mask = (mask==False)
					small_cells_dset = dset.filtered_by_mask(mask)
					if small_cells_dset.npoints == 0:
						continue
					sizes = self.convol_kernel.get_size(small_cells_dset)
					# Get projected (u,v,w) coordinates of the dataset points
					pts, w = camera.project_points(dset.points[mask], take_into_account_perspective=True)
				else:
					small_cells_dset = dset
					pts, w = camera.project_points(dset.points, take_into_account_perspective=True)
	
				# Weights fields
				weights = {}
				for key, func in op:
					weights[key] = func(small_cells_dset)
					if stars_age_instensity_dimming:
						ten_Myr_unit_time = self.info["unit_time"].express(C.Myr)*10
						stars_age = (self.info["time"] - dset["epoch"])*ten_Myr_unit_time
						oldStarsMask = stars_age > 1
						weights[key][oldStarsMask] *= (stars_age[oldStarsMask])**-.7
						
				print("Projection time = %.3fs"%(time() - tInit2))
				tInit3 = time()
				# Map 2D binning
				for size in N.unique(sizes):# Level-by-level cell processing
					mask = (size==sizes)
	
					w = dict.fromkeys(weights)
					for key, func in op:
						w[key] = weights[key][mask]
					if size not in list(cam_dict.keys()):# New size => get a new Camera
						r = N.ones((2,3))*size
						r[0,:] = -r[0,:]
						# Convolution kernel = gaussian type : extend the region by 3*sigma 
						# to get 99% of the gaussian into the extended region (anti-periodic effect)
						if isinstance(self.convol_kernel, GaussSplatterKernel):
							r = 3. * r
						cam_dict[size] = ExtendedCamera(camera, r)
	
					c = cam_dict[size]
					# Camera view area
					centered_map_box = c.get_map_box(reduce_u_v_to_PerspectiveRatio=True)
					map_range = N.array([[centered_map_box.min_coords[0],centered_map_box.max_coords[0]], \
						 [centered_map_box.min_coords[1],centered_map_box.max_coords[1]], \
						 [centered_map_box.min_coords[2],centered_map_box.max_coords[2]]])
	
					# Camera map size
					nx_map, ny_map = c.get_map_size()
	
					# 2D Binning of the dataset points with a dict. of weights.
					map0 = histo2D(pts[mask,:], [nx_map, ny_map], map_range, w)
					for key, func in op:
						ma = map_dict[key]
						if size in list(ma.keys()):
							ma[size] = ma[size] + map0[key]
						else:
							ma[size] = map0[key]
				print("Level-by-level cell processing time = %.3fs"%(time() - tInit3))
			big_cells_number = len(big_cells)
			if big_cells_number < big_cells_number_limit:
				break
			else :
				print("There are too many big cells (",big_cells_number,\
					") for this view, trying again with big_cells_size / 2 ...")
				big_cells_size /= 2
		# big_cells_size is fine so big_cells_number is not bigger than the limit
		if multiprocessing:
			try:
				from multiprocessing import Process, Pipe, cpu_count
				if cpu_count() == 1 or misc.NUMBER_OF_PROCESSES_LIMIT <= 1:
					raise(Exception) # don't use multiprocessing if there is only one cpu
				p={}
				parent_conn={}
				child_conn={}
				def process_dset(convol_kernel, md, cam_dict, conn):
					conn.send(convol_kernel.convol_fft(md, cam_dict))
					conn.close()
				# Start process
				for key, md in map_dict.items():
					parent, child = Pipe()
					parent_conn[key] = parent
					child_conn[key] = child
					p[key] = Process(target=process_dset, args=(self.convol_kernel, md, cam_dict, child_conn[key]))
					p[key].start()
				# Collect results
				for key, md in map_dict.items():
					if len(list(md.keys()))>0:
						maps[key] += parent_conn[key].recv()
						p[key].join()
			except Exception:
				print('WARNING: multiprocessing unavailable')
				multiprocessing = False
		if not multiprocessing:
			for key, md in map_dict.items():
				if len(list(md.keys()))>0:
					maps[key] += self.convol_kernel.convol_fft(md, cam_dict)
		del map_dict

		# Large points direct gaussian kernel addition
		if big_cells_number > 0:
			dset_bcells = big_cells[0].concatenate(big_cells)
			print("Big points : %i"%dset_bcells.npoints)

			# Map edges/center coordinates
			xedges, yedges = camera.get_pixels_coordinates_edges(take_into_account_perspective=True)
			xc = (xedges[1:] + xedges[:-1])/2.
			yc = (yedges[1:] + yedges[:-1])/2.

			# u/v/w Coordinates of the points
			pts, w = camera.project_points(dset_bcells.points, take_into_account_perspective=True)
			
			# Sizes of the points
			sizes = self.convol_kernel.get_size(dset_bcells)
			
			# change weights depending on depth, with a window function
			centered_map_box = camera.get_map_box()
			zmin = centered_map_box.min_coords[2]
			zmax = centered_map_box.max_coords[2]
			f = N.abs(w - (zmin+zmax)/2.) / (zmax-zmin) - 0.5
			weights_fact = win_func(f)

			for key, func in op:
				weights = func(dset_bcells)
				if stars_age_instensity_dimming:
					ten_Myr_unit_time = self.info["unit_time"].express(C.Myr)*10
					stars_age = (self.info["time"] - dset["epoch"])*ten_Myr_unit_time
					oldStarsMask = stars_age > 1
					weights[key][oldStarsMask] *= (stars_age[oldStarsMask])**-.7
				for i in range(pts.shape[0]):
					s = sizes[i]
					m = self.convol_kernel.ker_func((xc-pts[i,0]), (yc-pts[i,1]), s)
					maps[key] += m * weights[i] * weights_fact[i]
		if not data_already_prepared:
			self.filtered_source = None # Free filtered data
		print("Process fft map total time = %.3fs"%(time() - t0))
		# Final Operator process
		nproc = True
		for m in list(maps.values()):
			nproc *= (m == 0.0).all()
		if nproc:
			return map

		map = map + op.operation(maps)
		del maps
		
		if surf_qty:
			map = self._surf_qty_map(camera, map)
		
		return map
	#}}}


def win_func(x, l=0.1, c=9.):
	y = N.zeros_like(x)
	y[(x<=0.)]=1.0

	# exp 1
	mask = ((x<=l/2.)*(x>0.))
	y[mask] = 0.5 + 0.5 * (1.0- N.exp((x[mask] - l/2.)/(l/(2.*c)))) / (1.0-N.exp(-c))

	# exp 2
	mask = ((x<=l)*(x>l/2.))
	y[mask] = 0.5 - 0.5 * (1.0- N.exp(-(x[mask] - l/2.)/(l/(2.*c)))) / (1.0-N.exp(-c))

	return y


__all__ = ["MapFFTProcessor"]

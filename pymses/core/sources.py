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
:mod:`pymses.core.sources` --- PyMSES generic data source module
================================================================

"""
from time import time

class Source(object):#{{{
	r"""
	Base class for all data source objects

	"""
	# WARNING : need to update comments in the file pymses/sources/ramses/sources.py
	# if those definitions are changed
	AMR_SOURCE = 10
	PARTICLE_SOURCE = 11
	def __init__(self):
		self.dom_decomp = None
		self._data_list = []
		self.read_lmax = None
		self.ndim = None
		self.keep_cache_dset = False # this is used for the Filter remember_data option
		self.fields_to_read = None # this is used for the Filter remember_data option
		self.remember_data = False # this is used only by the CellsToPoints filter

	def __del__(self):
		del self.dom_decomp
		del self._data_list

	def set_read_lmax(self, max_read_level):
		r"""Sets the maximum AMR grid level to read in the datasource

		Parameters
		----------
		max_read_level : ``int``
			max. AMR level to read

		"""
		assert (max_read_level > 0)
		self.read_lmax = int(max_read_level)

	def get_source_type(self):
		raise NotImplementedError()

	def iter_dsets(self):
		r"""
		Datasets iterator method. Yield datasets from the datasource

		"""
		for idata in self._data_list:
			yield self.get_domain_dset(idata)

	def __iter__(self):
		return self.iter_dsets()


	def get_domain_dset(self, idomain):
		raise NotImplementedError


	def flatten(self):
		"""
		Read each data file and concatenate resulting dsets.
		This method tries to use multiprocessing if possible.
		This method uses cache_dset if this class is an instance
		of pymses Filter with self.remember_data==True

		Returns
		-------

		fdset : flattened dataset

		"""
		startingTime = time()
		try:
			from multiprocessing import Process, Queue, cpu_count
			from pymses.utils import misc
			NUMBER_OF_PROCESSES = min(len(self._data_list), cpu_count(), misc.NUMBER_OF_PROCESSES_LIMIT)
			
			if NUMBER_OF_PROCESSES == 1:
				raise(Exception) # don't use multiprocessing if there is only one cpu
			dsets = []
	
			# define long computing process method to do in parallel
			def read_dsets(cpu_task_queue, dsets_queue):
				"""Utility method for the Source.flatten() multiprocessing method. 
				It reads the given list of data file and concatenate resulting dsets
	
				Parameters
				----------
				cpu_task_queue : ``list`` of ``int``
					queue of data file number corresponding to data files that have to be read by a process
				dsets_queue : multiprocessing queue
					to send the result to parent process
				"""
				dsets = []
				if self.keep_cache_dset:
					cache_dset_update = {}
				for icpu in iter(cpu_task_queue.get, 'STOP'):
					keep_data_set_in_cache = self.keep_cache_dset and \
						not self.is_dset_in_cache(icpu)
					dset = self.get_domain_dset(icpu)
					dsets.append(dset)
					if keep_data_set_in_cache:
						cache_dset_update[icpu, self.source.read_lmax] = \
							self.cache_dset[icpu, self.source.read_lmax]
				dsets_queue.put((1,dsets))
				if self.keep_cache_dset:
					dsets_queue.put((0,cache_dset_update))
			
			# Create queues
			cpu_task_queue = Queue()
			dsets_queue = Queue()
			# Submit tasks
			for icpu in self._data_list:
				cpu_task_queue.put(icpu)
			# Tell child processes to stop when they have finished
			for i in range(NUMBER_OF_PROCESSES):
				cpu_task_queue.put('STOP')
			dsets_queue = Queue()
			# Starts process
			for i in range(NUMBER_OF_PROCESSES):
				Process(target=read_dsets, args=(cpu_task_queue, dsets_queue)).start()
			# Get results
			if self.keep_cache_dset:
				queue_size= 2 * NUMBER_OF_PROCESSES
			else :
				queue_size = NUMBER_OF_PROCESSES
			for i in range(queue_size):
				id, dsets_queue_got = dsets_queue.get()
				if id==1 and dsets_queue_got != []:
					dsets.extend(dsets_queue_got)
				elif id==0: # update the cache_dset
					self.cache_dset.update(dsets_queue_got)
		except Exception:
			print('WARNING: no multiprocessing')
			dsets = list(self.iter_dsets())
		print("Read and filter time : %.2f s"%(time()-startingTime))
		if dsets == []:
			return None
		else:
			if self.keep_cache_dset:
				# we need to take care of the
				# field list that may be different
				# from one cache dset to another !
				# So we concatenate dset with a trick
				# to keep only our field list:
				field_list_intersection_set = set(dsets[0].scalars)
				for dset in dsets:
					field_list_intersection_set = field_list_intersection_set.intersection(set(dset.scalars))
				field_list_intersection = list(field_list_intersection_set)
				dsets_0_scalars_saved = dsets[0].scalars
				dsets[0].scalars = field_list_intersection				
				from pymses.core import IsotropicExtPointDataset
				fdset = IsotropicExtPointDataset.concatenate(dsets)
				dsets[0].scalars = dsets_0_scalars_saved
				return fdset
			else:
				return dsets[0].concatenate(dsets)
#}}}

class Filter(Source):#{{{
	r"""
	Data source filter generic class.

	"""
	def __init__(self, source):
		Source.__init__(self)
		self.source = source
		self.fields_to_read = self.source.fields_to_read
		self.dom_decomp = source.dom_decomp
		self._data_list = source._data_list
		self.read_lmax = source.read_lmax

	def __del__(self):
		del self.source
		del self.dom_decomp
		del self._data_list

	def set_read_lmax(self, max_read_level):
		r"""
		Source inherited behavior + apply the set_read_lmax() method to the `source` param.

		Parameters
		----------
		max_read_level : ``int``
			max. AMR level to read

		"""
		self.read_lmax = max_read_level
		self.source.set_read_lmax(max_read_level)

	def get_source_type(self):
		r"""
		Returns
		-------
		type : ``int``
			the result of the `get_source_type()` method of the `source` param.

		"""
		return self.source.get_source_type()

	def filtered_dset(self, dset):
		r"""
		Abstract `filtered_dset()` method

		"""
		raise NotImplementedError

	
	def is_dset_in_cache(self, idomain, fields_to_read=None):
		if fields_to_read==None:
			fields_to_read = self.fields_to_read
		if (idomain, self.source.read_lmax) in self.cache_dset:
			dset = self.cache_dset[idomain, self.source.read_lmax]
			return set(fields_to_read).issubset(dset.scalars+dset.vectors)
		else :
			return False
	
	def get_domain_dset(self, idomain, fields_to_read=None):
		r"""
		Get the filtered result of `self.source.get_domain_dset(idomain)`

		Parameters
		----------
		idomain : ``int``
			number of the domain from which the data is required

		Returns
		-------
		dset : ``Dataset``
			the filtered dataset corresponding to the given `idomain`

		"""
		if self.remember_data:
			if not self.is_dset_in_cache(idomain, fields_to_read):
				# we don't have data for this (dset, lmax, field_list) in cache
				# we need to read data from disk
				self.cache_dset[idomain, self.source.read_lmax] = self.filtered_dset(
					self.source.get_domain_dset(idomain, fields_to_read=fields_to_read))
			return self.cache_dset[idomain, self.source.read_lmax]
		else:
			return self.filtered_dset(
				self.source.get_domain_dset(idomain, fields_to_read=fields_to_read))
#}}}

class SubsetFilter(Filter):#{{{
	r"""
	SubsetFilter class. Selects a subset of datasets to read from the
	datasource

	Parameters
	----------
	data_sublist : ``list`` of ``int``
		list of the selected dataset index to read from the datasource

	"""
	def __init__(self, data_sublist, source):
		Filter.__init__(self, source)
		if data_sublist is not None:
			self._data_list = [i for i in data_sublist\
					if i in self._data_list]
	
	def filtered_dset(self, dset):
		return dset
#}}}

__all__ = ["Source", "Filter", "SubsetFilter"]

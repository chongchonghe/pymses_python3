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
from pymses.utils.regions import Box
from pymses.filters import RegionFilter

def amr2cube(source, var, xmin, xmax, cubelevel, out=None):
	r"""
	amr2cube tool.

	"""
	# allow amr2cube to work with a vector
	op = var
	if isinstance(var, str):
		op = lambda dset: dset[var]
		
	# TODO :  add cache_dset reuse possibility
	# Region CPU-filtering
	b = Box([xmin, xmax])
	rsource = RegionFilter(b, source)
	rsource.set_read_lmax(cubelevel)
	try:
		from multiprocessing import Process, Queue, cpu_count
		if cpu_count() == 1:
			raise(Exception) # don't use multiprocessing if there is only one cpu
		dsets = []
		from pymses.utils import misc
		NUMBER_OF_PROCESSES = min(len(rsource._data_list), cpu_count(), misc.NUMBER_OF_PROCESSES_LIMIT)

		# define long computing process method to do in parallel
		def read_dsets_to_cartesian(cpu_task_queue, dsets_queue, rsource):
			"""Utility method for amr2cube multiprocessing method. 
			It reads the given list of data file and concatenate resulting dsets

			Parameters
			----------
			cpu_task_queue : ``list`` of ``int``
				queue of data file number corresponding to data files that have to be read by a process
			dsets_queue : multiprocessing queue
				to send the result to parent process
		
			"""
			len_dsets = 0
			out = None
			for icpu in iter(cpu_task_queue.get, 'STOP'):
				dset = rsource.get_domain_dset(icpu)
				active_mask = dset.get_active_mask()
				g_levels =  dset.get_grid_levels()
				# We do the processing only if needed, i.e. only if the
				# amr level cubelevel of active cells in the dset is >0
				if len(g_levels[active_mask]==cubelevel) > 0 :
					if out is None:
						out = dset.to_cartesian(var, xmin, xmax, cubelevel)
					else:
						dset.to_cartesian(var, xmin, xmax, cubelevel, dest=out)
					len_dsets += 1
			if len_dsets == 0:
				dsets_queue.put("NoDset")
			else:
				dsets_queue.put(out)

		# Create queues
		cpu_task_queue = Queue()
		dsets_queue = Queue()
		# Submit tasks
		for task in rsource._data_list:
			cpu_task_queue.put(task)
		# Start worker processes
		for i in range(NUMBER_OF_PROCESSES):
			Process(target=read_dsets_to_cartesian, args=(cpu_task_queue, dsets_queue, rsource)).start()
		# Tell child processes to stop when they have finished
		for i in range(NUMBER_OF_PROCESSES):
			cpu_task_queue.put('STOP')
		# Get results
		for i in range(NUMBER_OF_PROCESSES):
			outP = dsets_queue.get()
			if outP != "NoDset":
				if out is None:
					out = outP
				else :
					out += outP
	except Exception:
		print('WARNING: multiprocessing unavailable')
		for dset in rsource.iter_dsets():
			active_mask = dset.get_active_mask()
			g_levels =  dset.get_grid_levels()
			# We do the processing only if needed, i.e. only if the
			# amr level cubelevel of active cells in the dset is >0
			if len(g_levels[active_mask]==cubelevel) > 0 :
				if out is None:
					out = dset.to_cartesian(var, xmin, xmax, cubelevel)
				else:
					dset.to_cartesian(var, xmin, xmax, cubelevel, dest=out)

	return out

__all__ = ["amr2cube"]

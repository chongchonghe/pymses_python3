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
from .camera import Camera
from ..point_sampler import sample_points
import types
import numpy

def SliceMap(source, camera, op, z=0.0, interpolation=False, use_C_code=True,\
		use_openCL=False, verbose=False):
	"""
	Compute a map made of sampling points

	Parameters
	----------
	source : ``Source``
		data source
	camera : :class:`~pymses.analysis.visualization.Camera`
		camera handling the view parameters
	op     : :class:`~pymses.analysis.visualization.Operator`
		data sampling operator
	z      : ``float``
		position of the slice plane along the line-of-sight axis of the camera
	interpolation : ``boolean`` (default False)
		Experimental : A proper bi/tri-linear interpolation could be great!
		THIS IS NOT IMPLEMENTED YET : in this attempt we supposed corner cell data
		while ramses use centered cell data, letting alone the problem
		of different AMR level...
	use_C_code : ``boolean`` (default True)
		The pure C code is slightly faster than the (not well optimized) Cython code,
		and should give the same result
	use_openCL : ``boolean`` (default False)
		Experimental : use "pyopencl" http://pypi.python.org/pypi/pyopencl
	verbose : ``boolean`` (default False)
		some console printout...
	Returns
	-------
	map : ``array``
		sliced map

	"""
	# Map size
	nx, ny = camera.get_map_size()

	# Sammpling points
	p = camera.get_slice_points(z)

	# Get dataset
	if source.ndim == 2:
		p = p[:,0:2] # 3D -> 2D restriction
	
	add_level = op.use_cell_dx()
	max_lev = camera.get_required_resolution()
	dset = sample_points(source, p, interpolation=interpolation,
		use_C_code=use_C_code, use_openCL=use_openCL,
		add_level=add_level, max_search_level=max_lev, verbose=verbose)
	
	# Compute map values according to operator
	maps = {}
	for key, func in op:
		if add_level:
			# for MaxLevelOperator
			maps[key] = dset["level"]
		else:
			maps[key] = func(dset)
	map = op.operation(maps)

	map = map.reshape(nx, ny)
	
	return map

__all__ = ["SliceMap"]

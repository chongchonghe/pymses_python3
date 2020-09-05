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
from pymses.utils.point_utils import corner_points
from pymses.core.datasets import PointDataset

def sample_points(amr_source, points, add_cell_center=False, add_level=False,\
		  max_search_level=None, interpolation=False, use_C_code=True,\
		  use_openCL=False, verbose=False):#{{{
	r"""
	Create point-based data from AMR-based data by point sampling. Samples all available fields
	of the `amr_source` at the coordinates of the `points`.
	
	Parameters
	----------
	amr_source : :class:`~pymses.sources.ramses.output.RamsesAmrSource`
		data description
	points : (`npoints`, `ndim`) ``array``
		sampling points coordinates
	add_level : ``boolean`` (default False)
		whether we need to add a `level` field in the returned dataset containing
		the value of the AMR level the sampling points fall into
	add_cell_center : ``boolean`` (default False)
		whether we need to add a `cell_center` field in the returned dataset containing
		the coordinates of the AMR cell center the sampling points fall into
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
	dset : :class:`~pymses.core.datasets.PointDataset`
		Contains all these sampled values.
	
	"""
	points = numpy.asarray(points)
	npoints = points.shape[0]

	# Compute the domain ids of the points
	if amr_source.dom_decomp != None:
		points_datamap = amr_source.dom_decomp.map_points(points)
	else:
		points_datamap = numpy.ones(npoints)
	# Sort the points by datamap
	unique_domains = numpy.unique(points_datamap)
	
	ipoint_batches = []
	point_dsets = []

	# Loop over unique domains
	for idomain in unique_domains:
		# Select points from current domain only
		ipoint_batch = numpy.nonzero(points_datamap == idomain)[0]
		ipoint_batches.append(ipoint_batch)
		points_in_domain = points[ipoint_batch, :]
		if max_search_level is None:
			level_in_domain=None
		elif isinstance(max_search_level, int):
			level_in_domain = numpy.ones(len(points_in_domain)) * max_search_level
		else :
			level_in_domain = max_search_level[ipoint_batch]
		
		# Read the cell dataset of the current domain
		if level_in_domain is not None:
			amr_source.read_lmax = int(max(level_in_domain))
		dset = amr_source.get_domain_dset(idomain)
		# Sample on the current domain
		point_dsets.append(dset.sample_points(points_in_domain, add_cell_center=add_cell_center,\
						      add_level=add_level, max_search_level=level_in_domain,\
						      interpolation=interpolation, use_C_code=use_C_code,\
						      use_openCL=use_openCL, verbose=verbose))

	# Perform final reordering concatenation
	cat_method = point_dsets[0].concatenate
	return cat_method(point_dsets, reorder_indices=ipoint_batches)
#}}}

def CIC_point_sampling(amr_source, points, weight_func=None):#{{{
	dset = sample_points(amr_source, points, add_cell_center=True, add_level=True)
	npoints = points.shape[0]
	ndim = points.shape[1]
	dx_loc = 1./2.**dset["level"]
	cc = dset["cell_center"]
	dx = (points - cc)/dx_loc[:,numpy.newaxis] + 0.5
	vol = numpy.zeros((npoints, 2**ndim))
	lev = numpy.repeat(dset["level"], 2**ndim)
	cp = corner_points(points, dx_loc)
	dset = sample_points(amr_source, cp, max_search_level=lev)
	
	dr=dx + 0.5
	ir=dr.astype('i')
	dr=dr-ir
	dl=1.0-dr
	il=ir-1
	# Volume weights
	vol[:,0] = dl[:,0] * dl[:,1] * dl[:,2]
	vol[:,1] = dr[:,0] * dl[:,1] * dl[:,2]
	vol[:,2] = dl[:,0] * dr[:,1] * dl[:,2]
	vol[:,3] = dr[:,0] * dr[:,1] * dl[:,2]
	vol[:,4] = dl[:,0] * dl[:,1] * dr[:,2]
	vol[:,5] = dr[:,0] * dl[:,1] * dr[:,2]
	vol[:,6] = dl[:,0] * dr[:,1] * dr[:,2]
	vol[:,7] = dr[:,0] * dr[:,1] * dr[:,2]

	ds = PointDataset(points)
	for sfield in dset.scalars:
		field = dset[sfield].reshape((npoints, 2**ndim))
		field = numpy.sum(field * vol, axis=1)
		ds.add_scalars(sfield, field)
	for vfield in dset.vectors:
		field = dset[vfield].reshape((npoints, 2**ndim, ndim))
		field = numpy.sum(field * vol[:,:,numpy.newaxis], axis=1)
		ds.add_vectors(vfield, field)

	return ds
#}}}

__all__ = ["sample_points", "CIC_point_sampling"]

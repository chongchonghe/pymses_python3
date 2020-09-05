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
""" ray_trace.pyx -- Ray-tracer routine
-- Cython used to speed up the code with specific C code
"""
# cython: profile=True

cimport numpy
cimport cython
import numpy
from pymses.sources.ramses.tree_utils import tree_search
from pymses.analysis.visualization.transfer_functions import ColorLinesTransferFunction

# Define integer and double precision float types + ctypes {{{
INT_t = numpy.int
ctypedef numpy.int_t cINT_t
INT8_t = numpy.int8
ctypedef numpy.int8_t cINT8_t
DBL_t = numpy.float64
ctypedef numpy.float64_t cDBL_t
#}}}

cdef extern from "ray_trace_C_code.c":
	void ray_trace_C(double * maps, double * ray_length, double * ray_origins, double * ray_vects,\
	double * cell_centers, int * sons, int * g_levels, double * scal_data,\
	int * igrids, char * icells, long * active_mask, double * ray_depth_max,\
	long * param_info_int)

cdef extern from "ray_trace_C_code.c":
	void ray_trace_octree_C(double * I, double * ray_origins, double * ray_vects,\
	double * grid_centers, double * cell_centers, int * sons, int * cell_levels,\
	double * scalar_data, int * neighbors, double * ray_lengths,\
	long * param_info_int, double gamma, double * vlines, double * wlines,\
	double * rlines, double * glines, double * blines, double * alines)

## Math functions (C calls) {{{
#cdef inline double fmax (cDBL_t x, cDBL_t y):
#	if x>y:return x
#	return y

cdef inline cDBL_t gaussn(cDBL_t x, cDBL_t width):
	return exp(-x*x/(width*width))

cdef extern from "math.h":
	cDBL_t fabs(cDBL_t x)
	cDBL_t ceil(cDBL_t x)
	cDBL_t floor(cDBL_t)
	cDBL_t exp(cDBL_t)
#}}}

@cython.wraparound(False)
@cython.boundscheck(False)
def ray_trace_amr(dset, numpy.ndarray[cDBL_t, ndim=2] ray_origins, \
		numpy.ndarray[cDBL_t, ndim=2] ray_vects, \
		numpy.ndarray[cDBL_t, ndim=1] ray_depth_max, op, info, level_max=None, dset_active_mask=None,\
		dset_grid_levels=None , use_C_code=True, use_hilbert_domain_decomp=True):
	""" Ray tracer
	
	Parameters
	----------
	ray_origins : numpy.array (float64, ndim=2)
		Rays origins : array of vector, shape=(nrays,ndim)
	ray_vects : numpy.array (float64, ndim=2)
		Rays vectors : array of vector, shape=(nrays,ndim)
	ray_depth_max : numpy.array (float64, ndim=1)
		Rays length max : array of vector, shape=(nrays)
	op : :class:`~pymses.analysis.visualization.Operator`
		physical scalar quantity data operator
	info : python dictionary
		output info from source.info (or from pymses.source.ramses.info.read_ramses_info_file)
	level_max : int  (default None)
		level max used in processing : we stop the top down traversal of the octree to this depth
	dset_active_mask : numpy.array (bool, ndim=1)  (default None)
		octree active_mask array used to flag cells that have to be taken into account
		(i.e. not ghost cells), array of bool, shape=(ngrids)
	dset_grid_levels : numpy.array (int, ndim=1)  (default None)
		octree grid_levels array used to tell the octree level of each cell, array of int, shape=(ngrids)
	use_C_code : ``boolean`` (default True)
		Our pure C code is faster than the (not well optimized) Cython code,
		and should give the same result
	use_hilbert_domain_decomp : ``boolean`` (default True)
		If False, iterate on the whole octree for each cpu file(instead of iterating
		on the cpu minimal domain decomposition, which is faster)
	
	"""
	# integer variables
	cdef cINT_t nrays= ray_origins.shape[0]
	cdef cINT_t ndim_rays = ray_origins.shape[1]
	cdef cINT_t ndim = dset.amr_header["ndim"]
	assert (ndim_rays == ndim)
	cdef cINT_t max_read_level
	if level_max is not None:
		max_read_level = level_max 
	else:
		max_read_level = dset.amr_struct["readlmax"]
	cdef cINT_t nfunc = op.nscal_func()
	cdef cINT_t ngrids = dset.amr_struct["ngrids"]
	cdef cINT_t twotondim = (1 << ndim)
	cdef cINT_t psize = (twotondim-1) * max_read_level
	cdef cINT_t iray, idim, iblock, nblocks, level, ilevel, ifunc
	cdef cINT_t igrid_root, icell_root, igrid, icell, igrid_son
	cdef cINT_t ind_son, head, i, intersect, iface, sgn_face, idim2, iblock2
	cdef cINT_t ind_son1, ind_son2, ind_entry, ind_exit, add, block_already_done#, ncd, ncdnul
	cdef cINT_t max_op = op.is_max_alos()
	cdef cINT_t use_dx = op.use_cell_dx()

	# float variables
	cdef cDBL_t alpha, alpha_save, dx, dx_2, cube_depth, r2, l, d, v, vl
	cdef cDBL_t Rsphere, Rsphere2
	cdef cDBL_t sqrt3 = numpy.sqrt(3.0)
	cdef cDBL_t dm2, depth_max
	
	# Big AMR arrays
	cdef numpy.ndarray[int, ndim=1] igrids = numpy.empty(ngrids, 'i')
	cdef numpy.ndarray[char, ndim=1] icells = numpy.empty(ngrids, 'i1')
	cdef numpy.ndarray[int, ndim=2] sons = dset.amr_struct["son_indices"]
	cdef numpy.ndarray[int, ndim=1] g_levels
	if dset_grid_levels is None:
		g_levels = dset.get_grid_levels()
	else:
		g_levels = dset_grid_levels
	cdef numpy.ndarray[cINT_t, ndim=1] active_mask
	if dset_active_mask is None:
		active_mask = numpy.array(dset.get_active_mask(), INT_t)
	else:
		active_mask = numpy.array(dset_active_mask, INT_t)
	#assert numpy.max(sons) < len(active_mask)
	cdef numpy.ndarray[cDBL_t, ndim=3] cell_centers = dset.get_cell_centers()

	# Pile arrays	
	cdef numpy.ndarray[cINT_t, ndim=1] igrid_pile = numpy.empty(psize, INT_t)
	cdef numpy.ndarray[cINT_t, ndim=1] icell_pile = numpy.empty(psize, INT_t)
	cdef numpy.ndarray[cINT_t, ndim=1] level_pile = numpy.empty(psize, INT_t)

	# Integrand variables
	cdef numpy.ndarray[cDBL_t, ndim=3] scal_data = numpy.empty((ngrids, twotondim, nfunc), DBL_t)
	# Operand data
	ifunc = 0
	for (key, scal_func) in op.iter_scalar_func():
		scal_data[:,:,ifunc] = scal_func(dset)
		ifunc += 1
	
	# (ndim,) vector arrays
	cdef numpy.ndarray[cDBL_t, ndim=1] block_center = numpy.empty(ndim, DBL_t)
	cdef numpy.ndarray[cDBL_t, ndim=1] fc = numpy.empty(ndim, DBL_t)
	cdef numpy.ndarray[cDBL_t, ndim=1] I = numpy.empty(ndim, DBL_t)
	cdef numpy.ndarray[cINT_t, ndim=1] dim_constr = numpy.empty(ndim, INT_t)
	cdef numpy.ndarray[cINT_t, ndim=1] dim_side = numpy.empty(ndim, INT_t)
	
	# Output maps
	cdef numpy.ndarray[cDBL_t, ndim=2] maps = numpy.zeros((nrays, nfunc), DBL_t)
	cdef numpy.ndarray[cDBL_t, ndim=2] ray_length = numpy.zeros((nrays, nfunc), DBL_t)
	
	cdef numpy.ndarray[cINT_t, ndim=1] order
	if use_hilbert_domain_decomp and info["dom_decomp"] is not None:
		# iteration over the minimal grid description of the domain
		pos_blocks, order_list, noverlap = info["dom_decomp"].minimal_domain(dset.icpu)
		order = order_list.astype(INT_t)
		nblocks = pos_blocks.shape[0] - noverlap
		if nblocks>1 or order[0]!=0:
			search_dict = tree_search(dset.amr_struct, pos_blocks, order)
			igrids = search_dict["grid_indices"]
			icells = search_dict["cell_indices"]
		else:
			use_hilbert_domain_decomp = False
	if not use_hilbert_domain_decomp or info["dom_decomp"] is None:
		# iteration on the full octree
		nblocks = 8
		igrids[0:8] = 0
		icells[0:8] = range(8)
	
	cdef numpy.ndarray[cINT_t, ndim=1] param_info_int
	if use_C_code:
		param_info_int = numpy.empty(10, INT_t)
		param_info_int[0] = nrays
		param_info_int[1] = ndim
		param_info_int[2] = nblocks
		param_info_int[3] = max_read_level
		param_info_int[4] = nfunc
		param_info_int[5] = max_op
		param_info_int[6] = use_dx
		param_info_int[7] = ngrids
		ray_trace_C(<double *>maps.data, <double *>ray_length.data, <double *>ray_origins.data, <double *>ray_vects.data,\
			<double *>cell_centers.data, <int *>sons.data, <int *>g_levels.data, <double *>scal_data.data,\
			<int *>igrids.data, <char *>icells.data, <long *>active_mask.data, <double *>ray_depth_max.data,\
			<long *>param_info_int.data)
#		print("ray_length[0]", ray_length[0])
		return maps,ray_length
#	ncd = 0
#	ncdnul = 0
	for iblock in range(nblocks):
		igrid_root = igrids[iblock]
		icell_root = icells[iblock]
		block_already_done = 0
		for iblock2 in range(iblock):
			if igrids[iblock2] == igrids[iblock] and icells[iblock2] == icells[iblock]:
				block_already_done = 1
				break
		if 	block_already_done == 1:
			continue
		level = g_levels[igrid_root]
#		if level != order[iblock]:
#			print("Warning: according to the hilbert minimal domain decomposition for this domain,\
#				this block was supposed to be order", order[iblock]," but the corresponding block\
#				find in this cpu file (using block center and treesearch) is order:",level)
		dx = 1.0/(1<<level)
		Rsphere = sqrt3 * dx / 2.0
		Rsphere2 = Rsphere*Rsphere
		# Block center (ndim,) coordinates
		for idim in range(ndim):
			block_center[idim] = cell_centers[igrid_root, icell_root, idim]
		#################
		# Ray iteration #
		#################
		for iray in range(nrays):
			depth_max = ray_depth_max[iray]
			dm2 = depth_max/2.0
			# Coordinate parameter of the point of the ray nearest to the block center
			alpha = 0.0
			for idim in range(ndim):
				alpha += (block_center[idim]-ray_origins[iray,idim])*ray_vects[iray,idim]
			# square distances of the ray from the block center
			r2 = 0.0
			for idim in range(ndim):
				v = ray_origins[iray,idim] + alpha * ray_vects[iray, idim] - block_center[idim]
				r2 += v*v
			# Is this ray able to intersect the block ?
			if (r2 > Rsphere2): # no => continue
				continue
			# distance from the ray extreme points
			d = alpha-dm2
			if d<0.: d=-d
			d = d-dm2
			if d>0.:
				r2 += d*d
				# Is this ray still able to intersect the block ?
				if (r2 > Rsphere2):
					continue
			# Pile initialisation with current block
			head=0
			igrid_pile[head] = igrid_root
			icell_pile[head] = icell_root
			level_pile[head] = level
			while head !=-1 :
				igrid = igrid_pile[head]
				icell = icell_pile[head]
				ilevel = level_pile[head]
				head -= 1
#				print("head", head, "igrid", igrid, "icell", icell, "ilevel", ilevel)
				# Current cell size
				dx = 1.0/(1<<ilevel)
				dx_2 = dx/2.0
				# Cube optical depth computation
				cube_depth = 0.0
				intersect = 0
				for idim2 in range(ndim):
					if(ray_vects[iray, idim2] == 0.0):
						# ray is parallel to the faces along dimension idim2.
						# There is no  intersection.
						continue
					# All ray-cell intersections found => break loop
					if intersect==2: break
					for sgn_face in [-1,1]:
						# Coordinate parameter of the intersection of the ray with the face plane
						alpha = ((cell_centers[igrid, icell, idim2] + sgn_face * dx_2) \
								- ray_origins[iray, idim2])/ray_vects[iray, idim2]
						# Is this intersection within the face ?
						v=0.0
						vl=0.0
						ind_son = (twotondim-1) - (((-sgn_face+1)>>1)<<idim2)
						for idim in range(ndim):
							if idim == idim2: continue
							l = (ray_origins[iray,idim] + alpha * ray_vects[iray,idim] \
							     - cell_centers[igrid, icell, idim])/dx
							if ((l>vl) and (ray_vects[iray, idim] == 0.0)):vl=l
							if l<0.0:
								ind_son -= (1<<idim)
								l=-l
							if l>v: v=l

						if ((v <= 0.5) and (vl<0.5)):
							# The intersection is in the face of the cube
							intersect +=1
							if intersect == 1: # First intersection found
								if alpha < 0.0:
									alpha_save = 0.0
								elif alpha > depth_max:
									alpha_save = depth_max
								else:
									alpha_save = alpha
								ind_son1=ind_son
							else: # Second intersection found : done
								if alpha < 0.0:
									alpha = 0.0
								elif alpha > depth_max:
									alpha = depth_max
								cube_depth = alpha-alpha_save
								# Make sure ind_son1 = entry point son index
								#           ind_son2 = exit point son index
								if cube_depth < 0.0 : # ind_son is the entry point son index
									cube_depth=-cube_depth
									ind_son2 = ind_son1
									ind_son1 = ind_son
								else:# ind_son is the exit point son index
									ind_son2=ind_son
								break

				# Is the current cell actually intersected by the ray ?
				if cube_depth > 0.0:
#					ncd += 1
					# The ray cuts the block
					igrid_son = sons[igrid, icell]
					if (igrid_son < 0) or (ilevel >= max_read_level):
					# We're done searching: the subcell is the one we want
						# We skip inactive cells
						if active_mask[igrid] == False:
								continue
						for ifunc in range(nfunc):
							if use_dx:
								v = ilevel * 1.0
							else:
								v = scal_data[igrid, icell, ifunc]
							ray_length[iray, ifunc] += cube_depth
							if max_op:
								if v>maps[iray, ifunc]:
									maps[iray, ifunc] = v
							else:
								maps[iray, ifunc] += v * cube_depth

					else: # Go down the AMR tree
						#########################################################################################
						#                      Add the NECESSARY son cells to the pile                          # 
						#########################################################################################
						if ind_son1 != ind_son2:
							for idim in range(ndim):
								ind_entry = (ind_son1>>idim) & 1
								ind_exit = (ind_son2>>idim) & 1
								if (ind_entry==ind_exit):
									dim_constr[idim] = 1
									dim_side[idim] = ind_entry
								else:
									dim_constr[idim] = 0

							# Add the POSSIBLY intersecting son cells different from the entry/exit point son cells
							for ind_son in range(twotondim): # All he son cells
								add = 1
								# Different from entry/exit point son cells
								if (ind_son == ind_son1) or (ind_son== ind_son2):
									add = 0
								else:
									for idim in range(ndim):
										if dim_constr[idim] and (dim_side[idim] != (ind_son>>idim) & 1):
											add = 0
											break

								if add:
									head += 1
									level_pile[head] = ilevel+1
									igrid_pile[head] = igrid_son
									icell_pile[head] = ind_son

							# If different from the son cell of the entry point, add the exit point son cell
							head+=1
							level_pile[head] = ilevel+1
							igrid_pile[head] = igrid_son
							icell_pile[head] = ind_son2
						
						# Add the son cell of the entry point of the ray in the father grid
						head+=1
						level_pile[head] = ilevel+1
						igrid_pile[head] = igrid_son
						icell_pile[head] = ind_son1
#				else:
#					ncdnul += 1
						
						#########################################################################################
#	print ncd, ncdnul
	#cdef numpy.ndarray[cDBL_t, ndim=2] mapsTest = numpy.zeros((nrays, nfunc), DBL_t)
	#test = maps == mapsTest
	#if test.all():
	#	print "Useless CPU ",dset.icpu," for this map !!!"
	return maps,ray_length
#}}}

@cython.wraparound(False)
@cython.boundscheck(False)
def ray_trace_octree(dset, numpy.ndarray[cDBL_t, ndim=2] ray_origins,
			numpy.ndarray[cDBL_t, ndim=2] ray_vects,
			numpy.ndarray[cDBL_t, ndim=1] ray_lengths,
			op, tf=None, level_max=None, verbose=True,
			rgb=True, use_C_code=True):#{{{
	""" Octree Ray tracing function

	Parameters
	----------
	ray_origins : numpy.array (float64, ndim=2)
		Rays origins : array of vector, shape=(nrays,ndim)
	ray_vects : numpy.array (float64, ndim=2)
		Rays vectors : array of vector, shape=(nrays,ndim)
	ray_lengths : numpy.array (float64, ndim=1)
		Rays length max : array of vector, shape=(nrays)
	op : : class:`~pymses.analysis.visualization.Operator`
		physical scalar quantity data operator
	tf : : class:`~pymses.analysis.visualization.Camera.color_tf` (default None)
		camera color transfer function (used with rgb map only)
	level_max : int  (default None)
		level max used in processing : we limit the octree traversal
		to this depth
	verbose	: ``boolean`` (default False)
			show more console printouts
	rgb     : ``boolean`` (default True)
		if True, this code use the camera.color_tf to compute a rgb image
		if False, this code doesn't use the camera.color_tf, and works like the
		standard RayTracer. Then it returns two maps : the requested map,
		and the AMR levelmax map
	use_C_code : ``boolean`` (default True)
		Our pure C code is faster than the (not well optimized) Cython code,
		and should give the same result
	
	"""
	# integer variables
	cdef cINT_t nrays= ray_origins.shape[0]
	cdef cINT_t ndim_rays = ray_origins.shape[1]
	cdef cINT8_t ndim = dset.amr_header["ndim"]
	assert (ndim_rays == ndim)
	cdef cINT_t ngrids = dset.amr_struct["ngrids"]
	cdef cINT8_t twotondim = (1 << ndim)

	cdef cINT_t iray, igrid, igridn, igrid_son, ilevel, ileveln
	cdef cINT8_t idim, idim2, ind, indn, ibit, iface
	cdef cINT8_t sgn_face, isgn, ind_son

	#	cdef char test_ok
	cdef cINT_t idz, ndz
	cdef cINT_t max_read_level = dset.amr_struct["readlmax"]
	if level_max is not None and max_read_level>level_max:
		max_read_level = level_max
	cdef cDBL_t dx_min, dz, scalar_loc, zvar, x_ori, y_ori, z_ori
	cdef cDBL_t em, rem, gem, bem, alpha, gauss_value, absor, width, x, gamma
	cdef char irgb, iem, verbose_C, error, ifunc
	verbose_C = verbose
	error = 0
	
	# float variables
	cdef cDBL_t v, vl
	cdef cDBL_t scalar, scalar2, cc, scalarUp, scalarUp2
	cdef cDBL_t ray_vect_idim, ray_orig_idim, fc_idim, fcoord, fcoordn
	cdef cDBL_t zbefore, z, zin, zout, z_intersect, zmax

	cdef numpy.ndarray[cDBL_t ,ndim=1] dx = numpy.empty(max_read_level+1, DBL_t)
	cdef cDBL_t dx_loc, dx_2

	
	# Big AMR arrays
	cdef numpy.ndarray[int, ndim=2] sons = dset.amr_struct["son_indices"]
	cdef numpy.ndarray[int, ndim=1] cell_levels = dset.amr_struct["cell_levels"]
	cdef numpy.ndarray[int, ndim=2] neighbors = dset.amr_struct["neighbors"]
	cdef numpy.ndarray[cDBL_t, ndim=2] grid_centers = dset.amr_struct["grid_centers"]
	cdef numpy.ndarray[cDBL_t, ndim=3] cell_centers = dset.get_cell_centers()

	# Scalar operand data
	cdef int nfunc = 3 #op.nscal_func()
	cdef numpy.ndarray[cDBL_t, ndim=3] scalar_data = numpy.empty((nfunc, ngrids, twotondim), DBL_t)
	ifunc = 0
	for (key, scal_func) in op.iter_scalar_func():
		scalar_data[ifunc,:,:] = scal_func(dset)
		ifunc += 1
	cdef cINT_t max_op = op.is_max_alos()
	
	# Integrand variables
	cdef numpy.ndarray[cDBL_t, ndim=2] I = numpy.zeros((nrays, 3), dtype=DBL_t) # R, G, B ray intensity channels
	cdef numpy.ndarray[cDBL_t, ndim=1] vlines, wlines, rlines, glines, blines, alines
	cdef int nlines
	if not rgb:
		# init variables to avoid a bug on ubuntu 12.04
		tf = ColorLinesTransferFunction((0.0, 1.0))
	gamma = tf.gamma
	vlines, wlines, rlines, glines, blines, alines = tf.vwrgba
	nlines = vlines.size
	
	cdef numpy.ndarray[cDBL_t, ndim=1] vlinesC = vlines
	cdef numpy.ndarray[cDBL_t, ndim=1] wlinesC = wlines
	cdef numpy.ndarray[cDBL_t, ndim=1] rlinesC = rlines
	cdef numpy.ndarray[cDBL_t, ndim=1] glinesC = glines
	cdef numpy.ndarray[cDBL_t, ndim=1] blinesC = blines
	cdef numpy.ndarray[cDBL_t, ndim=1] alinesC = alines
	cdef numpy.ndarray[cINT_t, ndim=1] param_info_int
			
	if use_C_code:
		param_info_int = numpy.empty(10, INT_t)
		param_info_int[0] = nrays
		param_info_int[1] = ndim
		param_info_int[2] = nlines
		param_info_int[3] = max_read_level
		param_info_int[4] = nfunc
		param_info_int[5] = max_op
		param_info_int[6] = ngrids
		param_info_int[7] = rgb
		param_info_int[8] = verbose_C
		ray_trace_octree_C(<double *>I.data, <double *>ray_origins.data, <double *>ray_vects.data,\
			<double *>grid_centers.data, <double *>cell_centers.data, <int *>sons.data, <int *>cell_levels.data, <double *>scalar_data.data,\
			<int *>neighbors.data, <double *>ray_lengths.data,\
			<long *>param_info_int.data, <double> gamma, <double *> vlinesC.data, <double *> wlinesC.data, <double *> rlinesC.data\
			, <double *> glinesC.data, <double *> blinesC.data, <double *> alinesC.data)
		return I
	
	################### Precompute the cell sizes #########################
	dx[0]=0.5
	for ilevel in range(1,max_read_level+1):
		dx[ilevel] = dx[ilevel-1] / 2.0
	dx_min = dx[(max_read_level-1)]
	
	cdef numpy.ndarray[cINT_t, ndim=1] active_mask
	if not rgb:
		active_mask = numpy.array(dset.get_active_mask(), INT_t)
	##############################################################################################################
	#                                             Ray iteration                                                  #
	##############################################################################################################
	for iray in range(nrays): # Iteration over rays
		if verbose_C and (iray%100000 == 0):
			print("ray = %i/%i"%(iray+1, nrays))
		zmax = ray_lengths[iray]
		# Integration of the ray intensities (RGBA) between 0.0 (ray origin) and zmax
		
		######## Project ray origin inside the simulation box if it was outside ########
		if ray_origins[iray, 0] < 0 or ray_origins[iray, 0] > 1 or\
		ray_origins[iray, 1] < 0 or ray_origins[iray, 1] > 1 or\
		ray_origins[iray, 2] < 0 or ray_origins[iray, 2] > 1:
			# ray_origin is outside the simulation box
			t_0 = (0 - ray_origins[iray, 0]) / ray_vects[iray,0]
			t_1 = (1 - ray_origins[iray, 0]) / ray_vects[iray,0]
			tmin = min(t_0, t_1)
			tmax = max(t_0, t_1)
			t_0 = (0 - ray_origins[iray, 1]) / ray_vects[iray,1]
			t_1 = (1 - ray_origins[iray, 1]) / ray_vects[iray,1]
			tmin = max(tmin, min(t_0, t_1))
			tmax = min(tmax, max(t_0, t_1))
			t_0 = (0 - ray_origins[iray, 2]) / ray_vects[iray,2]
			t_1 = (1 - ray_origins[iray, 2]) / ray_vects[iray,2]
			tmin = max(tmin, min(t_0, t_1))
			tmax = min(tmax, max(t_0, t_1))
			if (0 < tmin < tmax):
				# intersect = 1
				# if tmin < 0:
				#	print "error : wrong ray",iray," direction"
				for idim in range(ndim):
					ray_origins[iray,idim] = ray_origins[iray,idim] + tmin * ray_vects[iray,idim] # * 1.00001
			else:
				# Projection not possible : we skip this ray !
				continue
				
		######## Tree-search in octree to find where is the current ray origin ########
		igrid = 0
		igrid_son = 0
		ilevel=-1
		while igrid_son >= 0:
			ilevel+=1
			if ilevel != 0: igrid = igrid_son
			# Find son index in octree
			ind = 0
			for idim in range(ndim):
				ibit = (ray_origins[iray, idim] > grid_centers[igrid, idim])
				ind += (ibit << idim)
			igrid_son = sons[igrid, ind]
		##############################################################################
		
		# Now the ray_origin is in a leaf-cell of the current grid
		zbefore = 0.0
		z=0.0 # Set the position along ray to 0.0 (ray origin)
		scalar = scalar_data[0, igrid, ind]
		scalarUp = scalar_data[1, igrid, ind]
		#print "ind", ind, "igrid", igrid
		#print "scalar", scalar
		##################  Initialisation : first cell ray/face intersection computation #################
		dx_loc = dx[ilevel]
		dx_2 = dx[ilevel+1] # dx_loc/2.0
		# Finding the exit intersection of the ray and the current cell
#		test_ok = 0
		for idim in range(ndim):
			ray_vect_idim = ray_vects[iray, idim]
			ray_orig_idim = ray_origins[iray, idim]
			if(ray_vect_idim == 0.0): # ray is parallel to the faces perpendicular to dimension idim.
				# There cannot be any intersection with thes faces.
				continue
			elif (ray_vect_idim > 0.0):
				sgn_face = 1
			else:
				sgn_face = -1
			# Coordinate of the cell face center along dimension idim
			fc_idim = cell_centers[igrid, ind, idim] + sgn_face * dx_2
			# Ray coordinate (z) of the intersection of the ray with the face plane
			z_intersect = (fc_idim - ray_orig_idim)/ ray_vect_idim
			# Is this intersection within the face ?
			# Position of the face in the neighbour cell along the dimension idim (0 or 1)
			isgn = (-sgn_face+1)>>1
			# Index of the son cell where the ray crosses the exit face (init.)
			ind_son = (twotondim-1) - (isgn<<idim)
			v=0.0
			vl=0.0
			for idim2 in range(ndim):
				if idim2 == idim: continue # Skip the dimension perpendicular to the face
				# Coordinate of the intersection
				fcoord = (ray_origins[iray,idim2] + z_intersect * ray_vects[iray, idim2])
				# Normalised coordinate within the face [ [-0.5, 0.5[, [-0.5, 0.5[ ]
				fcoordn = (fcoord - cell_centers[igrid, ind, idim2]) / dx_loc
				if (fcoordn>vl):vl=fcoordn
				if fcoordn<0.0:
					# Index of the son cell where the ray crosses the exit face (update)
					ind_son -= (1<<idim2)
					fcoordn=-fcoordn
				if fcoordn>v: v=fcoordn
			if ((v <= 0.5) and (vl<=0.5)): # Intersection found
#				test_ok=1
				break
#		if not test_ok:
#			print "Error : no intersection found"
#			return
		###################################################################################################

#		I[iray, 0] = ilevel+1
		
		if z_intersect>=zmax:
			# End of ray-tracing
			z = zmax
			zin = zmax
		else:
			zin = z_intersect
		
		while z<zmax:######################   Ray integration loop  #######################################
			# Record current level
			ileveln = ilevel
			indn = ind
			######################### Find neighbour cell in octree #############################
			if ((ind>>idim)&1) != isgn:
				iface = 2 * idim + 1 - isgn
				igridn = neighbors[igrid, iface]
				if cell_levels[igrid] != (ilevel+1):
					error = 1
					print(cell_levels[igrid], (ilevel+1))
				ileveln = cell_levels[igridn] - 1
				# son index from ray/face intersection computation
				if ileveln == ilevel: # Same resolution neighbor cell
					# Neighbour cell index
					indn -= sgn_face * (1<<idim)
					if ileveln+1 < max_read_level:
						# More refined neighbor cell ?
						igrid_son = sons[igridn, indn]
						if igrid_son > 0:
							igridn = igrid_son
							ileveln += 1
							# Neighbour cell index
							indn = ind_son - sgn_face * (1<<idim)
				elif ileveln == (ilevel-1):# Coarser resolution neighbor cell
					indn = 0
					for idim2 in range(ndim):
						cc = cell_centers[igrid, ind, idim2]
						if idim2==idim: cc += sgn_face * dx_loc
						if cc>grid_centers[igridn, idim2]: indn += (1<<idim2)
				else: # Sanity check : neighbour cells should never be more than 1 level of
					# refinement apart (RAMSES octree property).
					#print "level error : (ilevel, igridn, ilevel_neighbour) = ", ilevel, igridn, ileveln
					#print "              (zin, z, zout) = ", z, zin, zout
					error = 1
					break
				igrid = igridn
				ind = indn
				ilevel = ileveln
			else:
				# Neighbour cell index in the current grid
				ind += sgn_face * (1<<idim) # neighbour cell index
				if ilevel+1 < max_read_level:
					# More refined neighbor cell ?
					igrid_son = sons[igrid, ind]
					if igrid_son > 0:
						igrid = igrid_son
						ilevel += 1
						# Neighbour cell index
						ind = ind_son - sgn_face * (1<<idim)

			# Fetch scalar data in neighbour cell
			scalar2 = scalar_data[0, igrid, ind]
			scalarUp2 = scalar_data[1, igrid, ind]
			#####################################################################################

			
			######################### Neighbour cell ray/face intersection computation ##################
			dx_loc = dx[ilevel]
			dx_2 = dx[ilevel+1] # dx_loc/2.0
			# Finding the exit intersection of the ray and the neighbour cell

#			test_ok=0
			for idim in range(ndim):
				ray_vect_idim = ray_vects[iray, idim]
				ray_orig_idim = ray_origins[iray, idim]
				if(ray_vect_idim == 0.0): # ray is parallel to the faces perpendicular to dimension idim.
					# There cannot be any intersection with thes faces.
					continue
				elif (ray_vect_idim > 0.0):
					sgn_face = 1
				else:
					sgn_face = -1

				# Coordinate of the cell face center along dimension idim
				fc_idim = cell_centers[igrid, ind, idim] + sgn_face * dx_2
				# Ray coordinate (z) of the intersection of the ray with the face plane
				z_intersect = (fc_idim - ray_orig_idim)/ ray_vect_idim
				# Is this intersection within the face ?
				# Position of the face in the neighbour cell along the dimension idim (0 or 1)
				isgn = (-sgn_face+1)>>1
				# Index of the son cell where the ray crosses the exit face (init.)
				ind_son = (twotondim-1) - (isgn<<idim)
				v=0.0
				vl=0.0
				for idim2 in range(ndim):
					if idim2 == idim: continue # Skip the dimension perpendicular to the face
					# Coordinate of the intersection
					fcoord = (ray_origins[iray,idim2] + z_intersect * ray_vects[iray, idim2])
					# Normalised coordinate within the face [ [-0.5, 0.5[, [-0.5, 0.5[ ]
					fcoordn = (fcoord - cell_centers[igrid, ind, idim2]) / dx_loc
					if (fcoordn>vl):vl=fcoordn
					if fcoordn<0.0:
						# Index of the son cell where the ray crosses the exit face (update)
						ind_son -= (1<<idim2)
						fcoordn=-fcoordn
					if fcoordn>v: v=fcoordn
				if ((v <= 0.5) and (vl<=0.5)): # Intersection found
#					test_ok=1
					break
#			if not test_ok:
#				print "Error : no intersection found"
			#############################################################################################

			if z_intersect>=zmax:
				# End of ray-tracing
				z = zmax
			else:
				zout = z_intersect
				z = (zin +zout)/2.
			if rgb:
				# Linear evolution of the scalar data
				ndz = <cINT_t> floor((z-zbefore)/(dx_min))
				if ndz==0: ndz=1
				if ndz>5: ndz=5
	#			ndz=1
				dz = (z-zbefore) / ndz
	#			dz_2 = dz / 2.0
				zvar = zbefore
				for idz in range(ndz):
					scalar_loc = scalar
					for iem in range(nlines):
						x = scalar_loc - vlines[iem]
						width = wlines[iem]
						if (fabs(x) > (3.0*width)):
							if x<0.0 : break
						else:
	#						print gauss_value, scalar_loc
							gauss_value = gaussn(x, width)
							em = 10.**(scalar_loc* gamma)
							rem = em * gauss_value * rlines[iem]
							gem = em * gauss_value * glines[iem]
							bem = em * gauss_value * blines[iem]
							alpha = gauss_value * alines[iem]
							absor = 1.0 - alpha * dz
							if absor>0.0:
								for irgb in range(3):
									I[iray, irgb] = absor * I[iray, irgb]
							I[iray, 0] += dz * rem
							I[iray, 1] += dz * gem
							I[iray, 2] += dz * bem
							break
	
					
					zvar += dz
			elif active_mask[igrid]:
				# Piecewise constant evolution of the scalar field => dumb ray integration
				if max_op:
					if scalar > I[iray, 0]:
						I[iray, 0] = scalar
					if scalarUp > I[iray, 1]:
						I[iray, 1] = scalarUp
				else:
					I[iray, 0] += scalar * (zin-zbefore) + scalar2 * (z-zin)
					I[iray, 1] += scalarUp * (zin-zbefore) + scalarUp2 * (z-zin)
				if ilevel+1 > I[iray, 2]:
					I[iray, 2] = ilevel+1
			
			zin = zout
			scalar = scalar2
			scalarUp = scalarUp2
			zbefore = z
		############################## End of ray integration loop  #######################################

	########################################## End of ray iteration ##############################################
	if error:
		print("OCTREE ERRORS OCCURED DURING RAY TRACING...")
	return I
#}}}

@cython.wraparound(False)
@cython.boundscheck(False)
def ray_trace_cartesian(numpy.ndarray[cDBL_t, ndim=3] cube,
			numpy.ndarray[cDBL_t, ndim=2] ray_origins,
			numpy.ndarray[cDBL_t, ndim=2] ray_vectors,
			numpy.ndarray[cDBL_t, ndim=1] ray_lengths,
			bounding_box,
			cINT_t cube_size):#{{{
	""" Octree Ray tracing function

	Parameters
	----------
	cube : numpy.array (float64, ndim=3)
		cube with scalar values, shape=(cube_size, cube_size, cube_size),
		get this  from pymses.analysis import amr2cube
	ray_origins : numpy.array (float64, ndim=2)
		Rays origins : array of vector, shape=(nrays,ndim)
	ray_vectors : numpy.array (float64, ndim=2)
		Rays vectors : array of vector, shape=(nrays,ndim)
	ray_lengths : numpy.array (float64, ndim=1)
		Rays length max : array of vector, shape=(nrays)
	bounding_box : pymses camera bounding box
		pymses camera bounding box (bounding_box=camera.get_bounding_box())
	cube_size : integer
		size of the cube (cube_size=camera.map_max_size)
	
	"""
	# integer variables
	cdef cINT_t nrays= ray_origins.shape[0]
	cdef cINT_t iray, istep
	cdef cDBL_t x, y, z, step_x, step_y, step_z, scale_ratio, bb_xmin, bb_ymin, bb_zmin
	cdef numpy.ndarray[cDBL_t, ndim=1] map = numpy.zeros(nrays)
	scale_ratio = 1. / (bounding_box.max_coords[0] - bounding_box.min_coords[0])
	bb_xmin = bounding_box.min_coords[0]
	bb_ymin = bounding_box.min_coords[1]
	bb_zmin = bounding_box.min_coords[2]
	#print ray_origins[0, 0], bb_xmin
	for iray in range(nrays):
		x = (ray_origins[iray, 0] - bb_xmin) * scale_ratio
		#print x
		y = (ray_origins[iray, 1] - bb_ymin) * scale_ratio
		z = (ray_origins[iray, 2] - bb_zmin) * scale_ratio
		#step_x = ray_vectors[iray, 0] * ray_lengths[iray] * scale_ratio / cube_size
		step_x = ray_vectors[iray, 0] / cube_size
		step_y = ray_vectors[iray, 1] / cube_size
		step_z = ray_vectors[iray, 2] / cube_size
		for istep in range(cube_size):
			#print x,y,z
			#print int(x*cube_size)
			if 0<x<1 and 0<y<1 and 0<z<1:
				map[iray] = map[iray] + cube[int(x*cube_size), int(y*cube_size), int(z*cube_size)]
			x = x + step_x
			y = y + step_y
			z = z + step_z
	# z projection test :
	#map = numpy.zeros(nrays)
	#cdef cINT_t i,j,k
	#for j in range(cube_size):
	#	for i in range(cube_size):
	#		for k in range(cube_size):
	#			map[j*cube_size+i]=cube[i,j,k]
	return map


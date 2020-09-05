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

from . import info, tree_utils, filename_utils
from .amr import read_ramses_amr_file
from .amrdata import read_ramses_amr_data_file
from pymses.core.datasets import Dataset, PointDataset, _read_fields_from_HDF5
from pymses.filters import RegionFilter
from pymses.utils.point_utils import corner_points
from pymses.utils.regions import Box, Sphere
from pymses.utils import misc
from time import time
try:
	import tables
except:
	print("WARNING : Can't import tables module...")
from pymses.core import Source

class OctreeDataset(Dataset):#{{{
	def __init__(self, amr_dicts):
		self.amr_header, self.amr_struct = amr_dicts
		Dataset.__init__(self)
		self.active_mask = None

	def get_source_type(self):
		r"""
		Returns
		-------
		Source.AMR_SOURCE
		
		"""
		return Source.AMR_SOURCE

	def get_cell_centers(self, grid_mask=None):#{{{
		r"""
		Returns
		-------
		cell_centers : ``array``
			AMR cell center coordinates array

		"""
		# Grid centers
		gc = self.amr_struct["grid_centers"]
		if grid_mask is not None:
			gc = gc[grid_mask, :]
		# Grid levels
		gl = self.get_grid_levels(grid_mask)

		# Compute the cell centers
		cell_centers = tree_utils.cell_centers_from_grid_centers(gc, gl)
		return cell_centers
	#}}}
	
	def get_grid_levels(self, grid_mask=None):#{{{
		r"""
		Returns
		-------

		g_levels : ``array``
			the grid levels array

		"""
		if "cell_levels" in self.amr_struct:
			gl = self.amr_struct["cell_levels"]
		else:
			gl = tree_utils.grid_levels(self.amr_struct)
			self.amr_struct["cell_levels"] = gl
		if grid_mask is None:
			return gl
		else:
			return gl[grid_mask]
	#}}}

	def get_active_mask(self):#{{{
		r"""
		Returns
		-------
		mask = ``array`` of ``bool``
			Active grids mask

		"""
		if self.active_mask is None:
			self.active_mask = numpy.ones(self.amr_struct["ngrids"], 'bool')
		return self.active_mask
	#}}}

	def sample_points(self, points, max_search_level=None, add_level=False, add_cell_center=False,\
			  interpolation=False, use_C_code=True, use_openCL=False, verbose=True):#{{{
		r"""
		AMR grid point sampling method

		Parameters
		----------
		points : ``array``
			sampling points coordinates array
		max_search_level : ``int`` or ``ndarray`` (default None)
			max. AMR level to read during sampling
			(see :func:'~pymses.sources.ramses.tree_utils.tree_search' for details)
		add_level : ``boolean`` (default False)
			whether we need to add a `level` field in the returned dataset containing
			the value of the AMR level the sampling points fall into
		add_cell_center : ``boolean`` (default False)
			whether we need to add a `cell_center` field in the returned dataset containing
			the coordinates of the AMR cell center the sampling points fall into
		interpolation : ``boolean`` (default False)
			Experimental : attempt to compute bi/tri-linear interpolation
			Require use_C_code=True and use_openCL=False
		use_C_code : ``boolean`` (default True)
			The pure C code is slightly faster than the (not well optimized) Cython code,
			and should give the same result
		use_openCL : ``boolean`` (default False)
			Experimental : use "pyopencl" http://pypi.python.org/pypi/pyopencl
		verbose : ``boolean`` (default True)
			some console printout...
		Returns
		-------
		dset : ``PointDataset``
			point-based sampled values dataset of the available AMR fields

		"""
		# Create an empty PointDataset based on 'points'
		point_dset = PointDataset(points)
		t0 = time()
		if add_level or add_cell_center:
			if verbose: print("C and OpenCl code doesn't work yet with add_level or add_cell_center option")
			use_C_code = False
			use_openCL = False
		if use_openCL:
			import pyopencl as cl
			misc.init_OpenCl()
			t0 = time()
			if interpolation:
				print(" WARNING : openCL sampling with interpolation is not implemented yet !")
			extracted_data = self.sample_point_openCL(self.amr_struct, self[self.scalars[0]], points)
			point_dset.add_scalars(self.scalars[0], extracted_data)
			if verbose: print("OpenCL sampling time = ", time() - t0)
			return point_dset
		if use_C_code:
			ccenter_array = self.get_cell_centers()
			t0 = time()
			nscalars = len(self.scalars) + len(self.vectors) * 3
			big_scalars_array = None
			ngrids =  self.amr_struct["ngrids"]
			ndim = points.shape[1]
			for scal in self.scalars:
				if big_scalars_array is None:
					big_scalars_array = self[scal]
				else:
					big_scalars_array = numpy.concatenate((big_scalars_array, self[scal]),axis=0)
			for vect in self.vectors:
				if big_scalars_array is None:
					big_scalars_array = self[vect][:,:,0]
					for idim in range(1, ndim):
						big_scalars_array = numpy.concatenate((big_scalars_array,self[vect][:,:,idim]),axis=0)
				else:
					for idim in range(ndim):
						big_scalars_array = numpy.concatenate((big_scalars_array,self[vect][:,:,idim]),axis=0)
			extracted_data = tree_utils.sample_point(self.amr_struct, big_scalars_array, nscalars, ngrids,points,\
						 max_search_level=max_search_level, ccenter_array = ccenter_array, interpolation=interpolation)
			extracted_data = extracted_data.reshape((nscalars, points.shape[0]))
			i = 0
			for scal in self.scalars:
				point_dset.add_scalars(scal, extracted_data[:][i])
				i += 1
			for vect in self.vectors:
				point_dset.add_vectors(vect, extracted_data[:][i:i+3].transpose())
				i += 3
			if verbose: print("C code sampling time = ", time() - t0)
			return point_dset
		# First find the relevant cells in the tree
		search_dict = tree_utils.tree_search(self.amr_struct, points, max_search_level=max_search_level)
		
		# Now extract the relevant hydro data
		grids = search_dict["grid_indices"]
		cells = search_dict["cell_indices"]

		def process(kind_vars, add_func):
			for name in kind_vars:
				data = self[name]
				extracted_data = data[grids, cells]
				add_func(name, extracted_data)
		
		if interpolation:
			print("WARNING : interpolation option need use_C_code=True")
		# Extract the scalars
		process(self.scalars, point_dset.add_scalars)
		# Extract the vectors
		process(self.vectors, point_dset.add_vectors)
		
		# Extract the value of the cell level in which the points are sampled
		if add_level:
			levels = search_dict["levels"]
			point_dset.add_scalars("level", levels)

		if add_cell_center:
			ccenter = self.get_cell_centers() 
			point_dset.add_vectors("cell_center", ccenter[grids, cells])
		if verbose: print("Python code sampling time = ", time() - t0)
		return point_dset
	#}}}
	
	def sample_point_openCL(self, amr_struct, scalars, points):#{{{
		npoints = points.shape[0]
		ndim = numpy.uint32(points.shape[1])
		twotondim = numpy.uint32(2**ndim)
		C_points_array = numpy.asarray(points).astype(numpy.float32)
		C_scalars_array = numpy.asarray(scalars).astype(numpy.float32)
		max_read_level = amr_struct["readlmax"]
		#glevel = self.get_grid_levels()
		#max_read_level = min(max(glevel), 7)
		#print max_read_level
		max_search_level = numpy.uint32(max_read_level)
		
		# Big AMR arrays
		grid_centers = amr_struct["grid_centers"].astype(numpy.float32)
		son_indices = amr_struct["son_indices"].astype(numpy.uint32)
		# Output array:
		extracted_data = numpy.zeros(npoints).astype(numpy.float32)
		import pyopencl as cl
		grid_centers_buf = cl.Buffer(misc.OpenCL_ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=grid_centers)
		son_indices_buf = cl.Buffer(misc.OpenCL_ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=son_indices)
		scalars_buf = cl.Buffer(misc.OpenCL_ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=C_scalars_array)
		points_buf = cl.Buffer(misc.OpenCL_ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=C_points_array)
		extracted_data_buf = cl.Buffer(misc.OpenCL_ctx, cl.mem_flags.WRITE_ONLY, extracted_data.nbytes)
		prg = cl.Program(misc.OpenCL_ctx, """
		__kernel
		void sample(__global float *extracted_data, __global const float *grid_centers,
				__global const uint *son_indices, __global const float *scalars,
				__global const float *points, uint max_search_level, uint ndim, uint twotondim){
			uint ipoint = get_global_id(0);
			uint idim, ibit, ilevel, icell, son_ind;
			uint igrid = 0;
			for (ilevel = 1 ; ilevel <= max_search_level ; ilevel++){
				icell = 0;
				for (idim = 0 ; idim < ndim ; idim++){
					ibit = points[ipoint*ndim + idim] > grid_centers[igrid*ndim + idim];
					icell += (ibit << idim);
				}
				son_ind = son_indices[igrid*twotondim + icell];
				if ((son_ind == 4294967295) || // unsigned int - 1 = 4294967295
					(ilevel == max_search_level)){
					break;
				}
				else{
					igrid = son_ind;
				}
			}
			extracted_data[ipoint] = scalars[igrid*twotondim + icell];
		}
		""").build()
		prg.sample(misc.OpenCL_queue, (npoints,), None, extracted_data_buf, grid_centers_buf,
				son_indices_buf, scalars_buf,
				points_buf, max_search_level, ndim, twotondim)
		cl.enqueue_copy(misc.OpenCL_queue, extracted_data, extracted_data_buf).wait()

		return extracted_data
	#}}}

	def write_hdf5(self, h5file, where="/"):#{{{
		r"""

		"""
		if isinstance(h5file, tables.file.File):
			h5fobj = h5file
			close_at_end = False
		else:
			h5fobj = tables.openFile(h5file, "w")
			close_at_end = True
		
		# Save dataset icpu number
		h5fobj.createArray(where, "icpu", self.icpu)

		# Record amr_header and amr_struct dict contents
		for dict_name, d in [("amr_header", self.amr_header),("amr_struct", self.amr_struct)]:
			group = h5fobj.createGroup(where, dict_name)
			for data_name in list(d.keys()):
				h5fobj.createArray(group, data_name, d[data_name])

		# Save field data
		Dataset.write_hdf5(self, h5fobj, where)

		# Close if needed
		if close_at_end:
			h5fobj.close()
	#}}}

	@classmethod
	def from_hdf5(cls, h5file, where="/"):#{{{
		r"""

		"""
		if isinstance(h5file, tables.file.File):
			h5fobj = h5file
			close_at_end = False
		else:
			h5fobj = tables.openFile(h5file, "r")
			close_at_end = True

		
		# Read amr_header and amr_struct dict contents
		amr_head = {}
		amr_str = {}
		for dict_name, d in [("amr_header", amr_head),("amr_struct", amr_str)]:
			group = h5fobj.getNode(where, dict_name)
			for array in h5fobj.listNodes(group):
				data = array.read()
				d[array.name] = data
		# PointDataset initialisation
		odset = cls((amr_head, amr_str))
		odset.read_lmax = amr_str["readlmax"]
		
		# Set octree dataset icpu number
		odset.icpu = h5fobj.getNode(where, "icpu").read()

		# Read scalar and vector fields
		_read_fields_from_HDF5(odset, h5fobj, where)

		# Close HDF5 file if needed
		if close_at_end:
			h5fobj.close()

		return odset
	#}}}

#}}}

class RamsesOctreeDataset(OctreeDataset):#{{{
	r"""
	RAMSES octree dataset class

	contains all the relevant information about the AMR tree structure

	"""
#	def __init__(self, amr_dicts):
#		amr_head, amr_str = amr_dicts
#		OctreeDataset.__init__(self, amr_head, amr_str)

	def get_active_mask(self):#{{{
		r"""
		Returns
		-------
		mask = ``array`` of ``bool``
			Active grids mask

		"""
		if self.active_mask is None:
			self.active_mask = numpy.zeros(self.amr_struct["ngrids"], 'bool')
			offset = 0
			icpu = self.icpu
			ngridlevel = self.amr_struct["ngridlevel"]
			ngridbound = self.amr_struct["ngridbound"]
			for ilevel in range(self.amr_struct["readlmax"]):
				nbefore = ngridlevel[:icpu, ilevel].sum()
				nkeep = ngridlevel[icpu, ilevel].sum()
				ngrids = ngridlevel[:, ilevel].sum() + ngridbound[:, ilevel].sum()
				self.active_mask[offset+nbefore:offset+nbefore+nkeep] = True
				offset += ngrids
		return self.active_mask
	#}}}

	def get_boundary_mask(self):#{{{
		r"""
		Returns
		-------
		mask = ``array`` of ``bool``
			Boundary grids mask

		"""
		mask = numpy.zeros(self.amr_struct["ngrids"], 'bool')
		offset = 0
		ngridlevel = self.amr_struct["ngridlevel"]
		ngridbound = self.amr_struct["ngridbound"]
		for ilevel in range(self.amr_struct["readlmax"]):
			nskip = ngridlevel[:, ilevel].sum()
			ngrids = nskip + ngridbound[:, ilevel].sum()
			grid_mask[offset+nskip:offset+ngrids] = True
			offset += ngrids
		return mask
	#}}}

	def get_idomain_grid(self):#{{{
		r"""
		Returns
		-------
		idomain_grid = ``array`` of ``int``
			idomain grids array : every amr oct in amr_struct match one idomain and cpu file.
			! WARNING : icpu = idomain +1 So we have for icpu between 1 to nproc:
			dset = source.get_domain_dset(icpu)
			domain_grid = dset.get_idomain_grid()
			idomain = icpu - 1
			sum(domain_grid == idomain) == sum(dset.get_active_mask())

		"""
		ngrids = self.amr_struct["ngrids"]
		ngridlevel = self.amr_struct["ngridlevel"]
		ngridbound = self.amr_struct["ngridbound"]
		idomain_grid = numpy.zeros(ngrids, "i")
		ngridarray = numpy.zeros((ngridlevel.shape[0]+ngridbound.shape[0],ngridlevel.shape[1]), "i")
		ngridarray[:ngridlevel.shape[0],:] = ngridlevel
		ngridarray[ngridlevel.shape[0]:,:] = ngridbound
		
		i = 0
		for cpu_lvl in ngridarray.transpose():
			for idomain, noct in enumerate(cpu_lvl):
				if noct !=0:
					for x in range(i, i+noct):
						idomain_grid[x] = idomain
					i += noct
		
		
		assert(i == ngrids)
		return idomain_grid
	#}}}

	def to_cartesian(self, var, xmin, xmax, level, dest=None):#{{{
		return tree_utils.amr2array_3d(self.amr_struct, self[var], self.icpu,
				xmin, xmax, level, dest)
	#}}}

#}}}

class RamsesOctreeReader(object):#{{{
	r"""
	RamsesOctreeReader class

	"""
	def __init__(self, output_repos, iout, icpu, ivars_descrs_by_file, verbose=True):#{{{

		self.output_repos = output_repos
		self.iout = iout
		self.icpu = icpu
		self.ivars_descrs_by_file = ivars_descrs_by_file
		self.verbose = verbose


		self.amr_filename = filename_utils.amrlike_filename(
				"amr", output_repos, iout, icpu, check_exists=True)
	#}}}

	def read(self, read_lmax, fields_to_read=None):
		if fields_to_read != None:
			self.fields_to_read = fields_to_read
		# Load the AMR structure
		if self.verbose: print("Reading amr data  : %s" % self.amr_filename)
		amr = read_ramses_amr_file(
				self.amr_filename, max_read_level=read_lmax)
		
		amr_header, amr_struct = amr

		# Construct the octree
		dset = RamsesOctreeDataset(amr)
		for amr_file_type in list(self.ivars_descrs_by_file.keys()):
			ivars_to_read, descrs = self.ivars_descrs_by_file[amr_file_type]
			# Output file name
			fname = filename_utils.amrlike_filename(amr_file_type,
					self.output_repos, self.iout, self.icpu,
					check_exists=False)

			# Read the data
			if self.verbose: print("Reading %s : %s"%(amr_file_type.ljust(9), fname))
			try:
				data = read_ramses_amr_data_file(amr_file_type, fname, amr, ivars_to_read)
			except IOError:
				print("Reading %s failed, associated fields dropped" % fname)
				continue

			# Pack the fields into the octree thanks to the descriptors
			for desc in descrs:
				desc.gather(data, dset)

		# CPU id in 0-starting indexing
		dset.icpu = self.icpu - 1
		return dset
	#}}}

#}}}

class CameraOctreeDataset(OctreeDataset):#{{{
	r"""
	Camera filtered octree dataset class

	contains all the relevant information about the AMR tree structure in a camera region.

	"""
	def __init__(self, *args, **kwargs):#{{{
		if isinstance(args[0], tuple):# Initialisation with amr dicts
			amr_dicts = args[0]
			OctreeDataset.__init__(self, amr_dicts)
		else:
			assert isinstance(args[0], int) and isinstance(args[1], int)
			ngrid_max, ndim = args
			if "scalars" in list(kwargs.keys()):
				scalars=kwargs["scalars"]
			else:
				scalars = None
			if "vectors" in list(kwargs.keys()):
				vectors=kwargs["vectors"]
			else:
				vectors = None
			twotondim = (1 <<ndim)
			twondim = 2*ndim
			amr_head = {"ndim": ndim}
			amr_str  = {"son_indices":  -numpy.ones((ngrid_max, twotondim), dtype='i'),
						"neighbors":  -numpy.ones((ngrid_max, twondim), dtype='i'),
						"grid_centers": numpy.zeros((ngrid_max, ndim)),
						"cell_levels": numpy.zeros(ngrid_max, dtype='i'),
						"ngrids":ngrid_max,
						"o_igrid_max":1}

			# Set center coords of the root grid
			amr_str["grid_centers"][0,:]=0.5
			amr_str["cell_levels"][0]=1
			
			OctreeDataset.__init__(self, (amr_head, amr_str))

			if scalars is not None:
				for fs in scalars:
					self.add_scalars(fs, numpy.zeros((ngrid_max, twotondim)))
			if vectors is not None:
				for fv in vectors:
					self.add_vectors(fv, numpy.zeros((ngrid_max, twotondim, ndim)))

		self.icpu=0
	#}}}

	def restrict(self):#{{{
		ng = self.amr_struct["o_igrid_max"]
		new_dset = CameraOctreeDataset(ng, self.amr_header["ndim"])
		new_dset.amr_struct["readlmax"] = self.amr_struct["readlmax"]
		new_dset.amr_struct["son_indices"] = self.amr_struct["son_indices"][:ng,:].copy()
		new_dset.amr_struct["neighbors"] = self.amr_struct["neighbors"][:ng,:].copy()
		new_dset.amr_struct["grid_centers"] = self.amr_struct["grid_centers"][:ng,:].copy()
		new_dset.amr_struct["cell_levels"] = self.amr_struct["cell_levels"][:ng].copy()
		for field in self.scalars:
			new_dset.add_scalars(field, self[field][:ng,:].copy())
		for field in self.vectors:
			new_dset.add_vectors(field, self[field][:ng,:,:].copy())
		del self.amr_struct["o_igrid_max"]
		return new_dset
	#}}}
#}}}

class CameraOctreeDatasource(RegionFilter):#{{{
	r"""Camera octree dataset

	This call the octree_build function which fills an octree structure with a RAMSES AMR dataset
	by taking into account the original AMR refinement only in the camera specific region. This
	regroup	cpu distributed octree datasets in a single octree dataset.
	
	Parameters
	----------

	camera:
		camera that defines the region of interest for the octree dataset
	esize :
		extension size : extend the camera region
	ramses_amr_source
		ramses source to flatten
	radius (default None):
		define a sphere region like that : Sphere(camera.center, radius+esize)
		-> a RegionFilter is called on this region
	ngrid_max (default 2.000.000):
		Initial size of the created AMR array : 1e7 = 10 Millions <-> 3.8 GBytes of RAM memory
		This parameter has to be big enough to fit the local octree size.
	include_split_cells (default False):
		If True, the created octree will include all physical values of
		intermediary AMR resolution level (i.e. cells that are refined).
		If False, only leaf cell values are stored (this save memory
		and computation time for cell_to_points splatting rendering)

	Example (from pymses_repository/bin/pymses_tf_ray_tracing.py)
	--------
	ro = pymses.RamsesOutput(fileDir, outNumber)
	cam = Camera(center=[ 0.5, 0.5, 0.5 ], line_of_sight_axis="z", up_vector="y",\
			region_size=[5.0E-1, 5.0E-1], distance=2.5E-1,\
			far_cut_depth=2.5E-1, map_max_size=512)
	source = ro.amr_source(["rho"])
	esize = 0.5**(ro.info["levelmin"]+1)

	fullOctreeDataSource = CameraOctreeDatasource(cam, esize, source).dset

	# Here is how to save and load local octree option for faster reuse if wanted :
	#fullOctreeDataSource.write_hdf5("myDset")
	#fullOctreeDataSource = CameraOctreeDataset.from_hdf5("myDset")
	OctreeRT = OctreeRayTracer(fullOctreeDataSource)
	op = ScalarOperator(lambda dset: (dset["rho"]))
	map, levelmax_map = OctreeRT.process(op, cam, rgb=False)

	"""
	def __init__(self, camera, esize, ramses_amr_source, radius=None, ngrid_max=2000000, include_split_cells=False):#{{{
		t0 = time()
		self.ngrid_max = int(ngrid_max)
		# Extended camera
		r = numpy.ones((2,3)) * esize
		r[0,:] = -r[0,:]
		from pymses.analysis.visualization.camera import ExtendedCamera
		ecam = ExtendedCamera(camera, r)
		bb = ecam.get_bounding_box()

		if radius is None:
			# Init Filter with extended camera bounding box region
			reg=bb
		else:
			# Defined sphere region, checks the region includes the camera area
			zmax = numpy.max([camera.far_cut_depth, camera.distance])
			xmax = camera.region_size[0]/2.
			ymax = camera.region_size[1]/2.
			assert (radius**2 >= (xmax**2+ymax**2+zmax**2))
			reg = Sphere(camera.center, radius+esize)
		
		RegionFilter.__init__(self, reg, ramses_amr_source)

		# Max. read level setup
		lreq = camera.get_required_resolution()
		self.set_read_lmax(lreq)

		# Init. self.dset to None
		self.dset = None
		self.build_dset(ecam, radius, include_split_cells=include_split_cells)
		print("CameraOctreeDatasource loaded up to level", lreq,\
			"with ngrids =", self.dset.amr_struct["ngrids"],\
			"(loading time = %.2fs"%(time() - t0), ")")
	#}}}

	def build_dset(self, camera, radius, include_split_cells=0):#{{{
		# Fetch camera info
		cc = camera.center
		ca = camera.get_camera_axis()
		corners = corner_points(numpy.zeros((1,3)), numpy.ones(1))
		rcorners = camera.viewing_angle_rotation().transform_points(corners)
		bbmin = numpy.min(rcorners, axis=0)
		bbmax = numpy.max(rcorners, axis=0)
		cell_box = Box([bbmin, bbmax])
		cmb = camera.get_map_box()


		# Build octree dataset
		odset = None
		# Region-filtered octree structure creation
		for dset in self.iter_dsets():
			if odset is None:
				ndim = dset.amr_header["ndim"]
				odset = CameraOctreeDataset(self.ngrid_max, ndim, scalars=dset.scalars, vectors=dset.vectors)
				odset.amr_struct["readlmax"] = dset.amr_struct["readlmax"]
			tree_utils.octree_build(odset, dset, (cc, ca, cmb, cell_box), radius, include_split_cells=include_split_cells)
			assert (odset.amr_struct["o_igrid_max"]<odset.amr_struct["ngrids"]), "Memory error : increase ngrid_max"
		del dset

		# Neighbor indices computation
		tree_utils.octree_compute_neighbors(odset)

		# Set self.dset to new built octree
		# C'est moche, on restreint le tableau en memoire en redupliquant l'octree mais a la bonne taille 
		self.dset = odset.restrict()

		# Set the data cpu list to [0]. The octree is built in memory, the source now only returns the dataset
		self._data_list = [0]
	#}}}

	def get_domain_dset(self, idomain):#{{{
		if self.dset is None:
			return RegionFilter.get_domain_dset(self, idomain)
		else:
			return self.dset
	#}}}

#}}}

__all__ = ["RamsesOctreeDataset", "CameraOctreeDatasource", "CameraOctreeDataset"]

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

from pymses.utils.regions import Box
from pymses.core.transformations import *
from pymses.core.sources import Filter
from pymses.filters import RegionFilter
from pymses.utils.point_utils import meshgrid
import csv, json
try:
	import tables as T
except:
	print("WARNING : Can't import tables module...")
from .transfer_functions import ColorLinesTransferFunction

class Camera(object):#{{{
	r"""
	Camera class for 2D projected maps computing. Take a look at documentation figures to get a clearer definition.
	
	Parameters
	----------
	center             : region of interest center coordinates (default value is [0.5, 0.5, 0.5],
						 the simulation domain center).
	line_of_sight_axis : axis of the line of sight (z axis is the default_value) : not zero-normed
						 [ux, uy, uz] array or simulation domain specific axis key "x", "y" or "z"
	up_vector          : direction of the y axis of the camera (up). If None, the up vector is set
						 to the z axis (or y axis if the line-of-sight is set to the z axis). If
						 given a not zero-normed [ux, uy, uz] array is expected (or a simulation
						 domain specific axis key "x", "y" or "z").
	region_size        : projected size of the region of interest (default (1.0, 1.0))
	distance           : distance of the camera from the center of interest (along the line-of-sight
						 axis, default 0.5).
	far_cut_depth      : distance of the background (far) cut plane from the center of interest
						 (default 0.5). The region of interest is within the camera position and the
						 far cut plane.
	map_max_size       : maximal resolution of the camera (default 1024 pixels)
	log_sensitive      : whether the camera pixels are log sensitive or not (default True).
	perspectiveAngle   : (default 0 = isometric view) angle value in degree which can be used to
						 transfom the standard pymses isometric view into a perspective view.
						 Take a look at documentation figures to get a clearer definition.
	
	Examples
	--------
		>>> cam  = Camera(center=[0.5, 0.5, 0.5], line_of_sight_axis='z', region_size=[1., 1.], \
		... distance=0.5, far_cut_depth=0.5, up_vector='y', map_max_size=512, log_sensitive=True)
	
	"""
	special_axes = {"x": N.array([1.,0.,0.]), "y": N.array([0.,1.,0.]), "z": N.array([0.,0.,1.])}

	def __init__(self, center=None, line_of_sight_axis="z", up_vector=None, region_size=[1.,1.], \
			distance=0.5, far_cut_depth=0.5, map_max_size=1024, log_sensitive=True, perspectiveAngle=0):#{{{
		"""
		Camera builder
		
		"""
		# Region of interest center coordinates
		if center is None:
			self.center = N.array([0.5, 0.5, 0.5])
		else:
			self.center = N.asarray(center, dtype='d')

		# Axis of the line of sight
		if line_of_sight_axis in list(Camera.special_axes.keys()):
			self.los_axis = Camera.special_axes[line_of_sight_axis]
		else:
			self.los_axis = line_of_sight_axis / N.linalg.norm(line_of_sight_axis,2)

		# Up vector of the camera
		if up_vector is not None:
			if up_vector in list(Camera.special_axes.keys()):
				self.up_vector = Camera.special_axes[up_vector]
			else:
				self.up_vector= up_vector / N.linalg.norm(up_vector,2)
			assert (not (N.cross(self.up_vector,self.los_axis) == 0.).all()), \
					"Up vector must be different from line-of-sight axis"
		else:
			# Default up vector of the camera : z axis. if the line of sight is the z axis,
			# the up vector is set to y-axis
			if (self.los_axis == Camera.special_axes["z"]).all():
				self.up_vector = Camera.special_axes["y"]
			else:
				self.up_vector = Camera.special_axes["z"]

		# Map extent
		self.region_size = region_size

		# Region of interest depth
		self.distance = distance
		self.far_cut_depth = far_cut_depth

		# Map maximal size (pixels)
		self.map_max_size = map_max_size

		# Logarithmic sensitivity of the Camera captors
		self.log_sensitive = log_sensitive
		
		# Set the perspective angle 
		self.set_perspectiveAngle(perspectiveAngle)

		# Set the color transfer function to None
		self.color_tf=None
	#}}}
	
	def similar(self,cam):#{{{
		"""
		Draftly test if a camera is roughly equal to an other one, just to know in the amrviewer GUI 
		if we need to reload data or not.
		"""
		return (self.center == cam.center).all() and (self.los_axis == cam.los_axis).all() \
				 and self.map_max_size == cam.map_max_size
	#}}}	

	def set_perspectiveAngle(self, perspectiveAngle=0):#{{{
		"""
		Set the perspectiveAngle (default 0 = isometric view) angle value in degree which 
		can be used to transfom the standard pymses isometric view into a perspective view.

		"""
		self.perspectiveAngle = perspectiveAngle
		if perspectiveAngle != 0:
			centered_map_box = self.get_map_box()
			xmax = centered_map_box.max_coords[0]
			zmax = centered_map_box.max_coords[2]
			zmin = centered_map_box.min_coords[2]
			self.zFocalPoint = zmax - xmax / N.tan(perspectiveAngle / 2. * N.pi / 180)
			self.perspectiveRatio = (self.zFocalPoint -zmin) / (self.zFocalPoint - zmax)
		else:
			self.perspectiveRatio = 1 
		if self.perspectiveRatio>0:
			self.absolutePerspectiveRatio = self.perspectiveRatio
		else:
			self.absolutePerspectiveRatio = -self.perspectiveRatio
	#}}}
	
	def get_3D_right_eye_cam(self, z_fixed_point=0.0, ang_deg=1.0):#{{{
		"""
		Get the 3D right eye camera for stereoscopic view, which is made from the original 
		camera with just one rotation around the up vector (angle ang_deg)

		Parameters
		----------

		ang_deg         : float
						angle between self and the returned camera (in degrees, default 1.0)

		z_fixed_point   : float
						position (along w axis) of the fixed point in the right eye rotation

		Returns
		-------

		right_eye_cam   : the right eye Camera object for 3D image processing
		"""
		assert ang_deg != 0.0
		cam_axis = self.get_camera_axis()
		vaxis = cam_axis[1,:]
		if z_fixed_point==0.0:
			R = rot3d_axvector(vaxis, ang_deg*N.pi/180.)
			new_los_axis = R.transform_points([self.los_axis])[0]
			new_center = self.center
		else:
			fp = self.center+self.los_axis*z_fixed_point
			R = rot3d_axvector(vaxis, ang_deg*N.pi/180., rot_center=fp)
			new_los_axis = R.transform_vectors([self.los_axis], [fp])[0]
			new_center = fp - z_fixed_point * new_los_axis
		right_eye_cam = Camera(center=new_center, line_of_sight_axis=new_los_axis, up_vector=vaxis,\
				region_size=self.region_size, distance=self.distance, far_cut_depth=self.far_cut_depth,\
				map_max_size=self.map_max_size, log_sensitive=self.log_sensitive, perspectiveAngle=self.perspectiveAngle)
		return right_eye_cam
	#}}}
	
	def get_required_resolution(self):#{{{
		"""
		Returns
		-------
		
		lev : ``int``
			the level of refinement up to which one needs to read the data to compute the projection
			of the region of interest with the specified resolution.

		"""
		lev = int(N.ceil(N.log2(self.map_max_size / max(self.region_size))))
		return lev
	#}}}

	def get_map_size(self):#{{{
		"""
		Returns
		-------
		(nx, ny) : (``int``, ``int``) ``tuple``
			the size (nx,ny) of the image taken by the camera (pixels)

		"""
		aspect_ratio = float(self.region_size[0]) /float(self.region_size[1])
		if aspect_ratio > 1.:
			nx_map = self.map_max_size
			ny_map = int(N.round(self.map_max_size / aspect_ratio))
		else:
			nx_map = int(N.round(self.map_max_size * aspect_ratio))
			ny_map = self.map_max_size

		return (nx_map, ny_map)
	#}}}

	def contains_camera(self, cam):#{{{
		"""
		Parameters
		----------
		An other camera object
		
		Returns
		----------
		Boolean : True if data needed for this camera view include all data
			needed for the camera view given in argument.

		"""
		return (self.get_required_resolution() >= cam.get_required_resolution()) and \
			self.get_bounding_box().contains(cam.get_bounding_box())
	#}}}

	def get_map_box(self, reduce_u_v_to_PerspectiveRatio=False):#{{{
		"""Returns the (0.,0.,0.) centered cubic bounding box of the area covered by the camera
		Parameters
		----------
		reduce_u_v_to_PerspectiveRatio	: ``boolean`` (default False)
				take into account the camera.perspectiveAngle if it is defined to make
				a perspective projection. This reduce the map u and v (i.e.horizontal and vertical)
				size with the perspective ratio.
		"""
		dx, dy = self.region_size
		dz = self.distance + self.far_cut_depth
		if reduce_u_v_to_PerspectiveRatio and self.perspectiveAngle != 0 :
			bound_min = N.array([-dx/2.*self.absolutePerspectiveRatio, -dy/2.*self.absolutePerspectiveRatio, -self.far_cut_depth])
			bound_max = N.array([ dx/2.*self.absolutePerspectiveRatio,  dy/2.*self.absolutePerspectiveRatio, self.distance])
		else:
			bound_min = N.array([-dx/2., -dy/2., -self.far_cut_depth])
			bound_max = N.array([ dx/2.,  dy/2., self.distance])
		box = Box([bound_min, bound_max])
		return box
	#}}}

	def rotate_around_up_vector(self, ang_deg=1.0):#{{{
		"""
		"""
		assert ang_deg!=0.0
		R = rot3d_axvector(self.up_vector, ang_deg*N.pi/180.)
		self.los_axis = R.transform_points(N.array([self.los_axis]))[0]
	#}}}

	def get_pixel_surface(self):#{{{
		""" Returns the surface of any pixel of the camera
		"""
		dx, dy = self.region_size
		nx, ny = self.get_map_size()
		S = 1.0 * dx/nx * dy/ny
		return S
	#}}}

	def get_pixels_coordinates_edges(self, take_into_account_perspective=False):#{{{
		"""Returns the edges value of the camera pixels x/y coordinates
		The pixel coordinates of the center of the camera is (0,0)
		"""
		dx, dy = self.region_size
		nx, ny = self.get_map_size()
		if take_into_account_perspective and self.perspectiveAngle != 0 :
			xedges = N.linspace(-dx/2.*self.absolutePerspectiveRatio, dx/2.*self.absolutePerspectiveRatio, num=nx+1)
			yedges = N.linspace(-dy/2.*self.absolutePerspectiveRatio, dy/2.*self.absolutePerspectiveRatio, num=ny+1)
		else:
			xedges = N.linspace(-dx/2., dx/2., num=nx+1)
			yedges = N.linspace(-dy/2., dy/2., num=ny+1)
		return (xedges, yedges)
	#}}}

	def viewing_angle_rotation(self):#{{{
		"""Returns the rotation corresponding to the viewing angle of the camera
		"""
		cam_axis = self.get_camera_axis()
		rot = LinearTransformation(cam_axis)
		return rot
	#}}}

	def viewing_angle_transformation(self):#{{{
		"""Returns the transformation corresponding to the viewing angle
		of the camera
		"""
		rot = self.viewing_angle_rotation()
		tr = translation(-self.center)
		Tr = rot*tr
		return Tr
	#}}}

	def get_camera_axis(self):#{{{
		"""Returns the camera u, v and z axis coordinates
		"""
		z_axis = self.los_axis
		u_axis = N.cross(self.up_vector,self.los_axis)
		u_axis = u_axis / N.linalg.norm(u_axis)
		v_axis = N.cross(z_axis, u_axis)
		cam_axis = N.array([u_axis, v_axis, z_axis])
		return cam_axis
	#}}}

	def get_region_size_level(self):#{{{
		""" Returns the level of the AMR grid for which 
		the cell size ~ the region size
		"""
		lev = int(N.round(N.log2(1./max(self.region_size))))
		return lev
	#}}}
	
	def get_slice_points(self, z=0.0):#{{{
		"""
		Returns the (x, y, z) coordinates of the points contained in a slice plane
		perpendicular to the line-of-sight axis at a given position z.

		z --- slice plane position along line-of-sight (default 0.0 => center of the region)
		"""
		u, v = self.get_pixels_coordinates_edges()
		uc = (u[1:]+u[:-1])/2.
		vc = (v[1:]+v[:-1])/2.
		points = meshgrid([uc, vc, [z]]) 
		return self.deproject_points(points)
	#}}}

	def get_bounding_box(self):#{{{
		"""Returns the bounding box of the region of interest in the simulation
		domain corresponding of the area covered by the camera
		"""
		b = self.get_map_box()
		box_bounds = N.array([b.min_coords, b.max_coords]).transpose()
		corner_list = meshgrid(box_bounds)
		xform_corners = self.deproject_points(corner_list)
		xform_min_bounds = N.min(xform_corners, axis=0)
		xform_max_bounds = N.max(xform_corners, axis=0)
		min_bounds = N.max([xform_min_bounds, N.zeros(3)], axis=0)
		max_bounds = N.min([xform_max_bounds, N.ones(3)],  axis=0)
		box = Box([min_bounds, max_bounds])
		return box
	#}}}

	def deproject_points(self, uvw_points, origins=None):#{{{
		"""Return xyz_coords deprojected coordinates of a set of points from given [u,v,w] coordinates :
		- (u=0,v=0, w=0) is the center of the camera.
		- v is the coordinate along the vaxis
		- w is the depth coordinate of the points along the	line-of-sight of the camera.
		if origins is True, perform a vectorial transformation of the vectors described by uvw_points 
		anchored at positions 'origins'
		"""
		xform = self.viewing_angle_transformation().inverse()
		if origins is None:
			coords_xyz = xform.transform_points(uvw_points)
		else:
			coords_xyz = xform.transform_vectors(uvw_points, origins)
		return coords_xyz
	#}}}
	
	def project_points(self, points, take_into_account_perspective=False):#{{{
		"""Return a (coords_uv, depth) tuple where 'coord_uv' is the projected coordinates of
		a set of points on the camera plane. (u=0,v=0) is the center of the camera plane.
		'depth' is the depth coordinate of the points along the line-of-sight of the camera.
		Parameters
		----------
		points				: ``numpy array of floats``
				array of points(x,y,z) coordinates to project
		take_into_account_perspective	: ``boolean`` (default False)
				take into account the camera.perspectiveAngle if it is defined to make a perspective projection.
		"""
		xform = self.viewing_angle_transformation()
		xform_pts = xform.transform_points(points)
		coords_uv = xform_pts[:,:2]
		depth = xform_pts[:,2]
		if take_into_account_perspective and self.perspectiveAngle != 0 :
			zmin = -self.far_cut_depth
			if self.perspectiveRatio > 0:
				coords_uv[:,0] = coords_uv[:,0] * ((self.zFocalPoint - zmin) / (self.zFocalPoint - depth))
				coords_uv[:,1] = coords_uv[:,1] * ((self.zFocalPoint - zmin) / (self.zFocalPoint - depth))
			else:
				coords_uv[:,0] = coords_uv[:,0] * ((zmin - self.zFocalPoint) / (self.zFocalPoint - depth))
				coords_uv[:,1] = coords_uv[:,1] * ((zmin - self.zFocalPoint) / (self.zFocalPoint - depth))
		return (coords_uv, depth)
	#}}}

	def get_map_mask(self, float32=True):#{{{
		"""
		Returns the mask map of the camera. each pixel has an alpha :
		* 1, if the ray of the pixel intersects the simulation domain
		* 0, if not
		Parameters
		----------
		float32		``Boolean`` (default True)
			use "float32" numpy dtype array instead of float64 to save memory
			(when float type is not needed, int8 type will be used anyway)
		"""
		# Camera map size
		nx_map, ny_map = self.get_map_size()

		# bounding box of the region of interest in the simulation
		# domain corresponding of the area coverd by the camera
		b = self.get_map_box()
		box_bounds = N.array([b.min_coords, b.max_coords]).transpose()
		corner_list = meshgrid(box_bounds)
		xform_corners = self.deproject_points(corner_list)

		# The region of interest is completely inside the simulation domain
		if ((N.min(xform_corners)>=0.)*(N.max(xform_corners)<=1.0)):
			return N.ones((nx_map, ny_map), "int8")

		# Ray-simulation domain optical depth computation
		mask = N.zeros(nx_map*ny_map)
		
		# Pixels edges/center coordinates
		xedges, yedges = self.get_pixels_coordinates_edges()
		xcen = (xedges[1:]+xedges[:-1])/2.
		ycen = (yedges[1:]+yedges[:-1])/2.
		zmin = b.min_coords[2]
		zmax = b.max_coords[2]
		depth_max= zmax - zmin

		ray_origins = N.zeros((nx_map, ny_map, 3))
		ray_origins[:,:,0] = xcen[:, N.newaxis]
		ray_origins[:,:,1] = ycen[N.newaxis, :]
		ray_origins[:,:,2] = zmin
		ray_origins = ray_origins.reshape(nx_map * ny_map, 3)
		ray_vectors = N.zeros((nx_map*ny_map,3))
		ray_vectors[:,2]=1.0


		# Origin points + axis of the rays (in AMR grid coordinates : x, y, z)
		xform = self.viewing_angle_transformation().inverse()
		ray_vectors = xform.transform_vectors(ray_vectors, ray_origins)
		ray_origins = xform.transform_points(ray_origins)

#		# Distance of the rays from the center of the simulation domain
#		alpha = N.sum((0.5 - ray_origins) * ray_vectors, axis=1)
#		r2 = N.sqrt(N.sum((ray_origins + alpha[:,N.newaxis] * ray_vectors - 0.5)**2, axis=1))

		# We look for intersections with the 6 faces of the 3D simulation domain
		intersect = N.zeros_like(mask).astype('i')
		alpha = N.zeros_like(mask)
		alpha2 = N.zeros_like(mask)
		fc = 0.5 * N.ones((3, 2, 3))
		axes = N.arange(3)
		for idim in range(3):
			# rays not parallel to the faces along dimension idim for which there might be an intersection.
			m = (ray_vectors[:,idim] != 0.0)
			if (m==False).all(): continue
			for ifsign in range(2):
				# Face sign
				sgn_face = ifsign * 2 -1
				# Face center coordinates
				fc[idim, ifsign, idim] += sgn_face / 2.0
				face_center = fc[idim, ifsign, :]
				# Coordinate parameter of the intersection of the ray with the face plane
				a = (face_center[N.newaxis, idim]-ray_origins[:, idim])/ray_vectors[:, idim]
				dm = (axes != idim)
				fcoord = N.abs(ray_origins[:, dm] \
						+ a[:,N.newaxis] * ray_vectors[:, dm] \
						- face_center[dm])
				# mask of the rays for which the intersection falls into the simulation domain face
				m2 = (fcoord<0.5).all(axis=1)
				if (m2==False).all(): continue

				a2 = a[m2]

				# Clip the alpha value to [0., depth_max]
				N.clip(a2, 0., depth_max, out=a2)
				alpha2[m2] = a[m2]
				im = (intersect==0)
				alpha[(m2*im)] = a[(m2*im)]
				intersect[m2] += 1

		absalph = N.abs(alpha-alpha2)
		mmax = N.max(absalph)
		if mmax > 0.:
			absalph = absalph/mmax
		am = (absalph>0.)
		mask[am] = absalph[am]
		mask = mask.reshape(nx_map, ny_map)
		if float32:
			mask = mask.astype("float32")
		return mask
	#}}}

	def get_rays(self):#{{{
		"""
		Returns ray_vectors, ray_origins and ray_lengths arrays for ray tracing ray definition
		"""
		# Camera view area
		centered_map_box = self.get_map_box()
	
		# Camera map size
		nx_map, ny_map = self.get_map_size()
		n_rays = nx_map * ny_map
		# Pixels edges coordinates
		xedges, yedges = self.get_pixels_coordinates_edges()
		xcen = (xedges[1:]+xedges[:-1])/2.
		ycen = (yedges[1:]+yedges[:-1])/2.
		zmin = centered_map_box.min_coords[2]
		zmax = centered_map_box.max_coords[2]
		ray_length_maps_max = zmax - zmin

		ray_origins = N.zeros((nx_map, ny_map, 3))
		ray_origins[:,:,0] = xcen[:, N.newaxis]
		ray_origins[:,:,1] = ycen[N.newaxis, :]
		ray_origins[:,:,2] = zmin
		ray_origins = ray_origins.reshape(n_rays, 3)
		ray_vectors = N.zeros((n_rays,3))

		if self.perspectiveAngle != 0:
			perspectiveRatio = (self.zFocalPoint -zmin) / (self.zFocalPoint - zmax)
			
			ray_vectors = ray_origins.copy()
			ray_vectors[:,2] = zmax - self.zFocalPoint
			# Normalize vectors
			norms = N.sqrt(N.sum(ray_vectors**2, axis=1))
			ray_vectors = ray_vectors / norms[:,N.newaxis]

			ray_origins[:,0] = ray_origins[:,0] * perspectiveRatio
			ray_origins[:,1] = ray_origins[:,1] * perspectiveRatio
			ray_origins[:,2] = zmin
		
		else:
			ray_vectors[:,2] = 1.0
			
		# Origin points + axis of the rays (in AMR grid coordinates : x, y, z)
		ray_origins = self.deproject_points(ray_origins)
		ray_vectors = self.deproject_points(ray_vectors, ray_origins)

		# Simulation domain limitation
		ray_ends = N.clip((ray_origins + ray_vectors * ray_length_maps_max), 0.0, 1.0)
		# ray_origins = N.clip(ray_origins, 0.0, 1.0) # Doesn't work nicely for rotated images
		# See ray_trace_octree in raytracing/ray_trace.pyx for ray origin projections inside the simulation box
		ray_lengths = N.sqrt(N.sum((ray_ends-ray_origins)**2, axis=1))
		
		return (ray_vectors, ray_origins, ray_lengths)

	#}}}

	def set_color_transfer_function(self, tf):
		assert isinstance(tf, ColorLinesTransferFunction)
		self.color_tf = tf

	def save_HDF5(self, h5f):#{{{
		"""
		Saves the camera parameters into a HDF5 file.
		"""
		h5fname = None
		if not isinstance(h5f, T.file.File):
			h5fname = h5f
			h5f = T.openFile(h5f, 'w')
		b = self.get_map_box()
		bound_uv_depth = N.array([b.min_coords, b.max_coords])
		#nx, ny = self.get_map_size()
		#map_size = N.array([nx, ny])
		#map_range = N.array([N.min(map), N.max(map)])
		h5f.createGroup("/", "camera")
		h5f.createArray("/camera", "center", self.center)
		h5f.createArray("/camera", "axis_coordinates", self.get_camera_axis())
		h5f.createArray("/camera", "bound_uv_depth", bound_uv_depth)
		h5f.createArray("/camera", "map_max_size", self.map_max_size)
		h5f.createArray("/camera", "log_sensitive", self.log_sensitive)
		h5f.createArray("/camera", "perspectiveAngle", self.perspectiveAngle)
		if h5fname != None:
			h5f.close()
	#}}}

	@classmethod
	def from_HDF5(cls, h5f):#{{{
		"""
		Returns a camera from a HDF5 file.
		"""
		h5fname = None
		if not isinstance(h5f, T.file.File):
			h5fname = h5f
			h5f = T.openFile(h5f, 'r')
		center = h5f.getNode("/camera/center").read()
		uaxis, vaxis, los = h5f.getNode("/camera/axis_coordinates").read()
		bounds = h5f.getNode("/camera/bound_uv_depth").read()
		fcd = -bounds[0,2]
		dist = bounds[1,2]
		region_size = N.diff(bounds, axis=0)[0,:-1]
		mms = h5f.getNode("/camera/map_max_size").read()
		is_log_sens = h5f.getNode("/camera/log_sensitive").read()
		try :
			perspectiveAngle = h5f.getNode("/camera/perspectiveAngle").read()
		except T.exceptions.NoSuchNodeError:
			perspectiveAngle = 0
		if h5fname != None:
			h5f.close()
		cam = cls(center=center, line_of_sight_axis=los, up_vector=vaxis,\
			  region_size=region_size, distance=dist, far_cut_depth=fcd,\
			  map_max_size=mms, log_sensitive=is_log_sens,\
			  perspectiveAngle=perspectiveAngle)
		return cam
	#}}}
	
	def save_csv(self, csv_file):#{{{
		"""
		Saves the camera parameters into a csv (Comma Separated Values) file.
		"""
		with open(csv_file, 'wb') as csvfile:
			spamwriter = csv.writer(csvfile, delimiter=',')
			spamwriter.writerow(['PyMSES camera parameter csv file (this ordering is still required)'])
			spamwriter.writerow(['center', list(self.center)])
			spamwriter.writerow(['line_of_sight_axis', list(self.los_axis)])
			spamwriter.writerow(['up_vector', list(self.up_vector)])
			spamwriter.writerow(['region_size', list(self.region_size)])
			spamwriter.writerow(['distance', self.distance])
			spamwriter.writerow(['far_cut_depth', self.far_cut_depth])
			spamwriter.writerow(['map_max_size', self.map_max_size])
			spamwriter.writerow(['log_sensitive', self.log_sensitive])
			spamwriter.writerow(['perspectiveAngle', self.perspectiveAngle])
	#}}}
	
	@classmethod
	def from_csv(cls, csv_file):#{{{
		"""
		Returns a camera from a csv (Comma Separated Values) file.
		"""
		with open(csv_file, 'rb') as csvfile:
			spamreader = csv.reader(csvfile, delimiter=',')
			next(spamreader), " intro comment ! "
			cam = cls(center=json.loads(spamreader.next()[1]), line_of_sight_axis=json.loads(spamreader.next()[1]), up_vector=json.loads(spamreader.next()[1]),\
			  region_size=json.loads(spamreader.next()[1]), distance=float(spamreader.next()[1]), far_cut_depth=float(spamreader.next()[1]),\
			  map_max_size=int(spamreader.next()[1]), log_sensitive=bool(spamreader.next()[1]),\
			  perspectiveAngle=float(spamreader.next()[1]))
		return cam
	
	def copy(self):#{{{
		"""
		Returns a copy of this camera.
		"""
		return Camera(center=self.center, line_of_sight_axis=self.los_axis,\
				up_vector=self.up_vector, region_size=self.region_size,\
				distance=self.distance, far_cut_depth=self.far_cut_depth,\
				map_max_size=self.map_max_size, log_sensitive=self.log_sensitive,\
				perspectiveAngle=self.perspectiveAngle)
	#}}}
	
	def printout(self):
		"""
		Print camera parameters in the console
		"""
		print("Camera parameters: center =", self.center,"line of sight =", self.los_axis\
			,"\nup_vector =", self.up_vector, "region_size=", self.region_size, "distance =", \
			self.distance, "\nfar_cut_depth =", self.far_cut_depth, "map_max_size =",\
			self.map_max_size,"log_sensitive =",self.log_sensitive,"perspectiveAngle =",self.perspectiveAngle)

#}}}

class ExtendedCamera(Camera):#{{{
	def __init__(self, camera, ext_box):
		from .fft_projection.convolution_kernels import ConvolKernel

		# Camera view area
		filter_box = camera.get_map_box()
		map_range = N.array([[filter_box.min_coords[0],filter_box.max_coords[0]],\
					 [filter_box.min_coords[1],filter_box.max_coords[1]], \
					 [filter_box.min_coords[2],filter_box.max_coords[2]]])
		far_c = -filter_box.min_coords[2]
		dist  =  filter_box.max_coords[2]
		rsize = N.diff(map_range, axis=1).transpose()[0]
		# Camera map size + pixel size
		nx_map, ny_map = camera.get_map_size()
		dxy = rsize[:2] / N.array([nx_map, ny_map])
		
		if isinstance(ext_box, ConvolKernel): 
			ext = 3./2. * ext_box.max_size
			extension = N.zeros((2,3))
			ext_axes = ext_box.get_convolved_axes()
			extension[0,ext_axes] = -ext
			extension[1,ext_axes] =  ext
		else: # ext_box is an array
			extension = ext_box
	
		# Depths limit
		far_c = far_c - extension[0, 2]
		dist  = dist  + extension[1, 2]

		# Map extension computation : extended map_box
		nplus = (N.ceil(N.abs(extension[:,:2]) / dxy[N.newaxis, :])).astype('i')

		# Unextended map limit coordinates
		ix_min, ix_max = nplus[0,0] + N.array([0, nx_map])
		iy_min, iy_max = nplus[0,1] + N.array([0, ny_map])

		# Extended max map size
		mms = N.max([nx_map + N.sum(nplus[:,0]), ny_map + N.sum(nplus[:,1])])

		nplus[0,:] = -nplus[0,:]
		new_rsize = rsize[:2] + dxy * N.diff(nplus, axis=0)[0]


		Camera.__init__(self, camera.center, camera.los_axis, camera.up_vector, \
			region_size=new_rsize, distance=dist, far_cut_depth=far_c, map_max_size=mms, \
			log_sensitive=camera.log_sensitive)

		self.ix_min = ix_min
		self.ix_max = ix_max
		self.iy_min = iy_min
		self.iy_max = iy_max


	def get_window_mask(self):
		"""Returns the unextended map mask of the camera
		"""
		return (self.ix_min, self.ix_max, self.iy_min, self.iy_max)
#}}}

class CameraFilter(RegionFilter):#{{{
	def __init__(self, source, camera, ext_size=0.0, ext_3D=True,
		     keep_cache_dset=False, use_camera_lvlmax=True):
		"""
		Filter build to fit the camera bounding box

		Parameters
		----------
		source         : pymses data source
					
		camera   : pymses camera
						position (along w axis) of the fixed point in the right eye rotation
		ext_size	: float (default 0.0)
			extension to the camera box (in box unit between 0 and 1), used only if ext_3D==True
		ext_3D :	boolean (default True)
			if true, use an ExtendedCamera to extend the filtered region
		keep_cache_dset : boolean (default False)
			flag to keep the cache_dset dictonary of the source
			during the filtering process
			(used for PointsDataSet cache with the amrviewer GUI)
		use_camera_lvlmax : ``boolean`` (default True)
			Limit the transformation of the AMR grid to particles
			to AMR cells under the camera octree levelmax (so that visible cells
			are only the ones that have bigger size than the camera pixel size).
			Set this to False when using directly particle data from ".part"
			particles files (dark matter and stars particles), so as to get
			the cache_dset working without the levelmax specification
		"""
		self.cam = camera
		self.cam_box = camera.get_map_box()
		self.ext_3D = ext_3D
		if ext_3D:
			# Extended camera
			r = N.ones((2,3)) * ext_size
			r[0,:] = -r[0,:]
			ecam = ExtendedCamera(camera, r)
		else:
			ecam = camera

		# Init Filter with extended camera bounding box region
		RegionFilter.__init__(self, ecam.get_bounding_box(), source)
		
		# Max. read level setup
		if use_camera_lvlmax:
			lreq = min(camera.get_required_resolution(), source.read_lmax)
			self.set_read_lmax(lreq)
		
		if keep_cache_dset:
			self.keep_cache_dset = keep_cache_dset
			self.cache_dset = source.cache_dset
			
	
	def filtered_dset(self, dset):
		xform = self.cam.viewing_angle_transformation()
		rot_points = xform.transform_points(dset.points)
		if self.ext_3D:
			mask = N.zeros(dset.npoints, dtype=bool)
			sizes = dset.get_sizes()
			unique_sizes = N.unique(sizes)
			if len(unique_sizes) > 4:
				# limit extension size as too big (neighbouring?) cells
				# are normally less interesting as the user zoom on more refined levels
				max_ext_size = unique_sizes[4]
			elif len(unique_sizes) > 0:
				max_ext_size = unique_sizes[-1]
			for size in unique_sizes:
				if size > max_ext_size:
					ext_size = max_ext_size
				else:
					ext_size = size
				minb = self.cam_box.min_coords - ext_size
				maxb = self.cam_box.max_coords + ext_size
				b = Box([minb, maxb])
				mask = mask + (b.contains(rot_points) * (size == sizes))
		else:
			mask = self.cam_box.contains(rot_points)
	
		return dset.filtered_by_mask(mask)
#}}}

__all__ = ["Camera", "CameraFilter", "ExtendedCamera"]

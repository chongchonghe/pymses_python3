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
try:
	import tables
except:
	print("WARNING : Can't import tables module...")
import numpy
import pylab
try:
	import Image
except:
	print("WARNING : Can't import Image module...")
import os
from matplotlib.colors import Colormap, LinearSegmentedColormap
from pymses.utils.constants import Unit
from .camera import Camera
import pymses



def save_map_HDF5(map, camera, unit=None, scale_unit=None, hdf5_path="./",
		  map_name="my_map", float32=True, save_map_mask=True):#{{{
	r"""
	Saves the map and the camera into a HDF5 file
	Parameters
	----------
	map 		``numpy array``
		map to save
	camera		``Pymses camera``
		camera associated with the map to save
	unit 		``pymses unit`` (default None)
		map unit
	scale_unit 		``float?`` (default None)
		scale unit
	hdf5_path 		``string`` (default "./")
		path of the hdf5 file to create
	map_name 		``string`` (default "my_map")
		name of the map
	float32		``Boolean`` (default True)
		use "float32" numpy dtype array instead of float64 to save memory
	save_map_mask	``Boolean`` (default True)
		save camera map_mask or not (which is used to set
		transparency when there is no intersection
		with the simulation domain)

	"""
	
	nx, ny = camera.get_map_size()
	map_size = numpy.array([nx, ny])
	map_range = numpy.array([numpy.min(map), numpy.max(map)])
	fname = os.path.join(hdf5_path, "%s.h5"%map_name)
	if float32:
		# use appropriate type to save disk space :
		map = map.astype("float32")
	print("Saving map into '%s' HDF5 file"%fname)
	try:
		h5f = tables.openFile(fname, mode='w')
	except:
		os.mkdir(hdf5_path)
		h5f = tables.openFile(fname, mode='w')
	h5f.createArray("/", "name", map_name)
	camera.save_HDF5(h5f)
	h5f.createGroup("/", "map")
	h5f.createArray("/map", "map", map)
	if save_map_mask:
		map_mask = camera.get_map_mask(float32=float32)
		h5f.createArray("/map", "map_mask", map_mask)
	# uncomment those folowing 3 lines in order to always
	# be able to open this new hdf5 file with old pymses software
	# (retro compatibility)
	#else:
	#	map_mask = numpy.ones(map.shape, "float32")
	#	h5f.createArray("/map", "map_mask", map_mask)
	h5f.createArray("/map", "map_size", map_size)
	h5f.createArray("/map", "map_range", map_range)
	if unit is not None:
		h5f.createGroup("/map", "unit")
		h5f.createArray("/map/unit", "dimensions", unit.dimensions)
		h5f.createArray("/map/unit", "val", unit.val)
	if scale_unit is not None:
		h5f.createGroup("/map", "length_unit")
		h5f.createArray("/map/length_unit", "dimensions", scale_unit.dimensions)
		h5f.createArray("/map/length_unit", "val", scale_unit.val)
	
	h5f.close()
	return fname
#}}}

def save_HDF5_to_img(h5fname, img_path=None, cmap="jet", cmap_range=None, fraction=None,
		     discrete=False, ramses_output=None, ran=None, adaptive_gaussian_blur=False, 
		     drawStarsParam=None, verbose=True, log_sensitive=None, alpha_map_mask=True):#{{{
	"""
	Function that plots, from an HDF5 file, the map into a Image and saves it into a PNG file

	Parameters
	----------
	h5fname      : ``string``
		the name of the HDF5 file containing the map
	img_path     : ``string`` (default value)
		the path in wich the img file is to be saved. the image is returned
		(and not saved) if left to None (default value)
	cmap         : ``string`` or ``Colormap`` object (default "jet")
		colormap to use
	cmap_range   : [`vmin`, `vmax`] ``array`` (default None)
		value range for map values clipping (linear scale)
	fraction     : ``float`` (default None)
		fraction of the total map values below the min. map range (in percent)
	discrete     : ``boolean`` (default False)
		whether the colormap must be integer values only or not.
	ramses_output : ``Ramses_output``
		specify ramses output for additional csv star file (look for a "sink_%iout.csv" file
		with 3D coordinates in output directory) to add stars on the image
	ran           : ``boolean`` or (float, float) (default None)
		specify map range value to fix colormap during a movie sequence
		(same as the "Colormap range" values printed in console when verbose=True)
	adaptive_gaussian_blur : ``boolean`` (default False)
		experimental : compute local image resolution and apply an adaptive gaussian blur to the image
		where it is needed (usefull to avoid AMR big pixels with ray tracing technique)
		For rotated view : give the levelmax map in this parameter to get the good local image resolution
	drawStarsParam	: ``DrawStarsParameters`` (default None)
		if ramses_output is specified and if a star file is found,
		this may be used to specify some parameters
	verbose       : ``boolean`` (default True)
		if True, print colormap range in console.
	log_sensitive : ``boolean`` (default None)
		apply logarithmic value, if not precise,
		this code use the hdf5 camera.log_sensitive value to decide
	alpha_map_mask : ``boolean`` (default True)
		use the camera map_mask for the alpha band : ray that doesn't
		intersect the simulation domain are see-through (alpha = 0)
	Returns
	-------
	img : PIL ``Image``
		if `img_path` is left to None
	ran = (vmin, vmax)
		if `img_path` is specified : return the Colormap range that
		can be used as a ran parameter for future images

	"""
	h5f = tables.openFile(h5fname, 'r')
	cam = Camera.from_HDF5(h5f)
	if log_sensitive == None:
		log_sensitive = cam.log_sensitive
	map = h5f.getNode("/map/map").read()
	if log_sensitive and pymses.__revision__ >= '$Revision: 705$':
		map = apply_log_scale(map)
	if isinstance(adaptive_gaussian_blur, numpy.ndarray):
		# Assume we got the levelmax map inside adaptive_gaussian_blur:
		levelMax = adaptive_gaussian_blur.max()
		pixel_size_map = (2**(levelMax - adaptive_gaussian_blur-1)).astype("i")
		from pymses.utils.point_utils import adaptive_gaussian_blur
		map = adaptive_gaussian_blur(map, pixel_size_map)
	elif adaptive_gaussian_blur:
		from pymses.utils.point_utils import adaptive_gaussian_blur
		from pymses.utils.point_utils import compute_same_value_pixel_size_map
		same_value_pixel_size_map = compute_same_value_pixel_size_map(map)
		map = adaptive_gaussian_blur(map, same_value_pixel_size_map)
		#map = compute_same_value_pixel_size_map(map)
		#print map.min(), map.max()
		#discrete = True
		#log_sensitive = False
	try :
		map_mask = h5f.getNode("/map/map_mask").read()
	except Exception:
		map_mask = numpy.ones(map.shape, "int8")
	nx, ny = map.shape
	map = map.transpose()
	map_mask = map_mask.transpose()
	if ran != None:
		vmin, vmax = ran
		vrange = [vmin, vmax]
		if log_sensitive:
			vmin, vmax = numpy.log10(vmin), numpy.log10(vmax)
	else :
		vmin, vmax = get_map_range(map[(map_mask>0.)], log_sensitive, cmap_range, fraction)
		if log_sensitive:
			vrange = [10.**vmin, 10.**vmax]
		else:
			vrange = [vmin, vmax]
	if verbose: print("Colormap range is : [%e, %e]"%(vrange[0], vrange[1]))
	map = numpy.clip(map, vmin, vmax)

	colormap, vmin, vmax = get_Colormap(cmap, discrete, (vmin, vmax))
	
	map = (map - vmin) / (vmax - vmin)
	map = numpy.asarray(colormap(map)*255, dtype='i')
	if ramses_output != None:
		if drawStarsParam is None :
			drawStarsParam = DrawStarsParameters()
		draw_stars(map, ramses_output, cam, drawStarsParam)
	map = map.reshape(nx*ny,4)
	R_band = Image.new("L",(nx,ny))
	R_band.putdata(map[:,0])
	G_band = Image.new("L",(nx,ny))
	G_band.putdata(map[:,1])
	B_band = Image.new("L",(nx,ny))
	B_band.putdata(map[:,2])
	if alpha_map_mask:
		map_mask = numpy.asarray(map_mask, dtype='f') * 255
		map_mask = numpy.asarray(map_mask, dtype='i')
		map_mask = map_mask.reshape(nx*ny)
		A_band = Image.new("L",(nx,ny))
		A_band.putdata(map_mask)
		out_img = Image.merge("RGBA", (R_band, G_band, B_band, A_band)).transpose(Image.FLIP_TOP_BOTTOM)
	else:
		out_img = Image.merge("RGB", (R_band, G_band, B_band)).transpose(Image.FLIP_TOP_BOTTOM)
	if img_path is None:
		h5f.close()
		return out_img
	else:
		fname = "%s.png"%h5f.getNode("/name").read()
		fname = os.path.join(img_path, fname)
		print("Saving img into '%s'"%fname)
		try:
			out_img.save(fname)
		except:
			os.mkdir(img_path)
			out_img.save(fname)
		h5f.close()
		return (vrange[0], vrange[1])
#}}}

def get_Colormap(cmap, discrete, ran):#{{{
	vmin, vmax = ran
	if isinstance(cmap, Colormap):
		colormap = cmap
	else:
		colormap = pylab.cm.get_cmap(cmap)
	if discrete:
		ncols = int(round(vmax - vmin)) + 1
		colormap = LinearSegmentedColormap('my_cmap', colormap._segmentdata, ncols)
		vmin = vmin - 0.5
		vmax = vmax + 0.5
	return (colormap, vmin, vmax)
#}}}

def save_HDF5_to_plot(h5fname, img_path=None, axis_unit=None, map_unit=None, cmap="jet", cmap_range=None,\
		fraction=None, save_into_png=True, discrete=False, verbose=True):#{{{
	r"""
	Function that plots the map with axis + colorbar from an HDF5 file

	Parameters
	----------
	h5fname      : the name of the HDF5 file containing the map
	img_path     : the path in wich the plot img file is to be saved
	axis_unit    : a (length_unit_label, axis_scale_factor) tuple containing :
		 * the label of the u/v axes unit
		 * the scaling factor of the u/v axes unit, or a Unit instance
	map_unit     : a (map_unit_label, map_scale_factor) tuple containing :
		 * the label of the map unit
		 * the scaling factor of the map unit, or a Unit instance
	cmap         : a Colormap object or any default python colormap string
	cmap_range   : a [vmin, vmax] array for map values clipping (linear scale)
	fraction     : fraction of the total map values below the min. map range (in percent)
	save_into_png: whether the plot is saved into an png file or not (default True)
	discrete     : wheter the map values are discrete integer values (default False). for colormap

	"""
	h5f = tables.openFile(h5fname, 'r')
	cam = Camera.from_HDF5(h5f)
	map = h5f.getNode("/map/map").read()
	if cam.log_sensitive and pymses.__revision__ >= '$Revision: 705$':
		map = apply_log_scale(map)
	map_unit_label = None
	if map_unit is not None:
		map_unit_label, map_scale_factor = map_unit
		if isinstance(map_scale_factor, Unit):
			try:
				d = h5f.getNode("/map/unit/dimensions").read()
				v = h5f.getNode("/map/unit/val").read()
			except:
				raise ValueError("map unit record not found in '%s' HDF5 file"%h5fname)
			mu = Unit(d, v)
			map_scale_factor = mu.express(map_scale_factor)

		if cam.log_sensitive:
			map = map + numpy.log10(map_scale_factor)
		else:
			map = map * map_scale_factor

	nx, ny = map.shape
	map = map.transpose()
	
	vmin, vmax = get_map_range(map, cam.log_sensitive, cmap_range, fraction)
	colormap, vmin, vmax = get_Colormap(cmap, discrete, (vmin, vmax))
	vrange = [vmin, vmax]
	if cam.log_sensitive:
		vrange = [10.**vmin, 10.**vmax]
	if verbose: print("Colormap range is : [%e, %e]"%(vrange[0], vrange[1]))
	map = numpy.clip(map, vmin, vmax)

	uedges, vedges = cam.get_pixels_coordinates_edges()
	uaxis, vaxis, zaxis = cam.get_camera_axis()
	ulabel = 'u'
	vlabel = 'v'
	ax_dict={'x':0, 'y':1, 'z':2}
	for axname, axis in list(Camera.special_axes.items()):
		if  (numpy.abs(uaxis) == numpy.abs(axis)).all():
			ulabel = axname
			uedges = uedges + cam.center[ax_dict[axname]]
			if (uaxis != axis).any():
				uedges = uedges[::-1]
			break
	for axname, axis in list(Camera.special_axes.items()):
		if  (numpy.abs(vaxis) == numpy.abs(axis)).all():
			vlabel = axname
			vedges = vedges + cam.center[ax_dict[axname]]
			if (vaxis != axis).any():
				vedges = vedges[::-1]
			break

	uvaxis_unit_label = 'box size unit'
	if axis_unit is not None:
		uvaxis_unit_label, axis_scale_factor = axis_unit
		if isinstance(axis_scale_factor, Unit):
			try:
				d = h5f.getNode("/map/length_unit/dimensions").read()
				v = h5f.getNode("/map/length_unit/val").read()
			except:
				raise ValueError("length_unit record not found in '%s' HDF5 file"%h5fname)
			au = Unit(d, v)
			axis_scale_factor = au.express(axis_scale_factor)

		uedges = uedges * axis_scale_factor
		vedges = vedges * axis_scale_factor

	fig = pylab.figure()
	pylab.pcolormesh(uedges, vedges, map, cmap=colormap, vmin=vmin, vmax=vmax)
	pylab.xlabel("%s (%s)"%(ulabel, uvaxis_unit_label))
	pylab.xlim(uedges[0],uedges[-1])
	pylab.ylabel("%s (%s)"%(vlabel, uvaxis_unit_label))
	pylab.ylim(vedges[0],vedges[-1])
	pylab.gca().set_aspect('equal')
	
	# Pretty user-defined colorbar
	if cam.log_sensitive:
		fo = pylab.matplotlib.ticker.FormatStrFormatter("$10^{%d}$")
		offset = numpy.ceil(vmin)-vmin
		lo = pylab.matplotlib.ticker.IndexLocator(1.0,offset)
		cb = pylab.colorbar(ticks=lo,format=fo)
	else:
		if discrete:
			fo = pylab.matplotlib.ticker.FormatStrFormatter("%d")
			ncol = int(round(vmax-vmin))
			ti = numpy.linspace(vmin+0.5, vmax-0.5, ncol)
			cb = pylab.colorbar(format=fo, ticks=ti)
		else:
			cb = pylab.colorbar()

	
	if map_unit_label is not None:
		cb.set_label(map_unit_label)

	if img_path is None:
		h5f.close()
		return fig
	else:
		if save_into_png:
			fname = "%s.png"%h5f.getNode("/name").read()
			fname = os.path.join(img_path, fname)
			print("Saving plot into '%s'"%fname)
			try:
				pylab.savefig(fname)
			except:
				os.mkdir(img_path)
				pylab.savefig(fname)
			h5f.close()
			return
#}}}
	
def save_HDF5_seq_to_img(h5f_iter, *args, **kwargs):#{{{
	"""
	fraction : fraction (percent) of the total value of the map above the returned vmin value
			   (default 1 %)
	"""
	one_is_enough = False

	if "fraction" in list(kwargs.keys()):
		frac = kwargs["fraction"]
	else:
		frac = None

	if "cmap_range" in list(kwargs.keys()):
		cmap_range = kwargs["cmap_range"]
		one_is_enough = True
	else:
		cmap_range = None

	if "log_sensitive" in list(kwargs.keys()):
		log_sensitive = kwargs["log_sensitive"]
	else:
		log_sensitive = None
		
	vrange_seq = []
	for file in h5f_iter():
		h5f = tables.openFile(file, 'r')
		map = h5f.getNode("/map/map").read()
		try :
			map_mask = h5f.getNode("/map/map_mask").read()
		except Exception:
			map_mask = numpy.ones(map.shape, "int8")
		if log_sensitive==None:
			is_logscale = h5f.getNode("/camera/log_sensitive").read()
		else :
			is_logscale = log_sensitive
		h5f.close()
		if is_logscale and pymses.__revision__ >= '$Revision: 705$':
			map = apply_log_scale(map)
		vrange = get_map_range(map[(map_mask>0.)], is_logscale, cmap_range, frac)
		vrange_seq.append(vrange)
		if one_is_enough:
			break

	vrange_seq = numpy.array(vrange_seq)
	vrange = numpy.array([numpy.min(vrange_seq[:,0]), numpy.max(vrange_seq[:,1])])
	if is_logscale:
		vrange = 10.**vrange
	
	kwargs["cmap_range"] = vrange
	if "verbose" in list(kwargs.keys()):
		verbose = kwargs["verbose"]
	else:
		verbose = True
	if verbose:
		print("Colormap range is : [%e, %e]"%(vrange[0], vrange[1]))
		kwargs["verbose"] = False

	imglist=[]
	for h5fname in h5f_iter():
		imglist.append(save_HDF5_to_img(h5fname, *args, **kwargs))
	def img_iter():
		for img in imglist:
			yield img

	return img_iter
#}}}

def get_map_range(map, log_sensitive=False, cmap_range=None, fraction=None):#{{{
	"""
	Map range computation function. Computes the linear/log
	(according to the map values scaling) scale map range values
	of a given map :

	 * if a user-defined cmap_range is given, then it is used to compute the map range values
	 * if not, the map range values is computed from a fraction (percent) of the total value
	   of the map parameter. the min. map range value is defined as the value below which there
	   is a fraction of the map (default 1 %)


	Parameters
	----------
	map            : 2D map from wich the map range values are computed
	log_sensitive  : whether the map values are log-scaled or not (True or False)
	cmap_range     : user-defined map range values (linear scale)
	fraction       : fraction of the total map values below the min. map range (in percent)

	Returns
	-------
	map_range : [``float``, ``float``]
		the map range values [ min, max]

	"""
	if cmap_range is not None:
		vmin, vmax = cmap_range
		if log_sensitive:
			vmin = numpy.log10(vmin)
			vmax = numpy.log10(vmax)
		map_range = [vmin, vmax]
	else:
		map = map.copy() # bug fix with old numpy
		values = map.ravel()
		if fraction is not None:
			frac = fraction / 100.0
		else:
			frac = 0.01
		values.sort()
		if log_sensitive:
			val = 10.**values
			cumval = numpy.cumsum(val)
			cumval = cumval / cumval[-1]
			mask = (cumval >= frac)
			fvalues = values[mask]
			map_range = [fvalues[0], values[-1]]
		else:
			map_range = [values[0], values[-1]]
	
	return map_range
#}}}

def apply_log_scale(map, verbose=True):
	"""
	Used to apply log-scale map if the camera captors are log-sensitive.
	Takes care of null and negative values warning
	
	Parameters
	----------
	map            : original numpy array of map values

	Returns
	-------
	map : ~ numpy.log10(map) (takes care of null and negative values)

	"""
	# Apply log-scale map if the camera captors are log-sensitive
	not_null_mask = map > 0.0
	null_values = sum(sum(-not_null_mask))
	if not_null_mask.any():# test that there are positive values
		min_map = numpy.min(map[not_null_mask])
		if null_values > 0 and verbose:
			print("Warning:",null_values,"values <= 0 were replaced by min_map/10 =",\
				    min_map/10, "values to apply log scale") 
		numpy.clip(map, min_map/10, map.max(), out=map)
		map = numpy.log10(map)
	elif verbose:
		print("Warning: log scale not applied because all values are <= 0")
	return map


def draw_stars(map, ramses_output, cam, param=None):
	if ramses_output.sink == None: # read the file only once
		from  pymses.sources.ramses.filename_utils import sink_filename
		try:
			star_file = sink_filename(ramses_output.output_repos, ramses_output.iout)
			import csv
			csv_reader = csv.reader(open(star_file), delimiter=',')
			sink = []
			for line in csv_reader:
				sink.append([float(line[1]),float(line[2]),float(line[3]),float(line[4])])
			# sink.append([1, 0.1,0.1,0.1])
			sink = numpy.array(sink)
			# Normalize and log weigth
			sink[:,0] = numpy.log(sink[:,0])
			sink[:,0] = (sink[:,0] - sink[:,0].min())/(sink[:,0].max() - sink[:,0].min())
			ramses_output.sink = sink
		except Exception:
			return map
	
	# We need to filtered stars like dsets are filtered in mp_maps with a CameraFilter
	xform = cam.viewing_angle_transformation()
	rot_points = xform.transform_points(ramses_output.sink[:,1:4])
	cam_box = cam.get_map_box()
	mask = cam_box.contains(rot_points)
	pts, w = cam.project_points(ramses_output.sink[:,1:4][mask], take_into_account_perspective=True)
	
	if param is None :
		param = DrawStarsParameters()
	if param.adapt_intensity:
		intensity = ramses_output.sink[:,0][mask]
	else:
		intensity = numpy.ones(len(ramses_output.sink[:,0][mask]))
	if param.RT_instensity_dimming:
		try:
			from . import FractionOperator
			from .raytracing.ray_trace import ray_trace_amr 
			from time import time
			begin_time = time()
	#			ramses_output.verbose = False
			ray_origins = numpy.zeros((sink.shape[0],3))
			#ramses_output.sink[:,1:4] +  * cam.los_axis
			ray_origins[:] = cam.los_axis
			w = -w
			w += cam.distance
			ray_origins[:,0] *= -w
			ray_origins[:,1] *= -w
			ray_origins[:,2] *= -w
			ray_origins += ramses_output.sink[:,1:4][mask]
			ray_vectors = numpy.zeros((sink.shape[0],3))
			ray_vectors[:] =  cam.los_axis
			# AMR DataSource preparation
			rlev = cam.get_required_resolution()
			source = ramses_output.amr_source(["rho"])
			num_func = lambda dset: (dset["rho"]**2)
			denom_func = lambda dset: (dset["rho"])
			op = FractionOperator(num_func, denom_func)
			maps = numpy.zeros((sink.shape[0], op.nscal_func()), dtype='d')
			ray_length_maps = numpy.zeros((sink.shape[0], op.nscal_func()), dtype='d')
			for icpu in source._data_list:
				dset = source.get_domain_dset(icpu)
				active_mask = dset.get_active_mask()
				g_levels =  dset.get_grid_levels()
				# We do the processing only if needed, i.e. only if the amr
				# level min of active cells in the dset is <= rlev
				if len(g_levels[active_mask]) > 0 and numpy.min(g_levels[active_mask]) <= rlev:
					mapsDset,ray_length_mapsDset = ray_trace_amr(dset, ray_origins, ray_vectors,\
							0., op, ramses_output.info, rlev, active_mask, g_levels, w)
					ray_length_maps += ray_length_mapsDset
					if op.is_max_alos():
						maps = numpy.max(numpy.array([maps, mapsDset]), axis=0)
					else:
						maps += mapsDset
			print("Sink ray trace process time=", time() - begin_time)
	#			ramses_output.verbose = True
			ifunc=0
			map_dict = {}
			for key, func in op.iter_scalar_func():
				map_dict[key] = maps[:,ifunc]
				ifunc+=1
			dimming_map = op.operation(map_dict)

			min_map = numpy.min(dimming_map[(dimming_map > 0.0)])
			numpy.clip(dimming_map, min_map, dimming_map.max(), out=dimming_map)
			dimming_map = numpy.log10(dimming_map)
			dimming = (dimming_map - dimming_map.min())/(dimming_map.max() - dimming_map.min())
			intensity = intensity * dimming
		except Exception as e:
			print("Warning : Ray trace sink instensity dimming failed, error:", e)
	map_box = cam.get_map_box(reduce_u_v_to_PerspectiveRatio=True)
	xmin = map_box.min_coords[0]
	xmax = map_box.max_coords[0]
	ymin = map_box.min_coords[1]
	ymax = map_box.max_coords[1]
	nx_map, ny_map = cam.get_map_size()

	stars = []
	for pt in pts:
		stars.append([int((pt[0]-xmin)/(xmax-xmin)*nx_map), int((pt[1]-ymin)/(ymax-ymin)*ny_map)])
	
	for i, star in enumerate(stars):
		drawCross(map,star[0],star[1],intensity[i],param.rgb,param.PSF)
	return star, intensity

def drawCross(map, x, y, intensity, rgb, PSF):
#import Image, numpy
#starImage = Image.open("/home/itp/mlabaden/Documents/star_center.png")
#crossPixel = []
#for i in range(23):
#	for j in range(23):
#		 (nul,nul,nul,alpha) = starImage.getpixel((i,j))
#		 if alpha == 255 :
#		 	crossPixel.append((i-11,j-11)) # starImage.size = (23, 23)
#crossPixel = numpy.array(crossPixel)
#mask = crossPixel > 0  # 1/4 image using symetric property
#mask = mask[:,0] * mask[:,1]
#crossPixel[mask]
	#crossPixel = [[0, 0], [0, 1], [0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [1, 0], [1, 1],\
	#	[1, 2], [1, 6], [2, 0], [2, 1], [2, 6], [3, 0], [3, 5], [4, 0], [4, 4], [5, 0],\
	#	[5, 3], [6, 0], [6, 1], [6, 2]]
	firstCirclePixel = [[1, 6], [2, 6], [3, 5], [4, 4], [5, 3], [6, 1], [6, 2]]
	secondCirclePixel = [[ 1, 10], [ 2, 10], [ 3, 10], [ 4,  9], [ 5,  9], [ 6,  8],\
		[ 7,  7], [ 8,  6], [ 9,  4], [ 9,  5], [10,  1], [10,  2], [10,  3]]
	drawPixel(map, x, y, intensity * .5 + .5, rgb) # always draw the center pixel with double intensity
	drawPixel(map, x + 1, y + 1, intensity, rgb)
	drawPixel(map, x - 1, y + 1, intensity, rgb)
	drawPixel(map, x + 1, y - 1, intensity, rgb)
	drawPixel(map, x - 1, y - 1, intensity, rgb)
	for i in range(2):
		drawPixel(map, x + i, y, intensity, rgb)
		drawPixel(map, x - i, y, intensity, rgb)
		drawPixel(map, x, y + i, intensity, rgb)
		drawPixel(map, x, y - i, intensity, rgb)
	if PSF and intensity > .8:
		# We draw a simili PSF if intensity is beyond a threshold (staturated pixel)
		#i=6 
		for i in range(2,11):
			drawPixel(map, x + i, y, intensity * (11-i)/9., rgb)
			drawPixel(map, x - i, y, intensity * (11-i)/9., rgb)
			drawPixel(map, x, y + i, intensity * (11-i)/9., rgb)
			drawPixel(map, x, y - i, intensity * (11-i)/9., rgb)
		for (pixel_x,pixel_y) in firstCirclePixel:
			drawPixel(map, x + pixel_x, y + pixel_y, intensity * 6/9., rgb)
			drawPixel(map, x - pixel_x, y + pixel_y, intensity * 6/9., rgb)
			drawPixel(map, x + pixel_x, y - pixel_y, intensity * 6/9., rgb)
			drawPixel(map, x - pixel_x, y - pixel_y, intensity * 6/9., rgb)
		for (pixel_x,pixel_y) in secondCirclePixel:
			drawPixel(map, x + pixel_x, y + pixel_y, intensity * 1/9., rgb)
			drawPixel(map, x - pixel_x, y + pixel_y, intensity * 1/9., rgb)
			drawPixel(map, x + pixel_x, y - pixel_y, intensity * 1/9., rgb)
			drawPixel(map, x - pixel_x, y - pixel_y, intensity * 1/9., rgb)

def drawPixel(map, x, y, intensity, rgb):
	if y < map.shape[0] and y >= 0 and x < map.shape[1] and x >= 0:
		map[y, x, 0] += (rgb[0] - map[y, x, 0]) *  intensity
		map[y, x, 1] += (rgb[1] - map[y, x, 1]) *  intensity
		map[y, x, 2] += (rgb[2] - map[y, x, 2]) *  intensity

class DrawStarsParameters():
	r"""
	Utility class to store parameters for the draw_stars function
	
	Parameters
	----------
	adapt_intensity      : ``boolean``
		Whether to adpat the intensity or not with the sink csv 4th colum
	rgb     :  [R, G, B] ``list`` of int
		list of 3 integers between 0 and 255 corresponding to a RGB color 
	PSF         : ``boolean`` or ``Colormap`` object
		colormap to use
	RT_instensity_dimming  : ``boolean``
		experimental : this option add a ray tracing
		pass on data to compute star intensity dimming
	
	"""
	def __init__(self, adapt_intensity=True, rgb=[255,255,255],
		     PSF=True, RT_instensity_dimming=False):
		self.adapt_intensity = adapt_intensity
		self.rgb = rgb
		self.PSF = PSF
		self.RT_instensity_dimming = RT_instensity_dimming

__all__ = ["save_map_HDF5",
	   "save_HDF5_to_plot",
	   "save_HDF5_to_img",
	   "save_HDF5_seq_to_img",
	   "get_Colormap",
	   "get_map_range",
	   "apply_log_scale",
	   "DrawStarsParameters",
	   "draw_stars"]

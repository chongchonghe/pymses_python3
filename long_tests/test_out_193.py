import numpy
import tables
from pymses import RamsesOutput
from pymses.utils.regions import Cylinder, Sphere
from pymses.filters import RegionFilter, PointFunctionFilter
from pymses.analysis import sample_points, bin_cylindrical, bin_spherical, amr2cube
from pymses.utils import constants as C
from pymses.analysis.visualization import *


def test_cyl_profile():
	# Galactic cylinder parameters
	gal_center = [ 0.567811, 0.586055, 0.559156 ]        # in box units
	gal_radius = 0.00024132905460547268                  # in box units
	gal_thickn = 0.00010238202316595811                  # in box units
	gal_normal = [ -0.172935, 0.977948, -0.117099 ]      # Norm = 1
	
	# RamsesOutput
	ro = RamsesOutput("/data/Aquarius/output", 193)
	
	# Prepare to read the density field only
	source = ro.amr_source(["rho"])
	
	# Cylinder region
	cyl = Cylinder(gal_center, gal_normal, gal_radius, gal_thickn)
	
	# AMR density field point sampling
	numpy.random.seed(1652336)
	points = cyl.random_points(1.0e6) # 1M sampling points
	point_dset = sample_points(source, points)
	rho_weight_func = lambda dset: dset["rho"]
	r_bins = numpy.linspace(0.0, gal_radius, 200)
	
	# Profile computation
	rho_profile = bin_cylindrical(point_dset, gal_center, gal_normal,
			rho_weight_func, r_bins, divide_by_counts=True)
		
	# Plot
	# Geometrical midpoint of the bins
	length = ro.info["unit_length"].express(C.kpc)
	bins_centers = (r_bins[1:]+r_bins[:-1])/2. * length
	dens = ro.info["unit_density"].express(C.H_cc)
	
	#h5f = tables.openFile("./long_tests/cyl_profile.h5", mode='w')
	#h5f.createArray("/", "cyl_profile", rho_profile)	
	#h5f.close()
	
	h5f = tables.openFile("./long_tests/cyl_profile.h5", mode='r')
	rho_profileB = h5f.getNode("/cyl_profile").read()	
	h5f.close()
	
	print(rho_profile)
	assert sum(rho_profile - rho_profileB) < 10e-6
	
def test_sph_profile():
	# Halo parameters
	halo_center = [ 0.567811, 0.586055, 0.559156 ]        # in box units
	halo_radius = 0.00075                                  # in box units
	
	# RamsesOutput
	ro = RamsesOutput("/data/Aquarius/output", 193)
	
	# Prepare to read the mass/epoch fields only
	source = ro.particle_source(["mass", "epoch"])
	
	# Sphere region
	sph = Sphere(halo_center, halo_radius)
	
	# Filtering particles
	point_dset = RegionFilter(sph, source)
	dm_filter = lambda dset: dset["epoch"] == 0.0
	dm_parts = PointFunctionFilter(dm_filter, point_dset)
	
	# Profile computation
	m_weight_func = lambda dset: dset["mass"]
	r_bins = numpy.linspace(0.0, halo_radius, 200)
	
	# Mass profile
	# This triggers the actual reading of the particle data files from disk.
	mass_profile = bin_spherical(dm_parts, halo_center, m_weight_func, r_bins,
				     divide_by_counts=False)
	
	# Density profile
	sph_vol = 4.0/3.0 * numpy.pi * r_bins**3
	shell_vol = numpy.diff(sph_vol)
	rho_profile = mass_profile / shell_vol
	
	# Plot
	# Geometrical midpoint of the bins
	length = ro.info["unit_length"].express(C.kpc)
	bins_centers = (r_bins[1:]+r_bins[:-1])/2. * length
	dens = ro.info["unit_density"].express(C.Msun/C.kpc**3)
	
	#h5f = tables.openFile("./long_tests/sph_profile.h5", mode='w')
	#h5f.createArray("/", "sph_profile", rho_profile)	
	#h5f.close()
	
	h5f = tables.openFile("./long_tests/sph_profile.h5", mode='r')
	rho_profileB = h5f.getNode("/sph_profile").read()	
	h5f.close()
	
	#print rho_profile
	assert (rho_profile - rho_profileB).all() < 10e-6


def test_slice_density():
	# RamsesOutput
	ro = RamsesOutput("/data/Aquarius/output/", 193)
	
	# AMR data source
	amr = ro.amr_source(["rho"])
	
	# Defining a Camera object
	cam  = Camera(center=[0.5, 0.5, 0.5], line_of_sight_axis='z',
		      region_size=[1., 1.], up_vector='y', map_max_size=100,
		      log_sensitive=True)
	
	# Density field access operator
	rho_op = ScalarOperator(lambda dset: dset["rho"])
	
	# Slice map computation
	map = SliceMap(amr, cam, rho_op, z=0.4)
	# create a density slice map at z=0.4 depth position
	
	map = apply_log_scale(map)
	
	#h5f = tables.openFile("./long_tests/slice_density.h5", mode='w')
	#h5f.createArray("/", "map", map)	
	#h5f.close()
	
	h5f = tables.openFile("./long_tests/slice_density.h5", mode='r')
	mapB = h5f.getNode("/map").read()	
	h5f.close()
	
	#print map
	assert (map - mapB).all() < 10e-6
	
def test_rt_Tmin():
	# Ramses data
	ioutput = 193
	ro = RamsesOutput("/data/Aquarius/output/", ioutput)
	
	# Map operator : minimum temperature along line-of-sight
	class MyTempOperator(Operator):
		def __init__(self):
			def invT_func(dset):
				P = dset["P"]
				rho = dset["rho"]
				r = rho/P
	#			print r[(rho<=0.0)+(P<=0.0)]
	#			r[(rho<=0.0)*(P<=0.0)] = 0.0
				return r
			d = {"invTemp": invT_func}
			Operator.__init__(self, d, is_max_alos=True)
	
		def operation(self, int_dict):
				map = list(int_dict.values())[0]
				mask = (map == 0.0)
				mask2 = map != 0.0
				map[mask2] = 1.0 / map[mask2]
				map[mask] = 0.0
				return map
	scal_op = MyTempOperator()
	
	# Map region
	center = [ 0.567111, 0.586555, 0.559156 ]
	axes = {"los": "z"}
	
	# Map processing
	rt = raytracing.RayTracer(ro, ["rho", "P"])
	
	axname = "los"
	axis = "z"
	cam  = Camera(center=center, line_of_sight_axis=axis, up_vector="y", region_size=[3.0E-3, 3.0E-3], \
			distance=1.5E-3, far_cut_depth=1.5E-3, map_max_size=100)
	map = rt.process(scal_op, cam)
	
	#h5f = tables.openFile("./long_tests/rt_min.h5", mode='w')
	#h5f.createArray("/", "map", map)	
	#h5f.close()
	
	h5f = tables.openFile("./long_tests/rt_min.h5", mode='r')
	mapB = h5f.getNode("/map").read()	
	h5f.close()
	
	#print map
	assert (map - mapB).all() < 10e-6

def test_rt_rho():
	# Ramses data
	ioutput = 193
	ro = RamsesOutput("/data/Aquarius/output/", ioutput)
	
	# Map operator : mass-weighted density map
	up_func = lambda dset: (dset["rho"]**2)
	down_func = lambda dset: (dset["rho"])
	scal_op = FractionOperator(up_func, down_func)
	
	# Map region
	center = [ 0.567811, 0.586055, 0.559156 ]
	# Map processing
	rt = raytracing.RayTracer(ro, ["rho"])
	axname = "los"
	axis = [ -0.172935, 0.977948, -0.117099 ]
	cam  = Camera(center=center, line_of_sight_axis=axis, up_vector="z", region_size=[3.0E-2, 3.0E-2], \
			distance=2.0E-2, far_cut_depth=2.0E-2, map_max_size=100)
	map = rt.process(scal_op, cam)
	
	#h5f = tables.openFile("./long_tests/rt_rho.h5", mode='w')
	#h5f.createArray("/", "map", map)	
	#h5f.close()
	
	h5f = tables.openFile("./long_tests/rt_rho.h5", mode='r')
	mapB = h5f.getNode("/map").read()	
	h5f.close()
	
	#print map
	assert (map - mapB).all() < 10e-6

def test_rt_lmax():
	# Ramses data
	ioutput = 193
	ro = RamsesOutput("/data/Aquarius/output/", ioutput)
	
	# Map operator : max. AMR level of refinement along the line-of-sight
	scal_op = MaxLevelOperator()
	
	# Map region
	center = [ 0.567811, 0.586055, 0.559156 ]
	# Map processing
	rt = raytracing.RayTracer(ro, ["rho"])
	axname = "los"
	axis = [ -0.172935, 0.977948, -0.117099 ]
	cam  = Camera(center=center, line_of_sight_axis=axis, up_vector="z", region_size=[3.0E-2, 3.0E-2], \
			distance=2.0E-2, far_cut_depth=2.0E-2, map_max_size=100)
	map = rt.process(scal_op, cam)
	
	#h5f = tables.openFile("./long_tests/rt_lmax.h5", mode='w')
	#h5f.createArray("/", "map", map)	
	#h5f.close()
	
	h5f = tables.openFile("./long_tests/rt_lmax.h5", mode='r')
	mapB = h5f.getNode("/map").read()	
	h5f.close()
	
	#print map
	assert (map - mapB).all() < 10e-6

def test_fft_part():
	# Ramses data
	ioutput = 193
	ro = RamsesOutput("/data/Aquarius/output/", ioutput)
	parts = ro.particle_source(["mass", "level"])
	
	# Map operator : mass
	scal_func = ScalarOperator(lambda dset: dset["mass"])
	
	# Map region
	center = [ 0.567811, 0.586055, 0.559156 ]
	
	# Map processing
	mp = fft_projection.MapFFTProcessor(parts, ro.info)
	
	axname = "los"
	axis = [ -0.172935, 0.977948, -0.117099 ]
	
	cam  = Camera(center=center, line_of_sight_axis=axis, up_vector="z", region_size=[5.0E-1, 4.5E-1], \
			distance=2.0E-1, far_cut_depth=2.0E-1, map_max_size=100)
	map = mp.process(scal_func, cam, surf_qty=True)
	
	#h5f = tables.openFile("./long_tests/fft_part.h5", mode='w')
	#h5f.createArray("/", "map", map)	
	#h5f.close()
	
	h5f = tables.openFile("./long_tests/fft_part.h5", mode='r')
	mapB = h5f.getNode("/map").read()	
	h5f.close()
	
	#print map
	assert (map - mapB).all() < 10e-6
	
def test_fft_amr():
	# Ramses data
	ioutput = 193
	ro = RamsesOutput("/data/Aquarius/output/", ioutput)
	amr = ro.amr_source(["rho"])

	# Map operator : mass-weighted density map
	up_func = lambda dset: (dset["rho"]**2 * dset.get_sizes()**3)
	down_func = lambda dset: (dset["rho"] * dset.get_sizes()**3)
	scal_func = FractionOperator(up_func, down_func)
	
	# Map region
	center = [ 0.567811, 0.586055, 0.559156 ]
	
	# Map processing
	mp = fft_projection.MapFFTProcessor(amr, ro.info)
	
	axname = "los"
	axis = [ -0.172935, 0.977948, -0.117099 ]
	
	cam  = Camera(center=center, line_of_sight_axis=axis, up_vector="z", region_size=[5.0E-1, 4.5E-1], \
			distance=2.0E-1, far_cut_depth=2.0E-1, map_max_size=100)
	map = mp.process(scal_func, cam, surf_qty=True)
	
	#h5f = tables.openFile("./long_tests/fft_amr.h5", mode='w')
	#h5f.createArray("/", "map", map)	
	#h5f.close()
	
	h5f = tables.openFile("./long_tests/fft_amr.h5", mode='r')
	mapB = h5f.getNode("/map").read()	
	h5f.close()
	
	#print map
	assert (map - mapB).all() < 10e-6

def test_amr2cube_cartesian_rt():
	# backuped referenced data :
	cubeB = numpy.array([[[ 0.08186216, 0.03206611, 0.05075889, 0.21691999, 0.08395087]
		, [ 0.11306132, 0.05707518, 0.03212786, 0.33522855, 0.13602483]
		, [ 0.05632005, 0.04249893, 0.09034388, 0.1588526,  0.2080538 ]
		, [ 0.06852026, 0.03090859, 0.04515618, 0.17721034, 0.23618275]
		, [ 0.07367933, 0.11752801, 0.03341876, 0.2254721,  0.39768221]]
		,
		 [[ 0.05193184, 0.03975904, 0.05116311, 0.14412791, 0.21599674]
		, [ 0.08652996, 0.07561322, 0.06104241, 0.09306177, 0.06347754]
		, [ 0.03311921, 0.03681144, 0.04637034, 0.10137452, 0.02770789]
		, [ 0.06000543, 0.0630374,  0.07050136, 0.08983026, 0.06849894]
		, [ 0.11372988, 0.17594275, 0.09902632, 0.07525623, 0.10831208]]
		,
		 [[ 0.10366187, 0.12453692, 0.18768945, 0.16773106, 0.11033484]
		, [ 0.1043405,  0.13150752, 0.06345165, 0.02748383, 0.05059519]
		, [ 0.06553906, 0.07011834, 0.06567988, 0.04461206, 0.0677126 ]
		, [ 0.19513314, 0.10922475, 0.04116418, 0.09346056, 0.07560843]
		, [ 0.1156556,  0.11896267, 0.42504693, 0.14856789, 0.08862464]]
		,
		 [[ 0.0683512,  0.10387651, 0.08278252, 0.05064939, 0.11547664]
		, [ 0.08297042, 0.08370349, 0.03658147, 0.02292432, 0.05400176]
		, [ 0.14583478, 0.10265131, 0.10521284, 0.03336809, 0.05561544]
		, [ 0.10904973, 0.05832567, 0.04876777, 0.10397124, 0.14783567]
		, [ 0.21239959, 0.06806608, 0.28383805, 0.26950828, 0.46882107]]
		,
		 [[ 0.39340254, 0.21568631, 0.11401807, 0.09364021, 0.11969451]
		, [ 0.17342993, 0.0847922,  0.02964273, 0.07289768, 0.2268799 ]
		, [ 0.06048485, 0.09604769, 0.06068367, 0.06307055, 0.37759786]
		, [ 0.09158091, 0.05048944, 0.03639829, 0.25110933, 0.32988721]
		, [ 0.3167426,  0.17177987, 0.23479188, 0.42511884, 0.10431827]]])
	
	mapB = numpy.array([ 0.46555802,  0.67351774,  0.55606925,  0.55797812,  0.84778041,
		0.50297864,  0.37972491,  0.24538341,  0.35187338,  0.57226726,
		0.69395413,  0.37737869,  0.31366193,  0.51459106,  0.89685772,
		0.42113625,  0.28018146,  0.44268245,  0.46795008,  1.30263307,
		0.93644163,  0.58764244,  0.65788461,  0.75946518,  1.25275146])
	
	# Ramses data
	ioutput = 193
	ro = RamsesOutput("/data/Aquarius/output/", ioutput)
	cam = Camera(center=[ 0.5, 0.5, 0.5 ], line_of_sight_axis=[ 0.1, 0.1, 0.9 ],
		up_vector="y", region_size=[5.0E-1, 5.0E-1],
		distance=2.5E-1, far_cut_depth=2.5E-1, map_max_size=3)
	bb=cam.get_bounding_box()
	source = ro.amr_source(["rho"])
	from pymses.utils import misc
	misc.NUMBER_OF_PROCESSES_LIMIT = 1 # amr2cube no multiprocessing test
	cube = amr2cube(source, "rho", bb.min_coords, bb.max_coords, cam.get_required_resolution())
	print("cube", cube)
	print("cubeB", cubeB)
	assert ((cube - cubeB) < 10e-8).all()
	
	misc.NUMBER_OF_PROCESSES_LIMIT = 8 # amr2cube with multiprocessing test
	cube = amr2cube(source, "rho", bb.min_coords, bb.max_coords, cam.get_required_resolution())
	print("cube", cube)
	assert ((cube - cubeB) < 10e-8).all()
	# ray_trace_cartesian test
	cube_size = numpy.min(cube.shape)
	cam = Camera(center=[ 0.5, 0.5, 0.5 ], line_of_sight_axis=[ 0.1, 0.1, 0.9 ],
		up_vector="y", region_size=[5.0E-1, 5.0E-1],
		distance=2.5E-1, far_cut_depth=2.5E-1, map_max_size=cube_size)
	(ray_vectors, ray_origins, ray_lengths) = cam.get_rays()
	map = raytracing.ray_trace.ray_trace_cartesian(cube, ray_origins, ray_vectors, ray_lengths, bb, cube_size)
	print("map", map)
	print("mapB", mapB)
	assert ((map - mapB) < 10e-8).all()
	

	
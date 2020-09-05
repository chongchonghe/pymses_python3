#!/usr/bin/env python

from numpy.distutils.core import setup
from distutils.extension import Extension
from numpy.distutils.misc_util import Configuration

def make_config(parent_package='',top_path=None):
	
	config = Configuration(
			package_name = "pymses",
			parent_package = parent_package,
			top_path = top_path)

	config.add_extension(
			name = "sources.ramses.tree_utils",
			sources = ["pymses/sources/ramses/tree_utils.c"] )

	config.add_extension(
			name = "utils.point_utils",
			sources = ["pymses/utils/point_utils.c"] )

	config.add_extension(
			name = "sources.ramses._read_ramses",
			sources = ["src/fio.c", "src/read_amr.c", "src/read_cells.c",
				"src/read_parts.c", "src/py_read_ramses.c"] )
	config.add_extension(
			name = "sources.ramses.hilbert",
			sources = ["pymses/sources/ramses/hilbert.c"] )
	config.add_extension(
			name = "analysis.visualization.raytracing.ray_trace",
			sources = ["pymses/analysis/visualization/raytracing/ray_trace.c"] )

	return config
	
def setup_pymses():

    setup(
        name = "PyMSES",
        version = "4.0.0",
        description = "Analysis and visualization Python modules for RAMSES",
        classifiers = [ "Intended Audience :: Science/Research",
                        "License :: OSI Approved :: GNU General Public License (GPL)",
                        "Operating System :: MacOS :: MacOS X",
                        "Operating System :: POSIX :: AIX",
                        "Operating System :: POSIX :: Linux",
                        "Programming Language :: C",
                        "Programming Language :: Python",
                        "Topic :: Scientific/Engineering :: Astronophysics",
                        "Topic :: Scientific/Engineering :: Analysis",
                        "Topic :: Scientific/Engineering :: Visualization", ],
        keywords='astrophysics visualization amr ramses',
        author="Thomas GUILLET, Damien CHAPON, Marc LABADENS",
        author_email="damien.chapon@cea.fr, marc.labadens@cea.fr",
        url = "http://irfu.cea.fr/Projets/PYMSES/index.html",
        license="GPL-3",
        configuration=make_config,
        )
    return

if __name__ == '__main__':
    setup_pymses()

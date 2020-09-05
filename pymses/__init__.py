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
# Convenient imports

# Utils
from . import utils

# Sources and filters
from .sources.ramses.output import RamsesOutput
from . import sources
from . import filters

# Data transformation and analysis
from . import core
transformations = core.transformations
from . import analysis

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

__all__ = ["RamsesOutput", "sources", "filters", "transformations", "analysis", "utils"]

# Those following informations are automatically filled by ../update_version_number.py script 
# from ../setup.py version number and ../.hg/tags.cache informations
__version__  = '4.0.0'
__revision__ = '$Revision: 967$'
__date__     = '$Date: 2014-01-30 $'



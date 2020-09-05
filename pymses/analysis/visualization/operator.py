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
from numpy import zeros_like

class Operator:#{{{
	r"""
	Base Operator generic class

	"""
	def __init__(self, scalar_func_dict, is_max_alos=False, use_cell_dx=False):
		self.scalar_func_dict = scalar_func_dict
		self.max_alos = is_max_alos
		self.use_cell_size = use_cell_dx

	def __iter__(self):
		return self.iter_scalar_func()

	def iter_scalar_func(self):
		for (key, func) in self.scalar_func_dict.items():
			yield (key, func)
	
	def nscal_func(self):
		return len(list(self.scalar_func_dict.keys()))

	def is_max_alos(self):
		return self.max_alos
	
	def use_cell_dx(self):
		return self.use_cell_size

	def operation(self, maps):
		raise NotImplementedError()
#}}}

class ScalarOperator(Operator):#{{{
	r"""
	ScalarOperator class

	Parameters
	----------
	scalar_func : ``function``
		single `dset` argument function returning the scalar data ``array`` from this `dset` Dataset.

	Examples
	--------
	>>> # Density field scalar operator
	>>> op = ScalarOperator(lambda dset: dset["rho"])

	"""
	def __init__(self, scalar_func):
		Operator.__init__(self, {"scalar": scalar_func})

	def operation(self, maps):
		return maps["scalar"]
#}}}

class FractionOperator(Operator):#{{{
	r"""
	FractionOperator class

	Parameters
	----------
	up_func   : ``function``
		numerator function like `scalar_func` (see :class:`~pymses.analysis.visualization.ScalarOperator`)
	down_func : ``function``
		denominator function like `scalar_func` (see :class:`~pymses.analysis.visualization.ScalarOperator`)

	Examples
	--------
	>>> # Mass-weighted density scalar operator
	>>> num = lambda dset: dset["rho"]    * dset.get_sizes()**3
	>>> den = lambda dset: dset["rho"]**2 * dset.get_sizes()**3
	>>> op = FractionOperator(num, den)

	.. math::
		I = \frac{\int\limits_{V} \rho \times \rho \mathrm{d}V}{\int\limits_{V} \rho \mathrm{d}V}

	"""
	def __init__(self, num_func, denom_func):
		d = {"up": num_func, "down": denom_func}
		Operator.__init__(self, d)

	def operation(self, maps):
		md = maps["down"]
		mu = maps["up"]
		map = zeros_like(md)
		mask = (md != 0.0)
		map[mask] = mu[mask]/md[mask]
		return map
#}}}

class MaxLevelOperator(Operator):#{{{
	r"""
	Max. AMR level of refinement operator class

	"""
	def __init__(self):
		d = {"levelmax": (lambda dset: 0)}
		Operator.__init__(self, d, is_max_alos=True, use_cell_dx=True)

	def operation(self, int_dict):
		map = list(int_dict.values())[0]
		return map
#}}}	

__all__ = ["Operator", "ScalarOperator", "FractionOperator", "MaxLevelOperator"]

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
"""
:mod:`pymses.utils.constants` --- physical units and constants module
=====================================================================

"""
import numpy


class Unit(object):#{{{
	r"""
	Dimensional physical unit class

	Parameters
	----------
	dims : 6-``tuple`` of ``int``
		dimension of the unit object expressed in the international system units (``m``, ``kg``, ``s``, ``K``, ``h``, ``T``)
	val  : ``float``
		value of the unit object (in ISU)

	Examples
	--------

		>>> V_km_s = Unit((1,0,-1,0,0), 1000.0)
		>>> print "1.0 km/s = %.1e m/h"%(V_km_s.express(m/hour))
		1.0 km/s = 3.6e+06 m/h

	"""
	isu = ["m","kg","s","K","h","T"]
	
	def __init__(self, dims, val):
		if (len(dims) != len(Unit.isu)):
			raise ValueError("Wrong dimensions : must be a tuple of length "+str(len(Unit.isu)))
		self.dimensions = numpy.array(dims, 'i')
		self.val = val

	def __truediv__(self, other):
		r"""
		Returns the result of the division of `self` by `other`

		"""
		if numpy.isscalar(other):
			newdims = self.dimensions
			newval = self.val / other
			return Unit(newdims, newval)
		elif isinstance(other,Unit):
			newdims = self.dimensions - other.dimensions
			newval = self.val / other.val
			return Unit(newdims, newval)
		else:
			raise ValueError("Unable to divide a Unit instance by something which is neither a Unit object nor a scalar.")

	def __rdiv__(self, other):
		r"""
		Returns the result of the division of `other` by `self`

		"""
		if numpy.isscalar(other):
			newdims = -self.dimensions
			newval = other / self.val
			return Unit(newdims, newval)
		elif isinstance(other,Unit):
			newdims = other.dimensions - self.dimensions
			newval = other.val / self.val
			return Unit(newdims, newval)
		else:
			raise ValueError("Unable to divide something which is neither a Unit object nor a scalar by a Unit instance.")

	def __mul__(self, other):
		r"""
		Returns the result of the multiplication of `self` with `other`

		"""
		if numpy.isscalar(other):
			newdims = self.dimensions
			newval = self.val * other
			return Unit(newdims, newval)
		elif isinstance(other,Unit):
			newdims = self.dimensions + other.dimensions
			newval = self.val * other.val
			return Unit(newdims, newval)
		else:
			raise ValueError("Unable to multiply a Unit instance by something which is neither a Unit object nor a scalar.")
	
	def __rmul__(self, other):
		r"""
		Returns the result of the multiplication of `other` with `self`

		"""
		return self*other

	def __pow__(self, n):
		r"""
		Returns the result of the exponentiation of `self` by `n`

		"""
		return Unit(n*self.dimensions, self.val**n)

	def __repr__(self):
		r"""
		``string`` representation of `self`

		"""
		strrep = "("+str(self.val)+" "
		d = numpy.array(self.dimensions)
		for i in range(len(Unit.isu)):
			if d[i] != 0:
				if (d[0:i]!=0).any():
					strrep += "."
				strrep += Unit.isu[i]
				if d[i] != 1:
					strrep += "^"+str(d[i])
		strrep +=")"
		return strrep


	def express(self, unit):
		r"""
		Unit conversion method. Gives the conversion factor of this ``Unit`` object
		expressed into another (dimension-compatible) given `unit`.
		
		Checks that :
		
		* the `unit` param. is also a ``Unit`` instance
		* the `unit` param. is dimension-compatible
		
		Parameters
		----------
		unit : ``Unit`` object
			unit in which the conversion is made
	
		Returns
		-------
		fact : ``float``
			conversion factor of itself expressed in `unit`

		Examples
		--------
		Conversion of a kpc expressed in light-years :

			>>> factor = kpc.express(ly)
			>>> print "1 kpc = %f ly"%factor
			1 kpc = 3261.563163 ly

		Conversion of :math:`1 M_{\odot}` into kilometers :

			>>> print Msun.express(km)
			ValueError: Incompatible dimensions between (1.9889e+30 kg) and (1000.0 m)

		"""
		assert isinstance(unit, Unit), "argument must have units"
		dmax = len(Unit.isu)-1
		if tuple(self.dimensions[0:dmax]) != tuple(unit.dimensions[0:dmax]):
			raise ValueError("Incompatible dimensions between "+str(self)+" and "+str(unit))
		return (self/unit).val
#}}}


none  = Unit((0, 0, 0, 0, 0, 0), 1.0)

# Distance units
m     = Unit((1, 0, 0, 0, 0, 0), 1.0)
cm    = 1.0E-2 * m
km    = 1000.0 * m
pc    = 3.085677e16 * m
au    = 149598.0E6 * m
kpc   = 1.0E3 * pc
Mpc   = 1.0E6 * pc
Gpc   = 1.0E9 * pc

# Mass units
kg    = Unit((0, 1, 0, 0, 0, 0), 1.0)
g     = 1.0E-3 * kg
mH    = 1.66E-24 * g
Msun  = 1.9889E30 * kg

# Time units
s     = Unit((0, 0, 1, 0, 0, 0), 1.0)
hour  = 3600. * s
day   = 24. * hour
year  = 365.25 * day
Myr   = 1.0E6 * year
Gyr   = 1.0E9 * year

# Force units
N     = kg * m / s**2
dyne  = g * cm / s**2

# Pressure unit
barye = g / ( cm * s**2)

# Energy units
K     = Unit((0, 0, 0, 1, 0, 0), 1.0)
J     =  kg * (m/s)**2
W     = J/s
erg   = g  * (cm/s)**2
eV    = 1.602177e-19 * J

# Magnetic units
T      = Unit((0, 0, 0, 0, 0, 1), 1.0)
Gauss  = 1.0e-4 * T
uGauss = 1e-6 * Gauss

# Composite units
G       = 6.67428e-11 * m**3 / kg / (s**2)
kB      = 1.3806504E-23 * J / K
c       = 299792458.0 * m / s
ly      = c * year
H       = 70.0 * km / s / Mpc
rhoc    = 3.0 * H**2 / (8.0 * numpy.pi * G)
H_cc    = mH / cm**3 / 0.76
g_cc	= g / cm**3
# Planck constant
h       = 6.62606957E-34 * J * s
# Stefan-Boltzmann constant
sigmaSB = (2 * numpy.pi**5 * kB**4)/(15 * h**3 * c**2)
# Radiation constant
a_R     = 4*sigmaSB/c

def list_all():
	r"""
	Print all available constants list:
	
	``none``, ``m``, ``cm``, ``km``, ``pc``, ``au``, ``kpc``, ``Mpc``, ``Gpc``,
	``kg``, ``g``, ``mH``, ``Msun``, ``s``, ``hour``, ``day``, ``year``, ``Myr``, ``Gyr``,
	``N``, ``dyne``, ``K``, ``J``, ``W``, ``erg``, ``eV``, ``G``, ``kB``, ``c``, ``ly``, ``H``,
	``rhoc``, ``H_cc``, ``h``, ``sigmaSB``, ``a_R``, ``T``, ``Gauss``, ``uGauss``

	"""
	l = ['none',
		 'm', 'cm', 'km', 'pc', 'au', 'kpc', 'Mpc', 'Gpc',
		 'kg', 'g', 'mH', 'Msun',
		 's', 'hour', 'day', 'year', 'Myr', 'Gyr',
		 'N','dyne', 'K', 'J', 'W','erg','eV',
		 'G', 'kB', 'c', 'ly', 'H', 'rhoc', 'H_cc', 'h',
		'sigmaSB','a_R','T', 'Gauss', 'uGauss']
	print(l)

__all__ = ["Unit", "list_all"]

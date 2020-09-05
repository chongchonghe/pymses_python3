/******************************************************************************************
* License:                                                                                *
*   Copyright (C) 2011 Thomas GUILLET, Damien CHAPON, Marc LABADENS. All Rights Reserved. *
*                                                                                         *
*   This file is part of PyMSES.                                                          *
*                                                                                         *
*   PyMSES is free software: you can redistribute it and/or modify                        *
*   it under the terms of the GNU General Public License as published by                  *
*   the Free Software Foundation, either version 3 of the License, or                     *
*   (at your option) any later version.                                                   *
*                                                                                         *
*   PyMSES is distributed in the hope that it will be useful,                             *
*   but WITHOUT ANY WARRANTY; without even the implied warranty of                        *
*   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the                         *
*   GNU General Public License for more details.                                          *
*                                                                                         *
*   You should have received a copy of the GNU General Public License                     *
*   along with PyMSES.  If not, see <http://www.gnu.org/licenses/>.                       *
******************************************************************************************/
#include <Python.h>
#include "numpy/arrayobject.h"
#include "read_amr.h"
#include "read_cells.h"
#include "read_parts.h"
#include <strings.h>

/* Routines for easy dictionary insertion */

void dict_int(PyObject * dict, const char * key, int val) {
	PyObject *valobj;
	valobj = PyLong_FromLong(val);
	PyDict_SetItemString(dict, key, valobj);
	Py_DECREF(valobj);
}

void dict_float(PyObject * dict, const char * key, double val) {
	PyObject *valobj;
	valobj = PyFloat_FromDouble(val);
	PyDict_SetItemString(dict, key, valobj);
	Py_DECREF(valobj);
}

void dict_ndarr(PyObject * dict, const char * key, void * data,
		int nd, npy_intp * dims, int type) {

	/* Compute total element count */
	int size = 1;
	int i;
	for(i=0; i<nd; i++) size *= dims[i];

	/* Create empty array */
	PyObject *arrobj;
	arrobj = PyArray_SimpleNew(nd, dims, type);

	/* Copy into array buffer */
	void *npdata = PyArray_DATA((PyArrayObject*)arrobj);
	memcpy(npdata, data, PyArray_ITEMSIZE((PyArrayObject*)arrobj)*size);

	/* Add to dict */
	PyDict_SetItemString(dict, key, arrobj);
	Py_DECREF(arrobj);
}

void dict_1darr(PyObject * dict, const char * key, void * data,
		npy_intp n, int type) {
	dict_ndarr(dict, key, data, 1, &n, type);
}


void dict_2darr(PyObject * dict, const char * key, void * data,
		npy_intp nx, npy_intp ny, int type) {

	npy_intp dims[2] = {nx, ny};
	dict_ndarr(dict, key, data, 2, dims, type);
}




static PyObject *
read_amr(PyObject *self, PyObject *args) {
	const char *filename;
	int readlmax;

	if(!PyArg_ParseTuple(args, "si", &filename, &readlmax))
		return NULL;


	/* Read the AMR structure */
	RAMSES_AmrHeader_t * hdr = RAMSES_AmrHeader_New();
	RAMSES_AmrStruct_t * amr = RAMSES_AmrStruct_New();

	RAMSES_ReadAmr(hdr, amr, filename, readlmax);

	/* Assemble the header dictionary */
	PyObject *hdr_dict = PyDict_New();

	dict_int   (hdr_dict, "ndim"         , hdr->ndim       );
	dict_int   (hdr_dict, "ncpu"         , hdr->ncpu       );

	dict_1darr (hdr_dict, "coarse_shape" , hdr->coarse_shape,
		   	3, PyArray_INT);

	dict_int   (hdr_dict, "ncoarse"      , hdr->ncoarse    );
	dict_int   (hdr_dict, "nboundary"    , hdr->nboundary  );
	dict_int   (hdr_dict, "levelmax"     , hdr->levelmax   );
	dict_int   (hdr_dict, "ngridmax"     , hdr->ngridmax   );
	dict_int   (hdr_dict, "noutputs"     , hdr->noutputs   );

	dict_1darr (hdr_dict, "out_times"    , hdr->out_times,
			hdr->noutputs, PyArray_DOUBLE);

	dict_1darr (hdr_dict, "out_aexps"    , hdr->out_aexps,
			hdr->noutputs, PyArray_DOUBLE);

	dict_float (hdr_dict, "boxlen"       , hdr->boxlen     );
	dict_int   (hdr_dict, "iout"         , hdr->iout       );
	dict_float (hdr_dict, "aexp"         , hdr->aexp       );
	dict_float (hdr_dict, "time"         , hdr->time       );
	dict_int   (hdr_dict, "ngrids_lmax"  , hdr->ngrids_lmax);

	RAMSES_AmrHeader_Free(hdr);


	/* Assemble the AMR structure dictionary */
	PyObject *amr_dict = PyDict_New();

	dict_int   (amr_dict, "ndim"         , amr->ndim       );
	dict_int   (amr_dict, "twotondim"    , amr->twotondim  );
	dict_int   (amr_dict, "ncpu"         , amr->ncpu       );
	dict_int   (amr_dict, "nboundary"    , amr->nboundary  );
	dict_int   (amr_dict, "levelmax"     , amr->levelmax   );
	dict_int   (amr_dict, "ngrids"       , amr->ngrids     );
	dict_int   (amr_dict, "ngrids_lmax"  , amr->ngrids_lmax);
	dict_int   (amr_dict, "readlmax"     , amr->readlmax   );

	dict_2darr (amr_dict, "ngridlevel"  , amr->ngridlevel,
			amr->ncpu, amr->levelmax, PyArray_INT);

	dict_2darr (amr_dict, "ngridbound"  , amr->ngridbound,
			amr->nboundary, amr->levelmax, PyArray_INT);

	dict_1darr (amr_dict, "grid_indices" , amr->grid_indices,
			amr->ngrids, PyArray_INT);

	dict_1darr (amr_dict, "coarse_son_indices", amr->coarse_son_indices,
			amr->ncoarse, PyArray_INT);

	dict_2darr (amr_dict, "grid_centers" , amr->grid_centers,
			amr->ngrids, amr->ndim, PyArray_DOUBLE);

	dict_2darr (amr_dict, "son_indices"  , amr->son_indices,
			amr->ngrids, amr->twotondim, PyArray_INT);

	RAMSES_AmrStruct_Free(amr);


	return Py_BuildValue("(N,N)", hdr_dict, amr_dict);
}


static PyObject *
read_cells(PyObject *self, PyObject *args) {
	const char *filename;
	const char *filetype;
	int ngrids;
	int readlmax;

	PyObject * ivarlist;
	if(!PyArg_ParseTuple(args, "ssiiO",
				&filetype, &filename, &ngrids, &readlmax, &ivarlist))
		return NULL;

	/* Extract the list of ivars to read */
	Py_ssize_t nvarout = PyList_Size(ivarlist);
	if(nvarout <= 0) return NULL;

	int * ivarout_arr = malloc( ((int)nvarout) * sizeof(int) );

	Py_ssize_t ielem;
	for (ielem=0; ielem<nvarout; ielem++) {
		PyObject * item = PyList_GetItem(ivarlist, ielem);
		if(!PyLong_Check(item)) return NULL;
		long ivar = PyLong_AsLong(item);
		ivarout_arr[ielem] = (int)ivar;
	}

	/* AMR file type */
	RAMSES_CellData_FileType_t ftype;
	char gtype[4] = "grav";
	if (strcmp(filetype, gtype) == 0) ftype = RAMSES_CellData_GravFile;
	else ftype = RAMSES_CellData_HydroFile;
	
	/* Allocate and read the cell data */
	RAMSES_CellData_t * cell = RAMSES_CellData_New();
	RAMSES_CellData_Read(cell, filename, ftype, readlmax,
			ngrids, (int)nvarout, ivarout_arr);

	/* Initialize the variable dictionary */
	PyObject *cell_dict = PyDict_New();

	npy_intp dims[2] = {ngrids, 1<<cell->ndim};
	npy_intp size = ngrids * (1<<cell->ndim);

	int ivar;
	for(ivar=0; ivar<cell->nvar_data; ivar++) {
		/* Create a 2D Numpy array to store the cell data */
		PyObject *arrobj;
		arrobj = PyArray_SimpleNew(2, dims, PyArray_DOUBLE);

		/* Copy data into array buffer */
		void *npdata = PyArray_DATA((PyArrayObject*)arrobj);
		memcpy(npdata, cell->data[ivar],
				PyArray_ITEMSIZE((PyArrayObject*)arrobj)*size);

		/* Add to dict */
		PyObject *key = PyLong_FromLong(ivarout_arr[ivar]);
		PyDict_SetItem(cell_dict, key, arrobj); Py_DECREF(key);
		Py_DECREF(arrobj);
	}

	/* Cleanup */
	RAMSES_CellData_Free(cell);
	free(ivarout_arr);

	return Py_BuildValue("N", cell_dict);
}


static PyObject *
read_parts(PyObject *self, PyObject *args) {
	const char *filename;
	double boxlen;

	if(!PyArg_ParseTuple(args, "sd", &filename, &boxlen))
		return NULL;


	/* Open file and read particle data into memory */
	FIO_File * fiofile = FIO_Open(filename, "r", 4, FIO_DEFAULT);
	RAMSES_PartData_t * part = RAMSES_PartData_New();
	RAMSES_PartData_Read(part, boxlen, fiofile);
	FIO_Close(fiofile);

	/* Header */
	PyObject *hdr_dict = PyDict_New();
	dict_int(hdr_dict, "ncpu", part->ncpu);
	dict_int(hdr_dict, "ndim", part->ndim);
	dict_int(hdr_dict, "npart", part->npart);
	dict_int(hdr_dict, "nstar", part->nstar);
	dict_int(hdr_dict, "nsink", part->nsink);

	/* Particle data */
	PyObject *part_dict = PyDict_New();

	if(part->pos != NULL)
		dict_2darr(part_dict, "pos", part->pos,
				part->npart, part->ndim, PyArray_DOUBLE);

	if(part->vel != NULL)
		dict_2darr(part_dict, "vel", part->vel,
				part->npart, part->ndim, PyArray_DOUBLE);

	if(part->mass != NULL)
		dict_1darr(part_dict, "mass", part->mass,
				part->npart, PyArray_DOUBLE);

	if(part->id != NULL)
		dict_1darr(part_dict, "id", part->id,
				part->npart, PyArray_INT);

	if(part->level != NULL)
		dict_1darr(part_dict, "level", part->level,
				part->npart, PyArray_INT);

	if(part->epoch != NULL)
		dict_1darr(part_dict, "epoch", part->epoch,
				part->npart, PyArray_DOUBLE);

	/* Cleanup */
	RAMSES_PartData_Free(part);

	return Py_BuildValue("(N,N)", hdr_dict, part_dict);

}


static PyMethodDef ReadRamsesMethods[] = {
	{ "read_amr",      read_amr ,      METH_VARARGS, "Read an AMR structure" },
	{ "read_cells",    read_cells ,    METH_VARARGS, "Read data from AMR cells"},
	{ "read_parts",    read_parts ,    METH_VARARGS, "Read RAMSES particle data"},
	{ NULL, NULL, 0, NULL }
};

struct module_state {
    PyObject *error;
};

#if PY_MAJOR_VERSION >= 3
#define GETSTATE(m) ((struct module_state*)PyModule_GetState(m))
#else
#define GETSTATE(m) (&_state)
static struct module_state _state;
#endif

static PyObject *
error_out(PyObject *m) {
    struct module_state *st = GETSTATE(m);
    PyErr_SetString(st->error, "something bad happened");
    return NULL;
}

static PyMethodDef myextension_methods[] = {
    {"error_out", (PyCFunction)error_out, METH_NOARGS, NULL},
    {NULL, NULL}
};

#if PY_MAJOR_VERSION >= 3

static int myextension_traverse(PyObject *m, visitproc visit, void *arg) {
    Py_VISIT(GETSTATE(m)->error);
    return 0;
}

static int myextension_clear(PyObject *m) {
    Py_CLEAR(GETSTATE(m)->error);
    return 0;
}


static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        "_read_ramses",
        NULL,
        sizeof(struct module_state),
        ReadRamsesMethods,
        NULL,
        myextension_traverse,
        myextension_clear,
        NULL
};

#define INITERROR return NULL

PyObject *
PyInit__read_ramses(void)

#else
#define INITERROR return

PyMODINIT_FUNC
init_read_ramses(void)
#endif
{
#if PY_MAJOR_VERSION >= 3
    PyObject *module = PyModule_Create(&moduledef);
#else
    PyObject *module = Py_InitModule("_read_ramses", ReadRamsesMethods);
#endif
	import_array();

    if (module == NULL)
        INITERROR;
    struct module_state *st = GETSTATE(module);

    st->error = PyErr_NewException("myextension.Error", NULL, NULL);
    if (st->error == NULL) {
        Py_DECREF(module);
        INITERROR;
    }

#if PY_MAJOR_VERSION >= 3
    return module;
#endif
}

// PyMODINIT_FUNC
// init_read_ramses(void) {
// 	(void) Py_InitModule("_read_ramses", ReadRamsesMethods);
// 	import_array();
// }

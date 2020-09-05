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
#ifndef _H_RAMSES_READ_CELLS
#define _H_RAMSES_READ_CELLS

#include "fio.h"
#include "read_amr.h"


/* Type definitions */

typedef enum RAMSES_CellData_FileType_t {
	RAMSES_CellData_HydroFile,
	RAMSES_CellData_GravFile
} RAMSES_CellData_FileType_t;


typedef struct RAMSES_CellData_t {
	double ** data;
	int nvar_file;
	int nvar_data;
	int * ivar_arr;
	int ncpu;
	int ndim;
	int nboundary;
	int levelmax;
} RAMSES_CellData_t;


/* Method definitions */

RAMSES_CellData_t * RAMSES_CellData_New();

void RAMSES_CellData_Free(RAMSES_CellData_t * cell);

int RAMSES_CellData_Read(
		RAMSES_CellData_t * cell,
		const char * hydro_filename,
		RAMSES_CellData_FileType_t ftype,
		int readlmax,
		int ngrids,
		int nvarout,
		const int * ivarout_arr);

#endif


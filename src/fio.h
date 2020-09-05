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
#ifndef _FIO_H
#define _FIO_H

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#define FIO_DEFAULT     0
#define FIO_SWAP_BYTES  1

/* Fortran file struct */
typedef struct FIO_File {
	FILE *file;
	size_t markersize;
	int flags;
	char * _markerbuffer;
} FIO_File;


FIO_File *
FIO_Open(const char * filename, const char * mode, size_t markersize, int flags);

int
FIO_Close(FIO_File * fiofile);

size_t
_FIO_ReadMarker(FIO_File * fiofile);

int
FIO_SkipRecord(FIO_File * fiofile);

size_t
FIO_ReadRecord(void * buffer, size_t itemsize, size_t maxcount, FIO_File * fiofile);

#endif  /* #ifndef _FIO_H */

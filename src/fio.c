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
#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include "byteswap.h"
#include "fio.h"

FIO_File *
FIO_Open(const char * filename, const char * mode, size_t markersize, int flags) {
	FIO_File * f = malloc(sizeof(FIO_File));
	assert(markersize==sizeof(int));
	f->file = fopen(filename, mode);
	if (f->file == NULL) {
		printf("FIO_Open: failed to open file\n");
		return NULL;
	}
	f->markersize = markersize;
	f->flags = flags;
	return f;
}

int
FIO_Close(FIO_File * fiofile) {
	int ierr;
	ierr = fclose(fiofile->file);
	if (ierr) {
		printf("FIO_Close: Close error\n");
		return EIO;
	}
	free(fiofile);
	return ierr;
}

/* Low-level read function that handles byteswapping */
size_t
_FIO_FRead(FIO_File * fiofile, void * buf, size_t itemsize, size_t count) {
	size_t nread = fread(buf, itemsize, count, fiofile->file);

	if(fiofile->flags & FIO_SWAP_BYTES) {
		size_t i;
		switch(itemsize) {
			case 4:
				for(i=0; i<nread; i++) {
					unsigned int *ptr = (unsigned int*)buf + i;
					*ptr = __bswap_32(*ptr);
				}
				break;

			case 8:
				for(i=0; i<nread; i++) {
					unsigned long *ptr = (unsigned long*)buf + i;
					*ptr = __bswap_64(*ptr);
				}
				break;

			default:
				printf("_FIO_FRead: invalid item size\n");
				return 0;
		}
	}

	return nread;
}

size_t
_FIO_ReadMarker(FIO_File * fiofile) {
	size_t marker;
	size_t *ptr = &marker;

	_FIO_FRead(fiofile, ptr, fiofile->markersize, 1);

	switch (fiofile->markersize) {
		case 4:
			return (size_t)(*((unsigned int*)ptr));
			break;
		case 8:
			return (size_t)(*((unsigned long*)ptr));
			break;
		default:
			printf("_FIO_ReadMarker: invalid marker size\n");
			return -1;
			break;
	}
}

int
FIO_SkipRecord(FIO_File * fiofile) {
	size_t m1, m2;
	m1 = _FIO_ReadMarker(fiofile);
	int ierr = fseek(fiofile->file, m1, SEEK_CUR);
	if (ierr) {
		printf("FIO_SkipRecord: seek error\n");
		return EIO;
	}
	m2 = _FIO_ReadMarker(fiofile);
	if (m1 != m2) {
		printf("FIO_SkipRecord: marker mismatch\n");
		return EIO;
	}
	return 0;
}


size_t
FIO_ReadRecord(void * buffer, size_t itemsize, size_t maxcount, FIO_File * fiofile) {
	size_t m1, m2;
	m1 = _FIO_ReadMarker(fiofile);

	if(m1 < itemsize*maxcount) {
		printf("FIO_ReadRecord: read overflow\n");
		return 0;
	}

	size_t nread = _FIO_FRead(fiofile, buffer, itemsize, m1/itemsize);
	if(nread != m1/itemsize) {
		printf("FIO_ReadRecord: wrong number of items read\n");
		return 0;
	}

	m2 = _FIO_ReadMarker(fiofile);
	if (m1 != m2) {
		printf("FIO_ReadRecord: marker mismatch\n");
		return 0;
	}

	return nread;
}

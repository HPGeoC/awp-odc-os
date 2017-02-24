##
#@section LICENSE
#Copyright (c) 2013-2017, Regents of the University of California, San Diego State University
#All rights reserved.
#
#Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
#
#1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
#
#2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
#
#THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
##

CC 	= mpicc
CFLAGS	= -O3 -Wall
GFLAGS	= nvcc -use_fast_math -arch=sm_35 -O3 -lineinfo

INCDIR  = -I/usr/local/cuda/include -I/home/dmu/Tools/mpich-3.2/include
OBJECTS	= command.o pmcl3d.o grid.o source.o mesh.o cerjan.o swap.o kernel.o io.o
LIB	= -L/usr/local/cuda/lib64 -L/home/dmu/Tools/mpich-3.2/lib -lm -ldl -lcudart -lmpich


pmcl3d:	$(OBJECTS)
	$(CC) $(CFLAGS) $(INCDIR) -o	pmcl3d	$(OBJECTS)	$(LIB)

pmcl3d.o:	pmcl3d.c
	$(CC) $(CFLAGS) $(INCDIR) -c -o pmcl3d.o	pmcl3d.c

command.o:	command.c
	$(CC) $(CFLAGS) $(INCDIR) -c -o	command.o	command.c

io.o:	  io.c
	$(CC) $(CFLAGS) $(INCDIR) -c -o	io.o	  io.c

grid.o:		grid.c
	$(CC) $(CFLAGS) $(INCDIR) -c -o grid.o		grid.c

source.o:	source.c
	$(CC) $(CFLAGS) $(INCDIR) -c -o source.o	source.c

mesh.o:		mesh.c
	$(CC) $(CFLAGS) $(INCDIR) -c -o mesh.o		mesh.c

cerjan.o:	cerjan.c
	$(CC) $(CFLAGS) $(INCDIR) -c -o cerjan.o	cerjan.c

swap.o:		swap.c
	$(CC) $(CFLAGS) $(INCDIR) -c -o swap.o		swap.c

kernel.o:	kernel.cu
	$(GFLAGS) $(INCDIR) -c -o	kernel.o	kernel.cu

clean:
	rm *.o pmcl3d

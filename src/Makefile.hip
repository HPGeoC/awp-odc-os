CUDA_HOME = $(which hipcc)
MPI_HOME=$(which mpicc)
export OMPI_CXX=hipcc

CC      = mpic++
LD      = mpic++
LDFLAGS = -O3 -g -D__HIP --offload-arch=gfx908
# When we have access to MI200s, change to gfx90a
CFLAGS = -O3 -g -D__HIP -x hip --offload-arch=gfx908
GFLAGS = hipcc -D__HIP --offload-arch=gfx908

INCDIR  = -I$(MPI_HOME)/include
OBJECTS = command.o pmcl3d.o grid.o source.o mesh.o cerjan.o swap.o io.o kernel.o
LIB     = -L$(MPI_HOME)/lib

pmcl3d:	$(OBJECTS)
	[ -d bin ] || mkdir -p bin
	$(LD) $(LDFLAGS) $(INCDIR) -o	pmcl3d	$(OBJECTS)	$(LIB)

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
	rm *.o

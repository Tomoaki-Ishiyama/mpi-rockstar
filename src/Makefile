compiler=gnu
#compiler=intel
#compiler=fujitsu

ifeq ($(compiler), gnu)
CC = mpicc
CXX = mpicxx
OFLAGS = -O3 -fopenmp
endif

ifeq ($(compiler), intel)
CC = mpiicx
CXX = mpiicpx
OFLAGS = -O3 -fiopenmp
endif

ifeq ($(compiler), fujitsu)
CC = mpifccpx
CXX = mpiFCCpx
OFLAGS = -Nclang -Ofast -fopenmp # -I/usr/include/tirpc
endif

CFLAGS   = -m64 -D_LARGEFILE_SOURCE -D_LARGEFILE64_SOURCE -D_FILE_OFFSET_BITS=64 -D_DEFAULT_SOURCE -D_POSIX_SOURCE -D_POSIX_C_SOURCE=200809L -D_SVID_SOURCE -D_DARWIN_C_SOURCE -Wall -fno-math-errno -fPIC -std=c99 
CXXFLAGS = -Wall -fno-math-errno -fPIC -std=c++11
#ADDFLAGS = -DOUTPUT_RVMAX -DOUTPUT_NFW_CHI2 -DOUTPUT_INTERMEDIATE_AXIS
ADDFLAGS = -DOUTPUT_RVMAX -DOUTPUT_NFW_CHI2 -DOUTPUT_INERTIA_TENSOR
CFLAGS   += $(ADDFLAGS) $(OFLAGS) -DDO_CONFIG_MPI 
CXXFLAGS += $(ADDFLAGS) $(OFLAGS) -DDO_CONFIG_MPI 

HDF5_INCLUDE = -I/path/to/hdf5/include
HDF5_LIB = -L/path/to/hdf5/lib
HDF5_FLAGS = -DH5_USE_16_API -DENABLE_HDF5 $(HDF5_INCLUDE)

MPI_ROCKSTAR = rockstar.o check_syscalls.o fof.o groupies.o subhalo_metric.o potential.o nfw.o jacobi.o fun_times.o universe_time.o hubble.o integrate.o distance.o config_vars.o config.o bounds.o inthash.o io/read_config.o merger.o io/meta_io.o io/io_internal.o io/io_internal_hdf5.o io/io_ascii.o io/stringparse.o io/io_gadget.o io/io_generic.o io/io_art.o io/io_tipsy.o io/io_bgc2.o io/io_util.o io/io_arepo.o io/io_gadget4.o io/io_hdf5.o io/io_kyf.o interleaving.o mpi_main.o

MPI_ROCKSTAR_HDF5 = $(MPI_ROCKSTAR) io/io_internal_hdf5.o

FIND_PARENTS = find_parents.o io/stringparse.o check_syscalls.o

*.o: *.h Makefile*

.cpp.o: 
	$(CXX) $(CXXFLAGS) $(EXTRA_FLAGS) -c -o $@ $<

.c.o: 
	$(CC) $(CFLAGS) $(EXTRA_FLAGS) -c -o $@ $< -I/usr/include/tirpc

.PHONY: mpi-rockstar
mpi-rockstar: $(MPI_ROCKSTAR)
	$(CXX) $(CXXFLAGS) -o ../$@ $^ -lm -lstdc++ -ltirpc

.PHONY: mpi-rockstar_hdf5
mpi-rockstar_hdf5:
	make _mpi-rockstar_hdf5 EXTRA_FLAGS="$(HDF5_FLAGS)"

.PHONY: _mpi-rockstar_hdf5
_mpi-rockstar_hdf5: $(MPI_ROCKSTAR_HDF5)
	$(CXX) $(CXXFLAGS) $(EXTRA_FLAGS) -o ../mpi-rockstar_hdf5 $^ $(HDF5_LIB) -lm -lstdc++ -ltirpc -lhdf5 -lz

.PHONY: find_parents
find_parents: $(FIND_PARENTS)
	$(CC) $(CFLAGS) -o ../$@ $^ -lm

clean:
	rm -f *.o *~ io/*.o io/*~

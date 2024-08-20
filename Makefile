CFLAGS   = -m64 -D_LARGEFILE_SOURCE -D_LARGEFILE64_SOURCE -D_FILE_OFFSET_BITS=64 -D_BSD_SOURCE -D_POSIX_SOURCE -D_POSIX_C_SOURCE=200809L -D_SVID_SOURCE -D_DARWIN_C_SOURCE -Wall -fno-math-errno -fPIC -std=c99
CXXFLAGS = -Wall -fno-math-errno -fPIC -std=c++11
#ADDFLAGS = -DOUTPUT_RVMAX -DOUTPUT_INTERMEDIATE_AXIS
ADDFLAGS = -DOUTPUT_RVMAX -DOUTPUT_INERTIA_TENSOR -DDO_CONFIG_MPI
CFLAGS   += $(ADDFLAGS)
CXXFLAGS += $(ADDFLAGS)
LDFLAGS  = -shared
OFLAGS   = -O3 -fopenmp
DEBUGFLAGS = -lm -g -O0 -std=c99 -rdynamic
PROFFLAGS = -lm -g -pg -O2 -std=c99
CC = mpicc
CXX = mpic++
CFILES = rockstar.c check_syscalls.c fof.c groupies.c subhalo_metric.c potential.c nfw.c jacobi.c fun_times.c universe_time.c hubble.c integrate.c distance.c config_vars.c config.c bounds.c inthash.c io/read_config.c merger.c inet/socket.c inet/rsocket.c inet/address.c io/meta_io.c io/io_internal.c io/io_internal_hdf5.c io/io_ascii.c io/stringparse.c io/io_gadget.c io/io_generic.c io/io_art.c io/io_tipsy.c io/io_bgc2.c io/io_util.c io/io_arepo.c io/io_gadget4.c io/io_hdf5.c io/io_kyf.c interleaving.c
CPPFILES = mpi_main.cpp
OBJS = $(CFILES:.c=.o) $(CPPFILES:.cpp=.o)
DIST_FLAGS =
HDF5_INCLUDE = -I/path/to/hdf5/include
HDF5_LIB = -L/path/to/hdf5/lib
HDF5_FLAGS = -DH5_USE_16_API -DENABLE_HDF5 $(HDF5_INCLUDE)

all:
	@make rockstar EXTRA_FLAGS="$(OFLAGS)"

with_hdf5:
	@make rockstar_hdf5 EXTRA_FLAGS="$(OFLAGS) $(HDF5_FLAGS)"

debug:
	@make rockstar EXTRA_FLAGS="$(DEBUGFLAGS)"

with_hdf5_debug:
	@make rockstar EXTRA_FLAGS="$(DEBUGFLAGS) $(HDF5_FLAGS)"

prof:
	@make rockstar EXTRA_FLAGS="$(PROFFLAGS)"

.REMAKE:

dist: .REMAKE
	cd ../ ; perl -ne 'print "$$1\n" if (/VERSION\s*\"([^\"]+)/)' Rockstar/version.h > Rockstar/VERSION; tar -czvf rockstar.tar.gz Rockstar/Makefile Rockstar/*.[ch] Rockstar/examples/Makefile Rockstar/[^sto]*/*.[ch] Rockstar/quickstart.cfg Rockstar/parallel.cfg Rockstar/scripts/*.pbs Rockstar/scripts/*.cfg Rockstar/scripts/*.pl Rockstar/SOURCE_LAYOUT Rockstar/README.pdf Rockstar/README Rockstar/LICENSE Rockstar/VERSION Rockstar/ACKNOWLEDGMENTS Rockstar/CHANGELOG; mv rockstar.tar.gz Rockstar

versiondist:
	$(MAKE) dist DIST_FLAGS="$(DIST_FLAGS)"
	rm -rf dist
	mkdir dist
	cd dist; tar xzf ../rockstar.tar.gz ; perl -ne '/\#define.*VERSION\D*([\d\.rcRC-]+)/ && print $$1' Rockstar/version.h > NUMBER ; mv Rockstar Rockstar-`cat NUMBER`; tar czf rockstar-`cat NUMBER`.tar.gz Rockstar-`cat NUMBER`

rockstar: $(OBJS) main.o
	@$(CXX) -o $@ $^ $(EXTRA_FLAGS) -lm -lstdc++ -ltirpc

rockstar_hdf5: $(OBJS) main.o
	@$(CXX) -o $@ $^ $(EXTRA_FLAGS) $(HDF5_LIB) -lm -lstdc++ -ltirpc -lhdf5 -lz

%.o: %.c
	@$(CC) $(CFLAGS) -c $< -o $@ $(EXTRA_FLAGS) -I/usr/include/tirpc

%.o: %.cpp
	@$(CXX) $(CXXFLAGS) -c $< -o $@ $(EXTRA_FLAGS)

lib:
	$(CC) $(CFLAGS) $(LDFLAGS) $(CFILES) -o librockstar.so $(EXTRA_FLAGS)

bgc2:
	$(CC) $(CFLAGS) io/extra_bgc2.c util/redo_bgc2.c $(CFILES) -o util/finish_bgc2  $(OFLAGS)
	$(CC) $(CFLAGS) io/extra_bgc2.c util/bgc2_to_ascii.c $(CFILES) -o util/bgc2_to_ascii  $(OFLAGS)

parents:
	$(CC) $(CFLAGS) util/find_parents.c io/stringparse.c check_syscalls.c  -o util/find_parents $(OFLAGS) -lm

substats:
	$(CC) $(CFLAGS) util/subhalo_stats.c $(CFILES) -o util/subhalo_stats  $(OFLAGS)


clean:
	rm -f *~ io/*~ inet/*~ util/*~ rockstar rockstar_hdf5 util/redo_bgc2 util/subhalo_stats $(OBJS) main.o

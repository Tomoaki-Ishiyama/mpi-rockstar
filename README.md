# MPI-Rockstar: a Hybrid MPI and OpenMP Parallel Implementation of the Rockstar Halo finder

Copyright(c) 2024 Tomoyuki Tokuue, Tomoaki Ishiyama, Ken Osato, Satoshi Tanaka, and Peter Behroozi

License: GNU GPLv3
Paper: https://arxiv.org/abs/2412.18629

Base code: Rockstar ( (c) 2011 Peter Behroozi)
Repository: https://bitbucket.org/gfcstanford/rockstar/src/main/
Science/Documentation Paper: https://ui.adsabs.harvard.edu/abs/2013ApJ...762..109B/

MPI-Rockstar is a massively parallel halo finder based on the [Rockstar](https://bitbucket.org/gfcstanford/rockstar/) phase-space temporal halo finder code ([Behroozi et al., 2013](https://ui.adsabs.harvard.edu/abs/2013ApJ...762..109B/)), which is one of the most extensively used halo finding codes.  Compared to the original code, parallelized by a primitive socket communication library, we parallelized it in a hybrid way using MPI and OpenMP, which is suitable for analysis on the hybrid shared and distributed memory environments of modern supercomputers.  This implementation can easily handle the analysis of more than a trillion particles on more than 100,000 parallel processes, enabling the production of a huge dataset for the next generation of cosmological surveys. As new functions to the original Rockstar code, MPI-Rockstar supports HDF5 as an output format and can output additional halo properties such as the inertia tensor.

**Whenever you use this code, please cite the two papers shown above ([Tokuue+ 2024](https://arxiv.org/abs/2412.18629) and [Behroozi+ 2013](https://ui.adsabs.harvard.edu/abs/2013ApJ...762..109B/))**

**Most of functions provided in the original Rockstar can be used in MPI-Rockstar as they are. Only the differences between the original Rockstar and MPI-Rockstar are explained below.**


## Compiling and Running ##

You can compile MPI-Rockstar with this command at the root directory of the code.
```
make mpi-rockstar -C src
```
When you need the HDF5 support to file IO, use this command after setting appropriate path to `HDF5_INCLUDE` and `HDF5_LIB` in `Makefile`
```
make mpi-rockstar_hdf5 -C src
```
Then, you can run MPI-Rockstar as follows,
```
mpiexec -n <the number of MPI processes> mpi-rockstar -c <path to a configuration file>
```
You can find examples of configuration file in the `/examples` directory. Then, the number of OpenMP threads can be set by
```
export OMP_NUM_THREADS=<the number of OpenMP threads>
```
When you use a batch job system, please follow the instruction of the system.

In the original Rockstar, `PID` is the parent halo's ID and can be added by `find_parents` after running the Rockstar. You can also compile it by
```
make find_parents -C src
```

In some environments, due to a compatibility issue, you may encounter this or something similar error.
```
/usr/include/tirpc/rpc/xdr.h:111:52: error: unknown type name 'u_int'
```
You may be able to solve it by removing `-I/usr/include/tirpc` from `.c.o:` in the `Makefile`.


## HDF5 and Gadget-4 Format Supports for Output ##

MPI-Rockstar supports the HDF5 output of halo catalogs by adding this line in a configuration file.
```
OUTPUT_FORMAT="HDF5"
```

Gadget-4 (HDF5) format is also supported as input snapshots.
```
FILE_FORMAT="GADGET4"
```

The default number of Gadget-4 particle types is `NTYPES=6`. You can use other types by adding the line below to a configuration file.
```
GADGET4_NTYPES=<your ntypes>
```

The code assumes 8-byte (`uint64_t`) particle IDs by default (and recommended). When you use 4-byte (`uint32_t`) particle IDs, please also add the below line.
```
GADGET4_ID_BYTES=4
```
In Gadget4 `Config.sh`, you can set `IDS_64BIT` for 8-byte IDs or `IDS_32BIT` for 4-byte IDs. 6-byte ID (`IDS_48BIT`) is not supported.

The code also assumes a single precision (4-byte float) for particle positions. Ensure that `POSITIONS_IN_32BIT` is defined in Gadget4 `Config.sh`.


## Abolished Configuration Options ##

MPI-Rockstar can no-longer run on single process, therefore, only `PARALLEL_IO=1` is accepted (default value). The number of writer and reader processes are automatically set from the number of processes, therefore, `NUM_WRITERS`, `NUM_READERS`, `FORK_READERS_FROM_WRITERS`, and `FORK_PROCESSORS_PER_MACHINE` are abolished.



## New Compiling Options ##

MPI-Rockstar can output additional halo properties by adding the below options at compiling.

### OUTPUT_RVMAX ###
Output halo's $R_{\rm vmax}$, which is a radius of maximum circular velocity. This option is only applicable to ascii format, and is always output by default for binary and HDF5 format.

### OUTPUT_NFW_CHI2 ###
Output $\chi^2$ in the NFW fitting of the density profile of a halo. The NFW fitting is used to calculate the scale radius $r_{\rm s}$.
When the number of halo's bound particles is less than `MIN_SCALE_PART` (set in `nfw.c`, 100 by default), $r_{\rm s}$ is the same with $r_{\rm s}^{\rm klypin}$, which is the estimated scale radius using $V_{\rm max}$ and the halo mass. In this case, $\chi^2=0$ is set.
When this option is enabled, the binary file header will have the corresponding bit in `add_flag` set using `add_flag |= 1;`.

### OUTPUT_INTERMEDIATE_AXIS ###
Output six elements of halo's intermediate shape ellipsoid axis. Three and remaining three of them are calculated by using bound particles within $R_{\rm vir}$ and $R_{\rm 500c}$, respectively.
When this option is enabled, the binary file header will have the corresponding bit in `add_flag` set using `add_flag |= 2;`.

### OUTPUT_INERTIA_TENSOR ###
Output 12 elements of halo's inertia tensor. Six and remaining six of them are calculated by using bound particles within $R_{\rm vir}$ and $R_{\rm 500c}$, respectively. You can enable either `OUTPUT_INERTIA_TENSOR` and `OUTPUT_INTERMEDIATE_AXIS` at the same time.
When this option is enabled, the binary file header will have the corresponding bit in `add_flag` set using `add_flag |= 4;`.

## New Configuration Options ##

### FILES_PER_SUBDIR_INPUT and SUBDIR_DIGITS_INPUT ###

These options allow that snapshots are stored in multiple sub-directories. `FILES_PER_SUBDIR_INPUT` is the number of files in each sub-directory. The sub-directory name is expressed by a sequential number starting from zero, and its digit is set by `SUBDIR_DIGITS_INPUT` For example, if
```
INBASE="snapdir"
FILENAME="snap-<block>"
NUM_BLOCKS=16
FILES_PER_SUBDIR_INPUT=4
SUBDIR_DIGITS_INPUT=4
```
in a configuration file, the following directory structure of snapshots is assumed.
```
snapdir/
  0000/snap-[0-3]
  0001/snap-[4-7]
  0002/snap-[8-11]
  0003/snap-[12-15]
```
When `FILES_PER_SUBDIR_INPUT=0`, no sub-directories are assumed (by default).

### OUTLIST_PARALLEL ###
By default, MPI-Rockstar writes single halo catalog (out_X.list) using MPI_IO. When you want to write out_X.list per process, set `OUTLIST_PARALLEL=1` (0 by default).


### OUTPUT_SUBDIR and SNAPSHOT_SUBDIR_DIGITS ###
By defaults, MPI-Rockstar writes all output (bin, ascii, list) in single directory (set by `OUTBASE` in a configuration file). When you want to separate output directories for every snapshot, set `OUTPUT_SUBDIR=1`. The sub-directory name is expressed by the snapshot number with the digits set by `SNAPSHOT_SUBDIR_DIGITS`. For example, if
```
STARTING_SNAP=20
NUM_SNAPS=23
OUTBASE="Out"
OUTPUT_SUBDIR=1
SNAPSHOT_SUBDIR_DIGITS=3
```
in a configuration file, the output are stored in the following directory.
```
Out/
  020/ for snapshot 20
  021/ for snapshot 21
  022/ for snapshot 22
```
**Note that each sub-directory must be created before the running.**


### FILES_PER_SUBDIR_OUTPUT and SUBDIR_DIGITS_OUTPUT ###
These options are inverse of `FILES_PER_SUBDIR_INPUT` and `SUBDIR_DIGITS_INPUT`. These options allow that output of a snapshot is stored in multiple sub-directories. `FILES_PER_SUBDIR_OUTPUT` is the number of files in each sub-directory. The sub-directory name is expressed by a sequential number starting from zero, and its digit is set by `SUBDIR_DIGITS_OUTPUT` For example, if the number of MPI processes is 16 and
```
OUTBASE="Out"
FILES_PER_SUBDIR_OUTPUT=4
SUBDIR_DIGITS_OUTPUT=4
```
in a configuration file, the output are stored in the following directory.
```
Out/
  0000/ for processes [0-3]
  0001/ for processes [4-7]
  0002/ for processes [8-11]
  0003/ for processes [12-15]
```
**Note that each sub-directory is made automatically.***

**`OUTLIST_PARALLEL`, `OUTPUT_SUBDIR`, `SNAPSHOT_SUBDIR_DIGITS`, `FILES_PER_SUBDIR_OUTPUT` and `SUBDIR_DIGITS_OUTPUT` can work concurrently. In this case, sub-sub-directories are made.**


## Test Dataset ##
For tests, we provide tiny ($N=256^3$) and small ($N=1024^3$) particle dataset around $z\sim0$ from cosmological N-body simulations [here](https://hpc.imit.chiba-u.jp/~ishiymtm/Data/MPIRockstar/). The former consists of single file (one snapshot), and the latter consists of 640 files per snapshot (2 snapshots are provided). After downloading these files, you can analyze those data as follows using configuration files in the `/examples` directory.
```
mpiexec -n XXX mpi-rockstar -c parallel_256.cfg
mpiexec -n XXX mpi-rockstar -c parallel_1024.cfg
mpiexec -n XXX mpi-rockstar_hdf5 -c parallel_1024.cfg  #with HDF5 output and new configuration options
```

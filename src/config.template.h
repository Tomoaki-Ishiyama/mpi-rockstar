// string(a,b) real(a,b) integer(a,b)
// Sets strings, floats, and integers;
//      "a" is the variable name;
//      "b" is the default.

string(FILE_FORMAT, "GADGET2");
real(PARTICLE_MASS, 0); // Auto-generated or auto-read

string(MASS_DEFINITION, "vir");
string(MASS_DEFINITION2, "200b");
string(MASS_DEFINITION3, "200c");
string(MASS_DEFINITION4, "500c");
string(MASS_DEFINITION5, "2500c");
integer(STRICT_SO_MASSES, 0);
integer(MIN_HALO_OUTPUT_SIZE, 20);
real(FORCE_RES, 0.003); // In Mpc/h
real(FORCE_RES_PHYS_MAX, 0);

real(NON_COSMOLOGICAL, 0);
real(SCALE_NOW, 1.0);
real(h0, 0.7);
real(Ol, 0.73);
real(Om, 0.27);
real(W0, -1);
real(WA, 0);

integer(GADGET_ID_BYTES, 4);
real(GADGET_MASS_CONVERSION, 1e10);
real(GADGET_LENGTH_CONVERSION, 1.0);
integer(GADGET_SKIP_NON_HALO_PARTICLES, 1);
integer(GADGET_HALO_PARTICLE_TYPE, 1);
integer(RESCALE_PARTICLE_MASS, 0);

#ifdef ENABLE_HDF5
integer(AREPO_ID_BYTES, 8);
real(AREPO_MASS_CONVERSION, 1e10);
real(AREPO_LENGTH_CONVERSION, 1e-3);
integer(AREPO_DM_PARTTYPE, 1);

integer(GADGET4_ID_BYTES, 8);
integer(GADGET4_NTYPES, 6);
real(GADGET4_MASS_CONVERSION, 1e10);
real(GADGET4_LENGTH_CONVERSION, 1.0);
integer(GADGET4_DM_PARTTYPE, 1);
#endif /* ENABLE_HDF5 */

real(TIPSY_LENGTH_CONVERSION, 1.0);
real(TIPSY_VELOCITY_CONVERSION, 1.0);

integer(PARALLEL_IO, 1);
string(PARALLEL_IO_SERVER_ADDRESS, "auto");
string(PARALLEL_IO_SERVER_PORT, "auto");
integer(PARALLEL_IO_WRITER_PORT, 32001);
string(PARALLEL_IO_SERVER_INTERFACE, "");
integer(PARALLEL_IO_CATALOGS, 0);
string(RUN_ON_SUCCESS, "");
string(RUN_PARALLEL_ON_SUCCESS, "");
string(LOAD_BALANCE_SCRIPT, "");

string(INBASE, ".");
string(FILENAME, "tests/halo_nfw");
integer(STARTING_SNAP, 0);
integer(RESTART_SNAP, 0);
integer(NUM_SNAPS, 1);
integer(NUM_BLOCKS, 1);
integer(NUM_READERS, 0);
integer(PRELOAD_PARTICLES, 0);
string(SNAPSHOT_NAMES, "");
string(LIGHTCONE_ALT_SNAPS, "");
string(BLOCK_NAMES, "");

string(OUTBASE, ".");
real(OVERLAP_LENGTH, 3.0);
integer(NUM_WRITERS, 1);
integer(FORK_READERS_FROM_WRITERS, 0);
integer(FORK_PROCESSORS_PER_MACHINE, 1);

string(OUTPUT_FORMAT, "BOTH");
integer(DELETE_BINARY_OUTPUT_AFTER_FINISHED, 0);
integer(FULL_PARTICLE_CHUNKS, 0);
integer(OUTPUT_EVERY_N_PARTICLES, 1);
integer(UNFILTERED_HALO_OUTPUT, 0);
string(BGC2_SNAPNAMES, "");
real(WEAK_LENSING_FRACTION, 0);

integer(SHAPE_ITERATIONS, 10);
integer(WEIGHTED_SHAPES, 1);
integer(BOUND_PROPS, 1);
integer(BOUND_OUT_TO_HALO_EDGE, 0);
integer(DO_MERGER_TREE_ONLY, 0);
integer(IGNORE_PARTICLE_IDS, 0);
integer(EXACT_LL_CALC, 0);
real(TRIM_OVERLAP, 0);
real(ROUND_AFTER_TRIM, 1);
integer(LIGHTCONE, 0);
integer(PERIODIC, 1);

real3(LIGHTCONE_ORIGIN, "0 0 0");
real3(LIGHTCONE_ALT_ORIGIN, "0 0 0");

real3(LIMIT_CENTER, "0 0 0");
real(LIMIT_RADIUS, 0);

integer(SWAP_ENDIANNESS, 0);
integer(GADGET_VARIANT, 0);
integer(ART_VARIANT, 0);

real(FOF_FRACTION, 0.7);
real(FOF_LINKING_LENGTH, 0.28);
real(INITIAL_METRIC_SCALING, 1);
// real(CONTINUED_METRIC_SCALING, 1);
real(INCLUDE_HOST_POTENTIAL_RATIO, 0.3);
integer(TEMPORAL_HALO_FINDING, 1);
integer(MIN_HALO_PARTICLES, 10);
real(UNBOUND_THRESHOLD, 0.5);
integer(ALT_NFW_METRIC, 0);
integer(EXTRA_PROFILING, 1);

integer(TOTAL_PARTICLES, 8589934592);
real(BOX_SIZE, 250); // In Mpc/h
integer(OUTPUT_LEVELS, 0);
real3(DUMP_PARTICLES, "0 0 0");

string(ROCKSTAR_CONFIG_FILENAME, "");
real(AVG_PARTICLE_SPACING, 0); // Auto-generated
integer(SINGLE_SNAP, 0);

// Additional parameters for MPI Rockstar for input
integer(FILES_PER_SUBDIR_INPUT, 0); // The number of files in each subdir
integer(SUBDIR_DIGITS_INPUT, 4);    // and the length of subdir names
string(INBASE2, "snapdir_");

// Additional parameters for MPI Rockstar for output
integer(OUTLIST_PARALLEL, 0); // Output 'outlist' files in parallel
integer(OUTPUT_SUBDIR, 0);    // Organize output files by snapshot
integer(SNAPSHOT_SUBDIR_DIGITS, 3);  // The length of snapshot subdir numbers 
integer(FILES_PER_SUBDIR_OUTPUT, 0); // The number of files in each subdir (or sub-sub-dir)
integer(SUBDIR_DIGITS_OUTPUT, 4);    // and the length of subdir names (or sub-sub-dir)

// Additional parameters for MPI Rockstar for memory saving transfer
integer(MEMORY_SAVING_TRANSFER, 0); // Flag for using memory saving transfer

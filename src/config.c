#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <inttypes.h>
#include <string.h>
#include <sys/resource.h>
#include "config_vars.h"
#include "check_syscalls.h"
#include "io/read_config.h"
#include "universal_constants.h"

#ifdef DO_CONFIG_MPI
#include "mpi.h"
#endif


void setup_config(void) {
    if (!NUM_READERS)
        NUM_READERS = NUM_BLOCKS;

    if (!PARTICLE_MASS)
        PARTICLE_MASS = CRITICAL_DENSITY * BOX_SIZE * BOX_SIZE * BOX_SIZE * Om /
                        TOTAL_PARTICLES;
    if (!AVG_PARTICLE_SPACING)
        AVG_PARTICLE_SPACING = cbrt(PARTICLE_MASS / (Om * CRITICAL_DENSITY));

    if (LIGHTCONE || !PARALLEL_IO) {
        PERIODIC              = 0;
        TEMPORAL_HALO_FINDING = 0;
    }

    if (IGNORE_PARTICLE_IDS)
        TEMPORAL_HALO_FINDING = 0;

    if (!FORCE_RES_PHYS_MAX)
        FORCE_RES_PHYS_MAX = FORCE_RES;

    struct rlimit rlp;
    getrlimit(RLIMIT_NOFILE, &rlp);
    rlp.rlim_cur = rlp.rlim_max;
    setrlimit(RLIMIT_NOFILE, &rlp);
    getrlimit(RLIMIT_CORE, &rlp);
    rlp.rlim_cur = rlp.rlim_max;
    setrlimit(RLIMIT_CORE, &rlp);
    if (NUM_WRITERS < FORK_PROCESSORS_PER_MACHINE)
        NUM_WRITERS = FORK_PROCESSORS_PER_MACHINE;

    if (STARTING_SNAP >= NUM_SNAPS) {
        fprintf(stderr, "[Rockstar warning] No work will be done unless NUM_SNAPS > "
                        "STARTING_SNAP in config file!\n");
    }

    if (NUM_READERS > NUM_BLOCKS) {
        fprintf(stderr,
                "[Rockstar error] NUM_READERS must be <= NUM_BLOCKS in config file.\n");
        exit(1);
    }

    if ((strncmp(OUTPUT_FORMAT, "ASCII", 5) == 0) && STRICT_SO_MASSES) {
        fprintf(stderr, "[Rockstar warning] STRICT_SO_MASSES requires binary outputs; "
                        "setting OUTPUT_FORMAT=BOTH.\n");
        OUTPUT_FORMAT = "BOTH";
    }
}

void do_config(char *filename) {
  struct configfile c = {0};

  if (filename && strlen(filename)){
#ifdef DO_CONFIG_MPI
    int my_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    if( my_rank == 0)  load_config(&c, filename);
    MPI_Bcast( &c.num_entries, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if( my_rank != 0){
      c.keys = (char **)malloc( sizeof(char *)*(c.num_entries+10));
      c.values = (char **)malloc( sizeof(char *)*(c.num_entries+10));
      c.touched = (int *)malloc( sizeof(int)*(c.num_entries+10));
    }
    MPI_Bcast( c.touched, c.num_entries, MPI_INT, 0, MPI_COMM_WORLD);
    int i, len;
    for( i=0; i<c.num_entries; i++){
      if( my_rank == 0)  len = strlen( c.keys[i]) + 1;
      MPI_Bcast( &len, 1, MPI_INT, 0, MPI_COMM_WORLD);
      if( my_rank != 0)  c.keys[i] = (char *)malloc( sizeof(char)*len);
      MPI_Bcast( c.keys[i], len, MPI_CHAR, 0, MPI_COMM_WORLD);
      if( my_rank == 0)  len = strlen( c.values[i]) + 1;
      MPI_Bcast( &len, 1, MPI_INT, 0, MPI_COMM_WORLD);
      if( my_rank != 0)  c.values[i] = (char *)malloc( sizeof(char)*len);
      MPI_Bcast( c.values[i], len, MPI_CHAR, 0, MPI_COMM_WORLD);
      //fprintf( stderr, "%d\t%s\t%s\t%d\n", my_rank, c.keys[i], c.values[i], c.touched[i]);
    }
#else
    load_config(&c, filename);
#endif
  }

#define string(a, b)  a = config_to_string(&c, #a, b)
#define real(a, b)    a = config_to_real(&c, #a, b)
#define real3(a, b)   config_to_real3(&c, #a, a, b)
#define integer(a, b) a = config_to_real(&c, #a, b)
#include "config.template.h"
#undef string
#undef real
#undef real3
#undef integer
  
  syntax_check(&c, "[Rockstar warning]");
  setup_config();
  free_config(c);
  if (filename && strlen(filename)) {
    free(CONFIG_FILENAME);
    CONFIG_FILENAME = strdup(filename);
  }
}


void output_config(const char *filename) {
    make_directory_hir(OUTBASE);
    char  buffer[1024];
    FILE *output;
    if (!filename)
        filename = "rockstar.cfg";
    snprintf(buffer, 1024, "%s/%s", OUTBASE, filename);
    output = check_fopen(buffer, "w");

#define string(a, b) fprintf(output, "%s = \"%s\"\n", #a, a);
#define real(a, b)   fprintf(output, "%s = %g\n", #a, a);
#define real3(a, b)                                                            \
    fprintf(output, "%s = (%g, %g, %g)\n", #a, a[0], a[1], a[2]);
#define integer(a, b) fprintf(output, "%s = %" PRId64 "\n", #a, a);
#include "config.template.h"

    fclose(output);
}

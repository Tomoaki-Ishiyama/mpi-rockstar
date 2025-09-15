#include "para_qsort.hpp"

#include <algorithm>
#include <mpi.h>
#include <omp.h>
#include <numeric>
#include <random>
#include <vector>
#include <cassert>
#include <cstring>
#include <cstdarg>
#include <cstddef>
#include <ctime>
#include <cstdio>
#include <sys/stat.h>

#include "error.h"
extern "C" {
#include "config_vars.h"
#include "check_syscalls.h"
#include "merger.h"
#include "config.h"
#include "io/meta_io.h"
#include "io/io_util.h"
#include "io/io_bgc2.h"
#ifdef ENABLE_HDF5
#include "io/io_internal_hdf5.h"
#endif
#include "rockstar.h"
#include "bounds.h"
#include "distance.h"
#include "universe_time.h"
#include "interleaving.h"
#include "fun_times.h"
#include "merger.h"
#include "groupies.h"
#include "bitarray.h"
#include "version.h"
}

#include "mpi_rockstar.h"

#include "mpi_rockstar.h"

#define CLIENT_DEBUG 0
FILE  *profile_out = NULL;
double time_start;

#ifdef OUTPUT_INTERMEDIATE_AXIS
#ifdef OUTPUT_INERTIA_TENSOR
#error
#endif
#endif


int64_t accumulate(int64_t n){

  int64_t n_total = 0;
  MPI_Allreduce(&n, &n_total, 1, MPI_LONG_LONG_INT, MPI_SUM, MPI_COMM_WORLD);

  return n_total;

}


void timed_output(const char *format, ...) {
    MPI_Barrier(MPI_COMM_WORLD);
    int my_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    if (my_rank == 0) {
        std::va_list args;
        va_start(args, format);
        auto elapsed_time = static_cast<int64_t>(MPI_Wtime() - time_start);
        fprintf(stderr, "[%6" PRId64 "s] ", elapsed_time);
        vfprintf(stderr, format, args);
        va_end(args);
    }
}

template <typename T> T *reallocate(T *ptr, size_t num_elements) {
    const auto size = sizeof(T) * num_elements;
    if (size == 0) {
        std::free(ptr);
        return nullptr;
    }

    const auto res = static_cast<T *>(std::realloc(ptr, size));
    if (res == nullptr) {
        const auto size_in_mib = static_cast<double>(size) / (1024 * 1024);
        fprintf(stderr, "[Rockstar error] Failed to allocate %.2f MiB of memory!\n",
                size_in_mib);
        exit(1);
    }
    return res;
}

template <typename T> T *allocate(size_t num_elements) {
    return reallocate<T>(nullptr, num_elements);
}

template <typename Size> void clear_counts(Size counts[], size_t n) {
    std::fill_n(counts, n, 0);
}

template <typename Size>
Size calc_displs(const Size counts[], size_t n, Size displs[]) {
    displs[0] = 0;
    for (size_t i = 1; i < n; ++i) {
        displs[i] = displs[i - 1] + counts[i - 1];
    }
    return displs[n - 1] + counts[n - 1];
}

template <typename T>
int64_t exchange_data(const T send_data[], int counts[], T *&recv_data,
                      const int recv_offset, MPI_Datatype mpi_data_type) {
    static auto recv_counts = allocate<int>(NUM_WRITERS);
    static auto send_displs = allocate<int>(NUM_WRITERS);
    static auto recv_displs = allocate<int>(NUM_WRITERS);

    MPI_Alltoall(counts, 1, MPI_INT, recv_counts, 1, MPI_INT, MPI_COMM_WORLD);

    calc_displs(counts, NUM_WRITERS, send_displs);
    const auto num_to_recv = calc_displs(recv_counts, NUM_WRITERS, recv_displs);
    recv_data              = reallocate(recv_data, recv_offset + num_to_recv);

    MPI_Alltoallv(send_data, counts, send_displs, mpi_data_type,
                  recv_data + recv_offset, recv_counts, recv_displs,
                  mpi_data_type, MPI_COMM_WORLD);

    std::copy_n(recv_counts, NUM_WRITERS, counts);

    return num_to_recv;
}



void createDivision( const int n, int *ndiv){

  int nx, ny, nz;
  int n0, n1;
  n0 = (int)pow(n+0.1,0.33333333333333333333);
  while(n%n0)n0--;
  nx = n0;
  n1 = n/nx;
  n0 = (int)sqrt(n1+0.1);
  while(n1%n0)n0++;
  ny = n0; nz = n1/n0;
  int ntmp;
  if (nz > ny){
    ntmp = nz; nz = ny; ny = ntmp;
  }
  if (ny > nx){
    ntmp = nx; nx = ny; ny = ntmp;
  }
  if (nz > ny){
    ntmp = nz; nz = ny; ny = ntmp;
  }

  ndiv[0] = nx;
  ndiv[1] = ny;
  ndiv[2] = nz;

}


void check_num_writers(void) {
    if (NUM_WRITERS == 1) {
        if (PERIODIC) {
            fprintf(stderr,
                    "[Warning] Setting PERIODIC=0 since NUM_WRITERS=1.\n");
            fprintf(stderr, "[Warning] To enable periodic boundary conditions, "
                            "increase NUM_WRITERS to at least 8.\n");
            PERIODIC = 0;
        }
        return;
    } else {
        int factors[3] = {0};
        //MPI_Dims_create(NUM_WRITERS, 3, factors);
	createDivision( NUM_WRITERS, factors);
        if (((factors[0] < 2) || (factors[1] < 2) || (factors[2] < 2)) && (PERIODIC)) {
            fprintf(stderr,
                    "[Rockstar error] NUM_WRITERS should be the product of at least "
                    "three factors larger than 1 for periodic boundary "
                    "conditions to be enabled!\n");
            fprintf(stderr,
                    "[Rockstar error] (Currently, NUM_WRITERS = %" PRId64
                    " = %d x %d x %d)\n",
                    NUM_WRITERS, factors[0], factors[1], factors[2]);
            fprintf(stderr,
                    "[Rockstar error] Please adjust NUM_WRITERS or set PERIODIC=0 "
                    "in the config file.\n");
            exit(1);
        }
    }
}

template <typename RankConditionFunc>
MPI_Comm create_new_comm(RankConditionFunc rank_condition) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int      color = rank_condition(rank) ? 1 : MPI_UNDEFINED;
    MPI_Comm new_comm;
    MPI_Comm_split(MPI_COMM_WORLD, color, rank, &new_comm);
    return new_comm;
}

int get_reader_rank() {
    int  reader_rank      = -1;
    auto is_reader        = [](int rank) { return rank < NUM_READERS; };
    auto mpi_comm_readers = create_new_comm(is_reader);
    if (mpi_comm_readers != MPI_COMM_NULL) {
        MPI_Comm_rank(mpi_comm_readers, &reader_rank);
        MPI_Comm_free(&mpi_comm_readers);
    }
    return reader_rank;
}

void read_blocks(int64_t snap, int reader_rank, char buffer[]) {
    if (reader_rank != -1) {
        int64_t block_start, to_read;
        particle_range(NUM_BLOCKS, reader_rank, NUM_READERS, &block_start,
                       &to_read);
        const auto block_end = block_start + to_read;

        for (auto block = block_start; block < block_end; block++) {
            if (LIGHTCONE && strlen(LIGHTCONE_ALT_SNAPS) &&
                block >= (NUM_BLOCKS / 2)) {
                if (LIGHTCONE == 1)
                    read_input_names(LIGHTCONE_ALT_SNAPS, &snapnames,
                                     &NUM_SNAPS);
                LIGHTCONE = 2;
                get_input_filename(buffer, 1024, snap,
                                   block - (NUM_BLOCKS / 2));
            } else {
                if (LIGHTCONE == 2) {
                    LIGHTCONE = 1;
                    read_input_names(SNAPSHOT_NAMES, &snapnames, &NUM_SNAPS);
                }
                get_input_filename(buffer, 1024, snap, block);
            }
            read_particles(buffer);
            if (!block)
                output_config(NULL);
        }
    }

    timed_output("%" PRId64 " procs read %" PRId64
                 " blocks for snapshot %" PRId64 ".\n",
                 NUM_READERS, NUM_BLOCKS, snap);
}

void sync_config() {
    double config[] = {
        PARTICLE_MASS, AVG_PARTICLE_SPACING, SCALE_NOW, BOX_SIZE, Ol, Om, h0,
        TRIM_OVERLAP,  ROUND_AFTER_TRIM};

    MPI_Bcast(config, sizeof(config) / sizeof(double), MPI_DOUBLE, 0,
              MPI_COMM_WORLD);

    PARTICLE_MASS        = config[0];
    AVG_PARTICLE_SPACING = config[1];
    SCALE_NOW            = config[2];
    BOX_SIZE             = config[3];
    Ol                   = config[4];
    Om                   = config[5];
    h0                   = config[6];
    TRIM_OVERLAP         = config[7];
    ROUND_AFTER_TRIM     = config[8];

    if (strlen(LIGHTCONE_ALT_SNAPS)) {
        int64_t i;
        for (i = 0; i < 3; i++)
            if (LIGHTCONE_ORIGIN[i] || LIGHTCONE_ALT_ORIGIN[i])
                break;
        if (i == 3) {
            for (i = 0; i < 3; i++)
                LIGHTCONE_ORIGIN[i] = LIGHTCONE_ALT_ORIGIN[i] = BOX_SIZE / 2.0;
        }
    }

    int my_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    if (my_rank == 0) {
        if ((BOX_SIZE < OVERLAP_LENGTH * 5) && PERIODIC) {
            fprintf(stderr,
                    "[Rockstar error] Box size too small (%f) relative to overlap "
                    "length "
                    "(%f)!\n",
                    BOX_SIZE, OVERLAP_LENGTH);
            exit(1);
        }
    }
}

void calc_particle_bounds(float *bounds) {
    int64_t i, j;
    for (j = 0; j < 6; j++)
        bounds[j] = 0;
    if (!num_p)
        return;
    memcpy(bounds, p[0].pos, sizeof(float) * 3);
    memcpy(bounds + 3, p[0].pos, sizeof(float) * 3);
    for (i = 1; i < num_p; i++) {
        for (j = 0; j < 3; j++) {
            if (bounds[j] > p[i].pos[j])
                bounds[j] = p[i].pos[j];
            if (bounds[j + 3] < p[i].pos[j])
                bounds[j + 3] = p[i].pos[j];
        }
    }
}

void calc_particle_bounds_periodic(float *bounds) {
    int64_t i, j;
    for (j = 0; j < 6; j++)
        bounds[j] = 0;
    if (!num_p)
        return;
    memcpy(bounds, p[0].pos, sizeof(float) * 3);
    memcpy(bounds + 3, p[0].pos, sizeof(float) * 3);
    for (i = 1; i < num_p; i++) {
        for (j = 0; j < 3; j++) {
            float pos = p[i].pos[j];
            if (p[0].pos[j] - pos > BOX_SIZE / 2.0)
                pos += BOX_SIZE;
            else if (p[0].pos[j] - pos < -BOX_SIZE / 2.0)
                pos -= BOX_SIZE;
            if (bounds[j] > pos)
                bounds[j] = pos;
            if (bounds[j + 3] < pos)
                bounds[j + 3] = pos;
        }
    }
}

void trim_particles(float *bounds) {
    int64_t i;
    if (!TRIM_OVERLAP)
        return;
    for (i = 0; i < 3; i++) {
        bounds[i] += TRIM_OVERLAP;
        bounds[i + 3] -= TRIM_OVERLAP;
    }

    if (ROUND_AFTER_TRIM)
        for (i = 0; i < 6; i++)
            bounds[i] = ((int64_t)(bounds[i] / ROUND_AFTER_TRIM + 0.5)) *
                        ROUND_AFTER_TRIM;

    for (i = 0; i < num_p; i++)
        if (!_check_bounds_raw(p[i].pos, bounds)) {
            num_p--;
            p[i] = p[num_p];
            i--;
        }

    p = reallocate(p, num_p);
}

void decide_reader_bounds(float *reader_bounds, int reader_rank) {
    if (reader_rank != -1) {
        // Handle the case where the positions round to BOX_SIZE
        // due to floating-point precision
#if 1
      for (int64_t i = 0; i < num_p; i++) {
            for (int64_t j = 0; j < 3; j++) {
                if (p[i].pos[j] == static_cast<float>(BOX_SIZE)) {
                    p[i].pos[j] = 0;
                }
            }
      }
#endif
        calc_particle_bounds(reader_bounds);
        if (TRIM_OVERLAP)
            trim_particles(reader_bounds);
    }
}

void decide_chunks_for_volume_balance(const int chunks[],
                                      float (*writer_bounds)[6]) {
    int64_t i, n, idx[3];
    float   bounds[6];
    float   chunk_size[3];
    for (i = 0; i < 3; i++)
        chunk_size[i] = BOX_SIZE / (float)chunks[i];
    for (n = 0; n < NUM_WRITERS; n++) {
        idx[0] = n % chunks[0];
        idx[1] = ((int64_t)(n / chunks[0])) % chunks[1];
        idx[2] = n / (chunks[0] * chunks[1]);
        for (i = 0; i < 3; i++) {
            bounds[i]     = idx[i] * chunk_size[i];
            bounds[i + 3] = (idx[i] + 1 == chunks[i])
                                ? BOX_SIZE
                                : bounds[i] + chunk_size[i];
        }
        memcpy(writer_bounds[n], bounds, sizeof(float) * 6);
    }
}

void align_domain_particles(int      axis, const float (*all_samples)[3],
                            int64_t *particle_indices, int64_t num_particles,
                            int64_t first_proc, int64_t num_procs,
                            const int chunks[], float (*writer_bounds)[6]) {
    if (axis == 3) {
        return;
    }

    if (num_particles < chunks[axis]) {
        int my_rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
        if (my_rank == 0) {
            fprintf(stderr,
                    "[Rockstar error] Only %" PRId64
                    " sample particles available for %d chunks along axis %d.\n",
                    num_particles, chunks[axis], axis);
        }
        exit(2);
    }

    std::sort(particle_indices, particle_indices + num_particles,
              //[all_samples = all_samples, axis = axis](int64_t a, int64_t b) {
              [all_samples, axis](int64_t a, int64_t b) {
                  return all_samples[a][axis] < all_samples[b][axis];
              });

    const auto particles_per_group = num_particles / chunks[axis];
    const auto extra_particles     = num_particles % chunks[axis];

    const auto procs_per_group = num_procs / chunks[axis];
    const auto extra_procs     = num_procs % chunks[axis];

    auto particle_begin = particle_indices;
    auto proc_begin     = first_proc;

    for (int i = 0; i < chunks[axis]; i++) {
        const auto num_group_particles =
            particles_per_group + (i < extra_particles ? 1 : 0);
        const auto num_group_procs =
            procs_per_group + (i < extra_procs ? 1 : 0);

        const auto particle_end = particle_begin + num_group_particles;
        const auto proc_end     = proc_begin + num_group_procs;

        float min = (i == 0) ? 0
                             : 0.5 * (all_samples[*(particle_begin - 1)][axis] +
                                      all_samples[*particle_begin][axis]);
        float max = (i == chunks[axis] - 1)
                        ? BOX_SIZE
                        : 0.5 * (all_samples[*(particle_end - 1)][axis] +
                                 all_samples[*particle_end][axis]);

        for (auto j = proc_begin; j < proc_end; j++) {
            writer_bounds[j][axis]     = min;
            writer_bounds[j][axis + 3] = max;
        }

        particle_begin = particle_end;
        proc_begin     = proc_end;
    }

    particle_begin = particle_indices;
    proc_begin     = first_proc;

    for (int i = 0; i < chunks[axis]; i++) {
        const auto num_group_particles =
            particles_per_group + (i < extra_particles ? 1 : 0);
        const auto num_group_procs =
            procs_per_group + (i < extra_procs ? 1 : 0);

        align_domain_particles(axis + 1, all_samples, particle_begin,
                               num_group_particles, proc_begin, num_group_procs,
                               chunks, writer_bounds);

        particle_begin += num_group_particles;
        proc_begin += num_group_procs;
    }
}

void decide_chunks_for_memory_balance(const int chunks[],
                                      float (*writer_bounds)[6]) {
    int num_procs;
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    int my_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    int64_t num_local_samples = num_p / (10 * num_procs);
    // int64_t     num_local_samples = num_p;
    auto local_samples = allocate<float[3]>(num_local_samples);

    std::mt19937_64                        gen(num_p);
    std::uniform_int_distribution<int64_t> dist(0, num_p - 1);
    for (int64_t i = 0; i < num_local_samples; i++) {
        int64_t j = (num_local_samples == num_p) ? i : dist(gen);
        for (int64_t k = 0; k < 3; k++) {
            local_samples[i][k] = p[j].pos[k];
        }
    }

    auto recv_counts = allocate<int>(NUM_WRITERS);
    auto recv_displs = allocate<int>(NUM_WRITERS);

    int num_to_send = 3 * num_local_samples;
    MPI_Allgather(&num_to_send, 1, MPI_INT, recv_counts, 1, MPI_INT,
                  MPI_COMM_WORLD);
    auto num_all_samples =
        calc_displs(recv_counts, NUM_WRITERS, recv_displs) / 3;

    if (num_all_samples < NUM_WRITERS) {
        if (my_rank == 0) {
            fprintf(stderr,
                    "[Rockstar error] Only %" PRId64
                    " sample particles available for %" PRId64
                    " writers; at least %" PRId64
                    " samples are required to compute domain bounds.\n",
                    num_all_samples, NUM_WRITERS,
                    (int64_t)NUM_WRITERS);
        }
        exit(2);
    }
    auto all_samples = allocate<float[3]>(num_all_samples);

    MPI_Allgatherv(local_samples, num_to_send, MPI_FLOAT, all_samples,
                   recv_counts, recv_displs, MPI_FLOAT, MPI_COMM_WORLD);
    local_samples = reallocate(local_samples, 0);
    recv_counts   = reallocate(recv_counts, 0);
    recv_displs   = reallocate(recv_displs, 0);

    auto particle_indices = allocate<int64_t>(num_all_samples);
    std::iota(particle_indices, particle_indices + num_all_samples, 0);

    align_domain_particles(0, all_samples, particle_indices, num_all_samples, 0,
                           NUM_WRITERS, chunks, writer_bounds);

    all_samples      = reallocate(all_samples, 0);
    particle_indices = reallocate(particle_indices, 0);
}





void decide_writer_bounds(float (*writer_bounds)[6], const int my_rank) {
    int chunks[3] = {0};
    //MPI_Dims_create(NUM_WRITERS, 3, chunks);
    createDivision( NUM_WRITERS, chunks);

    // if (strlen(LOAD_BALANCE_SCRIPT)) decide_chunks_by_script();
    // else if (NUM_WRITERS == 1) decide_chunks_for_volume_balance();
    // else decide_chunks_for_memory_balance();
    /*
    int tmp = chunks[0];
    chunks[0] = chunks[2];
    chunks[2] = tmp;
    */

    decide_chunks_for_memory_balance(chunks, writer_bounds);

    if( my_rank == 0){
        if (CLIENT_DEBUG) fprintf( stderr, "#chunks=(%d,%d,%d)\n", chunks[0], chunks[1], chunks[2]);
    }
}

MPI_Datatype create_mpi_particle_type() {

    MPI_Datatype mpi_particle_type;
#if 0
    int          block_lengths[] = {1, 6};
    MPI_Aint     displacements[] = {offsetof(struct particle, id),
                                    offsetof(struct particle, pos)};
    MPI_Datatype types[]         = {MPI_INT64_T, MPI_FLOAT};
    MPI_Type_create_struct(2, block_lengths, displacements, types,
                           &mpi_particle_type);
#else
    MPI_Type_contiguous( sizeof(struct particle), MPI_CHAR, &mpi_particle_type);
#endif
    MPI_Type_commit(&mpi_particle_type);
    return mpi_particle_type;

}

MPI_Datatype create_mpi_bparticle_type() {

    MPI_Datatype mpi_bparticle_type;
#if 0
    int          block_lengths[] = {1, 6, 1, 1};
    MPI_Aint     displacements[] = {
        offsetof(struct bparticle, id), offsetof(struct bparticle, pos),
        offsetof(struct bparticle, bgid), offsetof(struct bparticle, chunk)};
    MPI_Datatype types[] = {MPI_INT64_T, MPI_FLOAT, MPI_INT64_T, MPI_INT64_T};
    MPI_Type_create_struct(4, block_lengths, displacements, types,
                           &mpi_bparticle_type);
#else
    MPI_Type_contiguous( sizeof(struct bparticle), MPI_CHAR, &mpi_bparticle_type);
#endif
    MPI_Type_commit(&mpi_bparticle_type);
    return mpi_bparticle_type;

}

MPI_Datatype create_mpi_bgroup_type() {

  MPI_Datatype mpi_bgroup_type;
#if 0
  int          block_lengths[] = {6};
  MPI_Aint     displacements[] = {offsetof(struct bgroup, id)};
  MPI_Datatype types[]         = {MPI_INT64_T};
  MPI_Type_create_struct(1, block_lengths, displacements, types,
			 &mpi_bgroup_type);
#else
  MPI_Type_contiguous( sizeof(struct bgroup), MPI_CHAR, &mpi_bgroup_type);
#endif
  MPI_Type_commit(&mpi_bgroup_type);
    return mpi_bgroup_type;

}

MPI_Datatype create_mpi_halo_type() {

    MPI_Datatype mpi_halo_type;
    MPI_Type_contiguous( sizeof(struct halo), MPI_CHAR, &mpi_halo_type);
    MPI_Type_commit(&mpi_halo_type);
    return mpi_halo_type;

}

MPI_Datatype create_mpi_ehi_type() {

    MPI_Datatype mpi_ehi_type;
    MPI_Type_contiguous( sizeof(struct extra_halo_info), MPI_CHAR, &mpi_ehi_type);
    MPI_Type_commit(&mpi_ehi_type);
    return mpi_ehi_type;

}

struct complete_halo {
    struct halo            h;
    struct extra_halo_info ehi;
    complete_halo(struct halo h_, struct extra_halo_info ehi_)
        : h(h_), ehi(ehi_) {}
};


MPI_Datatype create_mpi_complete_halo_type(MPI_Datatype mpi_halo_type,
                                           MPI_Datatype mpi_ehi_type) {
    MPI_Datatype mpi_complete_halo_type;
    MPI_Type_contiguous( sizeof(struct complete_halo), MPI_CHAR, &mpi_complete_halo_type);
    MPI_Type_commit(&mpi_complete_halo_type);
    return mpi_complete_halo_type;

}


MPI_Datatype create_mpi_eparticle_type() {

    MPI_Datatype mpi_eparticle_type;
    MPI_Type_contiguous( sizeof(struct extended_particle), MPI_CHAR, &mpi_eparticle_type);
    MPI_Type_commit(&mpi_eparticle_type);
    return mpi_eparticle_type;

}


void transfer_particles(int my_reader_rank, float *my_reader_bounds,
                        float (*writer_bounds)[6]) {
    auto send_counts = allocate<int>(NUM_WRITERS);
    clear_counts(send_counts, NUM_WRITERS);
    struct particle *send_buffer = nullptr;

    if (my_reader_rank != -1) {
        std::vector<int64_t> recipients;
        for (int64_t i = 0; i < NUM_WRITERS; i++) {
            float bounds[6];
            //if (bounds_overlap(my_reader_bounds, writer_bounds[i], bounds, 0)) {
	    if (bounds_overlap(my_reader_bounds, writer_bounds[i], bounds, FORCE_RES)) {
                recipients.emplace_back(i);
            }
        }

        auto dest_procs = allocate<int64_t>(num_p);
#if 0
        for (int64_t i = 0; i < num_p; i++) {
            for (auto j : recipients) {
                if (_check_bounds_raw(p[i].pos, writer_bounds[j])) {
                    dest_procs[i] = j;
                    ++send_counts[j];
                    break;
                }
            }
        }
#else
#pragma omp parallel for
	for (int64_t i = 0; i < num_p; i++) {
            for (auto j : recipients) {
                if (_check_bounds_raw(p[i].pos, writer_bounds[j])) {
                    dest_procs[i] = j;
                    break;
                }
            }
        }

	for (int64_t i = 0; i < num_p; i++){
	  send_counts[dest_procs[i]] ++;
	}
#endif
        std::vector<int64_t>().swap(recipients);

        auto send_displs = allocate<int>(NUM_WRITERS);
        auto num_to_send = calc_displs(send_counts, NUM_WRITERS, send_displs);
        assert(num_to_send == num_p);
        send_buffer = allocate<struct particle>(num_to_send);

        for (int64_t i = 0; i < num_p; i++) {
            send_buffer[send_displs[dest_procs[i]]++] = p[i];
        }
        send_displs = reallocate(send_displs, 0);
        dest_procs  = reallocate(dest_procs, 0);
    }
    p     = reallocate(p, 0);
    num_p = 0;
    timed_output("Transferring0...\n");

    auto mpi_particle_type = create_mpi_particle_type();
    num_p = exchange_data(send_buffer, send_counts, p, 0, mpi_particle_type);
    MPI_Type_free(&mpi_particle_type);
    send_buffer = reallocate(send_buffer, 0);
    send_counts = reallocate(send_counts, 0);

    timed_output("Transferring particles to writers...\n");
}

void transfer_particles_mem_save(int my_reader_rank, float *my_reader_bounds,
                                 float (*writer_bounds)[6]) {
    auto send_counts = allocate<int>(NUM_WRITERS);
    clear_counts(send_counts, NUM_WRITERS);

    // struct particle *send_buffer = nullptr;
    struct particle *recv_buffer = nullptr;

    if (my_reader_rank != -1) {
        std::vector<int64_t> recipients;
        for (int64_t i = 0; i < NUM_WRITERS; i++) {
            float bounds[6];
            if (bounds_overlap(my_reader_bounds, writer_bounds[i], bounds, 0)) {
                recipients.emplace_back(i);
            }
        }

        // auto dest_procs = allocate<int64_t>(num_p);
        auto dest_procs = allocate<int>(num_p);
#pragma omp parallel for
        for (int64_t i = 0; i < num_p; i++) {
            for (auto j : recipients) {
                if (_check_bounds_raw(p[i].pos, writer_bounds[j])) {
                    dest_procs[i] = j;
                    break;
                }
            }
        }

        for (int64_t i = 0; i < num_p; i++) {
            send_counts[dest_procs[i]]++;
        }

        std::vector<int64_t>().swap(recipients);

        /*
        auto send_displs = allocate<int>(NUM_WRITERS);
        auto num_to_send = calc_displs(send_counts, NUM_WRITERS, send_displs);
        assert(num_to_send == num_p);

                send_buffer = allocate<struct particle>(num_to_send);
                for (int64_t i = 0; i < num_p; i++) {
                    send_buffer[send_displs[dest_procs[i]]++] = p[i];
                }
        */

        // in-place sort by key
        // single_qsort_key(p, dest_procs, 0, num_p - 1);
        para_qsort_key(p, dest_procs, 0, num_p - 1);

        dest_procs = reallocate(dest_procs, 0);
    }

    // p   = reallocate(p, 0);
    num_p = 0;
    timed_output("Transferring...\n");

    recv_buffer = allocate<struct particle>(1024); // temporary

    auto mpi_particle_type = create_mpi_particle_type();
    num_p = exchange_data(p, send_counts, recv_buffer, 0, mpi_particle_type);
    MPI_Type_free(&mpi_particle_type);
    send_counts = reallocate(send_counts, 0);

    p = reallocate(p, 0);
    p = recv_buffer;
    // recv_buffer = reallocate(recv_buffer, 0);

    timed_output("Transferring particles to writers...\n");
}

void transfer_bparticles(int64_t my_rank, float (*writer_bounds)[6]) {
    auto send_counts = allocate<int>(NUM_WRITERS);
    clear_counts(send_counts, NUM_WRITERS);
    std::vector<struct bparticle> send_buffer;

    for (int64_t i = 0; i < NUM_WRITERS; i++) {
        if (i != my_rank) {
            float expanded_bounds[6];
            if (bounds_overlap(
                    writer_bounds[my_rank], writer_bounds[i], expanded_bounds,
                    1.01 * AVG_PARTICLE_SPACING * FOF_LINKING_LENGTH)) {
                for (int64_t j = 0; j < num_bp; j++) {
                    float dummy_pos[3];
                    if (_check_bounds(bp[j].pos, dummy_pos, expanded_bounds)) {
                        send_buffer.emplace_back(bp[j]);
                        ++send_counts[i];
                    }
                }
            }
        }
    }

    auto mpi_bparticle_type = create_mpi_bparticle_type();
    num_new_bp = exchange_data(send_buffer.data(), send_counts, bp, num_bp,
                               mpi_bparticle_type);
    MPI_Type_free(&mpi_bparticle_type);
    std::vector<struct bparticle>().swap(send_buffer);
    send_counts = reallocate(send_counts, 0);
    num_bp += num_new_bp;

    timed_output("Transferring boundary particles between writers...\n");
}

void distribute_halos(float (*writer_bounds)[6]) {
    auto halo_counts = allocate<int>(NUM_WRITERS);
    clear_counts(halo_counts, NUM_WRITERS);
    auto send_part_counts = allocate<int>(NUM_WRITERS);
    clear_counts(send_part_counts, NUM_WRITERS);
    std::vector<struct complete_halo> send_complete_halos;
    std::vector<struct particle>      send_particles;

    auto index_conversion = allocate<int64_t>(num_halos);
    for (int64_t i = 0; i < NUM_WRITERS; i++) {
        for (int64_t j = 0; j < num_halos; j++) {
            halos[j].flags -= (halos[j].flags & TAGGED_FLAG);
        }
        for (int64_t j = 0; j < num_halos; j++) {
            if (halos[j].flags & TAGGED_FLAG)
                continue;
            if (_check_bounds_raw(halos[j].pos, writer_bounds[i]))
                halo_counts[i] += tag_halo_as_in_bounds(halos + j);
        }
        for (int64_t j = 0; j < num_halos; j++) {
            if (halos[j].flags & TAGGED_FLAG) {
                send_part_counts[i] += halos[j].num_p;
            }
        }

        send_complete_halos.reserve(send_complete_halos.size() +
                                    halo_counts[i]);
        send_particles.reserve(send_particles.size() + send_part_counts[i]);

        std::fill_n(index_conversion, num_halos, -1);

        const auto h_start = send_complete_halos.size();
        for (int64_t j = 0; j < num_halos; j++) {
            if ((halos[j].flags & TAGGED_FLAG)) {
                const auto halo_index = send_complete_halos.size();
                send_complete_halos.emplace_back(halos[j], extra_info[j]);
                index_conversion[j] = halo_index - h_start;

                send_complete_halos[halo_index].h.p_start =
                    send_particles.size();
                send_particles.insert(send_particles.end(),
                                      p + halos[j].p_start,
                                      p + halos[j].p_start + halos[j].num_p);
            }
        }

#define IC(x) (x = ((x > -1) && (x < num_halos)) ? index_conversion[x] : -1)
        for (auto j = h_start; j < send_complete_halos.size(); j++) {
            IC(send_complete_halos[j].ehi.sub_of); // Note that parent may not
                                                   // exist for this chunk
            IC(send_complete_halos[j].ehi.child);
            IC(send_complete_halos[j].ehi.next_cochild);
            IC(send_complete_halos[j].ehi.prev_cochild);
        }
#undef IC
    }
    index_conversion = reallocate(index_conversion, 0);
    extra_info       = reallocate(extra_info, 0);

    // Store local particles not belonging to any halo in p[num_p]
    const auto total_num_p = num_p;
    auto       bitarray    = BIT_ALLOC(total_num_p);
    BIT_ALL_CLEAR(bitarray, total_num_p);
    for (int64_t i = 0; i < num_halos; i++) {
        for (auto j = 0; j < halos[i].num_p; j++) {
            BIT_SET(bitarray, halos[i].p_start + j);
        }
    }
    halos     = reallocate(halos, 0);
    num_halos = 0;
    num_p     = 0;
    for (int64_t i = 0; i < total_num_p; i++) {
        if (!BIT_TST(bitarray, i)) {
            p[num_p++] = p[i];
        }
    }
    bitarray = reallocate(bitarray, 0);
    p        = reallocate(p, num_p);

    auto mpi_halo_type = create_mpi_halo_type();
    auto mpi_ehi_type  = create_mpi_ehi_type();
    auto mpi_complete_halo_type =
        create_mpi_complete_halo_type(mpi_halo_type, mpi_ehi_type);
    struct complete_halo *recv_complete_halos = nullptr;
    num_halos = exchange_data(send_complete_halos.data(), halo_counts,
                              recv_complete_halos, 0, mpi_complete_halo_type);
    MPI_Type_free(&mpi_complete_halo_type);
    MPI_Type_free(&mpi_halo_type);
    MPI_Type_free(&mpi_ehi_type);
    std::vector<struct complete_halo>().swap(send_complete_halos);

    halos      = reallocate(halos, num_halos);
    extra_info = reallocate(extra_info, num_halos);
    for (int64_t i = 0; i < num_halos; i++) {
        halos[i]      = recv_complete_halos[i].h;
        extra_info[i] = recv_complete_halos[i].ehi;
    }
    recv_complete_halos = reallocate(recv_complete_halos, 0);

    int64_t hstart = 0;
    for (int64_t i = 0; i < NUM_WRITERS; i++) {
        for (auto j = hstart; j < hstart + halo_counts[i]; j++) {
            if (extra_info[j].child >= 0)
                extra_info[j].child += hstart;
            if (extra_info[j].next_cochild >= 0)
                extra_info[j].next_cochild += hstart;
            if (extra_info[j].prev_cochild >= 0)
                extra_info[j].prev_cochild += hstart;
            if (extra_info[j].sub_of >= 0)
                extra_info[j].sub_of += hstart;
        }
        hstart += halo_counts[i];
    }
    halo_counts = reallocate(halo_counts, 0);

    auto mpi_particle_type = create_mpi_particle_type();
    num_additional_p = exchange_data(send_particles.data(), send_part_counts, p,
                                     num_p, mpi_particle_type);
    MPI_Type_free(&mpi_particle_type);
    std::vector<struct particle>().swap(send_particles);
    send_part_counts = reallocate(send_part_counts, 0);

    int64_t cur_p = num_p;
    for (int64_t i = 0; i < num_halos; i++) {
        halos[i].p_start = cur_p;
        cur_p += halos[i].num_p;
    }
}

void distribute_sets(int my_rank) {
    auto send_set_counts = allocate<int>(NUM_WRITERS);
    clear_counts(send_set_counts, NUM_WRITERS);
    auto send_group_counts = allocate<int>(NUM_WRITERS);
    clear_counts(send_group_counts, NUM_WRITERS);

    auto set_chunks = allocate<int64_t>(num_bg_sets);

    // Sort bg sets by the chunk with max # of particles
    int64_t j = 0;
    for (int64_t i = 0; i < num_bg_sets; i++) {
        int64_t max_c = final_bg[j].chunk;
        int64_t max_p = final_bg[j].num_p;
        int64_t cur_c = max_c, cur_p = 0;

        assert(bg_set_sizes[i]);
        auto sort_by_chunk = [](const struct bgroup &a,
                                const struct bgroup &b) {
            if (a.chunk == b.chunk)
                return a.id < b.id;
            else
                return a.chunk < b.chunk;
        };
        std::sort(final_bg + j, final_bg + j + bg_set_sizes[i], sort_by_chunk);
        assert(final_bg[j].chunk == my_rank);

        cur_c = final_bg[j].chunk;
        for (auto k = j; k < j + bg_set_sizes[i]; k++) {
            if (cur_c != final_bg[k].chunk) {
                if (cur_p > max_p) {
                    max_c = cur_c;
                    max_p = cur_p;
                }
                cur_c = final_bg[k].chunk;
                cur_p = 0;
            }
            cur_p += final_bg[k].num_p;
        }
        if (cur_p > max_p)
            max_c = cur_c;
        set_chunks[i] = max_c;
        ++send_set_counts[max_c];
        send_group_counts[max_c] += bg_set_sizes[i];
        j += bg_set_sizes[i];
    }

    int64_t total_bg = 0;
    for (int64_t i = 0; i < num_bg_sets; i++)
        total_bg += bg_set_sizes[i];

    auto send_displs = allocate<int>(NUM_WRITERS);
    calc_displs(send_group_counts, NUM_WRITERS, send_displs);

    // Reorder sets
    auto new_groups = allocate<struct bgroup>(total_bg);
    j               = 0;
    for (int64_t i = 0; i < num_bg_sets; i++) {
        for (auto k = j; k < j + bg_set_sizes[i]; k++) {
            new_groups[send_displs[set_chunks[i]]++] = final_bg[k];
        }
        j += bg_set_sizes[i];
    }

    calc_displs(send_set_counts, NUM_WRITERS, send_displs);

    auto new_set_sizes = allocate<int64_t>(num_bg_sets);
    for (int64_t i = 0; i < num_bg_sets; i++) {
        new_set_sizes[send_displs[set_chunks[i]]++] = bg_set_sizes[i];
    }
    set_chunks  = reallocate(set_chunks, 0);
    send_displs = reallocate(send_displs, 0);

    num_bg_sets = exchange_data(new_set_sizes, send_set_counts, bg_set_sizes, 0,
                                MPI_INT64_T);
    new_set_sizes   = reallocate(new_set_sizes, 0);
    send_set_counts = reallocate(send_set_counts, 0);

    MPI_Datatype mpi_bgroup_type = create_mpi_bgroup_type();
    exchange_data(new_groups, send_group_counts, final_bg, 0, mpi_bgroup_type);
    MPI_Type_free(&mpi_bgroup_type);
    new_groups        = reallocate(new_groups, 0);
    send_group_counts = reallocate(send_group_counts, 0);
}

void collect_bgroups(int my_rank) {
    bgroups_to_setlist();

    auto num_recv_sets   = allocate<int64_t>(NUM_WRITERS);
    auto num_recv_groups = allocate<int64_t>(NUM_WRITERS);
    auto recv_set_sizes  = allocate<int64_t *>(NUM_WRITERS);
    auto recv_groups     = allocate<struct bgroup *>(NUM_WRITERS);

    auto set_counts   = allocate<int>(NUM_WRITERS);
    auto group_counts = allocate<int>(NUM_WRITERS);

    while (1) {
        auto dest         = calc_next_bgroup_chunk();
        int  is_finished  = (dest < 0);
        int  all_finished = 0;
        MPI_Allreduce(&is_finished, &all_finished, 1, MPI_INT, MPI_LAND,
                      MPI_COMM_WORLD);
        if (all_finished)
            break;

        clear_counts(set_counts, NUM_WRITERS);
        clear_counts(group_counts, NUM_WRITERS);

        if (dest >= 0) {
            int64_t total_bg = 0;
            for (int64_t i = 0; i < num_bg_sets; i++)
                total_bg += bg_set_sizes[i];

            set_counts[dest] += num_bg_sets;
            group_counts[dest] += total_bg;
        }

        int64_t *all_recv_set_sizes = nullptr;
        exchange_data(bg_set_sizes, set_counts, all_recv_set_sizes, 0,
                      MPI_INT64_T);
        int64_t set_start_index = 0;
        for (int64_t i = 0; i < NUM_WRITERS; i++) {
            num_recv_sets[i]  = set_counts[i];
            recv_set_sizes[i] = allocate<int64_t>(num_recv_sets[i]);
            std::copy_n(all_recv_set_sizes + set_start_index, set_counts[i],
                        recv_set_sizes[i]);
            set_start_index += set_counts[i];
        }
        all_recv_set_sizes = reallocate(all_recv_set_sizes, 0);

        MPI_Datatype mpi_bgroup_type = create_mpi_bgroup_type();

        struct bgroup *all_recv_groups = nullptr;
        exchange_data(final_bg, group_counts, all_recv_groups, 0,
                      mpi_bgroup_type);
        int64_t group_start_index = 0;
        for (int64_t i = 0; i < NUM_WRITERS; i++) {
            num_recv_groups[i] = group_counts[i];
            recv_groups[i]     = allocate<struct bgroup>(num_recv_groups[i]);
            std::copy_n(all_recv_groups + group_start_index, group_counts[i],
                        recv_groups[i]);
            group_start_index += group_counts[i];
        }
        all_recv_groups = reallocate(all_recv_groups, 0);

        for (int64_t i = 0; i < NUM_WRITERS; i++) {
            find_bgroup_sets(i, &num_recv_sets[i], &recv_set_sizes[i],
                             &recv_groups[i], &num_recv_groups[i]);
        }

        int64_t num_sets_to_send = 0;
        for (int64_t i = 0; i < NUM_WRITERS; i++) {
            set_counts[i] = num_recv_sets[i];
            num_sets_to_send += num_recv_sets[i];
        }

        auto send_set_sizes = allocate<int64_t>(num_sets_to_send);
        set_start_index     = 0;
        for (int64_t i = 0; i < NUM_WRITERS; i++) {
            std::copy_n(recv_set_sizes[i], num_recv_sets[i],
                        send_set_sizes + set_start_index);
            recv_set_sizes[i] = reallocate(recv_set_sizes[i], 0);
            set_start_index += num_recv_sets[i];
        }
        auto num_recv_sets = exchange_data(send_set_sizes, set_counts,
                                           all_recv_set_sizes, 0, MPI_INT64_T);
        send_set_sizes     = reallocate(send_set_sizes, 0);

        int64_t num_groups_to_send = 0;
        for (int64_t i = 0; i < NUM_WRITERS; i++) {
            group_counts[i] = num_recv_groups[i];
            num_groups_to_send += num_recv_groups[i];
        }
        auto send_groups  = allocate<struct bgroup>(num_groups_to_send);
        group_start_index = 0;
        for (int64_t i = 0; i < NUM_WRITERS; i++) {
            std::copy_n(recv_groups[i], num_recv_groups[i],
                        send_groups + group_start_index);
            recv_groups[i] = reallocate(recv_groups[i], 0);
            group_start_index += num_recv_groups[i];
        }
        auto num_recv_groups = exchange_data(
            send_groups, group_counts, all_recv_groups, 0, mpi_bgroup_type);
        MPI_Type_free(&mpi_bgroup_type);
        send_groups = reallocate(send_groups, 0);

        if (num_recv_sets > 0) {
            num_bg_sets  = num_recv_sets;
            bg_set_sizes = reallocate(bg_set_sizes, num_bg_sets);
            std::copy_n(all_recv_set_sizes, num_recv_sets, bg_set_sizes);
            all_recv_set_sizes = reallocate(all_recv_set_sizes, 0);
        }
        if (num_recv_groups > 0) {
            final_bg = reallocate(final_bg, num_recv_groups);
            std::copy_n(all_recv_groups, num_recv_groups, final_bg);
            all_recv_groups = reallocate(all_recv_groups, 0);
        }
    }
    num_recv_sets   = reallocate(num_recv_sets, 0);
    num_recv_groups = reallocate(num_recv_groups, 0);
    recv_set_sizes  = reallocate(recv_set_sizes, 0);
    recv_groups     = reallocate(recv_groups, 0);

    set_counts   = reallocate(set_counts, 0);
    group_counts = reallocate(group_counts, 0);

    clear_bg_data();

    num_bg_sets = prune_setlist();

    distribute_sets(my_rank);
}

int64_t collect_meta_fofs(struct fof *&meta_fofs) {
    auto counts = allocate<int>(NUM_WRITERS);
    clear_counts(counts, NUM_WRITERS);
    auto displs = allocate<int>(NUM_WRITERS);

    int64_t set_start = 0;
    for (int64_t i = 0; i < num_bg_sets; i++) {
        for (int64_t j = 0; j < bg_set_sizes[i]; j++) {
            ++counts[final_bg[set_start + j].chunk];
        }
        set_start += bg_set_sizes[i];
    }

    auto num_to_send = calc_displs(counts, NUM_WRITERS, displs);
    auto send_ids    = allocate<int64_t>(num_to_send);

    auto total_bg = set_start;
    for (int64_t i = 0; i < total_bg; i++) {
        send_ids[displs[final_bg[i].chunk]++] = final_bg[i].id;
    }

    int64_t *recv_ids = nullptr;
    exchange_data(send_ids, counts, recv_ids, 0, MPI_INT64_T);
    send_ids = reallocate(send_ids, 0);

    // All boundary FOFs are assumed to survive
    for (int64_t i = num_all_fofs - num_bfofs; i + 1 < num_all_fofs; i++) {
        assert(all_fofs[i + 1].particles ==
               all_fofs[i].particles + all_fofs[i].num_p);
    }

    int64_t id_start_index = 0;
    num_to_send            = 0;
    for (int64_t i = 0; i < NUM_WRITERS; i++) {
        int64_t total_num_p = 0;
        for (int64_t j = 0; j < counts[i]; j++) {
            auto id = recv_ids[id_start_index + j];
            assert(id >= num_all_fofs - num_bfofs);
            assert(id < num_all_fofs);
            total_num_p += all_fofs[id].num_p;
        }
        id_start_index += counts[i];
        num_to_send += total_num_p;
    }

    auto send_particles      = allocate<struct particle>(num_to_send);
    id_start_index           = 0;
    int64_t part_start_index = 0;
    for (int64_t i = 0; i < NUM_WRITERS; i++) {
        int64_t total_num_p = 0;
        for (int64_t j = 0; j < counts[i]; j++) {
            auto id = recv_ids[id_start_index + j];
            std::copy_n(all_fofs[id].particles, all_fofs[id].num_p,
                        send_particles + part_start_index);
            part_start_index += all_fofs[id].num_p;
            total_num_p += all_fofs[id].num_p;
            all_fofs[id].num_p = 0;
        }
        id_start_index += counts[i];
        counts[i] = total_num_p;
    }
    recv_ids = reallocate(recv_ids, 0);

    // Collapse sent boundary FOFs
    const auto num_local_fofs = num_all_fofs - num_bfofs;
    num_p                     = all_fofs[num_local_fofs].particles - p;
    num_bfofs                 = 0;
    for (auto i = num_local_fofs; i < num_all_fofs; i++) {
        if (all_fofs[i].num_p == 0)
            continue;
        if (all_fofs[i].particles != p + num_p) {
            std::memmove(p + num_p, all_fofs[i].particles,
                         sizeof(struct particle) * all_fofs[i].num_p);
            all_fofs[num_local_fofs + num_bfofs] = all_fofs[i];
        }
        num_p += all_fofs[i].num_p;
        ++num_bfofs;
    }
    num_all_fofs = num_local_fofs + num_bfofs;

    auto             mpi_particle_type  = create_mpi_particle_type();
    struct particle *recv_particles     = nullptr;
    auto             num_recv_particles = exchange_data(
        send_particles, counts, recv_particles, 0, mpi_particle_type);
    MPI_Type_free(&mpi_particle_type);
    send_particles = reallocate(send_particles, 0);

    auto num_meta_fofs = num_bg_sets;
    meta_fofs          = allocate<struct fof>(num_meta_fofs);

    auto orig_p = p;
    p           = reallocate(p, num_p + num_recv_particles);
    for (int64_t i = 0; i < num_all_fofs; i++) {
        all_fofs[i].particles = p + (all_fofs[i].particles - orig_p);
    }

    calc_displs(counts, NUM_WRITERS, displs);
    counts = reallocate(counts, 0);

    set_start = 0;
    for (int64_t i = 0; i < num_bg_sets; i++) {
        meta_fofs[i].particles = p + num_p;
        meta_fofs[i].num_p     = 0;
        for (int64_t j = 0; j < bg_set_sizes[i]; j++) {
            auto group_particles =
                recv_particles + displs[final_bg[set_start + j].chunk];
            std::copy_n(group_particles, final_bg[set_start + j].num_p,
                        p + num_p);
            displs[final_bg[set_start + j].chunk] +=
                final_bg[set_start + j].num_p;
            num_p += final_bg[set_start + j].num_p;
            meta_fofs[i].num_p += final_bg[set_start + j].num_p;
        }
        set_start += bg_set_sizes[i];
    }

    recv_particles = reallocate(recv_particles, 0);
    displs         = reallocate(displs, 0);
    return num_meta_fofs;
}

int get_max_threads() {
#ifdef _OPENMP
    return omp_get_max_threads();
#else
    return 1;
#endif
}

int get_threadID() {
#ifdef _OPENMP
    return omp_get_thread_num();
#else
    return 0;
#endif
}

void free_halos() {
    halos      = reallocate(halos, 0);
    extra_info = reallocate(extra_info, 0);
    num_halos  = 0;
}

void realloc_halos(int64_t new_num_halos) {
    halos      = reallocate(halos, new_num_halos);
    extra_info = reallocate(extra_info, new_num_halos);
}

void extend_halos(int64_t halo_offsets[], int num_threads) {
    halo_offsets[0] = num_halos;
    for (int i = 0; i < num_threads; i++) {
        halo_offsets[i + 1] += halo_offsets[i];
    }
    realloc_halos(halo_offsets[num_threads]);
    num_halos = halo_offsets[num_threads];
}

void copy_halos(const struct HaloInfo *haloinfo, int64_t offset) {
    assert(offset + haloinfo->num_halos <= num_halos);

    struct extra_halo_info *ei = haloinfo->extra_info;
    for (int64_t i = 0; i < haloinfo->num_halos; i++) {
        assert(-1 <= ei[i].sub_of && ei[i].sub_of < haloinfo->num_halos);
        if (ei[i].sub_of != -1)
            ei[i].sub_of += offset;

        assert(-1 <= ei[i].child && ei[i].child < haloinfo->num_halos);
        if (ei[i].child != -1)
            ei[i].child += offset;

        assert(-1 <= ei[i].next_cochild &&
               ei[i].next_cochild < haloinfo->num_halos);
        if (ei[i].next_cochild != -1)
            ei[i].next_cochild += offset;

        assert(-1 <= ei[i].prev_cochild &&
               ei[i].prev_cochild < haloinfo->num_halos);
        if (ei[i].prev_cochild != -1)
            ei[i].prev_cochild += offset;
    }

    std::copy_n(haloinfo->halos, haloinfo->num_halos, halos + offset);
    std::copy_n(haloinfo->extra_info, haloinfo->num_halos, extra_info + offset);
}

void do_halo_finding(struct fof *meta_fofs, int64_t num_metafofs) {
    num_halos = 0;

    const auto num_threads  = get_max_threads();
    auto       halo_offsets = allocate<int64_t>(num_threads + 1);

#pragma omp parallel \
  copyin(particle_thresh_dens, particle_rvir_dens, particle_rvir_dens_z0, min_dens_index, dynamical_time, scale_dx)
    {
        struct FOFInfo fofinfo;
        init_fofinfo(&fofinfo);
        struct HaloInfo haloinfo;
        init_haloinfo(&haloinfo);

#pragma omp for schedule(dynamic)
        for (int64_t i = 0; i < (num_all_fofs - num_bfofs); i++) {
            find_subs(&all_fofs[i], &fofinfo, &haloinfo);
        }
        const auto thread_id        = get_threadID();
        halo_offsets[thread_id + 1] = haloinfo.num_halos;

#pragma omp barrier
#pragma omp single
        { extend_halos(halo_offsets, num_threads); }

        copy_halos(&haloinfo, halo_offsets[thread_id]);
        haloinfo.num_halos = 0;
#pragma omp barrier

#pragma omp for schedule(dynamic)
        for (int64_t i = 0; i < num_metafofs; i++) {
            std::sort(meta_fofs[i].particles,
                      meta_fofs[i].particles + meta_fofs[i].num_p,
                      [](const struct particle &a, const struct particle &b) {
                          return a.id < b.id;
                      }); // Not necessarily
            if (PERIODIC)
                align_particles(meta_fofs[i]);
            find_subs(&meta_fofs[i], &fofinfo, &haloinfo);
        }

        free_fofinfo(&fofinfo);

        halo_offsets[thread_id + 1] = haloinfo.num_halos;

#pragma omp barrier
#pragma omp single
        { extend_halos(halo_offsets, num_threads); }

        copy_halos(&haloinfo, halo_offsets[thread_id]);
        free_haloinfo(&haloinfo);
    }
    halo_offsets = reallocate(halo_offsets, 0);
}

void calc_halo_bounds(float *bounds) {
    int64_t i, j;
    for (j = 0; j < 6; j++)
        bounds[j] = 0;
    if (!num_halos)
        return;
    memcpy(bounds, halos[0].pos, sizeof(float) * 3);
    memcpy(bounds + 3, halos[0].pos, sizeof(float) * 3);
    for (i = 0; i < num_halos; i++) {
        float r = BGC2_R * halos[i].r;
        if (STRICT_SO_MASSES)
            r = BGC2_R * max_halo_radius(halos + i);
        for (j = 0; j < 3; j++) {
            if (bounds[j] > halos[i].pos[j] - r)
                bounds[j] = halos[i].pos[j] - r;
            if (bounds[j + 3] < halos[i].pos[j] + r)
                bounds[j + 3] = halos[i].pos[j] + r;
        }
    }
}

void gather_spheres(int64_t my_rank, float (*writer_bounds)[6],
                    const std::vector<int64_t> &recipients) {
    std::vector<std::vector<struct sphere_request>> sphere_requests(
        NUM_WRITERS, std::vector<struct sphere_request>());

    for (int64_t i = 0; i < num_halos; i++) {
        if (!_should_print(halos + i, writer_bounds[my_rank]))
            continue;

        float r = BGC2_R * halos[i].r;
        if (STRICT_SO_MASSES)
            r = BGC2_R * max_halo_radius(halos + i);

        int64_t j;
        for (j = 0; j < 3; j++) {
            if (halos[i].pos[j] - r < writer_bounds[my_rank][j])
                break;
            if (halos[i].pos[j] + r > writer_bounds[my_rank][j + 3])
                break;
        }
        if (j == 3)
            continue;

        struct sphere_request sp;
        float                 bounds[6];

        for (auto recipient : recipients) {
            if (recipient == my_rank)
                continue;

            sp.r = halos[i].r * BGC2_R;
            if (STRICT_SO_MASSES)
                sp.r = BGC2_R * max_halo_radius(halos + i);
            for (int64_t k = 0; k < 3; k++) {
                bounds[k]     = writer_bounds[recipient][k] - sp.r;
                bounds[k + 3] = writer_bounds[recipient][k + 3] + sp.r;
            }

	    if (_check_bounds(halos[i].pos, sp.cen, bounds)) {
                sphere_requests[recipient].emplace_back(sp);
	    }

        }
    }

    auto counts = allocate<int>(NUM_WRITERS);
    clear_counts(counts, NUM_WRITERS);

    int64_t total_requests = 0;
    for (int64_t i = 0; i < NUM_WRITERS; i++) {
        counts[i] = sphere_requests[i].size();
        total_requests += counts[i];
    }

    auto    send_buffer = allocate<struct sphere_request>(total_requests);
    int64_t request_start_index = 0;
    for (int64_t i = 0; i < NUM_WRITERS; i++) {
        std::copy(sphere_requests[i].begin(), sphere_requests[i].end(),
                  send_buffer + request_start_index);
        std::vector<struct sphere_request>().swap(sphere_requests[i]);
        request_start_index += counts[i];
    }

    MPI_Datatype mpi_sp_type;
    //MPI_Type_contiguous(4, MPI_FLOAT, &mpi_sp_type);
    MPI_Type_contiguous( sizeof(struct sphere_request), MPI_CHAR, &mpi_sp_type);
    MPI_Type_commit(&mpi_sp_type);
    struct sphere_request *recv_requests = nullptr;
    exchange_data(send_buffer, counts, recv_requests, 0, mpi_sp_type);
    MPI_Type_free(&mpi_sp_type);
    send_buffer = reallocate(send_buffer, 0);

    std::vector<struct extended_particle> epbuffer;
    char *bitarray      = BIT_ALLOC(num_p + num_additional_p);
    request_start_index = 0;
    for (int64_t i = 0; i < NUM_WRITERS; i++) {
        BIT_ALL_CLEAR(bitarray, num_p + num_additional_p);
        int64_t k = 0;
        for (int64_t j = 0; j < counts[i]; j++) {
            int64_t                    num_sp;
            struct extended_particle **result = do_sphere_request(
                recv_requests[request_start_index + j].cen,
                recv_requests[request_start_index + j].r, &num_sp);

            for (int64_t l = 0; l < num_sp; l++) {
                if (BIT_TST(bitarray, result[l] - ep))
                    continue;
                BIT_SET(bitarray, result[l] - ep);
                epbuffer.emplace_back(*result[l]);
                epbuffer.back().hid = -1;
                k++;
            }
        }
        request_start_index += counts[i];
        counts[i] = k;
    }
    bitarray = reallocate(bitarray, 0);

    auto mpi_eparticle_type = create_mpi_eparticle_type();
    num_ep2 =
        exchange_data(epbuffer.data(), counts, ep2, 0, mpi_eparticle_type);
    MPI_Type_free(&mpi_eparticle_type);

}

void find_halos(int64_t snap, int64_t my_rank, char *buffer,
                float (*writer_bounds)[6]) {
    sync_config();
    if (LIGHTCONE)
        init_cosmology();
    init_time_table();

    if (!DUMP_PARTICLES[0]) {
        timed_output("Analyzing for FoF groups...\n");

        if (EXTRA_PROFILING && !profile_out) {
            snprintf(buffer, 1024, "%s/profiling", OUTBASE);
            mkdir(buffer, 0777);
            snprintf(buffer, 1024, "%s/profiling/profile.%" PRId64, OUTBASE,
                     my_rank);
            profile_out = check_fopen(buffer, "w"); // Truncate
            fclose(profile_out);
            profile_out = check_fopen(buffer, "a");
        }
        int64_t time_start = time(NULL);
        rockstar(writer_bounds[my_rank], 1);
        int64_t time_middle = time(NULL);
        set_bp_chunk(my_rank);
        int64_t time_end = time(NULL);
        if (CLIENT_DEBUG){
            fprintf(stderr, "Found %" PRId64 " fofs in chunk %" PRId64 "\n",
                    num_all_fofs, my_rank);
            
            fprintf(stderr, "find_halos: my_rank %" PRId64 " PARTICLE_MASS: %f\n", my_rank, PARTICLE_MASS);
            fprintf(stderr, "find_halos: my_rank %" PRId64 " BOX_SIZE: %f\n", my_rank, BOX_SIZE);
            fprintf(stderr, "find_halos: my_rank %" PRId64 " AVG_PARTICLE_SPACING: %f\n", my_rank, AVG_PARTICLE_SPACING);
            fprintf(stderr, "find_halos: my_rank %" PRId64 " num_p: %" PRId64 "\n", my_rank, num_p);
            fprintf(stderr, "find_halos: my_rank %" PRId64 " num_all_fofs: %" PRId64 "\n", my_rank, num_all_fofs);
        }


        if (profile_out) {
            fprintf(profile_out,
                    "[Prof] S%" PRId64 ",C%" PRId64 " %" PRId64 "s: %" PRId64
                    " fofs, %" PRId64 " particles, %" PRId64 "s for conf.\n",
                    snap, my_rank, (time_middle - time_start), num_all_fofs,
                    num_p, (time_end - time_middle));
            fflush(profile_out);
        }

        MPI_Barrier(MPI_COMM_WORLD);

        transfer_bparticles(my_rank, writer_bounds);
        build_bgroup_links();
        clear_bp_data();

        timed_output("Linking boundary particles...\n");

        collect_bgroups(my_rank);

        timed_output("Analyzing for halos / subhalos...\n");
        /*
        command_writers("rock");
        load_balance();
        if (server_error_state)
            return;
        */
        struct fof *meta_fofs     = nullptr;
        auto        num_meta_fofs = collect_meta_fofs(meta_fofs);

        float new_bounds[6];
        memcpy(new_bounds, writer_bounds[my_rank], sizeof(float) * 6);
        if (num_meta_fofs > 0) {
            if (!PERIODIC || !BOX_SIZE)
                calc_particle_bounds(new_bounds);
            else
                calc_particle_bounds_periodic(new_bounds);
        }
        if (TEMPORAL_HALO_FINDING) {
            load_previous_halos(snap, my_rank, new_bounds);
        }

        do_halo_finding(meta_fofs, num_meta_fofs);
        clear_prev_files();

        for (int64_t i = 0; i < num_halos; i++) {
            wrap_into_box(halos[i].pos);
        }

        distribute_halos(writer_bounds);

        timed_output("Output halos...\n");
        int64_t id_offset = 0;
        int64_t num_halos_print = count_halos_to_print(writer_bounds[my_rank]);
        MPI_Exscan(&num_halos_print, &id_offset, 1, MPI_INT64_T, MPI_SUM,
                   MPI_COMM_WORLD);
        output_halos(id_offset, snap, my_rank, writer_bounds[my_rank]);

#ifdef ENABLE_HDF5
        int64_t num_p_print = count_particles_to_print(writer_bounds[my_rank]);
        if (!strcasecmp(OUTPUT_FORMAT, "HDF5")) {
            int64_t tot_num_halos, tot_num_p;
            MPI_Allreduce(&num_halos_print, &tot_num_halos, 1, MPI_INT64_T, MPI_SUM,
                          MPI_COMM_WORLD);
            MPI_Allreduce(&num_p_print, &tot_num_p, 1, MPI_INT64_T, MPI_SUM,
                          MPI_COMM_WORLD);
            output_hdf5(id_offset, snap, my_rank, writer_bounds[my_rank],
                        tot_num_halos, tot_num_p, 1);
        }
#endif

        if (check_bgc2_snap(snap) || STRICT_SO_MASSES) {
            timed_output("Generating BGC2 files/SO Masses...\n");

            // p[0..num_p] are local particles not belonging to any halo
            // p[num_p..num_p + num_additional_p] are halo particles
            // total_num_p = num_p
            //             + # of particles in halos within my bounds
            int64_t total_num_p = num_p;
            for (int64_t i = 0; i < num_halos; i++) {
                int64_t j;
                for (j = 0; j < 3; j++) {
                    if (halos[i].pos[j] < writer_bounds[my_rank][j] ||
                        halos[i].pos[j] > writer_bounds[my_rank][j + 3])
                        break;
                }
                if (j == 3)
                    total_num_p += halos[i].num_p;
            }
            // The sum of total_num_p for all procs should equal TOTAL_PARTICLES

#if 1
	    for( int i=0; i<(num_p+num_additional_p); i++){
	      wrap_into_box(p[i].pos);
	    }
#endif

            float my_halo_bounds[6];
            calc_halo_bounds(my_halo_bounds);

            std::vector<int64_t> recipients;
            for (int64_t i = 0; i < NUM_WRITERS; i++) {
                float bounds[6];
                if (bounds_overlap(my_halo_bounds, writer_bounds[i], bounds, 0)) {
                    recipients.emplace_back(i);
		}
            }

            init_extended_particle_tree();
            gather_spheres(my_rank, writer_bounds, recipients);

            output_bgc2(id_offset, snap, my_rank, writer_bounds[my_rank]);
            free_extended_particle_tree();
        }
        free_halos();
        particle_cleanup();
        clear_final_bg_data();
        check_mtrim();
    } else {
        timed_output("Dumping particles...\n");
        free_halos();
        p     = reallocate(p, 0);
        num_p = 0;
    }
}

void get_bounds(int64_t snap, int64_t chunk, float bounds[]) {
    struct binary_output_header bheader;
    load_binary_header(snap, chunk, &bheader);
    memcpy(bounds, bheader.bounds, sizeof(float) * 6);
}

void transfer_halos(int64_t snap, int64_t my_rank, float (*writer_bounds)[6],
                    float (*bounds_prevsnap)[6]) {
    struct binary_output_header bheader;
    int64_t                    *part_ids = nullptr;
    load_binary_halos(snap, my_rank, &bheader, &halos, &part_ids, 0);

    auto send_halo_counts = allocate<int>(NUM_WRITERS);
    clear_counts(send_halo_counts, NUM_WRITERS);
    auto send_pid_counts = allocate<int>(NUM_WRITERS);
    clear_counts(send_pid_counts, NUM_WRITERS);
    std::vector<struct halo> send_halos;

    int64_t num_pids_to_send = 0;
    for (int64_t i = 0; i < NUM_WRITERS; i++) {
        float expanded_bounds[6];
        if (bounds_overlap(writer_bounds[my_rank], bounds_prevsnap[i],
                           expanded_bounds, OVERLAP_LENGTH)) {
            for (int64_t j = 0; j < bheader.num_halos; j++) {
                struct halo h = halos[j];
                if (_check_bounds(halos[j].pos, h.pos, expanded_bounds)) {
                    send_halos.emplace_back(h);
                    ++send_halo_counts[i];
                    send_pid_counts[i] += h.num_p;
                }
            }
        }
        num_pids_to_send += send_pid_counts[i];
    }

    std::vector<int64_t> send_pids;
    send_pids.reserve(num_pids_to_send);
    for (const auto &h : send_halos) {
        send_pids.insert(send_pids.end(), part_ids + h.p_start,
                         part_ids + h.p_start + h.num_p);
    }
    part_ids = reallocate(part_ids, 0);

    assert(head2.num_halos == 0);
    auto mpi_halo_type = create_mpi_halo_type();
    head2.num_halos = exchange_data(send_halos.data(), send_halo_counts, halos2,
                                    0, mpi_halo_type);
    MPI_Type_free(&mpi_halo_type);
    std::vector<struct halo>().swap(send_halos);
    send_halo_counts = reallocate(send_halo_counts, 0);

    // Redo particle pointers
    int64_t new_p_start = 0;
    for (int64_t j = 0; j < head2.num_halos; j++) {
        halos2[j].p_start = new_p_start;
        new_p_start += halos2[j].num_p;
    }

    assert(head2.num_particles == 0);
    head2.num_particles =
        exchange_data(send_pids.data(), send_pid_counts, part2, 0, MPI_INT64_T);
    std::vector<int64_t>().swap(send_pids);
    send_pid_counts = reallocate(send_pid_counts, 0);
}

void _do_merger_tree_part2(int64_t snap, int64_t my_rank) {
    float bounds[6];
    get_bounds(snap, my_rank, bounds);

    struct binary_output_header bheader;
    int64_t                    *part_ids = nullptr;
    load_binary_halos(snap, my_rank, &bheader, &halos, &part_ids, 0);

    std::vector<struct halo> new_halos;

    int64_t total_num_p = 0;
    for (int64_t i = 0; i < bheader.num_halos; i++) {
        struct halo h = halos[i];
        if (_check_bounds(halos[i].pos, h.pos, bounds)) {
            new_halos.emplace_back(h);
            total_num_p += h.num_p;
        }
    }

    // Redo particle pointers
    part1               = reallocate(part1, head1.num_particles + total_num_p);
    int64_t new_p_start = head1.num_particles;
    for (auto &h : new_halos) {
        std::copy_n(part_ids + h.p_start, h.num_p, part1 + new_p_start);
        h.p_start = new_p_start;
        new_p_start += h.num_p;
    }
    head1.num_particles = new_p_start;
    part_ids            = reallocate(part_ids, 0);

    halos1 = reallocate(halos1, head1.num_halos + new_halos.size());
    std::copy(new_halos.begin(), new_halos.end(), halos1 + head1.num_halos);
    head1.num_halos += new_halos.size();
    std::vector<struct halo>().swap(new_halos);

    init_descendants();

    timed_output("Constructing merger tree...\n");

    calculate_descendants();

    int64_t cat_length, head_length;
    char   *cat = gen_merger_catalog(snap, my_rank, halos1, head1.num_halos,
                                     &cat_length, &head_length);
    int64_t location = 0;
    if (!OUTLIST_PARALLEL) {
        assert(my_rank == 0 || head_length == 0);
        int64_t total_length = head_length + cat_length;
        MPI_Exscan(&total_length, &location, 1, MPI_INT64_T, MPI_SUM,
                   MPI_COMM_WORLD);
    }
    location += head_length;
    output_merger_catalog(snap, my_rank, location, cat_length, cat);
    clear_merger_tree();

    if (DELETE_BINARY_OUTPUT_AFTER_FINISHED)
        delete_binary(snap, my_rank);
}

void do_merger_tree(int64_t snap, int64_t my_rank, float (*writer_bounds)[6]) {
    int64_t starting_snap = 0;
    if ((SINGLE_SNAP && !snap) || (!SINGLE_SNAP && snap == STARTING_SNAP))
        starting_snap = 1;
    if (starting_snap && snap != NUM_SNAPS - 1)
        return;

    timed_output("Loading merger tree information...\n");
    if (!starting_snap) { // Load in the current snapshot, with overlap
        float bounds[6];
        get_bounds(snap - 1, my_rank, bounds);
        auto bounds_prevsnap = allocate<float[6]>(NUM_WRITERS);
        MPI_Allgather(bounds, 6, MPI_FLOAT, bounds_prevsnap, 6, MPI_FLOAT,
                      MPI_COMM_WORLD);

        get_bounds(snap, my_rank, bounds);
        if (!p_bounds) {
            p_bounds = reallocate(p_bounds, NUM_WRITERS);
        }
        MPI_Datatype mpi_pbounds_type;
#if 0
        MPI_Type_contiguous(6, MPI_FLOAT, &mpi_pbounds_type);
        MPI_Type_commit(&mpi_pbounds_type);
        MPI_Allgather(bounds, 6, MPI_FLOAT, p_bounds, 1, mpi_pbounds_type,
                      MPI_COMM_WORLD);
#else
	MPI_Type_contiguous( sizeof(struct prev_bounds), MPI_CHAR, &mpi_pbounds_type);
        MPI_Type_commit(&mpi_pbounds_type);
	struct prev_bounds bounds_snap;
	for( int i=0; i<6; i++)  bounds_snap.bounds[i] = bounds[i];
	MPI_Allgather( &bounds_snap, 1, mpi_pbounds_type, p_bounds, 1, mpi_pbounds_type, MPI_COMM_WORLD);
#endif
        MPI_Type_free(&mpi_pbounds_type);
        prev_snap = snap;

        transfer_halos(snap, my_rank, writer_bounds, bounds_prevsnap);
        bounds_prevsnap = reallocate(bounds_prevsnap, 0);
        connect_particle_ids_to_halo_ids();

        _do_merger_tree_part2(snap - 1, my_rank);
    }
    if (snap == NUM_SNAPS - 1)
        _do_merger_tree_part2(snap, my_rank);
}



void init_mpi(int argc, char *argv[]) {

  MPI_Init(&argc, &argv);

}



void check_config( const int my_rank){

  if(my_rank == 0){

    if (CLIENT_DEBUG) fprintf( stderr, "#The_number_of_threads= %d \n", get_max_threads());

#ifdef OUTPUT_RVMAX
    if (CLIENT_DEBUG) fprintf( stderr, "#OUTPUT_RVMAX on\n");
    
#endif

#ifdef OUTPUT_INTERMEDIATE_AXIS
    if (CLIENT_DEBUG) fprintf( stderr, "#OUTPUT_INTERMEDIATE_AXIS on\n");
#endif

#ifdef OUTPUT_INERTIA_TENSOR
    if (CLIENT_DEBUG) fprintf( stderr, "#OUTPUT_INERTIA_TENSOR on\n");
#endif

#ifdef OUTPUT_NFW_CHI2
    if (CLIENT_DEBUG) fprintf( stderr, "#OUTPUT_NFW_CHI2 on\n");
#endif

  }

  if( PARALLEL_IO != 1){
    if(my_rank == 0){
      fprintf( stderr, "PARALLEL_IO must be 1\n");
    }
    exit(1);
  }

}



void read_config_and_input(int argc, char **argv) {

    int64_t i, snap = -1, did_config = 0;

    srand(1);

    for (i = 1; i < argc - 1; i++) {
        if (!strcmp("-c", argv[i])) {
            do_config(argv[i + 1]);
            i++;
            did_config = 1;
        }
    }
    if (!did_config)
        do_config(NULL);
    if (strlen(SNAPSHOT_NAMES))
        read_input_names(SNAPSHOT_NAMES, &snapnames, &NUM_SNAPS);
    if (strlen(BLOCK_NAMES))
        read_input_names(BLOCK_NAMES, &blocknames, &NUM_BLOCKS);

    if (snap > -1) {
      STARTING_SNAP = snap;
      SINGLE_SNAP   = 1;
    }

}



extern "C" void rockstar_mpi_main(int argc, char *argv[]) {
    mpi_main(argc, argv);
}

void mpi_main(int argc, char *argv[]){

    read_config_and_input(argc, argv);

    char    buffer[1024];
    int64_t reload_parts = 0;

    int num_procs, my_rank;
    //MPI_Init(&argc, &argv);
#ifndef DO_CONFIG_MPI
    init_mpi( argc, argv);
#endif
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    #ifndef MPI_ROCKSTAR_LIBRARY
    if (my_rank==0 && argc < 2) {
        printf("MPI-Rockstar Halo Finder, Version %s\n", ROCKSTAR_VERSION);
        printf("(C) See the README file for redistribution details.\n");
        printf("Usage: %s [-c config]\n", argv[0]);
        exit(1);
    }
    #endif

    NUM_WRITERS = num_procs;
    NUM_READERS = (NUM_BLOCKS > num_procs) ? num_procs : NUM_BLOCKS;
    if (my_rank == 0)
        check_num_writers();

    check_config( my_rank);

    float my_reader_bounds[6];
    auto  writer_bounds = allocate<float[6]>(NUM_WRITERS);

    clear_merger_tree();
    if (LIGHTCONE && strlen(LIGHTCONE_ALT_SNAPS)) {
        memcpy(LIGHTCONE_ORIGIN, LIGHTCONE_ALT_ORIGIN, sizeof(double) * 3);
    }

    time_start = MPI_Wtime();
    if (STARTING_SNAP > RESTART_SNAP)
        RESTART_SNAP = STARTING_SNAP;
    reload_parts = 1;

    for (int64_t snap = RESTART_SNAP; snap < NUM_SNAPS; snap++) {
        RESTART_SNAP = snap;
        if (my_rank == 0)
            output_config("restart.cfg");
        timed_output("Start processing snapshot %" PRId64 " with %" PRId64
                     " procs.\n",
                     snap, NUM_WRITERS);
        if (!DO_MERGER_TREE_ONLY) {
            const auto my_reader_rank = get_reader_rank();
            if (!PRELOAD_PARTICLES || reload_parts) {
                read_blocks(snap, my_reader_rank, buffer);
                reload_parts = 0;
            }
            sync_config();
            decide_reader_bounds(my_reader_bounds, my_reader_rank);
            decide_writer_bounds(writer_bounds, my_rank);

            if (!MEMORY_SAVING_TRANSFER) {
              transfer_particles(my_reader_rank, my_reader_bounds, writer_bounds);
            }
            else {
              transfer_particles_mem_save(my_reader_rank, my_reader_bounds, writer_bounds);
            }

            std::sort(p, p + num_p,
                      [](const struct particle &a, const struct particle &b) {
                          return a.id < b.id;
                      }); // Not necessarily
            if (PRELOAD_PARTICLES && (snap < NUM_SNAPS - 1))
                read_blocks(snap + 1, my_reader_rank, buffer);
            find_halos(snap, my_rank, buffer, writer_bounds);
        }
        if (((strcasecmp(OUTPUT_FORMAT, "ASCII") != 0) ||
             TEMPORAL_HALO_FINDING) &&
            !DUMP_PARTICLES[0] && !IGNORE_PARTICLE_IDS)
            do_merger_tree(snap, my_rank, writer_bounds);
        timed_output("[Success] Done with snapshot %" PRId64 ".\n", snap);
        /*
        if (strlen(RUN_PARALLEL_ON_SUCCESS)) {
            timed_output("Running external parallel analysis process for "
                         "snapshot %" PRId64 "...\n",
                         snap);
            command_writers_and_confirm("rpos");
        }

        if (strlen(RUN_ON_SUCCESS)) {
            if (snapnames && snapnames[snap])
                snprintf(buffer, 1024, "%s %" PRId64 " %s", RUN_ON_SUCCESS,
                         snap, snapnames[snap]);
            else
                snprintf(buffer, 1024, "%s %" PRId64 " %" PRId64,
                         RUN_ON_SUCCESS, snap, snap);
            n = fork();
            if (n <= 0) {
                if (system(buffer) != 0)
                    fprintf(stderr,
                            "[Warning] Post-analysis command \"%s\" exited "
                            "abnormally.\n",
                            buffer);
            }
            if (n == 0)
                exit(0);
        }
        */
        if (SINGLE_SNAP)
            break;
    }
    writer_bounds = reallocate(writer_bounds, 0);

    timed_output("[Finished]\n");

#ifndef DO_CONFIG_MPI
    MPI_Finalize();
#endif
    
}




#ifndef MPI_ROCKSTAR_LIBRARY
int main(int argc, char **argv) {


#ifdef DO_CONFIG_MPI
    init_mpi( argc, argv);
#endif

    try {
        mpi_main(argc, argv);
    } catch (const rockstar_error &err) {
#ifdef DO_CONFIG_MPI
        MPI_Finalize();
#endif
        fprintf(stderr, "Rockstar error %d at %s:%d\n", err.code, err.file, err.line);
        return err.code;
    }


#ifdef DO_CONFIG_MPI
    MPI_Finalize();
#endif

    return 0;

}
#endif /* MPI_ROCKSTAR_LIBRARY */

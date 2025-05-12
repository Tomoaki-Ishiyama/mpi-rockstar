#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <strings.h>
#include <assert.h>
#include <inttypes.h>
#include <math.h>
#include <unistd.h>
#include <sys/stat.h>
#include "../check_syscalls.h"
#include "../config_vars.h"
#include "../config.h"
#include "../rockstar.h"
#include "../groupies.h"
#include "../universal_constants.h"
#include "io_art.h"
#include "io_ascii.h"
#include "io_bgc2.h"
#include "io_gadget.h"
#include "io_generic.h"
#include "io_internal.h"
#include "io_tipsy.h"
#include "io_kyf.h" // Added by TI 20160909
#include "meta_io.h"
#include "../distance.h"
#include "../version.h"

#ifdef ENABLE_HDF5
#include "io_internal_hdf5.h"
#include "io_arepo.h"
#include "io_gadget4.h"
#endif /* ENABLE_HDF5 */

char **snapnames  = NULL;
char **blocknames = NULL;

void read_input_names(char *filename, char ***stringnames, int64_t *num_names) {
    int64_t i = 0, j;
    char    buffer[1024], **names = NULL;
    FILE   *input;

    if (!strlen(filename))
        return;
    input = check_fopen(filename, "r");
    while (fgets(buffer, 1024, input)) {
        while (strlen(buffer) && buffer[strlen(buffer) - 1] == '\n')
            buffer[strlen(buffer) - 1] = 0;
        if (!strlen(buffer))
            continue;
        if (!(i % 10))
            names = check_realloc(names, sizeof(char *) * (i + 10),
                                  "Allocating snapshot names.");
        names[i] = strdup(buffer);
        i++;
    }
    fclose(input);

    if (*stringnames) {
        for (j = 0; j < *num_names; j++)
            free((*stringnames)[j]);
        free(*stringnames);
    }
    *num_names   = i;
    *stringnames = names;
}

void get_input_filename(char *buffer, int maxlen, int64_t snap, int64_t block) {
    int64_t i = 0, out = 0, l = strlen(FILENAME);
    assert(snap < NUM_SNAPS);
    snprintf(buffer, maxlen, "%s/", INBASE);
    out = strlen(buffer);
    if (FILES_PER_SUBDIR_INPUT > 0) {
        int64_t subdir = block / FILES_PER_SUBDIR_INPUT;
        snprintf(buffer + out, maxlen - out, "%s%03" PRId64 "/%0*ld/", INBASE2, snap, (int)SUBDIR_DIGITS_INPUT, subdir);
        out = strlen(buffer);
    }
    for (; (i < l) && (out < (maxlen - 1)); i++) {
        if (FILENAME[i] != '<') {
            buffer[out]     = FILENAME[i];
            buffer[out + 1] = 0;
        } else {
            if (!strncmp(FILENAME + i, "<snap>", 6)) {
                i += 5;
                if (snapnames)
                    snprintf(buffer + out, maxlen - out, "%s", snapnames[snap]);
                else {
                    if (!strncasecmp(FILE_FORMAT, "GADGET", 6) ||
                        !strncasecmp(FILE_FORMAT, "LGADGET", 7) ||
                        !strncasecmp(FILE_FORMAT, "KYF", 3) ||
                        !strncasecmp(FILE_FORMAT, "AREPO", 5) ||
                        !strncasecmp(FILE_FORMAT, "GADGET4", 7))
                        snprintf(buffer + out, maxlen - out, "%03" PRId64,
                                 snap);
                    else
                        snprintf(buffer + out, maxlen - out, "%" PRId64, snap);
                }
            } else if (!strncmp(FILENAME + i, "<block>", 7)) {
                i += 6;
                if (blocknames)
                    snprintf(buffer + out, maxlen - out, "%s",
                             blocknames[block]);
                else
                    snprintf(buffer + out, maxlen - out, "%" PRId64, block);
            } else
                buffer[out] = FILENAME[i];
        }
        out = strlen(buffer);
    }
    buffer[out] = 0;
}

int64_t get_output_dirname(char *buffer, int maxlen, int64_t snap,
                           int64_t chunk) {
    int64_t out = 0;
    snprintf(buffer, maxlen, "%s/", OUTBASE);
    out = strlen(buffer);
    if (OUTPUT_SUBDIR) {
        snprintf(buffer + out, maxlen - out, "%0*ld/", (int)SNAPSHOT_SUBDIR_DIGITS,
                 snap);
        out = strlen(buffer);
    }
    if (FILES_PER_SUBDIR_OUTPUT > 0) {
        int64_t subdir = chunk / FILES_PER_SUBDIR_OUTPUT;
        snprintf(buffer + out, maxlen - out, "%0*ld/", (int)SUBDIR_DIGITS_OUTPUT,
                 subdir);
        out = strlen(buffer);
        mkdir(buffer, 0777);
    }
    return out;
}

void get_output_filename(char *buffer, int maxlen, int64_t snap, int64_t chunk,
                         char *type) {
    int64_t out = get_output_dirname(buffer, maxlen, snap, chunk);
    if (snapnames)
        snprintf(buffer + out, maxlen - out, "halos_%s", snapnames[snap]);
    else
        snprintf(buffer + out, maxlen - out, "halos_%" PRId64, snap);
    out = strlen(buffer);
    snprintf(buffer + out, maxlen - out, ".%" PRId64 ".%s", chunk, type);
}

void read_particles(char *filename) {
    int64_t i, j, gadget = 0;
    int64_t p_start = num_p;
    float   dx, ds, z, a, vel_mul;
    double *origin, origin_offset[3] = {0};
    if (!strcasecmp(FILE_FORMAT, "ASCII"))
        load_particles(filename, &p, &num_p);
    else if (!strncasecmp(FILE_FORMAT, "GADGET4", 7)) {
#ifdef ENABLE_HDF5
        load_particles_gadget4(filename, &p, &num_p);
        gadget = 1;
#else
        fprintf(stderr, "[Error] GADGET4 needs HDF5 support.  Recompile Rockstar "
                        "using \"make with_hdf5\".\n");
        exit(1);
#endif
    } else if (!strncasecmp(FILE_FORMAT, "GADGET", 6) ||
             !strncasecmp(FILE_FORMAT, "LGADGET", 7)) {
        load_particles_gadget2(filename, &p, &num_p);
        gadget = 1;
    } else if (!strncasecmp(FILE_FORMAT, "ART", 3))
        load_particles_art(filename, &p, &num_p);
    else if (!strncasecmp(FILE_FORMAT, "INTERNAL", 8)) {
        load_particles_internal(filename, &p, &num_p);
    } else if (!strncasecmp(FILE_FORMAT, "GENERIC", 7)) {
        assert(load_particles_generic != NULL);
        load_particles_generic(filename, &p, &num_p);
    } else if (!strncasecmp(FILE_FORMAT, "TIPSY", 5)) {
        load_particles_tipsy(filename, &p, &num_p);
    } else if (!strncasecmp(FILE_FORMAT, "KYF", 3)) { // Added by TI 20160909
        load_particles_kyf(filename, &p, &num_p);
    } else if (!strncasecmp(FILE_FORMAT, "AREPO", 5)) {
#ifdef ENABLE_HDF5
        load_particles_arepo(filename, &p, &num_p);
#else
        fprintf(stderr, "[Error] AREPO needs HDF5 support.  Recompile Rockstar "
                        "using \"make with_hdf5\".\n");
        exit(1);
#endif
    } else {
        fprintf(stderr, "[Error] Unknown filetype %s!\n", FILE_FORMAT);
        exit(1);
    }

    if (PERIODIC) {
        for (i = p_start; i < num_p; i++) {
            for (j = 0; j < 3; j++) {
                if (p[i].pos[j] > BOX_SIZE)
                    p[i].pos[j] -= BOX_SIZE;
                else if (p[i].pos[j] < 0)
                    p[i].pos[j] += BOX_SIZE;
            }
        }
    }

    if (NON_COSMOLOGICAL) {
        SCALE_NOW = 1;
    }

    if (LIMIT_RADIUS) {
        for (i = p_start; i < num_p; i++) {
            for (j = 0, ds = 0; j < 3; j++) {
                dx = p[i].pos[j] - LIMIT_CENTER[j];
                ds += dx * dx;
            }
            if (ds > LIMIT_RADIUS * LIMIT_RADIUS) {
                num_p--;
                p[i] = p[num_p];
                i--;
            }
        }
    }

    if (LIGHTCONE) {
        init_cosmology();
        if (strlen(LIGHTCONE_ALT_SNAPS)) {
            for (i = 0; i < 3; i++)
                if (LIGHTCONE_ORIGIN[i] || LIGHTCONE_ALT_ORIGIN[i])
                    break;
            if (i < 3) { // Same box coordinates, different intended locations
                if (LIGHTCONE == 1) {
                    for (i = 0; i < 3; i++)
                        origin_offset[i] =
                            LIGHTCONE_ORIGIN[i] - LIGHTCONE_ALT_ORIGIN[i];
                }
            } else { // Offset everything
                for (i = 0; i < 3; i++)
                    origin_offset[i] = -BOX_SIZE;
            }
            BOX_SIZE *= 2.0;
        }
        origin = (LIGHTCONE == 2) ? LIGHTCONE_ALT_ORIGIN : LIGHTCONE_ORIGIN;
        for (i = p_start; i < num_p; i++) {
            if (LIGHTCONE == 2)
                p[i].id = -p[i].id; // Make ids different
            for (j = 0, dx = 0; j < 3; j++) {
                ds = p[i].pos[j] - origin[j];
                dx += ds * ds;
                p[i].pos[j] -= origin_offset[j];
            }
            if (!gadget)
                continue;
            dx      = sqrt(dx);
            z       = comoving_distance_h_to_redshift(dx);
            a       = scale_factor(z);
            vel_mul = sqrt(a);
            for (j = 0; j < 3; j++)
                p[i].pos[j + 3] *= vel_mul;
        }
    }
}

int _within_bounds(struct halo *h, float *bounds) {
    int64_t i;
    if (!bounds)
        return 1;
    for (i = 0; i < 3; i++)
        if (h->pos[i] < bounds[i] || h->pos[i] > bounds[i + 3])
            return 0;
    return 1;
}

int _should_print(struct halo *h, float *bounds) {
    if (!_within_bounds(h, bounds))
        return 0;
    if (h->flags & ALWAYS_PRINT_FLAG)
        return 1;
    if ((h->num_p < MIN_HALO_OUTPUT_SIZE) ||
        (h->m * UNBOUND_THRESHOLD >= h->mgrav) ||
        ((h->mgrav < 1.5 * PARTICLE_MASS) && UNBOUND_THRESHOLD > 0))
        return 0;
    return 1;
}

int64_t print_ascii_header_info(FILE *output, float *bounds, int64_t np) {
    int64_t chars = 0;
    chars += fprintf(output, "#a = %f\n", SCALE_NOW);
    if (bounds)
        chars +=
            fprintf(output, "#Bounds: (%f, %f, %f) - (%f, %f, %f)\n", bounds[0],
                    bounds[1], bounds[2], bounds[3], bounds[4], bounds[5]);
    chars += fprintf(output, "#Om = %f; Ol = %f; h = %f\n", Om, Ol, h0);
    chars += fprintf(output, "#FOF linking length: %f\n", FOF_LINKING_LENGTH);
    chars += fprintf(output,
                     "#Unbound Threshold: %f; FOF Refinement Threshold: %f\n",
                     UNBOUND_THRESHOLD, FOF_FRACTION);
    chars += fprintf(output, "#Particle mass: %.5e Msun/h\n", PARTICLE_MASS);
    chars += fprintf(output, "#Box size: %f Mpc/h\n", BOX_SIZE);
    if (np)
        chars +=
            fprintf(output, "#Total particles processed: %" PRId64 "\n", np);
    chars +=
        fprintf(output, "#Force resolution assumed: %g Mpc/h\n", FORCE_RES);
    if (STRICT_SO_MASSES && !np)
        chars +=
            fprintf(output, "#Using Strict Spherical Overdensity Masses\n");
    chars += fprintf(
        output,
        "#Units: Masses in Msun / h\n"
        "#Units: Positions in Mpc / h (comoving)\n"
        "#Units: Velocities in km / s (physical, peculiar)\n"
        "#Units: Halo Distances, Lengths, and Radii in kpc / h (comoving)\n"
        "#Units: Angular Momenta in (Msun/h) * (Mpc/h) * km/s (physical)\n"
        "#Units: Spins are dimensionless\n");
    if (np)
        chars += fprintf(
            output,
            "#Units: Total energy in (Msun/h)*(km/s)^2"
            " (physical)\n"
            "#Note: idx, i_so, and i_ph are internal debugging quantities\n");
    chars += fprintf(output, "#Np is an internal debugging quantity.\n");
    chars += fprintf(output, "#MPI-Rockstar Version: %s\n", ROCKSTAR_VERSION);
    return chars;
}

void output_ascii(int64_t id_offset, int64_t snap, int64_t chunk,
                  float *bounds) {
    char         buffer[1024];
    int64_t      i, id = 0;
    FILE        *output;
    struct halo *th;
    get_output_filename(buffer, 1024, snap, chunk, "ascii");
    output = check_fopen(buffer, "w");

    fprintf(output,
            "#id num_p m%s mbound_%s r%s vmax rvmax vrms x y z vx vy vz Jx Jy "
            "Jz E Spin PosUncertainty VelUncertainty bulk_vx bulk_vy bulk_vz "
            "BulkVelUnc n_core m%s m%s m%s m%s Xoff Voff spin_bullock b_to_a "
            "c_to_a A[x] A[y] A[z] b_to_a(%s) c_to_a(%s) A[x](%s) A[y](%s) "
            "A[z](%s) Rs Rs_Klypin T/|U| M_pe_Behroozi M_pe_Diemer "
            "Halfmass_Radius idx i_so i_ph num_cp mmetric\n",
            MASS_DEFINITION, MASS_DEFINITION, MASS_DEFINITION, MASS_DEFINITION2,
            MASS_DEFINITION3, MASS_DEFINITION4, MASS_DEFINITION5,
            MASS_DEFINITION4, MASS_DEFINITION4, MASS_DEFINITION4,
            MASS_DEFINITION4, MASS_DEFINITION4);
    print_ascii_header_info(output, bounds, num_p);

    for (i = 0; i < num_halos; i++) {
        if (!_should_print(halos + i, bounds))
            continue;
        th = halos + i;
        fprintf(output,
                "%" PRId64 " %" PRId64 " %.3e %.3e"
                " %f %f %f %f %f %f %f %f %f %f %g %g %g %g %g %f %f %f %f %f "
                "%f %" PRId64 " %e %e %e %e %f %f %f %f %f %f %f %f %f %f %f "
                "%f %f %f %f %f %e %e %f %" PRId64 " %" PRId64 " %" PRId64
                " %" PRId64 " %f\n",
                id + id_offset, th->num_p, th->m, th->mgrav, th->r, th->vmax,
                th->rvmax, th->vrms, th->pos[0], th->pos[1], th->pos[2],
                th->pos[3], th->pos[4], th->pos[5], th->J[0], th->J[1],
                th->J[2], th->energy, th->spin, sqrt(th->min_pos_err),
                sqrt(th->min_vel_err), th->bulkvel[0], th->bulkvel[1],
                th->bulkvel[2], sqrt(th->min_bulkvel_err), th->n_core,
                th->alt_m[0], th->alt_m[1], th->alt_m[2], th->alt_m[3],
                th->Xoff, th->Voff, th->bullock_spin, th->b_to_a, th->c_to_a,
                th->A[0], th->A[1], th->A[2], th->b_to_a2, th->c_to_a2,
                th->A2[0], th->A2[1], th->A2[2], th->rs, th->klypin_rs,
                th->kin_to_pot, th->m_pe_b, th->m_pe_d, th->halfmass_radius, i,
                extra_info[i].sub_of, extra_info[i].ph, th->num_child_particles,
                extra_info[i].max_metric);
        id++;
    }
    fclose(output);
}

void print_child_particles(FILE *output, int64_t i, int64_t pid, int64_t eid) {
    int64_t          j, child;
    struct particle *p2;
    for (j = 0; j < halos[i].num_p; j++) {
        p2 = p + halos[i].p_start + j;
        if (OUTPUT_EVERY_N_PARTICLES <= 1 ||
            !(p2->id % OUTPUT_EVERY_N_PARTICLES))
            fprintf(output,
                    "%f %f %f %f %f %f %" PRId64 " %" PRId64 " %" PRId64
                    " %" PRId64 "\n",
                    p2->pos[0], p2->pos[1], p2->pos[2], p2->pos[3], p2->pos[4],
                    p2->pos[5], p2->id, i, pid, eid);
    }
    child = extra_info[i].child;
    while (child > -1) {
        print_child_particles(output, child, pid, eid);
        child = extra_info[child].next_cochild;
    }
}

void output_full_particles(int64_t id_offset, int64_t snap, int64_t chunk,
                           float *bounds) {
    char         buffer[1024];
    FILE        *output;
    int64_t      i, id = 0;
    struct halo *th;

    if (chunk >= FULL_PARTICLE_CHUNKS)
        return;
    get_output_filename(buffer, 1024, snap, chunk, "particles");
    output = check_fopen(buffer, "w");

    fprintf(output, "#Halo table:\n");
    fprintf(output,
            "#id internal_id num_p m%s mbound_%s r%s vmax rvmax vrms x y z vx "
            "vy vz Jx Jy Jz energy spin\n",
            MASS_DEFINITION, MASS_DEFINITION, MASS_DEFINITION);
    fprintf(output, "#Particle table:\n");
    fprintf(output, "#x y z vx vy vz particle_id assigned_internal_haloid "
                    "internal_haloid external_haloid\n");

    if (OUTPUT_EVERY_N_PARTICLES > 1) {
        fprintf(output,
                "#Notes: Only particles with IDs divisible by %" PRId64
                " are printed.\n",
                OUTPUT_EVERY_N_PARTICLES);
    }

    fprintf(output,
            "#Notes: As not all halos are printed, some halos may not have "
            "external halo ids.  (Hence the need to print internal halo ids).  "
            "Each particle is assigned to a unique halo; however, some "
            "properties (such as halo bound mass) are calculated including all "
            "substructure.  As such, particles belonging to subhalos are "
            "included in outputs; to exclude substructure, verify that the "
            "internal halo id is the same as the assigned internal halo id.\n");

    print_ascii_header_info(output, bounds, num_p);
    fprintf(output, "#Halo table begins here:\n");

    for (i = 0; i < num_halos; i++) {
        th = halos + i;
        if (_should_print(th, bounds)) {
            th->id = id + id_offset;
            id++;
        } else {
            th->id = -1;
            if (!UNFILTERED_HALO_OUTPUT)
                continue;
        }

        fprintf(output,
                "#%" PRId64 " %" PRId64 " %" PRId64 " %.3e %.3e"
                " %f %f %f %f %f %f %f %f %f %f %g %g %g %g %g\n",
                th->id, i, th->num_p, th->m, th->mgrav, th->r, th->vmax,
                th->rvmax, th->vrms, th->pos[0], th->pos[1], th->pos[2],
                th->pos[3], th->pos[4], th->pos[5], th->J[0], th->J[1],
                th->J[2], th->energy, th->spin);
    }

    fprintf(output, "#Particle table begins here:\n");
    for (i = 0; i < num_halos; i++)
        if (UNFILTERED_HALO_OUTPUT || halos[i].id > -1)
            print_child_particles(output, i, i, halos[i].id);
    fclose(output);
    get_output_filename(buffer, 1024, snap, chunk, "particles");
    // gzip_file(buffer);
}

void delete_binary(int64_t snap, int64_t chunk) {
    char buffer[1024];
    get_output_filename(buffer, 1024, snap, chunk, "bin");
    unlink(buffer);
}

int64_t count_halos_to_print(float *bounds) {
    int64_t to_print = 0, i;
    for (i = 0; i < num_halos; i++)
        if (_should_print(halos + i, bounds))
            to_print++;
    return to_print;
}

int64_t count_particles_to_print(float *bounds) {
    int64_t to_print = 0, i;
    for (i = 0; i < num_halos; i++)
        if (_should_print(halos + i, bounds))
            to_print += halos[i].num_p;
    return to_print;
}

void output_halos(int64_t id_offset, int64_t snap, int64_t chunk,
                  float *bounds) {
    if (!strcasecmp(OUTPUT_FORMAT, "BOTH") ||
        !strcasecmp(OUTPUT_FORMAT, "ASCII"))
        output_ascii(id_offset, snap, chunk, bounds);
    if (!strcasecmp(OUTPUT_FORMAT, "BOTH") ||
        !strcasecmp(OUTPUT_FORMAT, "BINARY") ||
        (TEMPORAL_HALO_FINDING && !LIGHTCONE))
        output_binary(id_offset, snap, chunk, bounds, 1);

    if (chunk < FULL_PARTICLE_CHUNKS)
        output_full_particles(id_offset, snap, chunk, bounds);

    if (DUMP_PARTICLES[0] &&
        (chunk >= DUMP_PARTICLES[1] && chunk <= DUMP_PARTICLES[2]))
        output_particles_internal(snap, chunk, 1);

    if (WEAK_LENSING_FRACTION > 0)
        output_particles_internal(snap, chunk, WEAK_LENSING_FRACTION);
}

void get_outlist_filename(char *buffer, int maxlen, int64_t snap,
                          int64_t chunk) {
    int64_t out = get_output_dirname(buffer, maxlen, snap, chunk);
    snprintf(buffer + out, maxlen - out, "out_%" PRId64 ".list", snap);
    out = strlen(buffer);
    if (OUTLIST_PARALLEL) {
        snprintf(buffer + out, maxlen - out, "-%" PRId64, chunk);
        out = strlen(buffer);
    }
    else{
      snprintf(buffer, maxlen, "%s/out_%" PRId64 ".list", OUTBASE, snap);
      out = strlen(buffer);
    }

}

char *gen_merger_catalog(int64_t snap, int64_t chunk, struct halo *halos,
                         int64_t num_halos, int64_t *cat_length,
                         int64_t *header_length) {
    char        *cat     = NULL;
    char        *cur_pos = NULL;
    int64_t      chars = 0, hchars = 0, chars_a = 0;
    FILE        *output;
    int64_t      i, j;
    double       m;
    struct halo *th;

    if (OUTLIST_PARALLEL || chunk == 0) {
        char buffer[1024];
        get_outlist_filename(buffer, 1024, snap, chunk);
        output = check_fopen(buffer, "w");
        hchars += fprintf(
            output,
            "#ID DescID M%s Vmax Vrms R%s Rs Np X Y Z VX VY VZ JX JY JZ Spin "
            "rs_klypin M%s_all M%s M%s M%s M%s Xoff Voff spin_bullock b_to_a "
            "c_to_a A[x] A[y] A[z] b_to_a(%s) c_to_a(%s) A[x](%s) A[y](%s) "
            "A[z](%s) T/|U| M_pe_Behroozi M_pe_Diemer Halfmass_Radius"
#ifdef OUTPUT_RVMAX
	    " rvmax"
#endif
#ifdef OUTPUT_NFW_CHI2
	    " NFW_chi2"
#endif
#ifdef OUTPUT_INTERMEDIATE_AXIS
	    " B[x] B[y] B[z] B[x](%s) B[y](%s) B[z](%s)"
#endif
#ifdef OUTPUT_INERTIA_TENSOR
	    " Ixx Iyy Izz Ixy Iyz Izx Ixx(%s) Iyy(%s) Izz(%s) Ixy(%s) Iyz(%s) Izx(%s)"
#endif
	    "\n",
            MASS_DEFINITION, MASS_DEFINITION, MASS_DEFINITION, MASS_DEFINITION2,
            MASS_DEFINITION3, MASS_DEFINITION4, MASS_DEFINITION5,
            MASS_DEFINITION4, MASS_DEFINITION4, MASS_DEFINITION4,
            MASS_DEFINITION4, MASS_DEFINITION4
#ifdef OUTPUT_INTERMEDIATE_AXIS
	    ,MASS_DEFINITION4, MASS_DEFINITION4, MASS_DEFINITION4
#endif
#ifdef OUTPUT_INERTIA_TENSOR
	    ,MASS_DEFINITION4, MASS_DEFINITION4, MASS_DEFINITION4, MASS_DEFINITION4, MASS_DEFINITION4, MASS_DEFINITION4
#endif
			  );
        hchars += print_ascii_header_info(output, NULL, 0);
        fclose(output);
    }

    for (i = 0; i < num_halos; i++) {
        if (chars + 1024 > chars_a) {
            chars_a = chars + 10000;
            check_realloc_s(cat, sizeof(char), chars_a);
        }
        cur_pos = cat + chars;
        th      = halos + i;
        if (LIGHTCONE)
            for (j = 0; j < 3; j++)
                th->pos[j] -= LIGHTCONE_ORIGIN[j];
        m = (BOUND_PROPS) ? th->mgrav : th->m;
        chars += snprintf(
            cur_pos, 1024,
            "%" PRId64 " %" PRId64 " %.4e %.2f %.2f %.3f %.3f %" PRId64 " %.5f "
            "%.5f %.5f %.2f %.2f %.2f %.3e %.3e %.3e %.5f %.5f %.4e %.4e %.4e "
            "%.4e %.4e "
            "%.5f %.2f %.5f %.5f %.5f %.5f %.5f %.5f %.5f %.5f %.5f %.5f %.5f "
            "%.4f %.3e %.3e %.3f"
#ifdef OUTPUT_RVMAX
	    " %.3f"
#endif
#ifdef OUTPUT_NFW_CHI2
	    " %.4e"
#endif
#ifdef OUTPUT_INTERMEDIATE_AXIS
	    " %.5f %.5f %.5f %.5f %.5f %.5f"
#endif
#ifdef OUTPUT_INERTIA_TENSOR
	    " %.4e %.4e %.4e %.4e %.4e %.4e %.4e %.4e %.4e %.4e %.4e %.4e"
#endif
	    "\n",
            th->id, th->desc, m, th->vmax, th->vrms, th->r, th->rs, th->num_p,
            th->pos[0], th->pos[1], th->pos[2], th->pos[3], th->pos[4],
            th->pos[5], th->J[0], th->J[1], th->J[2], th->spin, th->klypin_rs,
            th->m, th->alt_m[0], th->alt_m[1], th->alt_m[2], th->alt_m[3],
            th->Xoff, th->Voff, th->bullock_spin, th->b_to_a, th->c_to_a,
            th->A[0], th->A[1], th->A[2], th->b_to_a2, th->c_to_a2, th->A2[0],
            th->A2[1], th->A2[2], th->kin_to_pot, th->m_pe_b, th->m_pe_d,
            th->halfmass_radius
#ifdef OUTPUT_RVMAX
	    ,th->rvmax
#endif
#ifdef OUTPUT_NFW_CHI2
	    ,th->chi2
#endif
#ifdef OUTPUT_INTERMEDIATE_AXIS
            ,
            th->A_I[0], th->A_I[1], th->A_I[2],
            th->A2_I[0], th->A2_I[1], th->A2_I[2]
#endif
#ifdef OUTPUT_INERTIA_TENSOR
            ,
            th->inertia_tensor[0], th->inertia_tensor[1], th->inertia_tensor[2],
            th->inertia_tensor[3], th->inertia_tensor[4], th->inertia_tensor[5],
            th->inertia_tensor2[0], th->inertia_tensor2[1], th->inertia_tensor2[2],
            th->inertia_tensor2[3], th->inertia_tensor2[4], th->inertia_tensor2[5]
#endif
	   );
    }
    *cat_length    = chars;
    *header_length = hchars;
    return cat;
}



#include <mpi.h>

void output_merger_catalog(int64_t snap, int64_t chunk, int64_t location,
                           int64_t length, char *cat) {
  /* File describing the header already exists. */
  char buffer[1024];
  get_outlist_filename(buffer, 1024, snap, chunk);

  if (OUTLIST_PARALLEL) {
    FILE *output;
    char  buffer[1024];
    get_outlist_filename(buffer, 1024, snap, chunk);
    output = check_fopen(buffer, "r+");
    check_lseek(fileno(output), location,
                SEEK_SET);                    // Increases length as needed
    check_fseeko(output, location, SEEK_SET); // Sets stream file pointer
    check_fwrite(cat, 1, length, output);
    fclose(output);
  }
  else{

    MPI_File   mfh;
    MPI_Info   info;
    MPI_Status mst;

    MPI_Info_create(&info);
    MPI_File_open(MPI_COMM_WORLD, buffer, MPI_MODE_WRONLY, info, &mfh);
    MPI_File_set_view(mfh, (MPI_Offset)location, MPI_CHAR, MPI_CHAR, "native", info);
    MPI_File_write_all(mfh, &cat[0], length, MPI_CHAR, &mst);

    MPI_File_close(&mfh);
    MPI_Info_free(&info);
  }

  free(cat);


}

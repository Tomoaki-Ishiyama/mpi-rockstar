#define VMAX_BINS 50
#include "io/io_generic.h"

float max_halo_radius(struct halo *h) {
    if (LIGHTCONE)
        lightcone_set_scale(h->pos);
    float thresh_dens = particle_thresh_dens[min_dens_index] * PARTICLE_MASS;
    float m           = (min_dens_index) ? h->alt_m[min_dens_index - 1] : h->m;
    return (cbrt((3.0 / (4.0 * M_PI)) * m / thresh_dens) * 1e3);
}

void _populate_mass_bins(struct halo *h, struct halo *cur_h, int64_t *bins,
                         int64_t num_bins, float r_scale, int64_t children,
                         const struct HaloInfo *haloinfo) {
    int64_t i, j, child, first_child, bin;
    float   ds, dx;
    for (i = 0; i < cur_h->num_p; i++) {
        for (ds = 0, j = 0; j < 3; j++) {
            dx = h->pos[j] - copies[cur_h->p_start + i].pos[j];
            ds += dx * dx;
        }
        bin = sqrt(ds) * r_scale;
        if (bin >= num_bins)
            continue;
        bins[bin]++;
    }

    if (!children)
        return;
    first_child = child = haloinfo->extra_info[cur_h - haloinfo->halos].child;
    while (child > -1) {
        _populate_mass_bins(h, haloinfo->halos + child, bins, num_bins, r_scale,
                            1, haloinfo);
        child = haloinfo->extra_info[child].next_cochild;
        assert(child != first_child);
    }
}

float _estimate_vmax(int64_t *bins, int64_t num_bins, float r_scale) {
    int64_t i, tp = 0;
    float   vmax = 0, vcirc, r;
    for (i = 0; i < num_bins; i++) {
        r = (i + 1.0) / r_scale;
        if (r < FORCE_RES)
            r = FORCE_RES;
        tp += bins[i];
        vcirc = tp / r;
        if (vcirc > vmax)
            vmax = vcirc;
    }

    float scale_now = LIGHTCONE ? scale_dx : SCALE_NOW;

    return sqrt(Gc * vmax * PARTICLE_MASS / scale_now);
}

void estimate_vmax(struct halo *h, const struct HaloInfo *haloinfo) {
    int64_t bins[VMAX_BINS] = {0};
    h->vmax = h->vmax_r = 0;
    if (!(h->child_r > 0))
        return;
    float r_scale = ((double)VMAX_BINS) / h->child_r;
    float scale_now = LIGHTCONE ? scale_dx : SCALE_NOW;

    _populate_mass_bins(h, h, bins, VMAX_BINS, r_scale, 0, haloinfo);
    h->vmax   = _estimate_vmax(bins, VMAX_BINS, r_scale);
    h->vmax_r = sqrt(scale_now) * _estimate_vmax(bins, VMAX_BINS, r_scale) *
                dynamical_time;
}

void calc_basic_halo_props(struct halo *h, const struct HaloInfo *haloinfo) {
    int64_t j, k;
    double  pos[6] = {0}, pos2[6] = {0}, x;
    double  pos_err, vel_err;
    h->r = h->vrms = 0;
    for (j = 0; j < h->num_p; j++)
        for (k = 0; k < 6; k++)
            pos[k] += copies[h->p_start + j].pos[k];

    for (k = 0; k < 6; k++)
        pos[k] /= (double)h->num_p;

    for (j = 0; j < h->num_p; j++)
        for (k = 0; k < 6; k++) {
            x = copies[h->p_start + j].pos[k] - pos[k];
            pos2[k] += x * x;
        }

    for (k = 0; k < 6; k++) {
        if (k < 3)
            h->r += pos2[k] / (double)h->num_p;
        else
            h->vrms += pos2[k] / (double)h->num_p;
    }

    pos_err = h->r / (double)h->num_p;
    vel_err = h->vrms / (double)h->num_p;

    if ((!h->min_pos_err) || (h->min_pos_err > pos_err)) {
        h->min_pos_err = pos_err;
        h->n_core      = h->num_p;
        for (k = 0; k < 3; k++)
            h->pos[k] = pos[k];
    }

    if ((!h->min_vel_err) || (h->min_vel_err > vel_err)) {
        h->min_vel_err = vel_err;
        for (k = 3; k < 6; k++)
            h->pos[k] = pos[k];
    }
    for (k = 3; k < 6; k++)
        h->bulkvel[k - 3] = pos[k];

    h->m = h->num_p;
    if (!h->num_child_particles)
        h->num_child_particles = h->num_p;

    h->r       = cbrt(h->num_p / ((4.0 * M_PI / 3.0) * particle_rvir_dens));
    h->child_r = cbrt(h->num_child_particles /
                      ((4.0 * M_PI / 3.0) * particle_rvir_dens));
    estimate_vmax(h, haloinfo);
    if (h->vmax_r)
        h->r = h->vmax_r;
    h->vrms = sqrt(h->vrms);
}

void add_ang_mom(double L[3], float c[6], float pos[6]) {
    // L = r x p;
#define cross(a, x, y, s) L[a] s(pos[x] - c[x]) * (pos[y + 3] - c[y + 3])
    cross(0, 1, 2, +=);
    cross(0, 2, 1, -=);
    cross(1, 2, 0, +=);
    cross(1, 0, 2, -=);
    cross(2, 0, 1, +=);
    cross(2, 1, 0, -=);
#undef cross
}

void _calc_num_child_particles(struct halo           *h,
                               const struct HaloInfo *haloinfo) {
    int64_t child, first_child;

    if (h->num_child_particles)
        return;
    h->num_child_particles = h->num_p;
    first_child = child = haloinfo->extra_info[h - haloinfo->halos].child;
    while (child > -1) {
        _calc_num_child_particles(haloinfo->halos + child, haloinfo);
        h->num_child_particles += haloinfo->halos[child].num_child_particles;
        child = haloinfo->extra_info[child].next_cochild;
        assert(child != first_child);
    }
}

void calc_num_child_particles(int64_t                h_start,
                              const struct HaloInfo *haloinfo) {
    int64_t i;
    for (i = h_start; i < haloinfo->num_halos; i++)
        haloinfo->halos[i].num_child_particles = 0;
    for (i = h_start; i < haloinfo->num_halos; i++)
        if (!haloinfo->halos[i].num_child_particles)
            _calc_num_child_particles(haloinfo->halos + i, haloinfo);
}

void calculate_corevel(struct halo *h, struct potential *po, int64_t total_p) {
    // Assumes po is already sorted.
    int64_t i, j;
    double  vel[3] = {0};
    int64_t core_max, rvir_max;
    double  var[3] = {0}, thisvar, bestvar = 0;
    double  rvir_thresh = particle_rvir_dens * (4.0 * M_PI / 3.0);

    for (j = total_p - 1; j >= 0; j--)
        if (j * j >
            (po[j].r2 * po[j].r2 * po[j].r2) * (rvir_thresh * rvir_thresh))
            break;
    rvir_max = j;
    if (rvir_max < 1)
        return;
    for (j = total_p - 1; j >= 0; j--)
        if (po[j].r2 * 100.0 < po[rvir_max].r2)
            break;
    core_max = j;

    if (core_max < 100)
        core_max = 100;

    for (i = 0; i < rvir_max; i++) {
        for (j = 0; j < 3; j++) {
            double delta = po[i].pos[j + 3] - vel[j];
            vel[j] += delta / ((double)(i + 1));
            var[j] += delta * (po[i].pos[j + 3] - vel[j]);
        }
        thisvar = (var[0] + var[1] + var[2]);
        if ((i < 10) || (thisvar < bestvar * (i - 3) * i)) {
            if (i > 3)
                bestvar = thisvar / (double)((i - 3) * i);
            else
                bestvar = 0;
            if (i < core_max) {
                h->n_core      = i;
                h->min_vel_err = bestvar;
                for (j = 0; j < 3; j++)
                    h->corevel[j] = vel[j];
            }
            for (j = 0; j < 3; j++)
                h->bulkvel[j] = vel[j];
            h->min_bulkvel_err = bestvar;
        }
    }

    for (j = 0; j < 3; j++)
        h->pos[j + 3] = h->corevel[j];
}

void calc_shape(struct halo *h, int64_t total_p, int64_t bound) {
    int64_t i, j, k, l, iter = SHAPE_ITERATIONS, analyze_p = 0, a, b, c;
    float   b_to_a, c_to_a, min_r            = FORCE_RES * FORCE_RES;
    double  mass_t[3][3], orth[3][3], eig[3] = {0}, r = 0, dr, dr2, weight = 0;
    h->b_to_a = h->c_to_a = 0;
    memset(h->A, 0, sizeof(float) * 3);

    if (!(h->r > 0))
        return;
    min_r *= 1e6 / (h->r * h->r);
    for (j = 0; j < total_p; j++) {
        if (bound && (po[j].pe < po[j].ke))
            continue;
        analyze_p++;
    }
    if (analyze_p < 3 || !(h->r > 0))
        return;
    if (analyze_p < iter)
        iter = analyze_p;

    for (i = 0; i < 3; i++) {
        memset(orth[i], 0, sizeof(double) * 3);
        orth[i][i] = 1;
        eig[i]     = (h->r * h->r) * 1e-6;
    }
    for (i = 0; i < iter; i++) {
        for (k = 0; k < 3; k++)
            memset(mass_t[k], 0, sizeof(double) * 3);
        weight = 0;
        for (j = 0; j < total_p; j++) {
            if (bound && (po[j].pe < po[j].ke))
                continue;
            r = 0;
            for (k = 0; k < 3; k++) {
                for (dr = 0, l = 0; l < 3; l++) {
                    dr += orth[k][l] * (po[j].pos[l] - h->pos[l]);
                }
                r += dr * dr / eig[k];
            }
            if (r < min_r)
                r = min_r;
            if (!(r > 0 && r <= 1))
                continue;
            double tw = (WEIGHTED_SHAPES) ? 1.0 / r : 1.0;
            weight += tw;
            for (k = 0; k < 3; k++) {
                dr = po[j].pos[k] - h->pos[k];
                mass_t[k][k] += dr * dr * tw;
                for (l = 0; l < k; l++) {
                    dr2 = po[j].pos[l] - h->pos[l];
                    mass_t[k][l] += dr2 * dr * tw;
                    mass_t[l][k] = mass_t[k][l];
                }
            }
        }

        if (!weight)
            return;
        for (k = 0; k < 3; k++)
            for (l = 0; l < 3; l++)
                mass_t[k][l] /= (double)weight;

#ifdef OUTPUT_INERTIA_TENSOR
	double mass_t2[3][3];
	memcpy( mass_t2, mass_t, sizeof(double)*9);
#endif

        jacobi_decompose(mass_t, eig, orth);
        a = 0;
        b = 1;
        c = 2;
        if (eig[1] > eig[0]) {
            b = 0;
            a = 1;
        }
        if (eig[2] > eig[b]) {
            c = b;
            b = 2;
        }
        if (eig[b] > eig[a]) {
            int64_t t = a;
            a         = b;
            b         = t;
        }
        if (!eig[a] || !eig[b] || !eig[c])
            return;
        b_to_a = sqrt(eig[b] / eig[a]);
        c_to_a = sqrt(eig[c] / eig[a]);
        if ((fabs(b_to_a - h->b_to_a) < 0.01 * h->b_to_a) &&
            (fabs(c_to_a - h->c_to_a) < 0.01 * h->c_to_a))
            return;
        h->b_to_a = (b_to_a > 0) ? b_to_a : 0;
        h->c_to_a = (c_to_a > 0) ? c_to_a : 0;
        r         = sqrt(eig[a]);
        for (k = 0; k < 3; k++) {
            h->A[k] = 1e3 * r * orth[a][k];
#ifdef OUTPUT_INTERMEDIATE_AXIS
	    h->A_I[k] = 1e3*r*orth[b][k];
#endif
           eig[k] *= (h->r * h->r * 1e-6) / (r * r);
        }
#ifdef OUTPUT_INERTIA_TENSOR
	h->inertia_tensor[0] = mass_t2[0][0];
	h->inertia_tensor[1] = mass_t2[1][1];
	h->inertia_tensor[2] = mass_t2[2][2];
	h->inertia_tensor[3] = mass_t2[0][1];
	h->inertia_tensor[4] = mass_t2[1][2];
	h->inertia_tensor[5] = mass_t2[2][0];
#endif
    }
}

float estimate_total_energy(int64_t total_p, float *energy_ratio) {
    int64_t i;
    double  phi = 0, total_phi = 0, ke = 0, r;
    for (i = total_p - 1; i >= 0; i--) {
        if (po[i].pe > po[i].ke) {
            ke += po[i].ke;
            r = sqrt(po[i].r2);
            if (r < FORCE_RES)
                r = FORCE_RES;
            total_phi += PARTICLE_MASS * i / r + phi;
            phi += PARTICLE_MASS / r;
        }
    }
    total_phi /= 2.0; // U = sum pe/2
    *energy_ratio = 0;
    if (total_phi)
        *energy_ratio = (ke / total_phi);

    float scale_now = LIGHTCONE ? scale_dx : SCALE_NOW;

    return ((ke - total_phi) * PARTICLE_MASS * Gc / scale_now);
}

void _calc_pseudo_evolution_masses(struct halo *h, int64_t total_p,
                                   int64_t bound) {
    int64_t j, num_part = 0, num_part_pe_d = 0;
    double  r, r32, max_pe_b               = 0;

    // Typical: R_s*4.0; Minimum thresh: R_halo/5.0
    double r_pe_d = h->rs * 4.0;
    // double r_pe_b = 0;
    if (r_pe_d < h->r / 5.0)
        r_pe_d = h->r / 5.0;
    r_pe_d *= 1e-3;
    for (j = 0; j < total_p; j++) {
        if (bound && (po[j].pe < po[j].ke))
            continue;
        num_part++;
        r = sqrt(po[j].r2);

        r32 = sqrt(r);
        r32 = r32 * r32 * r32; // r^(3/2)
        if ((double)(num_part * num_part) / r32 > max_pe_b) {
            max_pe_b = (double)(num_part * num_part) / r32;
            // r_pe_b = r;
        }

        if (r < r_pe_d)
            num_part_pe_d = num_part;
    }
    //  if (h->m > 1e13) fprintf(stderr, "%f %f\n", r_pe_b*1e3, h->rs);
    h->m_pe_d = num_part_pe_d * PARTICLE_MASS;
    h->m_pe_b = PARTICLE_MASS * pow(max_pe_b, 2.0 / 3.0) /
                cbrt(4.0 * M_PI * particle_rvir_dens_z0 / 3.0);
}

void _calc_additional_halo_props(struct halo *h, int64_t total_p,
                                 int64_t bound) {
    int64_t j, k, part_mdelta = 0, num_part = 0, np_alt[4] = {0}, np_vir = 0,
                  dens_tot = 0, parts_avgd = 0, num_part_half = 0;

    float scale_now = LIGHTCONE ? scale_dx : SCALE_NOW;

    double dens_thresh = particle_thresh_dens[0] * (4.0 * M_PI / 3.0);
    double d1          = particle_thresh_dens[1] * (4.0 * M_PI / 3.0);
    double d2          = particle_thresh_dens[2] * (4.0 * M_PI / 3.0);
    double d3          = particle_thresh_dens[3] * (4.0 * M_PI / 3.0);
    double d4          = particle_thresh_dens[4] * (4.0 * M_PI / 3.0);
    double rvir_thresh = particle_rvir_dens * (4.0 * M_PI / 3.0);
    double vmax_conv   = PARTICLE_MASS / scale_now;
    double r, circ_v, vmax = 0, rvmax = 0, L[3] = {0}, Jh, m = 0, ds;
    double vrms[3] = {0}, xavg[3] = {0}, vavg[3] = {0}, mdiff;
    double cur_dens, rvir, mvir;

    for (j = 0; j < total_p; j++) {
        if (bound && (po[j].pe < po[j].ke))
            continue;
        num_part++;
        r = sqrt(po[j].r2);
        if (r < FORCE_RES)
            r = FORCE_RES;
        cur_dens = ((double)num_part / (r * r * r));

        if (cur_dens > dens_thresh) {
            part_mdelta = num_part;
            dens_tot    = j;
        }

        if (cur_dens > d1)
            np_alt[0] = num_part;
        if (cur_dens > d2)
            np_alt[1] = num_part;
        if (cur_dens > d3)
            np_alt[2] = num_part;
        if (cur_dens > d4)
            np_alt[3] = num_part;

        if (cur_dens > rvir_thresh) {
            circ_v = (double)num_part / r;
            np_vir = num_part;
            if (part_mdelta && circ_v > vmax) {
                vmax  = circ_v;
                rvmax = r;
            }
        }
    }

    for (j = 0; j < dens_tot; j++) {
        if (bound && (po[j].pe < po[j].ke))
            continue;
        add_ang_mom(L, h->pos, po[j].pos);
        parts_avgd++;
        if (parts_avgd * 2 >= part_mdelta && !num_part_half)
            num_part_half = j;
        for (k = 0; k < 3; k++) { // Calculate velocity and position averages
            xavg[k] += (po[j].pos[k] - xavg[k]) / ((double)parts_avgd);
            mdiff = po[j].pos[k + 3] - vavg[k];
            vavg[k] += mdiff / (double)parts_avgd;
            vrms[k] += mdiff * (po[j].pos[k + 3] - vavg[k]);
        }
    }

    m = part_mdelta * PARTICLE_MASS;
    if (!bound)
        h->m = m;
    else
        h->mgrav = m;
    for (k = 0; k < 3; k++)
        vrms[k] = (parts_avgd > 0) ? (vrms[k] / parts_avgd) : 0;
    if ((!bound) == (!BOUND_PROPS)) { // Works even if BOUND_PROPS > 1
        h->Xoff = h->Voff = 0;
        for (k = 0; k < 3; k++) {
            ds = xavg[k] - h->pos[k];
            h->Xoff += ds * ds;
            ds = vavg[k] - h->pos[k + 3];
            h->Voff += ds * ds;
        }
        h->alt_m[0]        = np_alt[0] * PARTICLE_MASS;
        h->alt_m[1]        = np_alt[1] * PARTICLE_MASS;
        h->alt_m[2]        = np_alt[2] * PARTICLE_MASS;
        h->alt_m[3]        = np_alt[3] * PARTICLE_MASS;
        h->Xoff            = sqrt(h->Xoff) * 1e3;
        h->Voff            = sqrt(h->Voff);
        h->vrms            = sqrt(vrms[0] + vrms[1] + vrms[2]);
        h->vmax            = VMAX_CONST * sqrt(vmax * vmax_conv);
        h->rvmax           = rvmax * 1e3;
        h->halfmass_radius = sqrt(po[num_part_half].r2) * 1e3;
        h->r =
            cbrt((3.0 / (4.0 * M_PI)) * np_alt[2] / particle_thresh_dens[3]) *
            1e3;
        calc_shape(h, np_alt[2], bound);
        h->b_to_a2 = h->b_to_a;
        h->c_to_a2 = h->c_to_a;
        memcpy(h->A2, h->A, sizeof(float) * 3);
#ifdef OUTPUT_INERTIA_TENSOR
	memcpy(h->inertia_tensor2, h->inertia_tensor, sizeof(float)*6);
#endif
#ifdef OUTPUT_INTERMEDIATE_AXIS
	memcpy(h->A2_I, h->A_I, sizeof(float)*3);
#endif
        h->r =
            cbrt((3.0 / (4.0 * M_PI)) * part_mdelta / particle_thresh_dens[0]) *
            1e3;
        calc_shape(h, dens_tot, bound);

        rvir = cbrt((3.0 / (4.0 * M_PI)) * np_vir / particle_rvir_dens) * 1e3;
        mvir = np_vir * PARTICLE_MASS;
        calc_scale_radius(h, m, h->r, h->vmax, h->rvmax, scale_now, po,
                          dens_tot, bound);
        for (j = 0; j < 3; j++)
            h->J[j] = PARTICLE_MASS * scale_now * L[j];
        h->energy = estimate_total_energy(dens_tot, &(h->kin_to_pot));
        Jh        = PARTICLE_MASS * scale_now *
             sqrt(L[0] * L[0] + L[1] * L[1] + L[2] * L[2]);
        h->spin =
            (m > 0) ? (Jh * sqrt(fabs(h->energy)) / (Gc * pow(m, 2.5))) : 0;
        h->bullock_spin =
            (m > 0)
                ? (Jh / (mvir * sqrt(2.0 * Gc * mvir * rvir * scale_now / 1e3)))
                : 0;
        _calc_pseudo_evolution_masses(h, total_p, bound);
    }
}

// Assumes center + velocity already calculated.
void calc_additional_halo_props(struct halo           *h,
                                const struct HaloInfo *haloinfo) {
    int64_t j, total_p;
    double  dens_thresh;

    if (LIGHTCONE)
        lightcone_set_scale(h->pos);

    float scale_now = LIGHTCONE ? scale_dx : SCALE_NOW;

    dens_thresh = particle_thresh_dens[0] * (4.0 * M_PI / 3.0);
    if (h->num_p < 1)
        return;
    total_p = calc_particle_radii(h, h, h->pos, 0, 0, 0, haloinfo);
    if (BOUND_OUT_TO_HALO_EDGE) {
        qsort(po, total_p, sizeof(struct potential), dist_compare);
        for (j = total_p - 1; j >= 0; j--)
            if (j * j / (po[j].r2 * po[j].r2 * po[j].r2) >
                dens_thresh * dens_thresh)
                break;
        if (total_p)
            total_p = j + 1;
    }

    if (total_p > 1)
        compute_potential(po, total_p);
    for (j = 0; j < total_p; j++)
        if (po[j].ke < 0) {
            total_p--;
            po[j] = po[total_p];
            j--;
        }
    qsort(po, total_p, sizeof(struct potential), dist_compare);
    calculate_corevel(h, po, total_p);
    if (haloinfo->extra_info[h - haloinfo->halos].sub_of > -1)
        compute_kinetic_energy(po, total_p, h->corevel, h->pos, scale_now);
    else
        compute_kinetic_energy(po, total_p, h->bulkvel, h->pos, scale_now);

    _calc_additional_halo_props(h, total_p, 0);
    _calc_additional_halo_props(h, total_p, 1);
    if (analyze_halo_generic != NULL)
        analyze_halo_generic(h, po, total_p);
}

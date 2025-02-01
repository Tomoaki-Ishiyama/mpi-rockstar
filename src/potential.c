#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>
#include "config_vars.h"
#include "universal_constants.h"
#include "potential.h"
#include "hubble.h"

#define POTENTIAL_ERR_TOL 1.0
#define POTENTIAL_USE_BH  1
#ifndef POTENTIAL_HALT_AFTER_BOUND
#define POTENTIAL_HALT_AFTER_BOUND 0
#endif /* !def POTENTIAL_HALT_AFTER_BOUND */

// Note: If POTENTIAL_HALT_AFTER_BOUND is set, then algorithm does not compute
// full potential;
//  it does not need to compute additional potential terms for particles that
//  are bound---if they are bound, they can only become *more* bound by adding
//  more potential terms.

double _distance2(float *p1, float *p2) {
    double dx, r2 = 0;
    for (int64_t k = 0; k < 3; k++) {
        dx = p1[k] - p2[k];
        r2 += dx * dx;
    }
    return (r2);
}

double inv_distance(float *p1, float *p2) {
    double r = sqrt(_distance2(p1, p2));
    if (r < FORCE_RES)
        r = FORCE_RES;
    return (1.0 / r);
}

void _compute_direct_potential(struct potential *po, int64_t num_po) {
    int64_t i, j;
    double  dpo;
    for (i = 0; i < num_po; i++)
        for (j = i + 1; j < num_po; j++) {
            dpo = PARTICLE_MASS * inv_distance(po[i].pos, po[j].pos);
            po[i].pe += dpo;
            po[j].pe += dpo;
        }
}

void _compute_indirect_potential(struct potential *po, int64_t num_po,
                                 struct potential *po2, int64_t num_po2) {
    int64_t i, j;
    for (i = 0; i < num_po; i++)
        for (j = 0; j < num_po2; j++)
            po[i].pe += PARTICLE_MASS * inv_distance(po[i].pos, po2[j].pos);
}

#define POINTS_PER_LEAF  10
#define FAST3TREE_TYPE   struct potential
#define FAST3TREE_PREFIX POTENTIAL
#define FAST3TREE_EXTRA_INFO                                                   \
    float   mass_center[3];                                                    \
    float   m, dmin;                                                           \
    int64_t num_unbound;
#include "fast3tree.c"

struct fast3tree         *p_tree = NULL;
struct fast3tree_results *p_res  = NULL;

#pragma omp threadprivate(p_tree, p_res)

void _compute_dmin(struct tree3_node *n) {
    double sum_x2 = 0, bmax = 0, dx;
    for (int64_t i = 0; i < n->num_points; i++) {
        dx = _distance2(n->mass_center, n->points[i].pos);
        if (bmax < dx)
            bmax = dx;
        sum_x2 += dx;
    }
    bmax = sqrt(bmax) / 2.0;
    n->dmin =
        bmax + sqrt(bmax * bmax + sum_x2 / (n->num_points * POTENTIAL_ERR_TOL));
}

void _ignore_unbound(struct tree3_node *n) {
    struct potential tmp;
    assert(n->div_dim < 0);
    for (int64_t i = 0; i < n->num_unbound; i++) {
        if (n->points[i].pe > n->points[i].ke) {
            n->num_unbound--;
            tmp                       = n->points[n->num_unbound];
            n->points[n->num_unbound] = n->points[i];
            n->points[i]              = tmp;
            i--;
        }
    }
}

void _compute_mass_centers(struct tree3_node *n) {
    int64_t i, j;
    double  pos[3] = {0};
    if (n->div_dim < 0) { // Leaf node
        n->m = PARTICLE_MASS * n->num_points;
        for (i = 0; i < n->num_points; i++)
            for (j = 0; j < 3; j++)
                pos[j] += n->points[i].pos[j];
        for (j = 0; j < 3; j++)
            n->mass_center[j] = pos[j] ? pos[j] / (double)n->num_points : 0;
        _compute_dmin(n);
        n->num_unbound = n->num_points;
#if POTENTIAL_HALT_AFTER_BOUND
        _ignore_unbound(n);
#endif
        return;
    }
    _compute_mass_centers(n->left);
    _compute_mass_centers(n->right);
    n->m = n->left->m + n->right->m;
    for (j = 0; j < 3; j++)
        n->mass_center[j] = n->m ? ((n->right->mass_center[j] * n->right->m +
                                     n->left->mass_center[j] * n->left->m) /
                                    n->m)
                                 : 0;
    _compute_dmin(n);
}

float _monopole_acceptable(struct tree3_node *n, struct tree3_node *n2) {
    float r2 = _distance2(n->mass_center, n2->mass_center);
    if (r2 > n2->dmin * n2->dmin)
        return 1;
    return 0;
}

void _compute_monopole_potentials(struct tree3_node *n, struct tree3_node *n2) {
    int64_t i;
    float   r;
    assert(n->div_dim < 0);

#if POTENTIAL_HALT_AFTER_BOUND
    _ignore_unbound(n);
    if (!n->num_unbound)
        return;
#endif

    if (n == n2) {
        if (n->num_points > (2 * POINTS_PER_LEAF))
            return; // Degenerate node
        _compute_direct_potential(n->points, n->num_points);
        return;
    }
    for (i = 0; i < 3; i++)
        if (n->min[i] < n2->min[i] || n->max[i] > n2->max[i])
            break;
    if (i == 3 || !_monopole_acceptable(n, n2)) {
        if (n2->div_dim < 0) {
            assert(i < 3);
            _compute_indirect_potential(n->points, n->num_unbound, n2->points,
                                        n2->num_points);
        } else {
            _compute_monopole_potentials(n, n2->left);
            _compute_monopole_potentials(n, n2->right);
        }
    } else {
        for (i = 0; i < n->num_unbound; i++) {
            r = sqrt(_distance2(n->points[i].pos, n2->mass_center));
            n->points[i].pe += n2->m / r;
        }
    }
}

void _compute_barnes_hut_potential(struct potential *po, int64_t num_po) {
    if (!p_tree)
        p_tree = fast3tree_init(0, NULL);
    if (!p_res)
        p_res = fast3tree_results_init();

    fast3tree_rebuild(p_tree, num_po, po);
    _compute_mass_centers(p_tree->root);
    for (int64_t i = 0; i < p_tree->num_nodes; i++)
        if (p_tree->root[i].div_dim < 0)
            _compute_monopole_potentials(p_tree->root + i, p_tree->root);
}

void compute_potential(struct potential *po, int64_t num_po) {
    for (int64_t i = 0; i < num_po; i++) {
        po[i].pe = 0;
    }
#if POTENTIAL_USE_BH
    _compute_barnes_hut_potential(po, num_po);
#else
    _compute_direct_potential(po, num_po);
#endif /* POTENTIAL_USE_BH */
}

void compute_kinetic_energy(struct potential *po, int64_t num_po,
                            float *vel_cen, float *pos_cen, float scale_now) {
    int64_t i, j;
    double  dv = 0, conv_const = 0.5 * scale_now / Gc;
    double  z1     = 1.0 / scale_now;
    double  hubble = 100.0 * hubble_scaling(z1 - 1.0);
    for (i = 0; i < num_po; i++) {
        if (po[i].ke < 0)
            continue;
        po[i].ke = dv = 0;
        for (j = 0; j < 3; j++) {
            dv = po[i].pos[j + 3] - vel_cen[j] +
                 hubble * scale_now * (po[i].pos[j] - pos_cen[j]);
            po[i].ke += dv * dv;
        }
        po[i].ke *= conv_const;
    }
}

#ifndef _POTENTIAL_H_
#define _POTENTIAL_H_

#include <stdint.h>
#define POTENTIAL_DONT_CALCULATE_FLAG 1

struct potential {
    float   pos[6], r2;
    double  pe;
    float   ke;
    int32_t flags;

    /*The following fields are not included for the main halo finder. */
#ifdef POTENTIAL_COMPARISON
    float pe2, ke2, pe3, ke3;
    float v, r;
#endif /* POTENTIAL_COMPARISON */
#ifdef CALC_POTENTIALS
    int64_t id, hid;
#endif /* CALC_POTENTIALS */
};

void compute_kinetic_energy(struct potential *po, int64_t num_po,
                            float *vel_cen, float *pos_cen, float scale_now);
void compute_potential(struct potential *po, int64_t num_po);

#endif /* _POTENTIAL_H_ */

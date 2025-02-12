---
title: "MPI-Rockstar: a Hybrid MPI and OpenMP Parallel Implementation of the Rockstar Halo finder"
tags:
  - cosmology
  - dark matter
  - massively parallel
authors:
  - name: Tomoyuki Tokuue 
    affiliation: "1"
  - name: Tomoaki Ishiyama
    orcid: 0000-0002-5316-9171
    affiliation: "2"
  - name: Ken Osato
    orcid: 0000-0002-7934-2569
    affiliation: "3,4,5"
  - name: Satoshi Tanaka
    orcid: 0000-0003-2442-8784
    affiliation: "6"
  - name: Peter Behroozi
    orcid: 0000-0002-2517-6446
    affiliation: "7"
affiliations:
 - name: Department of Applied and Cognitive Informatics, Division of Mathematics and Informatics, Graduate School of Science and Engineering, Chiba University, Chiba 263-8522, Japan
   index: 1
 - name: Digital Transformation Enhancement Council, Chiba University, Chiba 263-8522, Japan
   index: 2
 - name: Center for Frontier Science, Chiba University, Chiba 263-8522, Japan
   index: 3
 - name: Department of Physics, Graduate School of Science, Chiba University, Chiba 263-8522, Japan
   index: 4
 - name: Kavli Institute for the Physics and Mathematics of the Universe (WPI), The University of Tokyo, Kashiwa, Chiba 277-8583, Japan
   index: 5
 - name: Centre for Gravitational Physics and Quantum Information, Yukawa Institute for Theoretical Physics, Kyoto University, Kyoto 606-8502, Japan
   index: 6
 - name: Department of Astronomy, University of Arizona, AZ 85721, USA
   index: 7
date: 16 December 2024
bibliography: paper.bib

---

# Summary

According to the concordance cosmological theory, structure
formation and evolution proceeds hierarchically in the Universe.
Smaller-scale dark matter structures gravitationally collapse first
everywhere in the Universe, then merge into larger-scale structures.
Such dense gravitationally bound structures of dark matter are called
halos.  Halos can host smaller halos, so-called subhalos (or
substructures).  Cosmological $N$-body simulations are vital in
understanding the formation and evolution of halos and subhalos.
Halo/subhalo finders are a post-processing step to identify those dense
structures in the particle dataset of cosmological $N$-body
simulations.

MPI-Rockstar is a massively parallel halo finder
based on the [Rockstar](https://bitbucket.org/gfcstanford/rockstar/)
phase-space temporal halo finder code [@Behroozi2013], which is one of
the most extensively used halo finding codes.  Compared to the
original code, parallelized by a primitive socket communication
library, we parallelized it in a hybrid way using MPI and OpenMP,
which is suitable for analysis on the hybrid shared and distributed memory
environments of modern supercomputers.  This implementation can easily
handle the analysis of more than a trillion particles on more than
100,000 parallel processes, enabling the production of a huge dataset
for the next generation of cosmological surveys.


# Statement of need 

Owing to the advance of supercomputing power and highly scalable
parallel gravitational $N$-body codes [@Ishiyama2009b; @Ishiyama2012;
@Potter2017; @Wang2018; @Garrison2021], the number of particles in
recent massive cosmological simulations exceeds a trillion
[@Potter2017; @Ishiyama2021; @Wang2022], posing significant challenges
for halo finding.  Several halo finding algorithms have been
suggested [@Knebe2013], and some of their implementation are publicly
available [@Knollmann2009; @Behroozi2013; @Elahi2019; @Springel2021].
The computational performance of these implementations differs substantially,
and they have not been uniformly tested yet on the large-scale hybrid shared and distributed
memory environments of modern supercomputers.  MPI-Rockstar addresses
these issues and is designed to run on more than 100,000 parallel
processes in a hybrid way using MPI and OpenMP.  As new functions to
the original Rockstar code, MPI-Rockstar supports HDF5 as an output format and can
output additional halo properties such as the inertia tensor.

# Acknowledgements 

This work has been supported by IAAR Research Support Program in Chiba
University Japan, MEXT/JSPS KAKENHI (Grant Number JP23H04002), MEXT as
``Program for Promoting Researches on the Supercomputer Fugaku''
(JPMXP1020230406 and JPMXP1020230407), and JICFuS.  This work was
partially supported by the Sasakawa Scientific Research Grant from The
Japan Science Society.  Numerical computations were carried out on the
supercomputer Fugaku provided by the RIKEN Center for Computational
Science (Project ID: hp230173, and hp240184).


# References
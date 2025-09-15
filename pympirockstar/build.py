import os
from pathlib import Path
from setuptools.command.build_ext import build_ext as _build_ext
from Cython.Build import cythonize


class build_ext(_build_ext):
    """Custom build_ext to cythonize and link against Rockstar."""

    def finalize_options(self):
        super().finalize_options()
        # Transform any Cython sources into C++ code so we can link against
        # the Rockstar library.  ``cythonize`` converts ``.pyx`` files to
        # ``.cpp`` sources; setting ``_needs_stub`` prevents setuptools from
        # expecting stub files on the resulting Extension objects.
        self.extensions = cythonize(self.extensions)
        for ext in self.extensions:
            setattr(ext, "_needs_stub", False)

    def build_extensions(self):
        root_dir = Path(__file__).resolve().parents[1]
        lib_dir = root_dir / "src"
        lib_spec = os.environ.get("ROCKSTAR_LIB", "mpi_rockstar")
        lib_name = Path(lib_spec).stem
        if lib_name.startswith("lib"):
            lib_name = lib_name[3:]
        # Use the MPI C++ compiler if provided so the extension can link
        # against MPI-enabled Rockstar builds.
        mpicxx = os.environ.get("MPICC") or os.environ.get("MPICXX")
        if mpicxx and hasattr(self, "compiler"):
            self.compiler.set_executable("compiler_cxx", mpicxx)
            self.compiler.set_executable("linker_so", mpicxx)

        # If the HDF5-flavoured Rockstar library is requested, ensure we also
        # link against the HDF5 dependencies so symbols like
        # ``H5T_NATIVE_INT_g`` are resolved at import time.  Additional
        # libraries can be provided via the ``HDF5_LIBS`` environment variable
        # (space separated).
        hdf5_libs = []
        if lib_name.endswith("_hdf5"):
            hdf5_libs = os.environ.get("HDF5_LIBS", "hdf5").split()

        for ext in self.extensions:
            ext.include_dirs.append(str(lib_dir))
            ext.library_dirs.append(str(lib_dir))
            ext.libraries.append(lib_name)
            ext.libraries.extend(hdf5_libs)
            ext.language = "c++"
            # The Rockstar library uses OpenMP; ensure the wrapper is compiled
            # and linked with the appropriate OpenMP flags so symbols like
            # ``omp_get_thread_num`` are resolved at runtime.
            ext.extra_compile_args = (ext.extra_compile_args or []) + ["-fopenmp"]
            ext.extra_link_args = (ext.extra_link_args or []) + ["-fopenmp"]
        super().build_extensions()
